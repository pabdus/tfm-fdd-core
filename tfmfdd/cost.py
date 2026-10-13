"""Coste computacional: tiempo de ajuste, tiempo de inferencia por muestra y pico de memoria.

Por que existe. La memoria compara las tres familias de metodos tambien por su
coste, y esa comparacion solo vale si los tres bloques miden igual, en la misma
maquina de referencia y con el mismo procedimiento. Cronometrar a mano en cada
cuaderno da cifras que no se pueden comparar: unas incluyen la carga de datos y
otras no, unas toman la primera ejecucion (que paga la inicializacion de las
bibliotecas) y otras una media, y casi nunca se apunta cuantos hilos tenia la
maquina, que con BLAS o con PyTorch en CPU cambia el tiempo.

Que mide cada herramienta y por que se eligio:

- Tiempo de reloj con `time.perf_counter`. Es monotono y de alta resolucion (en
  Windows, QueryPerformanceCounter). Es el tiempo que espera quien usa el
  modelo, con la espera y la competencia con otros procesos incluidas. Es la
  medida principal de tiempo.
- Tiempo de CPU con `time.process_time`. Suma la CPU de todos los hilos del
  proceso; comparado con el de reloj dice cuanto paralelismo uso el metodo. En
  Windows avanza a saltos del tic del sistema (normalmente 15,625 ms), asi que
  solo se da para pasadas completas, nunca por llamada. `machine_info` mide ese
  salto en la maquina y lo deja registrado.
- Pico de memoria del proceso con psutil (en Windows, el conjunto de trabajo,
  `peak_wset`; en Linux, `VmHWM`, porque `ru_maxrss` se hereda a traves de
  exec; en macOS, `ru_maxrss`, sin comprobar). Ve todo lo que ocupa el proceso,
  incluidas las reservas de PyTorch y de las bibliotecas en C, asi que es la
  medida comparable entre familias. Su trampa: el sistema da el maximo de TODA
  la vida del proceso, no el del tramo medido. Por eso se resta la memoria de
  justo antes de la llamada y se comprueba si la llamada supero el pico
  anterior. Si lo supero, la cifra es exacta; si no, el maximo del tramo esta
  escondido debajo de un pico viejo, y se usa el maximo de un muestreo periodico
  en un hilo aparte, que es una cota inferior. La columna `*_exact` dice cual de
  las dos es. Lo mas limpio es medir en un proceso nuevo (un kernel recien
  arrancado o un script) con solo los datos necesarios cargados.
- Memoria trazada con `tracemalloc`. Ve las reservas de Python y de numpy, pero
  no las de PyTorch ni las de las bibliotecas en C que no avisan (los arboles de
  un RandomForest de scikit-learn, por ejemplo), y frena mucho el codigo Python.
  Es un diagnostico opcional, en una pasada aparte, nunca a la vez que se
  cronometra.

Procedimiento de `measure_cost`, en este orden:

1. Una pasada de memoria del ajuste y otra de la inferencia (con el muestreo
   activo). Van primero porque la primera llamada del proceso es la que mas
   probabilidad tiene de superar el pico de vida y dar una cifra exacta. Estas
   pasadas tambien calientan las bibliotecas. Ojo con la de la inferencia: va
   justo detras del ajuste, que casi siempre deja el pico de vida mas alto, asi
   que en el mismo proceso solo sale exacta si la inferencia necesita mas
   memoria que el ajuste (y medir en un proceso nuevo no lo cambia). Con
   `infer_memory="subprocess"` se mide en un proceso aparte que solo carga el
   modelo y los datos (ver `measure_cost`).
2. Los tiempos de ajuste: `n_warmup_fit` ajustes sin cronometrar y despues
   `n_repeats_fit` cronometrados, cada uno con un modelo NUEVO.
3. Los tiempos de inferencia con el ultimo modelo: `n_warmup_infer` llamadas sin
   cronometrar con el primer lote y despues `n_repeats_infer` pasadas completas
   por lotes de `batch_size` muestras.

Ninguna de estas cantidades (repeticiones, calentamiento, tamano de lote) tiene
valor por defecto: son decisiones del protocolo que se declaran antes de medir.
Las cifras de coste NO se escriben a mano en ningun sitio: salen de esta
funcion, con la configuracion y la maquina al lado.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

# Relojes del modulo. Estan aqui (y no llamados directamente) para que las
# pruebas puedan sustituirlos por un reloj falso y comprobar la aritmetica de
# forma determinista.
_reloj = time.perf_counter
_cpu = time.process_time

#: Bibliotecas cuya version se registra: (clave, modulo, distribucion).
_BIBLIOTECAS = (
    ("numpy", "numpy", "numpy"),
    ("scipy", "scipy", "scipy"),
    ("pandas", "pandas", "pandas"),
    ("sklearn", "sklearn", "scikit-learn"),
    ("torch", "torch", "torch"),
    ("tfmfdd", "tfmfdd", "tfmfdd"),
    ("psutil", "psutil", "psutil"),
    ("threadpoolctl", "threadpoolctl", "threadpoolctl"),
)

#: Campos de `machine_info` que se copian en cada fila de `measure_cost`.
_CAMPOS_MAQUINA_EN_FILA = (
    "platform",
    "cpu_model",
    "cpu_count_logical",
    "cpu_count_physical",
    "ram_total_bytes",
    "python_version",
    "numpy_version",
    "scipy_version",
    "pandas_version",
    "sklearn_version",
    "torch_version",
    "tfmfdd_version",
    "blas_threads",
    "torch_num_threads",
    "torch_num_interop_threads",
)


# ---------------------------------------------------------------- maquina

def _version(modulo: str, distribucion: str) -> str | None:
    """Version de una biblioteca SIN importarla.

    Si el modulo ya esta cargado se toma su `__version__`, que es la que de
    verdad se esta ejecutando. Si no, se lee de los metadatos de instalacion,
    que no cargan el modulo (importante con torch, que no se debe importar solo
    para registrar su version).
    """
    cargado = sys.modules.get(modulo)
    if cargado is not None:
        v = getattr(cargado, "__version__", None)
        if v is not None:
            return str(v)
    try:
        return importlib.metadata.version(distribucion)
    except importlib.metadata.PackageNotFoundError:
        return None


def _modelo_cpu() -> str:
    """Nombre comercial de la CPU, con el mejor metodo de cada sistema."""
    try:
        if sys.platform == "win32":
            import winreg

            clave = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with clave:
                return str(winreg.QueryValueEx(clave, "ProcessorNameString")[0]).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for linea in f:
                    if linea.lower().startswith("model name"):
                        return linea.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            import subprocess

            salida = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if salida.stdout.strip():
                return salida.stdout.strip()
    except Exception:  # noqa: BLE001 - registrar la maquina nunca debe romper la medida
        pass
    return platform.processor() or platform.machine()


def _hilos_blas() -> tuple[list[dict], str | None]:
    """Hilos de BLAS y OpenMP de las bibliotecas YA cargadas (threadpoolctl).

    threadpoolctl lo instala scikit-learn como dependencia. Solo ve las
    bibliotecas que estan cargadas en el momento de la llamada, por eso
    `machine_info` se llama al medir y no al principio del cuaderno.
    """
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return [], None
    try:
        info = threadpool_info()
    except Exception:  # noqa: BLE001
        return [], None
    pools = [
        {k: d.get(k) for k in ("user_api", "internal_api", "num_threads", "version")}
        for d in info
    ]
    resumen = ";".join(f"{d['internal_api']}={d['num_threads']}" for d in pools) or None
    return pools, resumen


def _tic_de_cpu(limite_s: float = 0.25) -> float:
    """Salto minimo observable de `process_time` en esta maquina (en segundos).

    Se mide girando hasta ver dos cambios del reloj de CPU, con un limite de
    tiempo. Devuelve NaN si no se observa ningun salto antes del limite.
    """
    inicio = time.perf_counter()
    ultimo = time.process_time()
    saltos = []
    while len(saltos) < 2 and time.perf_counter() - inicio < limite_s:
        v = time.process_time()
        if v != ultimo:
            saltos.append(v - ultimo)
            ultimo = v
    return float(min(saltos)) if saltos else float("nan")


def machine_info() -> dict:
    """Describe la maquina y el entorno en el que se mide el coste.

    Por que existe. Una cifra de coste sin la maquina al lado no significa
    nada: el mismo ajuste tarda distinto con otra CPU, con otro numero de hilos
    o con otra version de numpy. La memoria promete medir "en una misma maquina
    de referencia", y este registro es lo que permite comprobarlo despues.

    Registra: sistema, CPU (modelo, nucleos logicos y fisicos), RAM, version de
    Python y de las bibliotecas, hilos de BLAS y OpenMP de lo que ya este
    cargado, y, si torch YA esta cargado, su version y sus hilos. No importa
    torch ni scikit-learn: lee las versiones de los metadatos de instalacion.
    No registra el nombre del equipo ni del usuario.

    Llamala en el mismo proceso y justo antes de medir, con el modelo ya
    importado, para que los hilos que registra sean los que se usan.

    Devuelve
    --------
    dict con claves en minusculas y en ingles. `threadpools` es una lista (no
    cabe en una columna); el resto son escalares.
    """
    info: dict = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu_model": _modelo_cpu(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_count_physical": None,
        "ram_total_bytes": None,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }

    try:
        import psutil

        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_bytes"] = int(psutil.virtual_memory().total)
    except ImportError:
        pass

    for clave, modulo, distribucion in _BIBLIOTECAS:
        info[f"{clave}_version"] = _version(modulo, distribucion)

    pools, resumen = _hilos_blas()
    info["threadpools"] = pools
    info["blas_threads"] = resumen

    # torch solo se consulta si el usuario ya lo cargo: importarlo aqui seria
    # caro y, en algunos equipos, imposible.
    torch = sys.modules.get("torch")
    info["torch_loaded"] = torch is not None
    info["torch_num_threads"] = None
    info["torch_num_interop_threads"] = None
    if torch is not None:
        try:
            info["torch_num_threads"] = int(torch.get_num_threads())
            info["torch_num_interop_threads"] = int(torch.get_num_interop_threads())
        except Exception:  # noqa: BLE001
            pass

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        info[f"env_{variable.lower()}"] = os.environ.get(variable)

    info["process_time_tick_s"] = _tic_de_cpu()
    return info


# ---------------------------------------------------------------- memoria

def _importar_psutil(alternativa: str | None = None):
    """Importa psutil o falla con un mensaje que solo ofrece lo que quien llama admite.

    `alternativa` es la salida que tiene la funcion que llama (por ejemplo, en
    `measure`, "pasa process_memory=False"); `measure_cost` no tiene ninguna,
    porque siempre mide la memoria del proceso.
    """
    try:
        import psutil
    except ImportError as exc:
        salida = f", o {alternativa}" if alternativa else ""
        raise ImportError(
            "La memoria del proceso se mide con psutil, que no esta instalado. "
            f"Instalalo en el entorno (es una dependencia de tfmfdd){salida}."
        ) from exc
    return psutil


def _memoria_proceso(proceso) -> tuple[int, int | None, str]:
    """(memoria actual, pico de toda la vida del proceso, nombre de la metrica) en bytes.

    En Windows, `rss` de psutil es el conjunto de trabajo y `peak_wset` su pico.
    En Linux, el pico sale de `VmHWM` de /proc/self/status: `ru_maxrss` se
    hereda a traves de exec, asi que un proceso lanzado desde otro arranca con
    el pico del padre (revision independiente del 12/10). En macOS sale de
    `ru_maxrss` (en bytes; sin comprobar). Todas son memoria fisica residente,
    que es lo comparable entre sistemas.
    """
    info = proceso.memory_info()
    actual = int(info.rss)
    if hasattr(info, "peak_wset"):
        return actual, int(info.peak_wset), "peak_wset"
    if sys.platform.startswith("linux") and proceso.pid == os.getpid():
        # En Linux, ru_maxrss se hereda a traves de exec: un proceso lanzado
        # desde otro arranca con el pico del padre, y el modo "subprocess" de
        # measure_cost nunca daria una cifra exacta. VmHWM es el pico del mapa
        # de memoria de este proceso y empieza de cero con cada exec.
        try:
            with open("/proc/self/status", encoding="ascii") as f:
                for linea in f:
                    if linea.startswith("VmHWM:"):
                        return actual, int(linea.split()[1]) * 1024, "VmHWM"
        except OSError:
            pass
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        factor = 1 if sys.platform == "darwin" else 1024
        return actual, int(maxrss) * factor, "ru_maxrss"
    except ImportError:
        return actual, None, "none"


class _Muestreador(threading.Thread):
    """Hilo que lee la memoria del proceso cada `intervalo` segundos y guarda el maximo.

    Sirve de red cuando la llamada medida no supera el pico de vida del proceso.
    Lo que dure menos que el intervalo se le puede escapar: por eso su maximo es
    una cota inferior del pico del tramo, nunca una cifra exacta.
    """

    def __init__(self, proceso, intervalo: float) -> None:
        super().__init__(daemon=True)
        self._proceso = proceso
        self._intervalo = float(intervalo)
        self._parar = threading.Event()
        self.maximo = 0
        self.n_lecturas = 0

    def _leer(self) -> None:
        rss = int(self._proceso.memory_info().rss)
        if rss > self.maximo:
            self.maximo = rss
        self.n_lecturas += 1

    def run(self) -> None:
        while not self._parar.is_set():
            self._leer()
            time.sleep(self._intervalo)
        self._leer()

    def parar(self) -> None:
        self._parar.set()
        self.join()


def _pico_del_tramo(actual_antes, pico_antes, pico_despues, maximo_muestreado):
    """Pico de memoria del tramo por encima de la memoria de partida, y si es exacto.

    Si el pico de vida subio durante el tramo, el maximo del tramo es justo ese
    pico nuevo: la cifra es exacta. Si no subio, se usa el maximo muestreado
    (cota inferior), y si no hubo muestreo, NaN.
    """
    if pico_antes is not None and pico_despues is not None and pico_despues > pico_antes:
        return float(pico_despues - actual_antes), True
    if maximo_muestreado is not None:
        return float(max(maximo_muestreado, actual_antes) - actual_antes), False
    return float("nan"), False


@contextmanager
def measure(
    *,
    process_memory: bool = True,
    traced_memory: bool = False,
    poll_interval_s: float | None = 0.001,
):
    """Mide un bloque de codigo: tiempo de reloj, tiempo de CPU y pico de memoria.

    Es la forma sencilla de medir algo suelto en cualquiera de los tres bloques,
    sea PCA de numpy, un modelo de scikit-learn o una red de PyTorch en CPU:

        with cost.measure() as m:
            modelo.fit(X)
        print(m["wall_s"], m["peak_process_bytes"], m["peak_process_exact"])

    Para la tabla de coste de la memoria usa `measure_cost`, que separa las
    pasadas de tiempo y de memoria, repite y calienta. Aqui todo se mide en una
    sola pasada, asi que el tiempo incluye la pequena interferencia del hilo de
    muestreo y, si `traced_memory` es True, la de tracemalloc, que es grande en
    codigo Python (`timing_with_tracemalloc` lo deja anotado).

    Parametros
    ----------
    process_memory : bool
        Medir el pico de memoria del proceso con psutil (necesita psutil).
    traced_memory : bool
        Medir tambien con tracemalloc (solo ve Python y numpy).
    poll_interval_s : float o None
        Intervalo del muestreo de memoria en segundos. None lo desactiva: si el
        bloque no supera el pico de vida del proceso, el pico sale NaN.

    Devuelve
    --------
    Un dict que se rellena al salir del bloque: `wall_s`, `cpu_s` y, segun las
    opciones, `peak_process_bytes` (pico por encima de la memoria de partida),
    `peak_process_exact`, `process_peak_lifetime_bytes`,
    `process_memory_metric`, `memory_poll_samples`, `peak_traced_bytes` y
    `retained_traced_bytes` (lo trazado que sigue vivo al salir).
    """
    if poll_interval_s is not None and not poll_interval_s > 0:
        raise ValueError("poll_interval_s debe ser positivo o None")
    resultado: dict = {}
    proceso = _importar_psutil("pasa process_memory=False").Process() if process_memory else None
    muestreador = None
    ya_trazaba = False
    traza_antes = 0

    gc.collect()
    if process_memory:
        actual_antes, pico_antes, metrica = _memoria_proceso(proceso)
        if poll_interval_s is not None:
            muestreador = _Muestreador(proceso, poll_interval_s)
            muestreador.start()
    if traced_memory:
        ya_trazaba = tracemalloc.is_tracing()
        if ya_trazaba:
            tracemalloc.reset_peak()
        else:
            tracemalloc.start()
        traza_antes = tracemalloc.get_traced_memory()[0]

    c0 = _cpu()
    t0 = _reloj()
    try:
        yield resultado
    finally:
        t1 = _reloj()
        c1 = _cpu()
        resultado["wall_s"] = float(t1 - t0)
        resultado["cpu_s"] = float(c1 - c0)
        resultado["timing_with_tracemalloc"] = bool(traced_memory)

        if traced_memory:
            actual_t, pico_t = tracemalloc.get_traced_memory()
            if not ya_trazaba:
                tracemalloc.stop()
            resultado["peak_traced_bytes"] = float(pico_t - traza_antes)
            resultado["retained_traced_bytes"] = float(actual_t - traza_antes)

        if process_memory:
            maximo = None
            if muestreador is not None:
                muestreador.parar()
                maximo = muestreador.maximo
                resultado["memory_poll_samples"] = muestreador.n_lecturas
            _, pico_despues, _ = _memoria_proceso(proceso)
            pico, exacto = _pico_del_tramo(actual_antes, pico_antes, pico_despues, maximo)
            resultado["peak_process_bytes"] = pico
            resultado["peak_process_exact"] = exacto
            resultado["process_peak_lifetime_bytes"] = (
                float(pico_despues) if pico_despues is not None else float("nan")
            )
            resultado["process_memory_metric"] = metrica


# ---------------------------------------------------------------- tiempos

def _entero(nombre: str, valor, minimo: int) -> int:
    if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)):
        raise TypeError(f"{nombre} debe ser un entero, llego {valor!r}")
    if valor < minimo:
        raise ValueError(f"{nombre} debe ser >= {minimo}, llego {valor}")
    return int(valor)


def _lotes(X, batch_size: int) -> list:
    """Parte X en lotes consecutivos de batch_size filas (el ultimo puede ser menor)."""
    try:
        n = len(X)
    except TypeError as exc:
        raise TypeError("X_infer debe admitir len() y cortes X[i:j]") from exc
    if n == 0:
        raise ValueError("X_infer esta vacio")
    if batch_size > n:
        raise ValueError(
            f"batch_size ({batch_size}) es mayor que el numero de muestras ({n})"
        )
    return [X[i : i + batch_size] for i in range(0, n, batch_size)]


def time_fit(fit, *, n_repeats: int, n_warmup: int):
    """Cronometra el ajuste de un modelo, repetido y tras un calentamiento.

    Parametros
    ----------
    fit : callable sin argumentos
        Crea un modelo NUEVO, lo ajusta y lo devuelve. Todo lo que haga cuenta
        como "ajuste": si incluye la calibracion de los limites o la parada
        temprana, entran en el tiempo; la carga de datos debe quedar fuera
        (hazla antes y captura los datos en la funcion). Ejemplo:
        `lambda: PCAMonitor(variance=0.85).fit(X_fit, X_cal=X_cal)`.
    n_repeats : int
        Ajustes cronometrados (>= 1). Sin valor por defecto: es una decision
        del protocolo.
    n_warmup : int
        Ajustes previos sin cronometrar (>= 0), que pagan la inicializacion de
        las bibliotecas. Sin valor por defecto.

    Devuelve
    --------
    (dict, modelo): el dict trae las listas `fit_wall_s` y `fit_cpu_s` (una
    entrada por repeticion) y sus resumenes; el modelo es el del ultimo ajuste.
    Antes de cada repeticion se recoge la basura, para que todas partan igual.
    """
    n_repeats = _entero("n_repeats", n_repeats, 1)
    n_warmup = _entero("n_warmup", n_warmup, 0)

    for _ in range(n_warmup):
        fit()

    reloj, cpu = [], []
    modelo = None
    for _ in range(n_repeats):
        modelo = None
        gc.collect()
        c0 = _cpu()
        t0 = _reloj()
        modelo = fit()
        t1 = _reloj()
        c1 = _cpu()
        reloj.append(float(t1 - t0))
        cpu.append(float(c1 - c0))

    r = np.asarray(reloj)
    return {
        "fit_wall_s": reloj,
        "fit_cpu_s": cpu,
        "fit_time_s_median": float(np.median(r)),
        "fit_time_s_min": float(r.min()),
        "fit_time_s_max": float(r.max()),
        "fit_cpu_s_median": float(np.median(cpu)),
        "n_repeats_fit": n_repeats,
        "n_warmup_fit": n_warmup,
    }, modelo


def time_inference(predict, X, *, batch_size: int, n_repeats: int, n_warmup: int) -> dict:
    """Cronometra la inferencia y la expresa por muestra.

    Por que por muestra. Los conjuntos de prueba de cada bloque no tienen por
    que tener el mismo tamano, y el uso real en planta es en linea: llega una
    muestra cada 3 minutos y hay que puntuarla. Dividir el tiempo total entre
    el numero de muestras hace comparables bloques que evaluan cantidades
    distintas. Con `batch_size=1` se mide la latencia en linea (una muestra por
    llamada); con lotes mayores, el rendimiento por lotes, que suele ser mucho
    mejor por muestra. Cual se reporta es una decision del protocolo.

    Parametros
    ----------
    predict : callable
        `predict(X_lote)` puntua un lote. La salida se descarta: el tiempo no
        incluye guardar resultados. En PyTorch, la funcion debe poner el modelo
        en modo evaluacion y usar `torch.no_grad()`; si convierte a tensor, la
        conversion entra en el tiempo (o pasa X ya convertido).
    X : array, tensor o cualquier secuencia con len() y cortes X[i:j]
        Entradas a puntuar, una por fila: muestras, o ventanas en los modelos
        con memoria (por ejemplo un array (n, ventana, variables)).
    batch_size : int
        Muestras por llamada (1 <= batch_size <= len(X)). Sin valor por defecto.
    n_repeats : int
        Pasadas completas cronometradas (>= 1). Sin valor por defecto.
    n_warmup : int
        Llamadas previas sin cronometrar con el primer lote (>= 0). Sin valor
        por defecto.

    Devuelve
    --------
    dict con `infer_time_per_sample_s_median` (mediana, entre pasadas, del
    tiempo total de la pasada dividido entre el numero de muestras), su minimo y
    su maximo, `infer_cpu_per_sample_s_median`, `infer_latency_s_median` e
    `infer_latency_s_p95` (tiempo de una llamada con un lote completo; el ultimo
    lote, si es menor, no entra), `infer_pass_wall_s` (lista por pasada) y la
    configuracion (`batch_size`, `n_infer_samples`, `n_repeats_infer`,
    `n_warmup_infer`).
    """
    batch_size = _entero("batch_size", batch_size, 1)
    n_repeats = _entero("n_repeats", n_repeats, 1)
    n_warmup = _entero("n_warmup", n_warmup, 0)
    lotes = _lotes(X, batch_size)
    n = len(X)
    completos = np.array([len(lote) == batch_size for lote in lotes])

    for _ in range(n_warmup):
        predict(lotes[0])

    por_muestra, cpu_por_muestra, pasadas = [], [], []
    latencias = []
    for _ in range(n_repeats):
        gc.collect()
        lat = np.empty(len(lotes))
        c0 = _cpu()
        inicio = _reloj()
        for j, lote in enumerate(lotes):
            t0 = _reloj()
            predict(lote)
            lat[j] = _reloj() - t0
        total = _reloj() - inicio
        cpu = _cpu() - c0
        pasadas.append(float(total))
        por_muestra.append(float(total) / n)
        cpu_por_muestra.append(float(cpu) / n)
        latencias.append(lat[completos])

    lat_todas = np.concatenate(latencias)
    pm = np.asarray(por_muestra)
    return {
        "infer_time_per_sample_s_median": float(np.median(pm)),
        "infer_time_per_sample_s_min": float(pm.min()),
        "infer_time_per_sample_s_max": float(pm.max()),
        "infer_cpu_per_sample_s_median": float(np.median(cpu_por_muestra)),
        "infer_latency_s_median": float(np.median(lat_todas)),
        "infer_latency_s_p95": float(np.percentile(lat_todas, 95)),
        "infer_pass_wall_s": pasadas,
        "batch_size": batch_size,
        "n_infer_samples": int(n),
        "n_repeats_infer": n_repeats,
        "n_warmup_infer": n_warmup,
    }


# ---------------------------------------------------------------- todo junto

def _pasar_lotes(predict, modelo, lotes) -> None:
    for lote in lotes:
        predict(modelo, lote)


#: Donde puede medir `measure_cost` la memoria de la inferencia.
INFER_MEMORY_MODES = ("same_process", "subprocess")


def _importar_cloudpickle():
    """cloudpickle, que serializa lambdas y clases definidas en un cuaderno.

    No es una dependencia nueva: joblib, que llega con scikit-learn, trae una
    copia (`joblib.externals.cloudpickle`). Si esta el paquete suelto, se usa.
    """
    try:
        import cloudpickle
    except ImportError:
        try:
            from joblib.externals import cloudpickle
        except ImportError as exc:
            raise ImportError(
                "infer_memory='subprocess' necesita cloudpickle (o joblib, que lo trae y "
                "llega con scikit-learn) para pasar el modelo y predict al proceso aparte."
            ) from exc
    return cloudpickle


def _hilos_actuales() -> dict:
    """Hilos de BLAS, OpenMP y PyTorch de este proceso, para repetirlos en el proceso aparte."""
    pools, _ = _hilos_blas()
    por_api: dict = {}
    for d in pools:
        if d.get("user_api") and d.get("num_threads"):
            por_api.setdefault(d["user_api"], int(d["num_threads"]))
    hilos_torch = None
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            hilos_torch = int(torch.get_num_threads())
        except Exception:  # noqa: BLE001
            hilos_torch = None
    return {"blas_openmp": por_api, "torch": hilos_torch}


def _inferencia_en_hijo(ruta_entrada: str, ruta_salida: str) -> None:
    """Lo que ejecuta el proceso aparte de `measure_cost(..., infer_memory="subprocess")`.

    Carga el modelo, `predict` y las entradas con `pickle.load` directamente
    del fichero: leerlo antes entero (`pickle.loads(f.read())`) deja un pico de
    carga del tamano del fichero, que volveria a tapar el de la inferencia.
    Repite los hilos del proceso que lo lanzo, mide una pasada completa por
    lotes con `measure` y escribe el resultado en un JSON.
    """
    with open(ruta_entrada, "rb") as f:
        paquete = pickle.load(f)
    hilos = paquete["hilos"]
    limites = None
    if hilos["blas_openmp"]:
        try:
            from threadpoolctl import threadpool_limits

            limites = threadpool_limits(limits=hilos["blas_openmp"])
        except Exception:  # noqa: BLE001 - sin threadpoolctl se mide con los hilos que haya
            limites = None
    torch = sys.modules.get("torch")
    if torch is not None and hilos["torch"] is not None:
        torch.set_num_threads(int(hilos["torch"]))
    lotes = _lotes(paquete["X_infer"], paquete["batch_size"])
    with measure(process_memory=True, traced_memory=False,
                 poll_interval_s=paquete["poll_interval_s"]) as m:
        _pasar_lotes(paquete["predict"], paquete["modelo"], lotes)
    del limites
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(m, f)


def _memoria_inferencia_aparte(modelo, predict, X_infer, batch_size: int, poll_interval_s: float) -> dict:
    """Mide la memoria de una pasada de inferencia en un proceso de Python aparte.

    Por que. Dentro de `measure_cost`, la pasada de memoria de la inferencia va
    justo despues del ajuste, que casi siempre deja el pico de vida del proceso
    mas alto: en el mismo proceso, la inferencia solo sale exacta si necesita
    mas memoria que el ajuste. El proceso aparte solo carga el modelo, `predict`
    y `X_infer` (serializados con cloudpickle en un fichero temporal) y repite
    los hilos de este, asi que el unico pico anterior es el de esa carga.

    Devuelve el dict de `measure` del proceso aparte. Si algo no se puede
    serializar o el proceso aparte falla, lanza un error con el motivo.
    """
    if not sys.executable:
        raise RuntimeError("infer_memory='subprocess': no se encuentra el ejecutable de Python")
    cloudpickle = _importar_cloudpickle()
    with tempfile.TemporaryDirectory(prefix="tfmfdd_coste_") as carpeta:
        entrada = os.path.join(carpeta, "entrada.pkl")
        salida = os.path.join(carpeta, "salida.json")
        rutas = os.path.join(carpeta, "sys_path.json")
        paquete = {
            "modelo": modelo, "predict": predict, "X_infer": X_infer,
            "batch_size": batch_size, "poll_interval_s": poll_interval_s,
            "hilos": _hilos_actuales(),
        }
        try:
            with open(entrada, "wb") as f:
                cloudpickle.dump(paquete, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise TypeError(
                "infer_memory='subprocess' no pudo serializar el modelo, predict o X_infer "
                f"({type(exc).__name__}: {exc}). Usa infer_memory='same_process' o un "
                "modelo que se pueda guardar con pickle."
            ) from exc
        with open(rutas, "w", encoding="utf-8") as f:
            json.dump(list(sys.path), f)
        # El proceso aparte ejecuta este mismo fichero como modulo suelto (no
        # importa el paquete entero) y ve las mismas rutas de importacion.
        codigo = (
            "import importlib.util, json, sys\n"
            "sys.path[:0] = json.load(open(sys.argv[2], encoding='utf-8'))\n"
            "spec = importlib.util.spec_from_file_location('_tfmfdd_cost_aparte', sys.argv[1])\n"
            "modulo = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(modulo)\n"
            "modulo._inferencia_en_hijo(sys.argv[3], sys.argv[4])\n"
        )
        proceso = subprocess.run(
            [sys.executable, "-c", codigo, os.path.abspath(__file__), rutas, entrada, salida],
            capture_output=True, encoding="utf-8", errors="replace", check=False,
            env=dict(os.environ, PYTHONIOENCODING="utf-8"), cwd=os.getcwd(),
        )
        if proceso.returncode != 0:
            ultimas = "\n".join(proceso.stderr.strip().splitlines()[-15:])
            raise RuntimeError(
                "La medida de la memoria de la inferencia en un proceso aparte fallo "
                f"(codigo {proceso.returncode}). Ultimas lineas del error:\n{ultimas}"
            )
        with open(salida, encoding="utf-8") as f:
            return json.load(f)


def measure_cost(
    fit,
    predict,
    X_infer,
    *,
    method: str,
    batch_size: int,
    n_repeats_fit: int,
    n_warmup_fit: int,
    n_repeats_infer: int,
    n_warmup_infer: int,
    traced_memory: bool = False,
    poll_interval_s: float = 0.001,
    infer_memory: str = "same_process",
) -> dict:
    """Mide el coste de un metodo con el procedimiento comun y devuelve una fila.

    Es la funcion para la tabla de coste: los tres bloques la llaman igual, en
    la misma maquina, y las filas se pueden juntar en un DataFrame. El
    procedimiento (pasadas de memoria primero, luego tiempos de ajuste y de
    inferencia) esta en el docstring del modulo.

    Ejemplos, uno por bloque (los datos se cargan ANTES, fuera de la medida; los
    nombres en mayusculas son los valores declarados, no sugerencias):

        # A: PCA del paquete
        fila = cost.measure_cost(
            fit=lambda: PCAMonitor(variance=0.85).fit(X_fit, X_cal=X_cal),
            predict=lambda modelo, xb: modelo.score(xb),
            X_infer=X_prueba, method="pca", batch_size=LOTE,
            n_repeats_fit=R_AJUSTE, n_warmup_fit=W_AJUSTE,
            n_repeats_infer=R_INFER, n_warmup_infer=W_INFER)

        # B: scikit-learn
        fit=lambda: KNeighborsClassifier(n_neighbors=5).fit(X, y)
        predict=lambda modelo, xb: modelo.predict(xb)

        # C: PyTorch en CPU (entrenar y crear_red son del bloque)
        fit=lambda: entrenar(crear_red(), X, y)
        def predict(red, xb):
            with torch.no_grad():
                return red(xb)          # X_infer ya convertido a tensor

    Antes de medir, fija los hilos y dejalos constantes en todos los metodos
    (por ejemplo `threadpoolctl.threadpool_limits` para BLAS y OpenMP y
    `torch.set_num_threads` en el bloque C): la fila registra los que hay.
    Necesita psutil, que siempre mide la memoria del proceso. La memoria de la
    GPU no se mide: la medida de referencia es en CPU.

    Cuando el pico de memoria es exacto y cuando no. Un pico solo es exacto si
    el tramo medido supera el pico de toda la vida del proceso; si no, la cifra
    es la cota inferior de un muestreo, la columna `*_exact` vale False y la
    funcion avisa con un RuntimeWarning que dice que hacer:

    - Ajuste: no sale exacto si antes hubo un pico mayor (la carga de los
      datos, otra medida). Lo arregla medir en un proceso nuevo con solo los
      datos necesarios cargados.
    - Inferencia: su pasada de memoria va justo despues de la del ajuste, que
      casi siempre deja el pico mas alto. Con `infer_memory="same_process"`
      solo sale exacta si la inferencia necesita mas memoria que el ajuste, y
      un proceso nuevo no lo cambia. Con `infer_memory="subprocess"` se mide en
      un proceso de Python aparte que solo carga el modelo, `predict` y
      `X_infer` (serializados con cloudpickle, que trae joblib) directamente
      del fichero y repite los hilos de BLAS, OpenMP y PyTorch de este: sale
      exacta si la inferencia supera lo que dejo esa carga (con arrays de numpy
      cargados asi, nada por encima del estado de llegada). Pide que el modelo
      y `predict` se puedan serializar y que el proceso aparte pueda importar
      sus bibliotecas; si no, falla con un error que lo dice. El tiempo de la
      inferencia se sigue midiendo en este proceso.
    - `traced_memory=True` da ademas, en los dos casos, la reserva exacta de
      Python y numpy (no la de PyTorch ni la que hacen en C otras bibliotecas).

    Parametros
    ----------
    fit : callable sin argumentos
        Crea un modelo NUEVO, lo ajusta y lo devuelve (ver `time_fit`).
    predict : callable
        `predict(modelo, X_lote)` puntua un lote con el modelo ajustado.
    X_infer : array, tensor o secuencia con len() y cortes
        Entradas de la inferencia (ver `time_inference`).
    method : str
        Nombre del metodo, la columna `method` del formato comun.
    batch_size, n_repeats_fit, n_warmup_fit, n_repeats_infer, n_warmup_infer : int
        Tamano de lote, repeticiones y calentamientos. Sin valores por defecto:
        se declaran antes de medir y son los mismos en los tres bloques.
    traced_memory : bool
        Si es True, anade una pasada aparte con tracemalloc (un ajuste y una
        inferencia mas) y rellena `fit_peak_traced_bytes` e
        `infer_peak_traced_bytes`. Si es False, esas columnas valen NaN.
    poll_interval_s : float
        Intervalo del muestreo de memoria de las pasadas de memoria.
    infer_memory : {"same_process", "subprocess"}
        Donde se mide la memoria de la inferencia (ver arriba). Por defecto, en
        este mismo proceso, como hasta la v0.3; la fila lo registra en
        `infer_memory_where`.

    Devuelve
    --------
    dict con columnas en minusculas y en ingles: `method`, la configuracion de
    la medida, los resumenes de tiempo (`fit_time_s_median`,
    `infer_time_per_sample_s_median`, `infer_latency_s_median`,
    `infer_latency_s_p95`...), la memoria (`fit_peak_process_bytes`,
    `infer_peak_process_bytes`, con sus columnas `*_exact`, `infer_memory_where`
    y `process_peak_lifetime_bytes`), un resumen de la maquina (CPU, RAM,
    versiones e hilos) y `measured_at` (hora UTC). La maquina completa, con la
    lista de bibliotecas de hilos, la da `machine_info`: guardala en un JSON al
    lado de la tabla.
    """
    if not isinstance(method, str) or not method:
        raise TypeError("method debe ser un texto no vacio")
    batch_size = _entero("batch_size", batch_size, 1)
    n_repeats_fit = _entero("n_repeats_fit", n_repeats_fit, 1)
    n_warmup_fit = _entero("n_warmup_fit", n_warmup_fit, 0)
    n_repeats_infer = _entero("n_repeats_infer", n_repeats_infer, 1)
    n_warmup_infer = _entero("n_warmup_infer", n_warmup_infer, 0)
    if infer_memory not in INFER_MEMORY_MODES:
        raise ValueError(f"infer_memory debe ser uno de {INFER_MEMORY_MODES}, llego {infer_memory!r}")
    lotes = _lotes(X_infer, batch_size)
    _importar_psutil()      # falla ya, antes de ajustar nada, y sin ofrecer salidas que no hay

    # 1. Memoria, primero (ver el docstring del modulo).
    with measure(process_memory=True, traced_memory=False, poll_interval_s=poll_interval_s) as m_fit:
        modelo = fit()
    if infer_memory == "subprocess":
        m_inf = _memoria_inferencia_aparte(modelo, predict, X_infer, batch_size, poll_interval_s)
    else:
        with measure(process_memory=True, traced_memory=False, poll_interval_s=poll_interval_s) as m_inf:
            _pasar_lotes(predict, modelo, lotes)
    del modelo
    gc.collect()

    # 2 y 3. Tiempos.
    t_fit, modelo = time_fit(fit, n_repeats=n_repeats_fit, n_warmup=n_warmup_fit)
    t_inf = time_inference(
        lambda xb: predict(modelo, xb), X_infer,
        batch_size=batch_size, n_repeats=n_repeats_infer, n_warmup=n_warmup_infer,
    )

    # Opcional: tracemalloc en pasadas aparte, despues de cronometrar.
    traza_fit = traza_inf = float("nan")
    if traced_memory:
        del modelo
        gc.collect()
        with measure(process_memory=False, traced_memory=True) as tf:
            modelo = fit()
        with measure(process_memory=False, traced_memory=True) as ti:
            _pasar_lotes(predict, modelo, lotes)
        traza_fit = tf["peak_traced_bytes"]
        traza_inf = ti["peak_traced_bytes"]

    # Pico de toda la vida del proceso al terminar: incluye Python, las
    # bibliotecas y los datos cargados; solo es la huella del metodo si el
    # proceso se arranco para medirlo.
    _, pico_final, _ = _memoria_proceso(_importar_psutil().Process())

    forma = getattr(X_infer, "shape", None)
    fila = {
        "method": method,
        "n_infer_samples": int(len(X_infer)),
        "infer_sample_shape": str(tuple(forma[1:])) if forma is not None else None,
        "batch_size": batch_size,
        "n_repeats_fit": n_repeats_fit,
        "n_warmup_fit": n_warmup_fit,
        "n_repeats_infer": n_repeats_infer,
        "n_warmup_infer": n_warmup_infer,
    }
    fila.update({k: v for k, v in t_fit.items() if not isinstance(v, list)})
    fila.update({k: v for k, v in t_inf.items() if not isinstance(v, list)})
    fila.update({
        "fit_peak_process_bytes": m_fit["peak_process_bytes"],
        "fit_peak_process_exact": m_fit["peak_process_exact"],
        "infer_peak_process_bytes": m_inf["peak_process_bytes"],
        "infer_peak_process_exact": m_inf["peak_process_exact"],
        "infer_memory_where": infer_memory,
        "process_peak_lifetime_bytes": float(pico_final) if pico_final is not None else float("nan"),
        "process_memory_metric": m_fit["process_memory_metric"],
        "memory_poll_interval_s": float(poll_interval_s),
        "fit_peak_traced_bytes": traza_fit,
        "infer_peak_traced_bytes": traza_inf,
    })
    maquina = machine_info()
    fila.update({k: maquina[k] for k in _CAMPOS_MAQUINA_EN_FILA})
    fila["measured_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    avisos = _avisos_de_memoria(fila["fit_peak_process_exact"], fila["infer_peak_process_exact"],
                                infer_memory)
    if avisos:
        warnings.warn(f"{method}: " + "; ".join(avisos) + ".", RuntimeWarning, stacklevel=2)
    return fila


def _avisos_de_memoria(fit_exacto: bool, infer_exacto: bool, infer_memory: str) -> list[str]:
    """Textos del aviso de `measure_cost` cuando un pico de memoria no es exacto.

    Cada texto dice por que la cifra es solo una cota inferior y que lo
    arreglaria de verdad; no se recomienda nada que no lo arregle.
    """
    avisos = []
    if not fit_exacto:
        avisos.append(
            "el pico de memoria del ajuste no supero un pico anterior del proceso (la carga "
            "de los datos u otra medida), asi que fit_peak_process_bytes es una cota inferior "
            "sacada del muestreo; para una cifra exacta, mide en un proceso nuevo con solo "
            "los datos necesarios cargados"
        )
    if not infer_exacto and infer_memory == "same_process":
        avisos.append(
            "el pico de memoria de la inferencia no supero el del ajuste, medido justo antes "
            "en el mismo proceso, asi que infer_peak_process_bytes es una cota inferior "
            "sacada del muestreo; un proceso nuevo no lo cambia: pasa "
            "infer_memory='subprocess' para medirla en un proceso aparte (o "
            "traced_memory=True para la reserva exacta de Python y numpy)"
        )
    elif not infer_exacto:
        avisos.append(
            "el pico de memoria de la inferencia, medido en un proceso aparte, no supero el "
            "que dejo alli la carga del modelo y de los datos, asi que "
            "infer_peak_process_bytes es una cota inferior sacada del muestreo"
        )
    return avisos
