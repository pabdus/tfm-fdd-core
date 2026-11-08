"""P1 - Piloto de la varianza de la ventaja del autoencoder sobre PCA y DPCA (Rieth, fallos 3, 9 y 15).

Por que existe. La regla de la hipotesis (memoria, parrafo 368) contrasta, con
dos Wilcoxon unilaterales sobre "la ventaja menos delta"
(`stats.non_superiority_test`), que el autoencoder no supera a PCA ni a DPCA en
los fallos 3, 9 y 15 de Rieth. Cuantas simulaciones confirmatorias hacen falta
(n) depende de cuanto varia esa ventaja de una simulacion a otra: su desviacion
sigma. La hoja de decision del 15/10 (Contexto, Revisiones_E2) propone fijar n
en el prerregistro con una regla que depende de la cota superior de sigma
estimada en un piloto sobre las simulaciones de desarrollo. Este script es ese
piloto: estima sigma por celda (fallo, baseline) con sus intervalos y deja la
regla de n al prerregistro.

Que hace, con la configuracion de configs/baseline.yaml (`tfmfdd.config`):

1. Reparte las simulaciones normales de train_fault00 (ids 1 a
   normales_ajuste + normales_calibracion, dentro de 1..MAXIMO_NORMALES, las 1
   a 40) UNA sola vez, con `data.split_runs` y la semilla del yaml, en tres
   bloques: entrenar el autoencoder, pararlo (parada temprana) y calibrar los
   limites. Las de parada (`particion.rieth.normales_parada`) salen de las de
   ajuste, y PCA y DPCA se ajustan con todas las de ajuste (entrenar + parar).
   Es el reparto del C00 (celda 35): con 14 de ajuste, 4 de parada y 6 de
   calibracion, el mismo split_runs(20, fractions=(0.5, 0.2, 0.3)) y los
   mismos ids (regla 3: la parada usa otro bloque de normales, distinto del de
   calibracion y de la prueba). Con las decisiones del 18/10 (14, 4 y 20) es
   split_runs(34, (10/34, 4/34, 20/34)) sobre los ids 1 a 34, el mismo reparto
   que cada replica confirmatoria hace sobre las posiciones de sus ids
   (particion.rieth.reparto).
2. Ajusta los baselines de `hipotesis.baselines` (PCAMonitor con los limites
   de `limites_protocolo.*`, `calibration="holdout"`) y el autoencoder del
   cuaderno notebooks/C00_punto_de_partida_deeplearning.ipynb (misma
   arquitectura, mismo bucle de entrenamiento con parada temprana, mismos
   hiperparametros declarados; umbral empirico al nivel protocolo.alpha sobre
   el error de reconstruccion de las normales reservadas). Un limite empirico
   se calibra con las normales reservadas; uno parametrico (el T2 por F o por
   Beta, el SPE de Jackson) sale de cantidades del ajuste, no de las
   reservadas: el piloto lee `limit_source_` de cada estadistico que usa, lo
   avisa y lo deja en el CSV (columnas limite_reservado_<baseline>).
3. Evalua cada metodo, simulacion a simulacion, en test_fault03, test_fault09 y
   test_fault15 (los fallos de `fallos_dificiles`) y en sus gemelas de
   test_fault00 (el mismo simulationRun), con `metrics.evaluate_run`: FDR sobre
   las muestras 161 a 960 y FAR de la gemela en la ventana
   `deteccion.ventana_far_normal`.
4. Construye la ventaja por simulacion con `stats.paired_advantage` y la metrica
   de `hipotesis.metrica_ventaja` ("fdr", o "exceso_far_gemela": FDR de la
   simulacion con fallo menos FAR de su gemela normal), con el sentido de
   `hipotesis.mayor_es_mejor`.
5. Por celda (fallo, baseline): n, la desviacion muestral sigma_hat de la
   ventaja (ddof = 1), su intervalo bilateral al 95 % y su cota superior
   unilateral al 95 % (los dos por chi cuadrado, que suponen normalidad de la
   ventaja), y la pseudomediana de Hodges-Lehmann con su intervalo al 90 %
   (`stats.hodges_lehmann`, que no la supone), el nivel del contraste de la
   fila 27 de docs/decisiones.md: dos unilaterales al 5 %.

Lo que NO hace. No decide delta ni la aplica; no aplica la regla de n (la fija
el prerregistro con la cota superior que este CSV reporta); no contrasta la
hipotesis; no da ninguna cifra confirmatoria. Es un piloto sobre simulaciones
de desarrollo, ya miradas, y sus cifras no se citan como resultado.

Regla de ids (regla 3, DIS-17). En los ficheros de prueba solo lee
simulaciones con id entre 1 y datos.runs_desarrollo (20); en train_fault00,
entre 1 y MAXIMO_NORMALES (40: las 1 a 20 por la regla 3, punto 1, y las 21 a
40 porque las cargo la verificacion del 27/09, punto 2; fila del 2026-10-18 de
docs/decisiones.md, que enmienda la 29, porque 14 + 20 normales no caben en
20). En los dos casos, ademas, registradas como miradas en
`data.RIETH_SEEN_RUNS`: cada carga pasa por `comprobar_ids_mirados`, que
consulta `data.seen_runs(split, fallo)` ANTES de abrir el Parquet y se niega si
algun id no cumple las dos cosas. Los ids que toco quedan en la procedencia.
Ninguno es confirmatorio: los confirmatorios salen de data.unseen_runs.

Configuracion y negativa. Como A0, A1 y V1, lee las decisiones del yaml y no
sustituye ninguna clave en null por un valor por defecto. Necesita estas seis:
hipotesis.metrica_ventaja, hipotesis.mayor_es_mejor, hipotesis.baselines,
deteccion.ventana_far_normal, particion.rieth.normales_ajuste y
particion.rieth.normales_calibracion. Si alguna sigue en null, las lista y
termina con codigo 2 sin abrir ningun fichero. Mientras Pablo las decide,
`--provisional ruta.yaml` acepta un yaml con valores DE PRUEBA para esas claves
(y solo para claves que en el yaml del protocolo esten en null: una clave con
valor no se pisa). Todo lo que sale de una ejecucion provisional lo dice:
columna `provisional` del CSV, `extra.provisional` y el fichero en
`config_files` del JSON de procedencia, y un aviso en la salida.

La ejecucion que cuenta (sin valores de prueba) exige ademas los cinco
`limites_protocolo.*` y `particion.rieth.normales_parada` (fila 29: el piloto
corre con las decisiones ya fijadas). Solo en una ejecucion provisional, si
siguen en null (o la clave de la parada aun no existe), usa los limites de
desarrollo (`pca.*`, con los dos limites de la regla T2 o SPE al mismo nivel y
el escalado de la matriz con retardos, "retardar_y_escalar") y para el autoencoder con
round(particion.fraccion_validacion * normales), la cuenta del C00; lo avisa y
lo deja en el CSV (columnas limites_fuente y normales_parada_fuente).

Formato de hipotesis.baselines (lista; cada elemento es un baseline de la
familia): nombre (pca, dpca...), estadistico ("t2", "spe" o "t2_o_spe", la regla
T2 o SPE con `PCAMonitor.combined_index`) y lags (0 = PCA; 1 o 2 = DPCA).

Valores de limites_protocolo que el piloto entiende: t2_method y spe_method,
los de `PCAMonitor` (T2: f, beta, empirical, kde; SPE: box, jackson, kde,
empirical); quantile_method, los de `limits.limit_empirical` (linear,
conformal), tambien para el umbral del autoencoder; nivel_por_estadistico,
"mismo_alpha" (cada limite al nivel protocolo.alpha) o "sidak" (en la regla T2
o SPE, cada uno a protocolo.alpha ** 0.5, para una FAR nominal conjunta); y
orden_escalado_dpca, con los nombres de configs/yin2012.yaml (S5):
"retardar_y_escalar" (PCAMonitor escala la matriz con retardos) o
"escalar_y_retardar" (un Scaler ajustado solo con las normales de ajuste,
sobre las variables originales, con el gancho scaler= de PCAMonitor.fit; es el
orden de Yin et al., 2012, p. 1570, y el de la configuracion principal de V1).
Los cinco quedan en cada fila del CSV.

Salidas (en --out; por defecto results/summary y, en una ejecucion
provisional, results/raw/P1_provisional, que git no versiona; una provisional
se niega a escribir en results/summary): P1_piloto_varianza.csv (una fila por
fallo y baseline, con la configuracion completa en cada fila),
P1_piloto_varianza_runs.csv (una fila por metodo, fallo y simulacion, en el
formato comun de la regla 11, con las columnas de evaluate_run) y el
.provenance.json de cada uno (`tfmfdd.provenance`): configuracion, ficheros de
configuracion con su huella, ids tocados y los Parquet usados segun
SHA256SUMS.txt de la carpeta de datos. Antes de abrir ningun Parquet, el piloto
compara los bytes de cada fichero en el disco con los del manifiesto y se
niega si no coinciden; la SHA-256 no se recalcula (seria volver a leer 1,3 GB)
y va como sha256_segun_manifiesto.

Entorno. PyTorch (el `.venv` del repositorio lo tiene); la carpeta de los
Parquet de Rieth en --root o en la variable de entorno TFM_RIETH_DIR.

Uso:
    python experiments/P1_piloto_varianza.py                             # se niega con claves en null
    python experiments/P1_piloto_varianza.py --provisional p1_prov.yaml  # valores de prueba, a results/raw/P1_provisional
    python experiments/P1_piloto_varianza.py --provisional p1_prov.yaml --n-ids 8 --epocas-max 5 --out results/raw/P1
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from tfmfdd import config, data, limits, metrics, provenance, stats
from tfmfdd.pca import SPE_METHODS, T2_METHODS, PCAMonitor
from tfmfdd.preprocessing import Scaler

#: Claves del yaml que este piloto necesita con valor (en null, se niega).
CLAVES_NECESARIAS = (
    "hipotesis.metrica_ventaja",
    "hipotesis.mayor_es_mejor",
    "hipotesis.baselines",
    "deteccion.ventana_far_normal",
    "particion.rieth.normales_ajuste",
    "particion.rieth.normales_calibracion",
)
#: Metricas de la ventaja que el piloto sabe calcular. "fdr_a_far_igualada" (la
#: tercera opcion del comentario del yaml) exige recalibrar cada metodo a la FAR
#: del otro y no esta implementada aqui.
METRICAS = ("fdr", "exceso_far_gemela")
#: Estadisticos admitidos en hipotesis.baselines.
ESTADISTICOS = ("t2", "spe", "t2_o_spe")
#: Nombre del autoencoder en la columna method.
METODO_AE = "ae"
#: Hiperparametros del autoencoder del C00 en Rieth (celdas 14 y 36 del cuaderno):
#: declarados, no ajustados. --epocas-max los cambia solo para pruebas de humo.
HP_AE = {"capas": (32, 16), "lr": 1e-3, "lote": 64, "epocas_max": 200, "paciencia": 15,
         "mejora_min": 1e-3}
#: Nivel del intervalo bilateral de sigma y de su cota superior unilateral.
NIVEL_SIGMA = 0.95
#: Nivel del intervalo de Hodges-Lehmann de la ventaja: el del contraste de la
#: fila 27 de docs/decisiones.md (dos unilaterales al 5 %, intervalo al 90 %).
NIVEL_HL = 0.90
#: Nombres de los ficheros de salida.
SALIDA = "P1_piloto_varianza.csv"
SALIDA_RUNS = "P1_piloto_varianza_runs.csv"
#: Manifiesto de la carpeta de Rieth (nombre, bytes, sha256 por linea).
MANIFIESTO = "SHA256SUMS.txt"
#: Limites del resultado (limites_protocolo, fijados el 2026-10-18). En null
#: (un yaml anterior), solo una ejecucion provisional sigue, con los de
#: desarrollo (pca.*).
CLAVES_LIMITES = (
    "limites_protocolo.t2_method",
    "limites_protocolo.spe_method",
    "limites_protocolo.quantile_method",
    "limites_protocolo.nivel_por_estadistico",
    "limites_protocolo.orden_escalado_dpca",
)
#: Valores de limites_protocolo.nivel_por_estadistico y de orden_escalado_dpca.
NIVELES = ("mismo_alpha", "sidak")
ORDENES_ESCALADO = ("retardar_y_escalar", "escalar_y_retardar")    # los de yin2012.yaml, S5
#: Normales de train_fault00 que paran el autoencoder (salen de las de ajuste).
#: Existe en el yaml desde el 2026-10-18; un yaml anterior no la tiene.
CLAVE_PARADA = "particion.rieth.normales_parada"
#: Ultimo id de train_fault00 que el piloto puede leer (commit 2b; fila del
#: 2026-10-18 de docs/decisiones.md, que enmienda la 29). Las normales 1 a 40 ya
#: estan en el registro como miradas (regla 3, puntos 1 y 2); las de prueba
#: siguen en 1..datos.runs_desarrollo.
MAXIMO_NORMALES = 40
#: Raiz del repositorio y carpetas de salida.
RAIZ = Path(__file__).resolve().parents[1]
SALIDA_DEFINITIVA = Path("results") / "summary"
SALIDA_PROVISIONAL = Path("results") / "raw" / "P1_provisional"


class DecisionPendiente(RuntimeError):
    """Falta una decision del yaml (o el provisional no vale): el piloto no se ejecuta."""


# ------------------------------------------------------------ configuracion

def _aplanar(d: dict, prefijo: str = "") -> dict:
    """{"a": {"b": 1}} -> {"a.b": 1}. Una lista o un escalar son hojas."""
    plano = {}
    for k, v in d.items():
        ruta = f"{prefijo}.{k}" if prefijo else str(k)
        if isinstance(v, dict):
            plano.update(_aplanar(v, ruta))
        else:
            plano[ruta] = v
    return plano


def aplicar_provisional(cfg: dict, provisional: dict) -> tuple[dict, list[str]]:
    """Copia de cfg con los valores del yaml provisional en las claves que esten en null.

    Solo se admite una clave que exista en cfg y este en null: una clave con
    valor es una decision tomada y el yaml del protocolo manda; una que no
    existe es un error de escritura. Devuelve (cfg nuevo, claves aplicadas).
    """
    nuevo = copy.deepcopy(cfg)
    aplicadas = []
    for clave, valor in _aplanar(provisional).items():
        try:
            actual = config.get(nuevo, clave)
        except KeyError:
            raise DecisionPendiente(
                f"--provisional trae la clave '{clave}', que no existe en el yaml del protocolo."
            ) from None
        if actual is not None:
            raise DecisionPendiente(
                f"--provisional quiere fijar '{clave}', que ya tiene valor en el yaml del protocolo "
                f"({actual!r}): el yaml manda y un provisional no lo pisa."
            )
        if valor is None:
            continue
        tramos = clave.split(".")
        destino = nuevo
        for t in tramos[:-1]:
            destino = destino[t]
        destino[tramos[-1]] = valor
        aplicadas.append(clave)
    return nuevo, aplicadas


def pendientes(cfg: dict) -> list[str]:
    """Las claves de CLAVES_NECESARIAS que siguen en null (o que faltan)."""
    faltan = []
    for clave in CLAVES_NECESARIAS:
        try:
            if config.get(cfg, clave) is None:
                faltan.append(clave)
        except KeyError:
            faltan.append(f"{clave} (no existe en el yaml)")
    return faltan


def _ventana(valor, nombre: str) -> tuple[int, int | None]:
    """[inicio, fin] del yaml -> tupla para far_window (fin null = el final)."""
    if not isinstance(valor, (list, tuple)) or len(valor) != 2:
        raise DecisionPendiente(f"{nombre} debe ser [inicio, fin] (fin null = el final); llego {valor!r}")
    inicio, fin = valor
    if isinstance(inicio, bool) or not isinstance(inicio, int) or inicio < 0:
        raise DecisionPendiente(f"{nombre}: el inicio es un entero >= 0 en base cero; llego {valor!r}")
    if fin is not None and (isinstance(fin, bool) or not isinstance(fin, int) or fin <= inicio):
        raise DecisionPendiente(f"{nombre}: el fin es null o un entero mayor que el inicio; llego {valor!r}")
    return (inicio, fin)


def _entero_positivo(valor, nombre: str) -> int:
    if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)) or valor < 1:
        raise DecisionPendiente(f"{nombre} debe ser un entero >= 1; llego {valor!r}")
    return int(valor)


def validar_decisiones(cfg: dict) -> dict:
    """Lee y valida las seis decisiones. Devuelve un dict con sus valores ya tipados."""
    faltan = pendientes(cfg)
    if faltan:
        raise DecisionPendiente(
            "quedan decisiones pendientes (claves en null de configs/baseline.yaml; se fijan en "
            "docs/decisiones.md o, mientras tanto, con --provisional):\n  - " + "\n  - ".join(faltan)
        )
    metrica = config.require(cfg, "hipotesis.metrica_ventaja")
    if metrica not in METRICAS:
        raise DecisionPendiente(
            f"hipotesis.metrica_ventaja = {metrica!r} no es una de las metricas que el piloto calcula "
            f"{METRICAS} ('fdr_a_far_igualada' no esta implementada en el piloto)."
        )
    mayor = config.require(cfg, "hipotesis.mayor_es_mejor")
    if not isinstance(mayor, bool):
        raise DecisionPendiente(f"hipotesis.mayor_es_mejor debe ser true o false; llego {mayor!r}")
    baselines = config.require(cfg, "hipotesis.baselines")
    if not isinstance(baselines, list) or not baselines:
        raise DecisionPendiente(
            "hipotesis.baselines debe ser una lista de baselines, cada uno con nombre, estadistico "
            f"(uno de {ESTADISTICOS}) y lags; llego {baselines!r}"
        )
    vistos = set()
    limpios = []
    for b in baselines:
        if not isinstance(b, dict) or set(b) != {"nombre", "estadistico", "lags"}:
            raise DecisionPendiente(
                f"cada baseline lleva exactamente las claves nombre, estadistico y lags; llego {b!r}"
            )
        nombre = str(b["nombre"])
        if nombre == METODO_AE or nombre in vistos:
            raise DecisionPendiente(f"nombre de baseline repetido o reservado: {nombre!r}")
        if b["estadistico"] not in ESTADISTICOS:
            raise DecisionPendiente(
                f"baseline {nombre}: estadistico {b['estadistico']!r} no es uno de {ESTADISTICOS}"
            )
        lags = b["lags"]
        if isinstance(lags, bool) or not isinstance(lags, int) or lags < 0:
            raise DecisionPendiente(f"baseline {nombre}: lags debe ser un entero >= 0; llego {lags!r}")
        vistos.add(nombre)
        limpios.append({"nombre": nombre, "estadistico": b["estadistico"], "lags": lags})
    return {
        "metrica": metrica,
        "mayor_es_mejor": mayor,
        "baselines": limpios,
        "ventana_far_normal": _ventana(config.require(cfg, "deteccion.ventana_far_normal"),
                                       "deteccion.ventana_far_normal"),
        "normales_ajuste": _entero_positivo(config.require(cfg, "particion.rieth.normales_ajuste"),
                                            "particion.rieth.normales_ajuste"),
        "normales_calibracion": _entero_positivo(config.require(cfg, "particion.rieth.normales_calibracion"),
                                                 "particion.rieth.normales_calibracion"),
    }


def _ventana_texto(ventana: tuple[int, int | None]) -> str:
    inicio, fin = ventana
    return f"[{inicio}, {'null' if fin is None else fin}]"


def parametros_limites(cfg: dict, provisional: bool) -> dict:
    """Los limites de PCA y DPCA (y el metodo del cuantil del umbral del autoencoder).

    Con los cinco `limites_protocolo.*` fijados, esos, validados. Con los cinco
    en null, solo en una ejecucion provisional: los de desarrollo (`pca.*`), los
    dos limites de la regla T2 o SPE al mismo nivel y el escalado de la matriz
    con retardos ("retardar_y_escalar"), que es lo que hacia el piloto del 15/10. Fijados a medias, o en null
    en la ejecucion que cuenta, es una decision pendiente.
    """
    valores = {}
    for clave in CLAVES_LIMITES:
        try:
            valores[clave.split(".", 1)[1]] = config.get(cfg, clave)
        except KeyError:
            raise DecisionPendiente(f"{clave} no existe en el yaml del protocolo.") from None
    fijadas = [k for k, v in valores.items() if v is not None]
    if not fijadas:
        if not provisional:
            raise DecisionPendiente(
                "limites_protocolo.* sigue en null. El piloto que cuenta corre con las decisiones ya fijadas "
                "(fila 29 de docs/decisiones.md); los limites de desarrollo (pca.*) solo valen en una "
                "ejecucion provisional:\n  - " + "\n  - ".join(CLAVES_LIMITES)
            )
        return {"fuente": "pca_desarrollo", "t2_method": config.require(cfg, "pca.t2_method"),
                "spe_method": config.require(cfg, "pca.spe_method"),
                "quantile_method": config.require(cfg, "pca.quantile_method"),
                "nivel_por_estadistico": "mismo_alpha", "orden_escalado_dpca": "retardar_y_escalar"}
    if len(fijadas) < len(valores):
        raise DecisionPendiente(
            "limites_protocolo.* esta fijado a medias; faltan:\n  - "
            + "\n  - ".join(f"limites_protocolo.{k}" for k, v in valores.items() if v is None)
        )
    admitidos = {"t2_method": T2_METHODS, "spe_method": SPE_METHODS, "quantile_method": limits.EMPIRICAL_METHODS,
                 "nivel_por_estadistico": NIVELES, "orden_escalado_dpca": ORDENES_ESCALADO}
    for k, opciones in admitidos.items():
        if valores[k] not in opciones:
            raise DecisionPendiente(f"limites_protocolo.{k} = {valores[k]!r} no es uno de {opciones}")
    return {"fuente": "limites_protocolo", **valores}


def alpha_por_estadistico(alpha: float, estadistico: str, nivel: str) -> float:
    """Nivel de confianza de cada limite de un baseline.

    Con la regla T2 o SPE y "sidak", cada limite va a alpha ** 0.5: con dos
    estadisticos independientes, la FAR nominal conjunta es la de alpha
    (DIS-10). En los demas casos, alpha tal cual.
    """
    if estadistico == "t2_o_spe" and nivel == "sidak":
        return float(alpha) ** 0.5
    return float(alpha)


def resolver_normales_parada(cfg: dict, n_ajuste: int, n_cal: int, provisional: bool) -> tuple[int, str]:
    """(normales que paran el autoencoder, de donde sale ese numero).

    Con `particion.rieth.normales_parada` fijada, ese valor. Si sigue en null o
    la clave aun no existe, solo en una ejecucion provisional:
    round(particion.fraccion_validacion * (n_ajuste + n_cal)), la cuenta del C00
    (0,2 de 20 = 4).
    """
    try:
        valor = config.get(cfg, CLAVE_PARADA)
        existe = True
    except KeyError:
        valor, existe = None, False
    if valor is not None:
        return _entero_positivo(valor, CLAVE_PARADA), CLAVE_PARADA
    if not provisional:
        estado = "sigue en null" if existe else "no existe en el yaml"
        raise DecisionPendiente(
            f"{CLAVE_PARADA} {estado}: la ejecucion que cuenta necesita fijado cuantas normales paran el "
            "autoencoder (fila 29 de docs/decisiones.md)."
        )
    fraccion = config.require(cfg, "particion.fraccion_validacion")
    return int(round(fraccion * (n_ajuste + n_cal))), "particion.fraccion_validacion (provisional)"


def carpeta_salida(out, provisional: bool) -> Path:
    """Donde escribe el piloto: --out, o results/summary, o results/raw/P1_provisional si es provisional.

    Una ejecucion provisional no escribe en results/summary ni dentro: lo que
    hay alli se versiona, y las cifras de una provisional no se citan.
    """
    if out is None:
        return SALIDA_PROVISIONAL if provisional else SALIDA_DEFINITIVA
    destino = Path(out)
    if provisional:
        resuelto = destino.resolve()
        for prohibida in {(Path.cwd() / SALIDA_DEFINITIVA).resolve(), (RAIZ / SALIDA_DEFINITIVA).resolve()}:
            if resuelto == prohibida or prohibida in resuelto.parents:
                raise DecisionPendiente(
                    f"una ejecucion provisional no escribe en {SALIDA_DEFINITIVA.as_posix()} ({resuelto}): sus "
                    f"cifras no se citan. Quita --out (va a {SALIDA_PROVISIONAL.as_posix()}) o elige otra carpeta."
                )
    return destino


# ---------------------------------------------------------------- ids

def comprobar_ids_mirados(split: str, fallo: int, ids, maximo: int) -> list[int]:
    """Ids que el piloto puede leer de {split}_fault{fallo}: entre 1 y `maximo` y registrados como mirados.

    Se llama ANTES de abrir el Parquet. Un id mayor que `maximo` (datos.runs_desarrollo,
    20, en la prueba; MAXIMO_NORMALES, 40, en train_fault00) o que no este en
    `data.seen_runs(split, fallo)` es un error: una simulacion no mirada queda
    inservible para el resultado confirmatorio en cuanto se lee (regla 3).
    """
    ids = sorted({int(i) for i in ids})
    if not ids or ids[0] < 1:
        raise ValueError(f"{split}_fault{fallo:02d}: los ids son enteros >= 1; llego {ids[:10]}")
    altos = [i for i in ids if i > maximo]
    if altos:
        raise ValueError(
            f"{split}_fault{fallo:02d}: los ids {altos[:10]} pasan de {maximo}, el ultimo que el piloto puede "
            "leer en este fichero. El piloto solo lee simulaciones de desarrollo; no se carga nada."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)      # el aviso de RIETH_SEEN_PENDING ya se dio
        mirados = set(data.seen_runs(split, fallo).tolist())
    nuevos = [i for i in ids if i not in mirados]
    if nuevos:
        raise ValueError(
            f"{split}_fault{fallo:02d}: los ids {nuevos[:10]} no estan en el registro de simulaciones "
            "miradas (data.RIETH_SEEN_RUNS). El piloto no lee simulaciones nuevas; no se carga nada."
        )
    return ids


def cargar_mirados(fallo: int, split: str, ids, root, maximo: int) -> pd.DataFrame:
    """`data.load_rieth` solo de ids que pasan `comprobar_ids_mirados`."""
    ids = comprobar_ids_mirados(split, fallo, ids, maximo)
    return data.load_rieth(fallo, split, runs=ids, root=root)


def repartir_normales(n_ajuste: int, n_cal: int, n_parada: int, semilla: int, maximo: int) -> dict:
    """Ids de train_fault00 para entrenar, parar y calibrar, dentro de 1..maximo: UN solo reparto.

    `data.split_runs` reparte los ids 1..(n_ajuste + n_cal) una sola vez, con la
    semilla del yaml, en entrenar (n_ajuste - n_parada), parar (n_parada) y
    calibrar (n_cal), como el C00 (celda 35). Las fracciones son tamano / total,
    asi que los tamanos son exactos. PCA y DPCA se ajustan con todos los de
    ajuste (entrenar + parar).
    """
    total = n_ajuste + n_cal
    if total > maximo:
        raise DecisionPendiente(
            f"normales_ajuste + normales_calibracion = {total} pasa de {maximo}: el piloto solo puede tocar "
            f"las normales 1 a {maximo} de train_fault00 (MAXIMO_NORMALES; fila del 2026-10-18 de "
            "docs/decisiones.md)."
        )
    n_tr = n_ajuste - n_parada
    if n_parada < 1 or n_tr < 1:
        raise DecisionPendiente(
            f"con normales_ajuste = {n_ajuste} y normales_parada = {n_parada} no queda al menos una normal "
            "para entrenar y otra para parar el autoencoder."
        )
    tr, es, cal = data.split_runs(total, fractions=(n_tr / total, n_parada / total, n_cal / total), seed=semilla)
    assert tr.size == n_tr and es.size == n_parada and cal.size == n_cal
    return {"tr": tr.tolist(), "es": es.tolist(), "cal": cal.tolist(),
            "ajuste": sorted(tr.tolist() + es.tolist())}


# -------------------------------------------------------------- autoencoder

def _capas_simetricas(capas):
    """(32, 16) -> (32, 16, 32): el decodificador es el espejo del codificador (C00)."""
    return tuple(capas) + tuple(capas[-2::-1])


def construir_ae(n_vars: int, capas):
    """La red del C00: Linear + Tanh por capa y salida lineal que reconstruye z."""
    from torch import nn

    bloques, previo = [], n_vars
    for h in _capas_simetricas(capas):
        bloques += [nn.Linear(previo, h), nn.Tanh()]
        previo = h
    bloques.append(nn.Linear(previo, n_vars))
    return nn.Sequential(*bloques)


def entrenar_autoencoder(Z_tr: np.ndarray, Z_es: np.ndarray, hp: dict, semilla: int):
    """El bucle del C00: entrena con Z_tr, parada temprana con Z_es (normales), vuelve a la mejor epoca.

    Devuelve (red, historia del error medio en Z_es por epoca).
    """
    import torch

    torch.manual_seed(semilla)
    red = construir_ae(Z_tr.shape[1], hp["capas"])
    opt = torch.optim.Adam(red.parameters(), lr=hp["lr"])
    A = torch.tensor(np.asarray(Z_tr), dtype=torch.float32)
    V = torch.tensor(np.asarray(Z_es), dtype=torch.float32)
    gen = torch.Generator().manual_seed(semilla)
    historia, mejor, referencia, mejor_estado, sin_mejora = [], np.inf, np.inf, None, 0
    for _ in range(hp["epocas_max"]):
        red.train()
        for idx in torch.randperm(len(A), generator=gen).split(hp["lote"]):
            opt.zero_grad()
            perdida = ((red(A[idx]) - A[idx]) ** 2).mean()
            perdida.backward()
            opt.step()
        red.eval()
        with torch.no_grad():
            val = float(((red(V) - V) ** 2).mean())
        historia.append(val)
        if val < mejor:
            mejor, mejor_estado = val, copy.deepcopy(red.state_dict())
        if val < referencia * (1 - hp["mejora_min"]):
            referencia, sin_mejora = val, 0
        else:
            sin_mejora += 1
            if sin_mejora >= hp["paciencia"]:
                break
    red.load_state_dict(mejor_estado)
    return red, historia


def error_reconstruccion(red, Z: np.ndarray) -> np.ndarray:
    """||z - z_hat||^2 por muestra, la misma escala que el SPE (C00)."""
    import torch

    red.eval()
    with torch.no_grad():
        T = torch.tensor(np.asarray(Z), dtype=torch.float32)
        return ((red(T) - T) ** 2).sum(dim=1).numpy().astype(float)


# ------------------------------------------------------------------ sigma

def estimar_sigma(ventaja, nivel: float = NIVEL_SIGMA) -> dict:
    """Desviacion muestral de la ventaja por simulacion, con intervalo y cota superior por chi cuadrado.

    sigma_hat es la desviacion con ddof = 1 de las n ventajas (los NaN se
    descartan). Si la ventaja fuera normal, (n - 1) s^2 / sigma^2 sigue una chi
    cuadrado con n - 1 grados de libertad, y de ahi salen el intervalo bilateral
    [sigma_ci_low, sigma_ci_high] al nivel `nivel` y la cota superior unilateral
    sigma_upper al mismo nivel (la que usa la regla de n de la hoja del 15/10).
    Con una ventaja asimetrica o con muchos ceros (lo plausible en los fallos
    dificiles) el nivel es aproximado: por eso el CSV lleva al lado la
    pseudomediana de Hodges-Lehmann y su intervalo, que no suponen normalidad.
    Con n < 2 no hay desviacion y todo es NaN.
    """
    if not (0.0 < nivel < 1.0):
        raise ValueError(f"nivel debe estar entre 0 y 1, no {nivel!r}")
    d = np.asarray(ventaja, dtype=float).reshape(-1)
    d = d[~np.isnan(d)]
    n = int(d.size)
    vacio = {"n": n, "mean_advantage": np.nan, "sigma_hat": np.nan, "sigma_ci_low": np.nan,
             "sigma_ci_high": np.nan, "sigma_upper": np.nan, "sigma_level": float(nivel),
             "min_advantage": np.nan, "max_advantage": np.nan, "n_nonzero": 0}
    if n < 2:
        if n == 1:
            vacio.update(mean_advantage=float(d[0]), min_advantage=float(d[0]), max_advantage=float(d[0]),
                         n_nonzero=int(d[0] != 0))
        return vacio
    s = float(np.std(d, ddof=1))
    gl = n - 1
    cola = (1.0 - nivel) / 2.0
    return {
        "n": n,
        "mean_advantage": float(d.mean()),
        "sigma_hat": s,
        "sigma_ci_low": s * float(np.sqrt(gl / sps.chi2.ppf(1.0 - cola, gl))),
        "sigma_ci_high": s * float(np.sqrt(gl / sps.chi2.ppf(cola, gl))),
        "sigma_upper": s * float(np.sqrt(gl / sps.chi2.ppf(1.0 - nivel, gl))),
        "sigma_level": float(nivel),
        "min_advantage": float(d.min()),
        "max_advantage": float(d.max()),
        "n_nonzero": int(np.count_nonzero(d)),
    }


# -------------------------------------------------------------- evaluacion

def _puntuar_baseline(modelo: PCAMonitor, estadistico: str, X, runs, samples):
    """(estadistico por muestra, limite) de un baseline segun hipotesis.baselines."""
    t2, spe = modelo.score(X, runs=runs, samples=samples)
    if estadistico == "t2":
        return t2, modelo.t2_limit_
    if estadistico == "spe":
        return spe, modelo.spe_limit_
    return modelo.combined_index(t2, spe), 1.0


def _trayectoria(df: pd.DataFrame):
    """(runs, samples) de un DataFrame del contrato: la trayectoria es (faultNumber, simulationRun)."""
    return df[["faultNumber", "simulationRun"]].to_numpy(), df["sample"].to_numpy()


def fuentes_limite(modelo: PCAMonitor, estadistico: str) -> dict:
    """De donde sale el limite de cada estadistico que usa el baseline (`limit_source_` de PCAMonitor)."""
    usados = ("t2", "spe") if estadistico == "t2_o_spe" else (estadistico,)
    return {s: modelo.limit_source_[s] for s in usados}


def ajustar_baselines(decisiones: dict, cfg: dict, normales: pd.DataFrame, ids: dict, lim: dict) -> dict:
    """Un PCAMonitor por baseline, ajustado con las normales de ajuste.

    Los limites empiricos se calibran con las normales reservadas
    (`calibration="holdout"`); los parametricos salen de cantidades del ajuste.
    `limit_source_` dice cual es cual, y `fuentes_limite` lo lee para cada
    estadistico que usa el baseline.
    """
    calibration = config.require(cfg, "pca.calibration")
    if calibration != "holdout":
        raise DecisionPendiente(
            f"pca.calibration = {calibration!r}: el piloto calibra los limites con normales reservadas "
            "(regla 3) y solo admite 'holdout'."
        )
    ajuste = normales[normales["simulationRun"].isin(ids["ajuste"])]
    cal = normales[normales["simulationRun"].isin(ids["cal"])]
    X_fit, X_cal = data.values(ajuste), data.values(cal)
    runs_fit, samples_fit = _trayectoria(ajuste)
    runs_cal, samples_cal = _trayectoria(cal)
    alpha = config.require(cfg, "protocolo.alpha")
    # Escalar y retardar: un Scaler de las variables originales, solo con las normales de ajuste (regla 2).
    escalador = Scaler().fit(X_fit) if lim["orden_escalado_dpca"] == "escalar_y_retardar" else None
    modelos = {}
    for b in decisiones["baselines"]:
        modelo = PCAMonitor(variance=config.require(cfg, "pca.variance"),
                            alpha=alpha_por_estadistico(alpha, b["estadistico"], lim["nivel_por_estadistico"]),
                            lags=b["lags"], spe_method=lim["spe_method"], t2_method=lim["t2_method"],
                            quantile_method=lim["quantile_method"])
        modelo.fit(X_fit, runs=runs_fit, X_cal=X_cal, runs_cal=runs_cal, calibration="holdout",
                   samples=samples_fit, samples_cal=samples_cal, scaler=escalador)
        modelos[b["nombre"]] = modelo
    return modelos


def ajustar_autoencoder(cfg: dict, normales: pd.DataFrame, ids: dict, hp: dict, quantile_method: str) -> dict:
    """Escalador (solo normales de ajuste), red (entrena con tr, para con es) y umbral (reservadas)."""
    import torch

    semilla = config.require(cfg, "protocolo.semilla")
    bloque = lambda sel: data.values(normales[normales["simulationRun"].isin(sel)])  # noqa: E731
    escalador = Scaler().fit(bloque(ids["ajuste"]))                                   # regla 2
    Z_tr, Z_es, Z_cal = (escalador.transform(bloque(ids[k])) for k in ("tr", "es", "cal"))
    red, historia = entrenar_autoencoder(Z_tr, Z_es, hp, semilla)
    e_cal = error_reconstruccion(red, Z_cal)
    umbral = limits.limit_empirical(e_cal, alpha=config.require(cfg, "protocolo.alpha"), method=quantile_method)
    return {"escalador": escalador, "red": red, "umbral": umbral, "epocas": len(historia),
            "mejor_epoca": int(np.argmin(historia)) + 1, "n_tr": len(Z_tr), "n_es": len(Z_es),
            "n_cal": len(Z_cal), "torch_version": torch.__version__,
            "torch_threads": int(torch.get_num_threads())}


def evaluar(decisiones: dict, cfg: dict, modelos: dict, ae: dict, fallos, ids_prueba, root,
            maximo: int) -> pd.DataFrame:
    """Una fila por metodo, fallo (0 y los pedidos) y simulacion, con las columnas de evaluate_run."""
    onset_test = data.fault_onset("test", "rieth")
    deteccion = {"k": config.require(cfg, "protocolo.k_alarmas"),
                 "sample_minutes": config.require(cfg, "protocolo.sample_minutes"),
                 "min_fdr": config.require(cfg, "protocolo.min_fdr"),
                 "delay_convention": config.require(cfg, "protocolo.retardo_convencion")}
    filas = []
    for fallo in [0, *fallos]:
        d = cargar_mirados(fallo, "test", ids_prueba, root, maximo)
        onset = None if fallo == 0 else onset_test
        ventana = decisiones["ventana_far_normal"] if fallo == 0 else None
        for sim, g in d.groupby("simulationRun", sort=True):
            g = g.sort_values("sample")
            X = data.values(g)
            runs, samples = _trayectoria(g)
            e = error_reconstruccion(ae["red"], ae["escalador"].transform(X))
            puntuaciones = [(METODO_AE, e, ae["umbral"])]
            for b in decisiones["baselines"]:
                s, lim = _puntuar_baseline(modelos[b["nombre"]], b["estadistico"], X, runs, samples)
                puntuaciones.append((b["nombre"], s, lim))
            for metodo, s, lim in puntuaciones:
                r = metrics.evaluate_run(s, lim, onset, far_window=ventana, **deteccion)
                filas.append({"method": metodo, "faultNumber": int(fallo), "simulationRun": int(sim),
                              **r, "limit": lim})
    return pd.DataFrame(filas)


def ventaja_por_celda(res: pd.DataFrame, decisiones: dict, fallos) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anade la metrica de la ventaja a `res` y estima sigma y Hodges-Lehmann por (fallo, baseline).

    Con "exceso_far_gemela", a cada fila con fallo se le resta la FAR de la fila
    (mismo metodo, fallo 0, misma simulacion): la gemela normal, en la ventana
    del yaml. Devuelve (res con las columnas nuevas, tabla por celda).
    """
    res = res.copy()
    metrica = decisiones["metrica"]
    gemelas = res[res["faultNumber"] == 0].set_index(["method", "simulationRun"])["far"]
    clave = pd.MultiIndex.from_arrays([res["method"], res["simulationRun"]])
    res["far_gemela"] = gemelas.reindex(clave).to_numpy()
    res["exceso_far_gemela"] = res["fdr"] - res["far_gemela"]
    con_fallo = res[res["faultNumber"] > 0]
    filas = []
    for fallo in fallos:
        for b in decisiones["baselines"]:
            ventaja = stats.paired_advantage(con_fallo, metric=metrica, method=METODO_AE, baseline=b["nombre"],
                                             higher_is_better=decisiones["mayor_es_mejor"], fault=fallo)
            hl = stats.hodges_lehmann(ventaja.to_numpy(), ci_level=NIVEL_HL)
            sub = con_fallo[con_fallo["faultNumber"] == fallo]
            filas.append({
                "faultNumber": int(fallo), "method": METODO_AE, "baseline": b["nombre"],
                "metric": metrica, "higher_is_better": decisiones["mayor_es_mejor"],
                **estimar_sigma(ventaja.to_numpy(), NIVEL_SIGMA),
                "hl_estimate": hl["hl_estimate"], "hl_ci_low": hl["ci_low"], "hl_ci_high": hl["ci_high"],
                "hl_ci_level": hl["ci_level"],
                "fdr_mean_ae": float(sub.loc[sub["method"] == METODO_AE, "fdr"].mean()),
                "fdr_mean_baseline": float(sub.loc[sub["method"] == b["nombre"], "fdr"].mean()),
                "far_gemela_mean_ae": float(sub.loc[sub["method"] == METODO_AE, "far_gemela"].mean()),
                "far_gemela_mean_baseline": float(sub.loc[sub["method"] == b["nombre"], "far_gemela"].mean()),
                "baseline_statistic": b["estadistico"], "baseline_lags": b["lags"],
            })
    return res, pd.DataFrame(filas)


# ----------------------------------------------------------------- manifiesto

def huellas_manifiesto(root: Path, nombres) -> dict:
    """Los Parquet usados frente a SHA256SUMS.txt: bytes comprobados en el disco y SHA-256 del manifiesto.

    Los bytes de cada fichero se miran en el disco y se comparan con los del
    manifiesto, que es barato. La SHA-256 no se recalcula (seria volver a leer
    1,3 GB): va como sha256_segun_manifiesto, para que nadie la tome por medida.
    `discrepancias` lista los ficheros cuyos bytes no coinciden (o que faltan).
    """
    huellas = {n: None for n in nombres}
    manifiesto = Path(root) / MANIFIESTO
    if not manifiesto.is_file():
        return {"manifiesto": None, "ficheros": huellas, "discrepancias": []}
    for linea in manifiesto.read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if len(partes) == 3 and partes[0] in huellas and partes[1].isdigit():
            fichero = Path(root) / partes[0]
            en_disco = fichero.stat().st_size if fichero.is_file() else None
            huellas[partes[0]] = {"bytes_segun_manifiesto": int(partes[1]), "bytes_en_disco": en_disco,
                                  "bytes_coinciden": en_disco == int(partes[1]),
                                  "sha256_segun_manifiesto": partes[2]}
    discrepancias = [n for n, h in huellas.items() if h is not None and not h["bytes_coinciden"]]
    return {"manifiesto": MANIFIESTO, "manifiesto_sha256": provenance.file_sha256(manifiesto, text=True),
            "ficheros": huellas, "discrepancias": discrepancias}


# ---------------------------------------------------------------- ejecucion

def _argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="baseline.yaml (por defecto, el orden de tfmfdd.config.load_baseline)")
    p.add_argument("--provisional", default=None,
                   help="yaml con valores DE PRUEBA para las claves que sigan en null (queda en la procedencia)")
    p.add_argument("--root", default=None,
                   help="carpeta de los Parquet de Rieth (por defecto, la variable TFM_RIETH_DIR)")
    p.add_argument("--out", default=None,
                   help="carpeta de salida (por defecto results/summary; en una ejecucion provisional, "
                        "results/raw/P1_provisional)")
    p.add_argument("--fallos", type=int, nargs="+", default=None,
                   help="fallos del piloto (por defecto, fallos_dificiles del yaml: 3, 9 y 15)")
    p.add_argument("--n-ids", type=int, default=None,
                   help="ids de prueba 1..N (por defecto y como maximo, datos.runs_desarrollo)")
    p.add_argument("--epocas-max", type=int, default=None,
                   help="epocas maximas del autoencoder (solo para pruebas de humo; queda registrado)")
    p.add_argument("--hilos-torch", type=int, default=None,
                   help="hilos de PyTorch (por defecto, min(4, nucleos), como el C00)")
    return p.parse_args(argv)


def ejecutar(argv=None) -> int:
    args = _argumentos(argv)
    if args.epocas_max is not None and args.epocas_max < 1:
        print(f"P1 no se ejecuta: --epocas-max tiene que ser un entero >= 1; llego {args.epocas_max}.",
              file=sys.stderr)
        return 2
    cfg, info_cfg = config.load_baseline(args.config, with_info=True)
    archivos_cfg = [info_cfg]
    aplicadas: list[str] = []
    valores_prov = {}
    try:
        if args.provisional:
            ruta_prov = Path(args.provisional)
            prov = config.load_yaml(ruta_prov)
            cfg, aplicadas = aplicar_provisional(cfg, prov)
            valores_prov = {c: config.get(cfg, c) for c in aplicadas}
            archivos_cfg.append({"path": str(ruta_prov), "source": "provisional",
                                 "sha256": provenance.file_sha256(ruta_prov, text=True)})
        decisiones = validar_decisiones(cfg)
    except DecisionPendiente as exc:
        print(f"P1 no se ejecuta: {exc}", file=sys.stderr)
        return 2

    maximo = config.require(cfg, "datos.runs_desarrollo")
    n_ids = maximo if args.n_ids is None else args.n_ids
    if not 1 <= n_ids <= maximo:
        print(f"P1 no se ejecuta: --n-ids {n_ids} tiene que estar entre 1 y {maximo} (datos.runs_desarrollo).",
              file=sys.stderr)
        return 2
    ids_prueba = list(range(1, n_ids + 1))
    fallos = list(config.require(cfg, "fallos_dificiles")) if args.fallos is None else args.fallos
    if any(f not in range(1, 21) for f in fallos) or len(set(fallos)) != len(fallos):
        print(f"P1 no se ejecuta: --fallos deben ser fallos de Rieth entre 1 y 20, sin repetir; llego {fallos}.",
              file=sys.stderr)
        return 2
    hp = dict(HP_AE)
    if args.epocas_max is not None:
        hp["epocas_max"] = args.epocas_max
    provisional = bool(aplicadas)
    try:
        lim = parametros_limites(cfg, provisional)
        n_parada, fuente_parada = resolver_normales_parada(cfg, decisiones["normales_ajuste"],
                                                           decisiones["normales_calibracion"], provisional)
        ids_normales = repartir_normales(decisiones["normales_ajuste"], decisiones["normales_calibracion"],
                                         n_parada, config.require(cfg, "protocolo.semilla"), MAXIMO_NORMALES)
        out = carpeta_salida(args.out, provisional)
    except DecisionPendiente as exc:
        print(f"P1 no se ejecuta: {exc}", file=sys.stderr)
        return 2

    root = Path(args.root) if args.root else None
    ficheros = ["train_fault00.parquet", "test_fault00.parquet"] + [f"test_fault{f:02d}.parquet" for f in fallos]
    carpeta_datos = data._carpeta_rieth(root)[0]          # la misma carpeta que leera data.load_rieth
    huellas = huellas_manifiesto(carpeta_datos, ficheros)
    if huellas["discrepancias"]:
        def _bytes(h):
            disco = "no esta en el disco" if h["bytes_en_disco"] is None else f"{h['bytes_en_disco']} bytes en el disco"
            return f"{disco}, {h['bytes_segun_manifiesto']} en el manifiesto"
        print(f"P1 no se ejecuta: estos Parquet no coinciden con {carpeta_datos / MANIFIESTO} y no se abre "
              "ninguno:\n  - " + "\n  - ".join(f"{n}: {_bytes(huellas['ficheros'][n])}"
                                                for n in huellas["discrepancias"]), file=sys.stderr)
        return 2

    try:
        import torch
    except ImportError as exc:
        print(f"P1 no se ejecuta: hace falta PyTorch para el autoencoder del C00 ({exc}).", file=sys.stderr)
        return 2
    torch.set_num_threads(args.hilos_torch if args.hilos_torch else min(4, os.cpu_count() or 1))

    if aplicadas:
        print("AVISO: ejecucion PROVISIONAL. Estas claves llevan valores de prueba de "
              f"{args.provisional}, no decisiones del registro:\n  - "
              + "\n  - ".join(f"{c} = {valores_prov[c]!r}" for c in aplicadas)
              + "\nLas cifras no se citan; sirven para probar el script.\n")
    if lim["fuente"] == "pca_desarrollo":
        print("AVISO: limites_protocolo.* sigue en null; esta ejecucion provisional usa los limites de desarrollo "
              f"(pca.*: T2 {lim['t2_method']}, SPE {lim['spe_method']}, cuantil {lim['quantile_method']}), la regla "
              "T2 o SPE con los dos al mismo nivel y el escalado de la matriz con retardos.\n")
    if fuente_parada != CLAVE_PARADA:
        print(f"AVISO: {CLAVE_PARADA} no esta fijada; esta ejecucion provisional para el autoencoder con "
              f"round(particion.fraccion_validacion x normales) = {n_parada}, la cuenta del C00.\n")
    if huellas["manifiesto"] is None:
        print(f"AVISO: no hay {MANIFIESTO} en {carpeta_datos}: los Parquet no se comprueban contra el manifiesto.\n")
    else:
        sin_entrada = [n for n, h in huellas["ficheros"].items() if h is None]
        if sin_entrada:
            print(f"AVISO: estos Parquet no estan en {MANIFIESTO}: {', '.join(sin_entrada)}.\n")
    data.seen_runs()                       # el aviso de las entradas pendientes del registro, una vez
    print(f"Ids de prueba: {ids_prueba[0]} a {ids_prueba[-1]} de test_fault00 y de test_fault"
          + ", ".join(f"{f:02d}" for f in fallos) + f" (maximo permitido: {maximo}; en train_fault00, "
          f"{MAXIMO_NORMALES})")
    print(f"train_fault00: entrenar {ids_normales['tr']} | parar {ids_normales['es']} | calibrar {ids_normales['cal']}")

    contexto = provenance.run_context(__file__)          # git y maquina, antes de escribir nada
    normales = cargar_mirados(0, "train", ids_normales["ajuste"] + ids_normales["cal"], root, MAXIMO_NORMALES)
    modelos = ajustar_baselines(decisiones, cfg, normales, ids_normales, lim)
    ae = ajustar_autoencoder(cfg, normales, ids_normales, hp, lim["quantile_method"])
    print(f"Autoencoder: {ae['epocas']} epocas (mejor, la {ae['mejor_epoca']}), umbral empirico {ae['umbral']:.3f} "
          f"con {ae['n_cal']} muestras reservadas; hilos de torch: {ae['torch_threads']}")
    for b in decisiones["baselines"]:
        m = modelos[b["nombre"]]
        print(f"{b['nombre']}: lags = {b['lags']}, {m.n_components_} componentes, estadistico {b['estadistico']}; "
              f"T2 {m.t2_limit_:.2f} ({m.limit_source_['t2']}), SPE {m.spe_limit_:.2f} ({m.limit_source_['spe']})")
        no_reservados = {s: f for s, f in fuentes_limite(m, b["estadistico"]).items() if f != "holdout"}
        if no_reservados:
            print(f"AVISO: en {b['nombre']}, " + "; ".join(f"el limite del {s.upper()} sale de {f!r}"
                                                         for s, f in no_reservados.items())
                  + ": no se calibra con las normales reservadas (es parametrico, con cantidades del ajuste).")

    res = evaluar(decisiones, cfg, modelos, ae, fallos, ids_prueba, root, maximo)
    res, celdas = ventaja_por_celda(res, decisiones, fallos)

    configuracion = {
        "fuente": "rieth", "metric": decisiones["metrica"], "higher_is_better": decisiones["mayor_es_mejor"],
        "baselines": decisiones["baselines"], "far_window_normal": _ventana_texto(decisiones["ventana_far_normal"]),
        "normales_ajuste": decisiones["normales_ajuste"], "normales_calibracion": decisiones["normales_calibracion"],
        "normales_entrenar_ae": len(ids_normales["tr"]), "normales_parar_ae": len(ids_normales["es"]),
        "normales_parada_fuente": fuente_parada,
        "replicas_ajuste": 1, "n_ids_prueba": n_ids, "max_id": maximo, "max_id_normales": MAXIMO_NORMALES,
        "faults": fallos,
        "alpha": config.require(cfg, "protocolo.alpha"), "k": config.require(cfg, "protocolo.k_alarmas"),
        "min_fdr": config.require(cfg, "protocolo.min_fdr"),
        "delay_convention": config.require(cfg, "protocolo.retardo_convencion"),
        "sample_minutes": config.require(cfg, "protocolo.sample_minutes"),
        "semilla": config.require(cfg, "protocolo.semilla"),
        "variance": config.require(cfg, "pca.variance"), "limites_fuente": lim["fuente"],
        "spe_method": lim["spe_method"], "t2_method": lim["t2_method"],
        "quantile_method": lim["quantile_method"], "nivel_por_estadistico": lim["nivel_por_estadistico"],
        "orden_escalado_dpca": lim["orden_escalado_dpca"], "calibration": "holdout",
        "alpha_por_estadistico": {b["nombre"]: alpha_por_estadistico(config.require(cfg, "protocolo.alpha"),
                                                                    b["estadistico"], lim["nivel_por_estadistico"])
                                  for b in decisiones["baselines"]},
        "n_components": {b: int(m.n_components_) for b, m in modelos.items()},
        "t2_limit": {b: float(m.t2_limit_) for b, m in modelos.items()},
        "spe_limit": {b: float(m.spe_limit_) for b, m in modelos.items()},
        "limit_source": {b: dict(m.limit_source_) for b, m in modelos.items()},
        "limite_reservado": {b["nombre"]: all(f == "holdout" for f in
                                              fuentes_limite(modelos[b["nombre"]], b["estadistico"]).values())
                             for b in decisiones["baselines"]},
        "scaler_source": {b: m.scaler_source_ for b, m in modelos.items()},
        "n_vars": data.N_VARS, "ae_arquitectura": [data.N_VARS, *_capas_simetricas(hp["capas"]), data.N_VARS],
        "hp_ae": hp, "ae_limit": float(ae["umbral"]), "ae_limit_method": "empirical",
        "ae_epochs": ae["epocas"], "ae_best_epoch": ae["mejor_epoca"],
        "torch_version": ae["torch_version"], "torch_threads": ae["torch_threads"],
        "sigma_level": NIVEL_SIGMA, "hl_ci_level": NIVEL_HL,
        "provisional": bool(aplicadas), "provisional_keys": ";".join(aplicadas),
    }
    # Las mismas columnas de configuracion en cada fila del CSV (los dicts, aplanados
    # un nivel; listas, tuplas y dicts anidados, como texto).
    def _celda(v):
        return v if v is None or isinstance(v, (str, bool, int, float, np.integer, np.floating)) else str(v)

    columnas_cfg = {}
    for k, v in configuracion.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                columnas_cfg[f"{k}_{kk}"] = _celda(vv)
        else:
            columnas_cfg[k] = _celda(v)
    for k, v in columnas_cfg.items():
        celdas[k] = v
        res[k] = v

    procedencia = dict(
        config=configuracion, config_files=archivos_cfg, script=__file__, **contexto,
        ids={"fuente": "rieth", "max_id": maximo, "max_id_normales": MAXIMO_NORMALES,
             "comprobado_con": "data.seen_runs (antes de cada carga)",
             "ajuste": {"fichero": "train_fault00.parquet", "entrenar_ae": ids_normales["tr"],
                        "parar_ae": ids_normales["es"], "ajuste_pca_dpca": ids_normales["ajuste"],
                        "calibracion": ids_normales["cal"]},
             "prueba": {"ficheros": ficheros[1:], "simulationRun": ids_prueba}},
        extra={"provisional": {"fichero": args.provisional, "claves": aplicadas, "valores": valores_prov},
               "limites": lim, "normales_parada": {"n": n_parada, "fuente": fuente_parada},
               "rieth": huellas},
    )
    out.mkdir(parents=True, exist_ok=True)
    for tabla, nombre in ((celdas, SALIDA), (res, SALIDA_RUNS)):
        tabla.to_csv(out / nombre, index=False)
        provenance.write_provenance(out / nombre, **procedencia)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\nVentaja por simulacion (" + decisiones["metrica"] + ", positivo = el autoencoder es mejor):")
        print(celdas[["faultNumber", "baseline", "n", "mean_advantage", "sigma_hat", "sigma_ci_low", "sigma_ci_high",
                      "sigma_upper", "hl_estimate", "hl_ci_low", "hl_ci_high", "n_nonzero"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsigma_upper es la cota superior unilateral de sigma al {NIVEL_SIGMA:.0%} (chi cuadrado); la regla de n "
          f"la aplica el prerregistro, no este script. El intervalo de Hodges-Lehmann es al {NIVEL_HL:.0%}.")
    if aplicadas:
        print("AVISO: ejecucion PROVISIONAL (ver arriba); el CSV lo dice en la columna provisional.")
    print(f"Guardado en {out / SALIDA} y {out / SALIDA_RUNS} (con su .provenance.json)")
    return 0


def main(argv=None) -> None:
    codigo = ejecutar(argv)
    if codigo:
        sys.exit(codigo)


if __name__ == "__main__":
    main()
