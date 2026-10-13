"""Mecanismos de las cuatro ablaciones del protocolo.

El eje del TFM es medir, con ablaciones controladas, cuanto mueven las metricas
publicadas cuatro fallos de diseno experimental: calibrar los limites en
muestra, estandarizar con datos de prueba, partir al azar por muestra series
autocorrelacionadas y evaluar sobre una unica simulacion (memoria E2, 2.2.4 y
OE1). Una ablacion es controlada "cuando cambia un solo factor, con el mismo
metodo y los mismos datos, sobre muchas simulaciones independientes y con un
contraste estadistico" (memoria, 2.2.4), y el signo del efecto no se
presupone: se mide y se reporta.

Las tres familias (PCA y DPCA; clasificadores; autoencoder y red recurrente)
tienen que construir los dos brazos de cada factor de la misma forma, o sus
ablaciones no mediran lo mismo. Por eso los mecanismos viven aqui, en el
paquete comun, y no en el repositorio de cada bloque.

Que hay para cada factor:

- Factor 1, calibracion en muestra frente a reservada: no esta en este modulo.
  Lo dan `PCAMonitor.fit(..., X_cal=..., calibration=...)` y los estimadores de
  `limits`, que sirven para cualquier puntuacion.
- Factor 2, estandarizacion con datos de prueba. Brazo S1 (ajuste y prueba
  juntos): `PCAMonitor.fit(..., scaler=Scaler().fit(np.vstack([X_fit, X_prueba])))`.
  Brazo S2 (cada simulacion de prueba con sus propios estadisticos):
  `standardize_by_run` y, en PCA y DPCA, `score_standardized_by_run`.
- Factor 3, particion por muestra frente a por simulacion: `split_by_run` (el
  brazo limpio), `split_by_sample` (el sucio, por muestra o por bloques) y
  `block_ids` (los mismos bloques como grupos de una validacion cruzada).
- Factor 4, una simulacion frente a muchas: `block_bootstrap_ci` (el intervalo
  de quien solo tiene una simulacion), `interval_coverage`,
  `contrary_conclusion_rate` y `subsample_runs` (ajustar con 1 frente a K
  simulaciones). Cada una mide algo distinto, y ninguna mide sola el efecto de
  usar una simulacion: sus docstrings dicen que mezcla cada una y que
  referencia necesita. Cual es la estimanda del factor lo decide el registro
  de decisiones.
- Ventanas por trayectoria (OE3, y factor 3 en modelos con ventanas):
  `window_ends`, `stack_windows` y `align_to_samples`.

Dos claves distintas, como en el resto del paquete:

- Trayectoria: filas SEGUIDAS con el mismo identificador, la misma definicion
  que `preprocessing.lag_by_run`. En Rieth es el par (faultNumber,
  simulationRun), porque la simulacion r de cada clase se numera igual que la
  normal r: pasa las dos columnas. Los bloques, las ventanas y la
  estandarizacion por simulacion se cortan por trayectoria, y las filas de cada
  trayectoria tienen que venir seguidas y en orden temporal.
- Grupo de la particion: simulationRun solo, comun a todas las clases (regla de
  las gemelas, `data.split_runs`). El brazo limpio reparte grupos enteros.

Nada de esto ejecuta una ablacion ni fija ids. Que simulaciones se usan, las
fracciones, la longitud de los bloques y de las ventanas, el numero de
remuestreos y la estimanda de cada factor se fijan en el registro de
decisiones y llegan como parametros, sin valores por defecto que escondan la
decision.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from . import data
# Misma definicion de trayectoria que lag_by_run (y, por tanto, que el DPCA):
# filas seguidas con el mismo id, runs 1D o 2D.
from .preprocessing import Scaler, trajectory_starts


# ---------------------------------------------------------------- utilidades


def _entero_positivo(valor, nombre: str) -> int:
    if (
        isinstance(valor, (bool, np.bool_))
        or not isinstance(valor, (int, np.integer))
        or valor < 1
    ):
        raise ValueError(f"{nombre} debe ser un entero >= 1; llego {valor!r}")
    return int(valor)


def _trayectorias(runs) -> tuple[np.ndarray, np.ndarray]:
    """Valida runs y devuelve (runs como array, inicio de cada trayectoria y el final)."""
    runs = np.asarray(runs)
    if runs.ndim not in (1, 2):
        raise ValueError(f"runs debe ser 1D o 2D (una o varias columnas de ids); llego {runs.ndim}D")
    if len(runs) == 0:
        raise ValueError("runs no tiene filas")
    return runs, trajectory_starts(runs)


def _nivel(ci_level) -> float:
    nivel = float(ci_level)
    if not 0.0 < nivel < 1.0:
        raise ValueError(f"ci_level es el nivel del intervalo, entre 0 y 1 (por ejemplo 0.95); llego {ci_level}")
    return nivel


# ------------------------------------------------- factor 3: particiones


class RowSplit(NamedTuple):
    """Reparto de filas en ajuste, calibracion y evaluacion (mascaras booleanas).

    Lo devuelven los dos brazos del factor 3 (`split_by_run` y
    `split_by_sample`), para que el resto del codigo sea el mismo en los dos:
    `X[r.fit]`, `X[r.cal]`, `X[r.eval]` (o `df[r.fit]`; para indices,
    `np.flatnonzero(r.fit)`). Las mascaras son disjuntas, cubren todas las filas
    y conservan el orden de entrada. Con ids ya repartidos (por ejemplo, los de
    `data.split_runs`), se construye a mano:
    `RowSplit(np.isin(g, ids_fit), np.isin(g, ids_cal), np.isin(g, ids_eval))`.

    En un detector, "evaluacion" es la evaluacion interna, la que reportaria
    quien parte asi; la externa (simulaciones de prueba que ninguno de los dos
    brazos toca) se mide aparte e igual en los dos. Una fraccion puede ser 0
    (por ejemplo, ajuste y calibracion sin evaluacion interna); cuantas
    simulaciones van a cada parte, redondeo incluido, lo decide
    `data.split_runs`.
    """

    fit: np.ndarray
    cal: np.ndarray
    eval: np.ndarray

    @property
    def sizes(self) -> tuple[int, int, int]:
        """Filas de ajuste, calibracion y evaluacion."""
        return int(np.sum(self.fit)), int(np.sum(self.cal)), int(np.sum(self.eval))


def split_by_run(groups, fractions, seed) -> RowSplit:
    """Brazo limpio del factor 3: simulaciones enteras en ajuste, calibracion y evaluacion.

    Es la regla del protocolo (regla 1 de docs/reglas_de_medida.md; memoria,
    3.3): una simulacion no puede aparecer en dos partes, porque sus muestras
    estan correlacionadas en el tiempo. Reparte los ids presentes en `groups`
    llamando a `data.split_runs` (la misma funcion, no una copia): los ids de
    1 a max(groups) que no aparecen van como `exclude`, asi que con los mismos
    ids, fracciones y semilla el reparto es el del protocolo, redondeo
    incluido, y cualquier cambio de `split_runs` llega aqui solo. Devuelve las
    mascaras de filas.

    Parametros
    ----------
    groups : array (n,) de enteros >= 1
        simulationRun de cada fila, el MISMO id en todas las clases (regla de
        las gemelas: nunca el par (fallo, simulacion)). Con ventanas, el
        simulationRun de la ultima muestra de cada ventana.
    fractions : (float, float, float)
        Fracciones de SIMULACIONES de ajuste, calibracion y evaluacion; suman 1
        y alguna puede ser 0. Sin valor por defecto: es una decision del diseno.
        Las valida `data.split_runs`.
    seed : int
        Semilla del reparto.

    Devuelve
    --------
    RowSplit con las tres mascaras; `.sizes` da sus filas, que son las que
    tiene que recibir `split_by_sample` para que solo cambie el factor.
    """
    g = np.asarray(groups)
    if g.ndim != 1:
        raise ValueError(
            "groups es el simulationRun de cada fila (1D). El reparto limpio agrupa por "
            "simulationRun en todas las clases a la vez; el par (fallo, simulacion) "
            "separaria a las gemelas (regla 1)."
        )
    if g.size == 0:
        raise ValueError("groups no tiene filas")
    if g.dtype.kind not in "iu":
        raise TypeError(
            f"groups son ids de simulacion (simulationRun), enteros >= 1; llego dtype {g.dtype}"
        )
    ids = np.unique(g)
    if ids[0] < 1:
        raise ValueError(f"groups son ids de simulacion, enteros >= 1; hay ids < 1: {ids[ids < 1][:10].tolist()}")

    # data.split_runs reparte los ids 1..n_runs menos exclude: excluir los que
    # no estan en groups deja exactamente los de groups.
    n_runs = int(ids[-1])
    fuera = np.setdiff1d(np.arange(1, n_runs + 1, dtype=np.int64), ids)
    ids_fit, ids_cal, ids_eval = data.split_runs(
        n_runs, fractions, seed, exclude=fuera if fuera.size else None
    )
    return RowSplit(np.isin(g, ids_fit), np.isin(g, ids_cal), np.isin(g, ids_eval))


def block_ids(runs, block_length) -> np.ndarray:
    """Bloque de cada fila: tramos de `block_length` filas seguidas dentro de cada trayectoria.

    Es la unidad del brazo sucio del factor 3 con dosis-respuesta: con
    block_length = 1 cada fila es su propio bloque (reparto por muestra); al
    crecer el bloque, menos muestras quedan separadas de sus vecinas
    temporales; con block_length mayor o igual que la trayectoria, cada
    trayectoria es un bloque. Ningun bloque cruza de una trayectoria a otra, y
    el ultimo de cada trayectoria puede ser mas corto.

    Tambien sirve de grupo en una validacion cruzada, que es como el OE2
    construye sus dos brazos con la misma funcion de scikit-learn:
    `GroupKFold(n_splits=k, shuffle=True, random_state=s).split(X, y, groups=g)`
    con g = simulationRun (limpio) o g = block_ids(trayectoria, L) (sucio; con
    L = 1, la validacion cruzada por muestra). Sin shuffle=True, GroupKFold
    reparte los grupos por tamano y en orden, que no es el reparto al azar que
    se quiere imitar. El parametro shuffle de GroupKFold existe desde
    scikit-learn 1.6 y el paquete admite >= 1.3: con una version anterior,
    baraja antes las etiquetas, `np.random.default_rng(s).permutation(g.max() + 1)[g]`.

    Parametros
    ----------
    runs : array (n,) o (n, k)
        Trayectoria de cada fila (ver el modulo).
    block_length : int
        Filas por bloque, >= 1.

    Devuelve
    --------
    Array int64 (n,) con ids de bloque de 0 en adelante, en el orden de las filas.
    """
    largo = _entero_positivo(block_length, "block_length")
    _, inicios = _trayectorias(runs)
    longitudes = np.diff(inicios)
    posicion = np.arange(int(inicios[-1])) - np.repeat(inicios[:-1], longitudes)
    bloques_por_trayectoria = -(-longitudes // largo)
    primero = np.concatenate([[0], np.cumsum(bloques_por_trayectoria)[:-1]])
    return (np.repeat(primero, longitudes) + posicion // largo).astype(np.int64)


def _tamanos(sizes, n: int) -> tuple[int, int, int]:
    if isinstance(sizes, RowSplit):
        raise TypeError("sizes son tres enteros: pasa limpio.sizes, no el RowSplit entero")
    try:
        t = tuple(sizes)
    except TypeError:
        t = ()
    if len(t) != 3 or any(
        isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 0 for x in t
    ):
        raise ValueError(f"sizes son tres enteros no negativos (ajuste, calibracion, evaluacion); llego {sizes!r}")
    if sum(t) != n:
        raise ValueError(
            f"sizes suman {sum(t)} y hay {n} filas. Para que solo cambie el factor, pasa los "
            "tamanos del brazo limpio sobre las mismas filas: split_by_run(...).sizes."
        )
    return int(t[0]), int(t[1]), int(t[2])


def split_by_sample(runs, sizes, seed, *, block_length) -> RowSplit:
    """Brazo sucio del factor 3: las mismas filas repartidas al azar, por muestra o por bloques.

    Que practica reproduce. Repartir al azar las muestras de unas mismas
    simulaciones entre entrenamiento y prueba: Askarian et al. (2016) reparten
    asi, 60:40, las muestras de cada estado, y Ragab et al. (2018) y la serie
    de Rodriguez Ramos y colaboradores validan con diez particiones sobre las
    observaciones (memoria E2, 2.2.2). Con ventanas, repartir al azar ventanas
    solapadas (Shamim et al., 2026, protocolo B; arXiv 2607.16493, preprint).
    Es la fuga L3.2 de Kapoor y Narayanan (2023), la falta de independencia
    entre entrenamiento y prueba: una muestra de evaluacion tiene a sus vecinas
    temporales, casi iguales, en el ajuste (memoria, 2.2.4).

    Por que no se usa fuera de la ablacion. La cifra que sale mide lo bien que
    el modelo reconoce simulaciones que ya vio, no lo que haria con una
    nueva, y el protocolo parte por simulacion (`split_by_run`, regla 1). Aqui
    solo sirve para medir cuanto cambia la metrica al partir asi, con su
    signo, que no se presupone: en Rieth la autocorrelacion y las gemelas
    empujan en sentidos contrarios (regla 1).

    Como. Las filas se agrupan en bloques de `block_length` filas seguidas de
    una misma trayectoria (`block_ids`); los bloques se barajan con la semilla
    y, en ese orden y con las filas de cada bloque en su orden temporal, las
    sizes[0] primeras van a ajuste, las sizes[1] siguientes a calibracion y el
    resto a evaluacion. Con block_length = 1 es el reparto por muestra; con
    bloques mas largos, cada vez menos muestras de evaluacion tienen vecinas
    en el ajuste (dosis-respuesta, anexo de diseno, DIS-8). Para que los
    tamanos sean exactamente los del brazo limpio, el bloque que cae en la
    frontera entre dos partes se corta en dos: como mucho dos bloques por
    reparto.

    Con block_length mayor o igual que la trayectoria, cada trayectoria es un
    bloque. Con clases que comparten ids eso es partir por el par (fallo,
    simulacion), que separa a las gemelas: NO es el brazo limpio.

    Con DPCA o con ventanas, reparte las filas ya apiladas (las de
    `preprocessing.lag_by_run` o las ventanas de `window_ends`), nunca las
    muestras antes de apilar: con huecos en el tiempo, cada fila con retardos
    juntaria muestras que no son consecutivas (Shamim et al., 2026, protocolo
    A), y, sin la columna del tiempo, ni `lag_by_run` ni `PCAMonitor` pueden
    detectarlo. Para el DPCA:
    `Xl, _, pos = lag_by_run(X, runs, lags, return_index=True)`, repartir las
    filas de Xl (con runs[pos]) y ajustar `PCAMonitor(lags=0)` sobre ellas, que
    es el mismo modelo que `PCAMonitor(lags=lags)` sin gancho del escalador.

    Parametros
    ----------
    runs : array (n,) o (n, k)
        Trayectoria de cada fila (ver el modulo): con una sola clase,
        simulationRun; con varias, faultNumber y simulationRun.
    sizes : (int, int, int)
        Filas de ajuste, calibracion y evaluacion; suman n. Para que solo
        cambie el factor, las del brazo limpio: `split_by_run(...).sizes`.
    seed : int
        Semilla del barajado.
    block_length : int
        Filas por bloque, >= 1. Obligatorio y sin valor por defecto: 1 es el
        reparto por muestra, y la longitud es una decision del diseno.

    Devuelve
    --------
    RowSplit con las tres mascaras.
    """
    bloque = block_ids(runs, block_length)
    n = bloque.size
    n_fit, n_cal, _ = _tamanos(sizes, n)

    # Posicion de cada bloque tras barajar; dentro del bloque, orden temporal.
    rango = np.random.default_rng(seed).permutation(int(bloque[-1]) + 1)
    orden = np.lexsort((np.arange(n), rango[bloque]))
    parte = np.empty(n, dtype=np.int8)
    parte[orden[:n_fit]] = 0
    parte[orden[n_fit : n_fit + n_cal]] = 1
    parte[orden[n_fit + n_cal :]] = 2
    return RowSplit(parte == 0, parte == 1, parte == 2)


# ------------------------------------- factor 2: brazo S2 (por simulacion)


def standardize_by_run(X, runs) -> np.ndarray:
    """Brazo S2 del factor 2: cada trayectoria estandarizada con sus propios estadisticos.

    Que practica reproduce. Deng et al. (2021, 3.1.4 y Tabla 7) comparan, con
    el mismo clasificador y los mismos datos del TEP clasico, la prueba
    estandarizada con la media y la desviacion del entrenamiento frente a cada
    conjunto de prueba estandarizado con sus propios estadisticos
    ("independent standardization"); cada conjunto de prueba es un fichero
    dXX_te, una simulacion con 160 muestras normales y 800 con fallo. La
    memoria (2.2.4) cubre las dos formas del factor 2: estandarizar con
    entrenamiento y prueba juntos (L1.2; el brazo S1, `PCAMonitor.fit(...,
    scaler=...)`) o con la prueba sola (esta).

    Por que no se usa fuera de la ablacion. La transformacion de cada
    simulacion de prueba depende de la propia simulacion, fallo incluido: un
    desplazamiento de media que dura casi toda la simulacion se resta en parte
    con su propia media, y la escala del fallo entra en su desviacion. Es usar
    la prueba para preprocesarla. El protocolo estandariza solo con las
    normales del ajuste (regla 2; memoria, 3.3). El signo del efecto no se
    presupone: con SVM, en Deng et al., esta forma bajo la exactitud en seis de
    los siete fallos (memoria, 2.2.4).

    Cada trayectoria (filas seguidas con el mismo id; ver el modulo) se
    estandariza con un `preprocessing.Scaler` ajustado solo con sus filas: la
    misma media, la desviacion con ddof = 1 y la desviacion 1 en variables
    constantes que el resto del paquete. Las demas filas no intervienen.

    Para PCA y DPCA usa `score_standardized_by_run`, porque `PCAMonitor`
    escala por dentro. Para el resto de modelos, pasa la matriz que devuelve
    esta funcion donde iria `escalador.transform(X_prueba)`.

    Parametros
    ----------
    X : array (n, m)
        Datos de prueba.
    runs : array (n,) o (n, k)
        Trayectoria de cada fila; en Rieth, faultNumber y simulationRun.

    Devuelve
    --------
    Matriz (n, m) en el orden de entrada. Conserva el tipo de X si es de coma
    flotante (float32 ocupa la mitad con las simulaciones de prueba de Rieth);
    si no, float64. Los estadisticos se calculan siempre en float64.
    """
    X = np.asarray(X)
    if X.dtype.kind != "f":
        X = X.astype(float)
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz 2D (muestras, variables); llego {X.ndim}D")
    runs, inicios = _trayectorias(runs)
    if len(runs) != len(X):
        raise ValueError(f"runs tiene {len(runs)} filas y X {len(X)}")
    Z = np.empty_like(X)
    for ini, fin in zip(inicios[:-1], inicios[1:]):
        if fin - ini < 2:
            raise ValueError(
                f"La trayectoria de las filas {ini} a {fin - 1} tiene {fin - ini} muestra(s) y no "
                "admite desviacion. Comprueba que las filas de cada trayectoria vengan seguidas."
            )
        Z[ini:fin] = Scaler().fit(X[ini:fin]).transform(X[ini:fin])
    return Z


class _YaEstandarizado:
    """Escalador identidad: los datos llegan ya estandarizados (brazo S2)."""

    def transform(self, X):
        return np.asarray(X, dtype=float)


def score_standardized_by_run(model, X, runs):
    """Brazo S2 del factor 2 en PCA y DPCA: (T2, SPE) con cada trayectoria estandarizada consigo misma.

    El modelo (cargas, autovalores y limites) es el del brazo limpio; solo
    cambia como se escala la prueba, que es lo que midieron Deng et al. (2021)
    (ver `standardize_by_run`). Estandariza con `standardize_by_run` y puntua
    con `PCAMonitor.score(..., scaler=...)` sin volver a escalar. Con lags > 0,
    el modelo tiene que haberse ajustado con el gancho del escalador
    (`fit(..., scaler=Scaler().fit(X_fit))`) para que los dos brazos escalen
    igual, antes de apilar retardos; si no, `score` lo dice con un error.

    Parametros
    ----------
    model : PCAMonitor ajustado
    X : array (n, m)
        Datos de prueba, en variables originales.
    runs : array (n,) o (n, k)
        Trayectoria de cada fila; se usa para estandarizar y para los retardos.

    Devuelve
    --------
    (t2, spe), como `PCAMonitor.score`: una fila por fila de X, en el mismo
    orden, con NaN en las `lags` primeras de cada trayectoria.
    """
    if runs is None:
        raise ValueError("runs es obligatorio: define que filas forman cada simulacion de prueba")
    Z = standardize_by_run(X, runs)
    return model.score(Z, runs, scaler=_YaEstandarizado())


# ----------------------------------------- factor 4: una frente a muchas


def _medias_circulares(v: np.ndarray, largo: int, replicas: int, rng) -> np.ndarray:
    """Medias de las replicas del bootstrap de bloques circulares (ver block_bootstrap_ci)."""
    n = v.size
    k = -(-n // largo)                  # bloques por remuestreo
    resto = n - (k - 1) * largo         # valores que se usan del ultimo bloque
    # La serie se prolonga con sus largo - 1 primeros valores: un bloque puede
    # empezar en cualquiera de las n posiciones y dar la vuelta al final.
    acumulada = np.concatenate([[0.0], np.cumsum(np.concatenate([v, v[: largo - 1]]))])
    inicios = rng.integers(0, n, size=(replicas, k))
    completos = acumulada[inicios[:, :-1] + largo] - acumulada[inicios[:, :-1]]
    ultimo = acumulada[inicios[:, -1] + resto] - acumulada[inicios[:, -1]]
    return (completos.sum(axis=1) + ultimo) / n


def block_bootstrap_ci(x, *, block_length, n_boot, seed, ci_level=0.95) -> dict:
    """Intervalo para la media de la serie de UNA simulacion, por bootstrap de bloques circulares.

    Es el intervalo que construiria quien solo tiene una simulacion (factor
    4): sus unicas replicas son las muestras de esa simulacion, y estan
    autocorrelacionadas. Remuestrear muestra a muestra (block_length = 1) las
    trata como independientes; remuestrear bloques de muestras seguidas
    conserva la dependencia dentro de cada bloque.

    Que estima. La variabilidad de la media temporal de la serie, suponiendo
    que la serie es estacionaria y que la dependencia a mas de L muestras es
    despreciable. Su objetivo es la media del proceso que genera la serie; si
    todas las simulaciones salen del mismo proceso (en Rieth, el mismo
    simulador con otra semilla), es la misma media que estiman las N. Aun asi,
    su cobertura real puede quedar lejos de la nominal sin que las
    simulaciones difieran en nada:
    - con autocorrelacion fuerte y pocos bloques, el bootstrap de bloques
      infraestima la varianza de la media;
    - con sucesos raros (alarmas con una FAR baja) la serie tiene pocos
      episodios, el intervalo es muy discreto y asimetrico, y una simulacion
      con pocas alarmas o ninguna da un intervalo que se queda por debajo (con
      ninguna, [0, 0]);
    - si la serie no es estacionaria (la de un fallo que se detecta tras un
      retardo pasa de no alarmar a alarmar), el intervalo no tiene esa lectura.
    Lo unico que una serie no puede mostrar es la variacion mas lenta que su
    longitud o que L. Por eso la falta de cobertura de este intervalo no es,
    sin mas, el efecto de usar una sola simulacion: `interval_coverage` dice
    que mezcla y que referencia necesita.

    Por que bloques circulares. Los bloques pueden empezar en cualquiera de
    las n posiciones y dan la vuelta al final de la serie, asi que cada
    muestra entra en el mismo numero de bloques y el centro del bootstrap (la
    media esperada de las replicas) es exactamente la media de la serie
    (Politis y Romano, 1992 [VERIFICAR: no hay PDF en la bibliografia]). Con
    bloques moviles, que no dan la vuelta (Kunsch, 1989 [VERIFICAR]), las
    muestras de los extremos entran en menos bloques: en una serie con un
    retardo de deteccion, el centro se desplaza hacia el tramo final y el
    intervalo con el.

    Sirve para metricas que son la media de una serie por muestra: la FAR (x =
    alarma, 0 o 1, en las muestras normales), la FDR (en las muestras con
    fallo) y la diferencia entre dos metodos en la misma simulacion (x =
    alarma de A menos alarma de B, muestra a muestra). La serie de alarmas
    tiene que conservar los NaN de las muestras no evaluadas:
    `np.where(np.isnan(stat), np.nan, stat > limite)`, no `metrics.alarms`,
    que devuelve False en un NaN. Asi, en una diferencia solo cuentan las
    muestras que evaluan los dos metodos (por ejemplo, PCA frente a DPCA). El
    retardo no es una media por muestra y no sirve aqui.

    Como. Con n valores (sin NaN) y bloques de longitud L, la serie se
    prolonga con sus L - 1 primeros valores; cada remuestreo concatena
    ceil(n / L) bloques de L valores seguidos, con inicio al azar entre las n
    posiciones, y se queda con los n primeros valores; su media es una
    replica. El intervalo es el de percentiles de las n_boot replicas. Los NaN
    (muestras no evaluadas) se quitan antes; lo que queda tiene que seguir en
    orden temporal (pasa con el DPCA y con ventanas).

    Parametros
    ----------
    x : array (n,)
        Serie de una simulacion. NaN = muestra no evaluada.
    block_length : int
        L, entre 1 y el numero de valores. Obligatorio: la longitud del bloque
        decide el resultado (con L = 1 es el bootstrap por muestra).
    n_boot : int
        Numero de remuestreos, >= 1. Obligatorio.
    seed : int, SeedSequence o Generator
        Lo que acepta numpy.random.default_rng. Con un Generator se encadenan
        las llamadas de un bucle sobre simulaciones con una sola semilla.
    ci_level : float
        Nivel del intervalo, en (0, 1). El 0,95 por defecto corresponde al
        alpha de 0,05 de los contrastes (memoria, 3.2).

    Devuelve
    --------
    dict con estimate (media de x sin NaN), ci_low, ci_high, ci_level, se
    (desviacion tipica de las replicas), n (valores usados), block_length y
    n_boot.
    """
    v = np.asarray(x, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"x es la serie de UNA simulacion (1D); llego {v.ndim}D")
    if np.isinf(v).any():
        raise ValueError("x tiene valores infinitos")
    v = v[~np.isnan(v)]
    n = v.size
    largo = _entero_positivo(block_length, "block_length")
    replicas = _entero_positivo(n_boot, "n_boot")
    nivel = _nivel(ci_level)
    if n == 0:
        raise ValueError("x no tiene ningun valor evaluado (todo NaN)")
    if largo > n:
        raise ValueError(f"block_length = {largo} es mayor que los {n} valores evaluados de x")

    medias = _medias_circulares(v, largo, replicas, np.random.default_rng(seed))
    bajo, alto = np.quantile(medias, [(1.0 - nivel) / 2.0, (1.0 + nivel) / 2.0])
    return {
        "estimate": float(v.mean()),
        "ci_low": float(bajo),
        "ci_high": float(alto),
        "ci_level": nivel,
        "se": float(medias.std(ddof=1)) if replicas > 1 else np.nan,
        "n": int(n),
        "block_length": largo,
        "n_boot": replicas,
    }


def interval_coverage(ci_low, ci_high, target) -> dict:
    """Con que frecuencia los intervalos de una sola simulacion contienen su referencia.

    La cobertura del factor 4 (informe 06, 2.2): para cada una de las N
    simulaciones, el intervalo que construiria quien solo tiene esa
    (`block_bootstrap_ci`), frente a una referencia de lo que deberia cubrir.

    Que mide de verdad. La falta de cobertura (nominal menos medida) mezcla
    tres cosas, y la cobertura sola no las separa:
    1. el error del propio intervalo con una serie autocorrelacionada de
       longitud finita, que aparece aunque todas las simulaciones salgan del
       mismo proceso estacionario (varianza infraestimada con autocorrelacion
       fuerte y pocos bloques; sucesos raros, como las alarmas con una FAR
       baja; ver `block_bootstrap_ci`);
    2. la no estacionariedad de la serie, si la tiene (un retardo de
       deteccion);
    3. lo que varia de una simulacion a otra a una escala que una sola serie
       no ve (componentes mas lentas que la serie o que L), que es lo que
       propiamente cuesta usar una sola simulacion.
    Leer toda la falta de cobertura como pseudorreplicacion (DIS-14) atribuye
    a la 3 lo que puede ser la 1 o la 2.

    Que referencia necesita.
    - La misma cantidad que estima el intervalo: la media de la MISMA serie
      (la misma metrica, ventana y muestras evaluadas) en la poblacion de
      simulaciones, estimada con la media de las N estimaciones. No su
      mediana: es otro parametro y, con sucesos raros, otro numero. Para que
      la propia simulacion no entre en su referencia, una por intervalo:
      `(est.sum() - est) / (len(est) - 1)`.
    - Para separar la 3 de la 1 y la 2, una cobertura de calibracion: el mismo
      procedimiento (misma L, n_boot, nivel, longitud de la serie y metrica)
      sobre series sinteticas de media conocida y con la autocorrelacion de
      los datos, que solo tienen la 1 (y la 2, si se le pone), o la misma
      cobertura con varias L declaradas. Lo que la cobertura con los datos
      queda por debajo de la de calibracion es lo que se puede atribuir a usar
      una sola simulacion.
    Si la cobertura es la estimanda del factor 4, y con que calibracion, lo
    fija el registro de decisiones.

    Parametros
    ----------
    ci_low, ci_high : array (N,)
        Extremos de cada intervalo.
    target : float o array (N,)
        La referencia: una para todos los intervalos o una por intervalo (por
        ejemplo, la media de las demas simulaciones, o la media verdadera de
        cada serie sintetica en la calibracion). Los intervalos con algun NaN,
        en sus extremos o en su referencia, no cuentan.

    Devuelve
    --------
    dict con coverage (fraccion de intervalos con ci_low <= referencia <=
    ci_high; NaN si no hay ninguno), n, n_below (intervalos enteros por debajo
    de su referencia: ci_high < referencia) y n_above (ci_low > referencia).
    """
    bajo = np.asarray(ci_low, dtype=float)
    alto = np.asarray(ci_high, dtype=float)
    if bajo.shape != alto.shape or bajo.ndim != 1:
        raise ValueError(f"ci_low y ci_high deben ser 1D y de la misma longitud; llegaron {bajo.shape} y {alto.shape}")
    ref = np.asarray(target, dtype=float)
    if ref.ndim == 0:
        if not np.isfinite(ref):
            raise ValueError(f"target debe ser finito; llego {target}")
        ref = np.full(bajo.shape, float(ref))
    elif ref.shape != bajo.shape:
        raise ValueError(
            f"target es un numero o una referencia por intervalo ({bajo.size}); llego forma {ref.shape}"
        )
    if np.isinf(ref).any():
        raise ValueError("target tiene valores infinitos")
    validos = ~(np.isnan(bajo) | np.isnan(alto) | np.isnan(ref))
    bajo, alto, ref = bajo[validos], alto[validos], ref[validos]
    if (bajo > alto).any():
        raise ValueError("Hay intervalos con ci_low > ci_high")
    n = int(bajo.size)
    dentro = int(((bajo <= ref) & (ref <= alto)).sum())
    return {
        "coverage": dentro / n if n else np.nan,
        "n": n,
        "n_below": int((alto < ref).sum()),
        "n_above": int((bajo > ref).sum()),
    }


def contrary_conclusion_rate(single, reference) -> dict:
    """Con que frecuencia una sola simulacion da el signo contrario al de las N.

    La tasa de conclusiones contrarias del factor 4 (DIS-9, d1): la
    diferencia entre dos metodos medida en una sola simulacion, frente a la
    conclusion con las N. Por ejemplo, single = FDR_A(r) - FDR_B(r) en cada
    simulacion r, y reference = la estimacion con las N:
    `stats.paired_test(a, b)["hl_estimate"]` o la media de las diferencias.

    Si quien solo tiene una simulacion concluye solo cuando su intervalo
    excluye el 0, pasa como single el signo de esa conclusion, por ejemplo
    `np.where(lo > 0, 1, np.where(hi < 0, -1, 0))` con los extremos de
    `block_bootstrap_ci` de cada simulacion. Que estimacion y que regla
    definen la conclusion con las N (el signo del estimador, o solo si el
    contraste pareado es significativo) es una decision del diseno: con una
    referencia nula no hay signo que contradecir y la tasa es NaN.

    Que mide de verdad. Sobre todo, el tamano del efecto frente a lo que varia
    la diferencia de una simulacion a otra: con un efecto cerca de 0 la tasa
    se acerca a la mitad aunque cada simulacion este bien medida, y con uno
    grande se acerca a 0. No es una propiedad de usar una simulacion, sino de
    esa dispersion, que existe aunque todas salgan del mismo proceso. Si single
    es la conclusion con intervalo, la tasa depende ademas de la calibracion de
    ese intervalo (ver `block_bootstrap_ci`). Y la referencia tambien es una
    estimacion: si con las N no hay una conclusion firme, su signo no sirve de
    referencia.

    Parametros
    ----------
    single : array (N,)
        Diferencia (o signo de la conclusion) en cada simulacion; NaN no cuenta.
    reference : float
        Estimacion o conclusion con las N; solo importa su signo.

    Devuelve
    --------
    dict con rate (n_contrary / n), n (simulaciones sin NaN), n_contrary,
    n_same, n_zero (diferencia o conclusion nula en esa simulacion: ni una
    cosa ni otra) y reference_sign (-1, 0 o 1).
    """
    s = np.asarray(single, dtype=float)
    if s.ndim != 1:
        raise ValueError(f"single es un valor por simulacion (1D); llego {s.ndim}D")
    ref = float(reference)
    if np.isnan(ref):
        raise ValueError("reference es NaN: no hay conclusion con las N simulaciones")
    s = s[~np.isnan(s)]
    signo_ref = int(np.sign(ref))
    signos = np.sign(s)
    n = int(s.size)
    n_cero = int((signos == 0).sum())
    if signo_ref == 0:
        return {"rate": np.nan, "n": n, "n_contrary": 0, "n_same": 0,
                "n_zero": n_cero, "reference_sign": 0}
    contrarias = int((signos == -signo_ref).sum())
    return {
        "rate": contrarias / n if n else np.nan,
        "n": n,
        "n_contrary": contrarias,
        "n_same": int((signos == signo_ref).sum()),
        "n_zero": n_cero,
        "reference_sign": signo_ref,
    }


def subsample_runs(candidates, n_runs, n_replicates, seed, *, disjoint) -> np.ndarray:
    """Replicas de ajuste de n_runs simulaciones cada una (factor 4: ajustar con 1 frente a K).

    El lado de ajuste del factor 4 (DIS-9, d2): el mismo modelo ajustado con
    una simulacion o con K, con el mismo numero total de muestras (por
    ejemplo 1 x 500, 2 x 250, 5 x 100 y 10 x 50 en el anexo de diseno). Esta
    funcion solo elige que simulaciones entran en cada replica; cuantas
    muestras se toman de cada una, y cuales (por ejemplo las primeras,
    `df.groupby("simulationRun").head(m)`), lo decide el diseno.

    Que compara de verdad. Con el total de muestras fijo, pasar de 1 a K
    simulaciones cambia dos cosas a la vez: el ajuste ve la variacion entre
    simulaciones, y, con autocorrelacion positiva, K tramos de m muestras de K
    simulaciones distintas equivalen a mas muestras independientes que K * m
    muestras seguidas de una sola. Ademas, el resultado depende de que tramo
    se tome de cada simulacion. Si el diseno quiere separar esas partes, tiene
    que decirlo antes de ejecutar.

    Parametros
    ----------
    candidates : iterable de int
        Ids disponibles (por ejemplo, ids no mirados de train_fault00; ver
        `data.unseen_runs`). Sin repetidos.
    n_runs : int
        K, simulaciones por replica, >= 1.
    n_replicates : int
        R, numero de replicas, >= 1.
    seed : int
        Semilla del sorteo.
    disjoint : bool
        True: ninguna simulacion se repite entre replicas (replicas
        independientes; exige R * K <= numero de candidatos). False: cada
        replica se sortea por separado y dos replicas pueden compartir
        simulaciones. Obligatorio: la independencia de las replicas es una
        decision del diseno (DIS-3).

    Devuelve
    --------
    Array int64 (n_replicates, n_runs); cada fila ordenada y sin repetidos.
    """
    if not isinstance(disjoint, (bool, np.bool_)):
        raise TypeError(f"disjoint debe ser True o False; llego {disjoint!r}")
    ids = np.asarray(list(candidates))
    if ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu":
        raise ValueError("candidates es una lista no vacia de ids enteros")
    if np.unique(ids).size != ids.size:
        raise ValueError("candidates tiene ids repetidos")
    k = _entero_positivo(n_runs, "n_runs")
    r = _entero_positivo(n_replicates, "n_replicates")
    if k > ids.size:
        raise ValueError(f"n_runs = {k} y solo hay {ids.size} candidatos")
    if disjoint and r * k > ids.size:
        raise ValueError(
            f"{r} replicas disjuntas de {k} simulaciones necesitan {r * k} candidatos y hay "
            f"{ids.size}: reduce R o K, o usa disjoint=False (replicas que pueden compartir)"
        )

    rng = np.random.default_rng(seed)
    if disjoint:
        sorteo = rng.permutation(ids)[: r * k].reshape(r, k)
    else:
        sorteo = np.stack([rng.choice(ids, size=k, replace=False) for _ in range(r)])
    return np.sort(sorteo, axis=1).astype(np.int64)


# ------------------------------------------------ ventanas por trayectoria


def window_ends(runs, length, step) -> np.ndarray:
    """Ventanas deslizantes que nunca cruzan de una trayectoria a otra: posicion de su ultima muestra.

    Un modelo con ventanas (autoencoder, red recurrente) ve las `length`
    ultimas muestras y la ventana se etiqueta con la ultima, como en Lyu et
    al. (2026, p. 8) y el C00. La regla que no se negocia: una ventana nunca
    cruza de una trayectoria a otra (regla 1). Como en `lag_by_run`, una
    trayectoria son filas seguidas con el mismo id; en Rieth, pasa
    faultNumber y simulationRun.

    Devuelve solo posiciones, sin copiar datos: la ventana i son las filas
    ends[i] - length + 1 a ends[i] de X. Asi un Dataset de PyTorch puede
    construirla al vuelo, y `stack_windows` la materializa cuando cabe en
    memoria. En cada trayectoria, la primera ventana termina en su fila
    length - 1 y las siguientes cada `step` filas. Con length = lags + 1 y
    step = 1 son las mismas posiciones que `lag_by_run(..., return_index=True)`.

    Para medir la deteccion, puntua con step = 1 (una ventana por muestra a
    partir de la length-esima de cada trayectoria) y lleva la puntuacion a las
    muestras con `align_to_samples`: con step > 1 quedarian NaN entre ventanas
    y una racha de k alarmas no podria atravesarlos. Un step > 1 sirve para
    entrenar con menos memoria, como hace el C00.

    El brazo sucio con ventanas (factor 3) reparte al azar estas ventanas
    solapadas: `split_by_sample(runs[ends], sizes, seed, block_length=1)`; el
    limpio, por simulacion: `split_by_run(simulationRun[ends], fractions, seed)`.

    Parametros
    ----------
    runs : array (n,) o (n, k)
        Trayectoria de cada fila.
    length : int
        Muestras por ventana, >= 1. Sin valor por defecto (decision del bloque C).
    step : int
        Paso entre ventanas consecutivas de una trayectoria, >= 1.

    Devuelve
    --------
    Array int64 creciente. Error si alguna trayectoria tiene menos de `length` filas.
    """
    largo = _entero_positivo(length, "length")
    paso = _entero_positivo(step, "step")
    _, inicios = _trayectorias(runs)
    finales = []
    for ini, fin in zip(inicios[:-1], inicios[1:]):
        if fin - ini < largo:
            raise ValueError(
                f"La trayectoria de las filas {ini} a {fin - 1} tiene {fin - ini} muestras y no "
                f"admite ventanas de {largo}. Comprueba que las filas de cada trayectoria vengan "
                "seguidas y en orden temporal."
            )
        finales.append(np.arange(ini + largo - 1, fin, paso))
    return np.concatenate(finales).astype(np.int64)


def _comprobar_finales(ends, n: int, largo: int) -> np.ndarray:
    e = np.asarray(ends)
    if e.ndim != 1 or (e.size and e.dtype.kind not in "iu"):
        raise ValueError("ends son posiciones enteras (1D), como las de window_ends")
    if e.size and (e.min() < largo - 1 or e.max() >= n):
        raise ValueError(
            f"ends fuera de rango: cada posicion tiene que estar entre {largo - 1} y {n - 1}"
        )
    return e.astype(np.int64)


def stack_windows(X, ends, length) -> np.ndarray:
    """Materializa las ventanas: (n_ventanas, length, n_variables), de la mas antigua a la ultima.

    La ventana i es X[ends[i] - length + 1 : ends[i] + 1]. Copia datos, asi
    que con muchas ventanas no cabe en memoria: cuenta ventanas x length x
    variables x bytes antes de llamarla, o construye las ventanas por lotes o
    al vuelo con `window_ends`. Conserva el tipo de X si es float32 (como hace
    el C00 para ahorrar memoria); si no es de coma flotante, pasa a float64.

    Las posiciones deben salir de `window_ends` con el mismo `length`: con
    otras, nada impide que una ventana cruce de una trayectoria a otra.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz 2D (muestras, variables); llego {X.ndim}D")
    if X.dtype.kind != "f":
        X = X.astype(float)
    largo = _entero_positivo(length, "length")
    e = _comprobar_finales(ends, len(X), largo)
    filas = e[:, None] + np.arange(1 - largo, 1)
    return X[filas]


def align_to_samples(values, ends, n_samples) -> np.ndarray:
    """Lleva una puntuacion por ventana a las muestras originales, con NaN donde no termina ninguna.

    Cada valor se coloca en la posicion de la ULTIMA muestra de su ventana
    (la que la etiqueta), y el resto queda en NaN ("muestra no evaluada", la
    semantica de NaN de `metrics`). Con ventanas de `window_ends(runs, L, 1)`,
    las L - 1 primeras muestras de cada trayectoria quedan en NaN, igual que
    las `lags` primeras del DPCA, y `metrics.evaluate_run(stat, limite,
    onset=160)` vale con el mismo onset para PCA, DPCA y los modelos con
    ventanas (MET-1). Alinear por la primera muestra de la ventana
    adelantaria el retardo L - 1 muestras.

    Parametros
    ----------
    values : array (n_ventanas,) o (n_ventanas, k)
        Una puntuacion (o k) por ventana, en el orden de `ends`.
    ends : array (n_ventanas,)
        Posiciones de `window_ends`, sin repetidos.
    n_samples : int
        Filas de X (las muestras originales).

    Devuelve
    --------
    Array float (n_samples,) o (n_samples, k).
    """
    n = _entero_positivo(n_samples, "n_samples")
    valores = np.asarray(values, dtype=float)
    e = _comprobar_finales(ends, n, 1)
    if valores.ndim not in (1, 2) or valores.shape[0] != e.size:
        raise ValueError(
            f"values tiene forma {valores.shape} y hay {e.size} ventanas: una fila por ventana"
        )
    if np.unique(e).size != e.size:
        raise ValueError("ends tiene posiciones repetidas: cada muestra puede cerrar una sola ventana")
    salida = np.full((n,) + valores.shape[1:], np.nan)
    salida[e] = valores
    return salida
