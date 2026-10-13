"""Metricas de deteccion y diagnostico de fallos.

Las tres metricas basicas de deteccion responden a tres preguntas distintas y
hay que reportar las tres, porque optimizar una sola engana:

- FAR (false alarm rate): de las muestras que estaban sanas, cuantas marco el
  detector. Un detector que siempre dice "fallo" tiene FDR perfecto y FAR del
  100 %: inutil en planta, porque los operarios acaban ignorando las alarmas.
- FDR (fault detection rate): de las muestras con fallo, cuantas detecto.
- Retardo de deteccion: cuanto tarda desde que el fallo entra hasta que se
  dispara la alarma. Es la metrica que la literatura mas omite y la que mas le
  importa a quien opera la planta, porque decide si da tiempo a reaccionar.

Para el retardo se exige confirmacion: no basta una muestra por encima del
limite, hacen falta k consecutivas. Una sola exceedencia suele ser ruido, y sin
esa regla el retardo medido es optimista.

Un NaN en el estadistico significa "muestra no evaluada" (por ejemplo, las
`lags` primeras de cada trayectoria en DPCA), no "muestra sin alarma": FAR y
FDR lo excluyen del numerador y del denominador, una racha de alarmas no lo
atraviesa y `evaluate_run` dice cuantas muestras entran en cada tramo. Un
limite NaN o infinito es un error, nunca un resultado.

El diagnostico (que clase de fallo es) se mide distinto: F1 por clase sobre
TODA la prueba y su media, la macro-F1 (`diagnosis_metrics`), una macro-F1 por
replica para poder contrastarla (`macro_f1_by_replicate`), y el recall por
simulacion para el formato comun (`diagnosis_recall_by_run`).

La regla de si un metodo detecta por encima de su FAR (`detects_above_far`)
vive en tfmfdd.stats, con el mismo Wilcoxon que la hipotesis; aqui sigue
disponible con su nombre de siempre.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Los contrastes del paquete (tfmfdd.stats; aqui `stats` es scipy.stats).
# stats.py no importa este modulo, asi que no hay importacion circular.
from . import stats as _contrastes


def _limite(limit: float) -> float:
    """Convierte el limite a float y falla si no es finito.

    Con un limite NaN todas las comparaciones dan False: saldrian FAR = 0,
    FDR = 0 y "no detectado" con aspecto de resultado (MET-11). Un limite
    infinito tampoco es un limite de control.
    """
    lim = float(limit)
    if not np.isfinite(lim):
        raise ValueError(
            f"El limite de control no es finito ({lim}). Revisa su calibracion "
            "(X_cal vacio, nivel de confianza fuera de (0, 1)...): un limite NaN "
            "daria FAR y FDR nulas con aspecto de resultado."
        )
    return lim


def _comprobar_onset(onset, n: int) -> int:
    """Indice de inicio del fallo valido para un estadistico de n muestras."""
    o = int(onset)
    if o != onset or not 0 <= o <= n:
        raise ValueError(
            f"onset = {onset} no es un indice valido para un estadistico de {n} "
            "muestras: debe ser un entero entre 0 y el numero de muestras."
        )
    return o


def alarms(statistic: np.ndarray, limit: float) -> np.ndarray:
    """Vector booleano: True donde el estadistico supera el limite.

    Un NaN en el estadistico es una muestra no evaluada: no hay alarma (False),
    pero tampoco cuenta como muestra sana. `far` y `fdr` la excluyen y una
    racha de alarmas no puede atravesarla. Un limite NaN o infinito lanza
    ValueError.
    """
    return np.asarray(statistic, dtype=float) > _limite(limit)


def _tasa_de_alarma(statistic: np.ndarray, limit: float) -> tuple[float, int]:
    """Proporcion de muestras evaluadas (no NaN) en alarma, y cuantas son."""
    stat = np.asarray(statistic, dtype=float)
    a = alarms(stat, limit)
    evaluadas = ~np.isnan(stat)
    n = int(evaluadas.sum())
    if n == 0:
        return np.nan, 0
    return float(a[evaluadas].mean()), n


def far(statistic: np.ndarray, limit: float) -> float:
    """Tasa de falsas alarmas sobre muestras que se saben sanas.

    Se calcula sobre las muestras ANTERIORES al inicio del fallo, o sobre una
    simulacion completa de operacion normal: es la proporcion de muestras
    sanas en las que salta una alarma (la definicion de la memoria, 2.2.4, y
    la de Yin et al., 2012).

    Las muestras con NaN (no evaluadas) no cuentan ni en el numerador ni en el
    denominador. Si no queda ninguna, devuelve NaN. `evaluate_run` devuelve en
    `n_far` cuantas entraron.
    """
    return _tasa_de_alarma(statistic, limit)[0]


def fdr(statistic: np.ndarray, limit: float) -> float:
    """Tasa de deteccion sobre muestras que se saben defectuosas.

    Mismo trato de los NaN que `far`: no cuentan, y si no queda ninguna
    muestra evaluada devuelve NaN (`n_fdr` en `evaluate_run`).
    """
    return _tasa_de_alarma(statistic, limit)[0]


def _rachas(a: np.ndarray) -> np.ndarray:
    """Longitud de cada racha maxima de True de un vector booleano."""
    borde = np.diff(np.concatenate(([0], np.asarray(a, dtype=np.int8), [0])))
    return np.flatnonzero(borde == -1) - np.flatnonzero(borde == 1)


def _horas_evaluadas(stat: np.ndarray, sample_minutes: float) -> float:
    if not sample_minutes > 0:
        raise ValueError(f"sample_minutes debe ser positivo (es {sample_minutes}).")
    return float((~np.isnan(stat)).sum()) * float(sample_minutes) / 60.0


def false_alarm_samples_per_hour(statistic: np.ndarray, limit: float,
                                 sample_minutes: float = 3.0) -> float:
    """Muestras en alarma por hora de operacion normal evaluada.

    El OE3 de la memoria pide "la FAR (o falsas alarmas por hora)", y "por
    hora" admite dos unidades que dan cifras distintas. Esta funcion da la
    primera: cuenta MUESTRAS en alarma. Es la FAR por muestra pasada a tiempo,
    FAR * 60 / sample_minutes: con muestreo de 3 minutos hay 20 muestras por
    hora, asi que una FAR del 1 % son 0,2 muestras en alarma por hora. La otra
    unidad, episodios por hora, esta en `false_alarm_episodes_per_hour`. Cual
    de las dos se reporta lo decide el equipo; aqui solo se calculan, cada una
    con su nombre.

    Se aplica a un tramo sano (el anterior al fallo o una simulacion normal
    entera). Las horas son las de las muestras evaluadas (sin NaN). Devuelve
    NaN si no hay ninguna.
    """
    stat = np.asarray(statistic, dtype=float)
    horas = _horas_evaluadas(stat, sample_minutes)
    a = alarms(stat, limit)
    return float(a.sum() / horas) if horas > 0 else np.nan


def false_alarm_episodes_per_hour(statistic: np.ndarray, limit: float, k: int = 3,
                                  sample_minutes: float = 3.0) -> float:
    """Episodios de alarma confirmada por hora de operacion normal evaluada.

    Es la segunda unidad de "falsas alarmas por hora" (la primera esta en
    `false_alarm_samples_per_hour`). Un episodio es una racha maxima de al
    menos k alarmas seguidas: confirma una alarma (la primera de una serie de
    k seguidas, como el retardo) y se cuenta UNA vez, dure lo que dure, como la
    alarma que un operario ve saltar y no se apaga hasta que el estadistico
    vuelve bajo el limite. Dos rachas separadas por una sola muestra sin alarma
    son dos episodios. Un NaN no es alarma y corta la racha.

    Una alarma aislada no cuenta como episodio, asi que esta cifra no es FAR por
    una constante: depende de como se agrupan las alarmas en el tiempo, y con
    estadisticos autocorrelacionados puede quedar muy por debajo de la de
    muestras. Las horas son las de las muestras evaluadas. Devuelve NaN si no
    hay ninguna.
    """
    if int(k) != k or k < 1:
        raise ValueError(f"k debe ser un entero positivo (es {k}).")
    stat = np.asarray(statistic, dtype=float)
    horas = _horas_evaluadas(stat, sample_minutes)
    episodios = int(np.sum(_rachas(alarms(stat, limit)) >= int(k)))
    return float(episodios / horas) if horas > 0 else np.nan


def detection_delay(
    statistic: np.ndarray,
    limit: float,
    onset: int,
    k: int = 3,
    sample_minutes: float = 3.0,
    convention: str = "appearance",
) -> dict:
    """Retardo hasta la primera racha de k alarmas consecutivas tras el fallo.

    Convencion (se documenta en la v0.3; el valor por defecto es el de
    siempre):

    - "appearance" (por defecto): el retardo es la posicion, contada desde
      `onset`, de la PRIMERA alarma de la primera racha de k alarmas seguidas
      que empieza en el tramo con fallo. Vale 0 si la racha empieza en la
      primera muestra con fallo (el indice `onset`, que en los ficheros de
      prueba del TEP es la muestra 161). Es la "alarma confirmada" de la
      memoria (2.2.4: la primera de una serie de tres alarmas seguidas).
      Russell et al. (2000) usan la misma convencion, con seis valores
      seguidos en vez de tres.
    - "confirmation": el instante en que la racha se completa y la deteccion
      se puede confirmar de verdad, aparicion + k - 1. Es el retardo que vive
      un operario que espera las k alarmas antes de actuar.

    Las dos cuentan posiciones en la escala de las muestras originales: una
    muestra con NaN (no evaluada) ocupa su posicion, no es alarma y corta la
    racha. `delay_minutes` es `delay_samples * sample_minutes`, sin sumar media
    muestra: al comparar con otro trabajo hay que mirar su convencion.

    Parametros
    ----------
    statistic : array
        Estadistico para la simulacion completa, en orden temporal.
    limit : float
        Limite de control. Si no es finito, ValueError.
    onset : int
        Numero de muestras normales al principio. En los ficheros de prueba del
        TEP son 160: el fallo afecta a partir de la muestra 161, es decir al
        indice 160 en base cero.
    k : int
        Alarmas consecutivas necesarias para confirmar la deteccion.
    sample_minutes : float
        Minutos entre muestras. En el TEP son 3.
    convention : {"appearance", "confirmation"}
        Que instante de la racha se toma como deteccion (ver arriba).

    Devuelve
    --------
    dict con 'delay_samples', 'delay_minutes' y 'detected'. Si el fallo nunca se
    confirma, el retardo es NaN y 'detected' es False.
    """
    if convention not in ("appearance", "confirmation"):
        raise ValueError(
            f"convention = {convention!r}: usa 'appearance' (primera alarma de la "
            "racha, la de siempre) o 'confirmation' (aparicion + k - 1)."
        )
    if int(k) != k or k < 1:
        raise ValueError(f"k debe ser un entero positivo (es {k}).")
    k = int(k)
    stat = np.asarray(statistic, dtype=float)
    onset = _comprobar_onset(onset, stat.size)
    a = alarms(stat, limit)[onset:]
    if a.size < k:
        return {"delay_samples": np.nan, "delay_minutes": np.nan, "detected": False}

    # Ventana deslizante: posicion i vale si desde ahi hay k Trues seguidos.
    ventanas = np.lib.stride_tricks.sliding_window_view(a, k).all(axis=1)
    idx = np.flatnonzero(ventanas)

    if idx.size == 0:
        return {"delay_samples": np.nan, "delay_minutes": np.nan, "detected": False}

    d = int(idx[0])
    if convention == "confirmation":
        d += k - 1
    return {
        "delay_samples": d,
        "delay_minutes": d * float(sample_minutes),
        "detected": True,
    }


def stratified_fdr(statistic: np.ndarray, limit: float, onset: int) -> dict:
    """FDR partida en dos mitades del periodo con fallo, con regla fijada a priori.

    Un unico numero de FDR puede promediar regimenes distintos. En un fallo que
    el control compensa (por ejemplo IDV 5), un estadistico puede acertar casi
    siempre al principio y casi nunca despues, y entonces la media no describe
    ninguno de los dos. La particion es la MITAD del periodo con fallo, la
    unica que se puede justificar sin mirar los datos. No se ajusta fallo a
    fallo.

    Devuelve fdr_h1 (primera mitad), fdr_h2 (segunda mitad) y persistence_gap,
    que es fdr_h1 - fdr_h2: cerca de 0 si el fallo se ve igual todo el tiempo,
    grande si se detecta y luego se pierde.

    Las mitades se cortan por posicion en la escala de las muestras originales,
    para que PCA y DPCA partan en el mismo instante; dentro de cada mitad, los
    NaN no cuentan. `n_fdr_h1` y `n_fdr_h2` dicen cuantas muestras evaluadas
    entran en cada mitad.
    """
    stat = np.asarray(statistic, dtype=float)
    lim = _limite(limit)
    onset = _comprobar_onset(onset, stat.size)
    tramo = stat[onset:]
    if tramo.size < 2:
        return {"fdr_h1": np.nan, "fdr_h2": np.nan, "persistence_gap": np.nan,
                "n_fdr_h1": 0, "n_fdr_h2": 0}
    mitad = tramo.size // 2
    h1, n1 = _tasa_de_alarma(tramo[:mitad], lim)
    h2, n2 = _tasa_de_alarma(tramo[mitad:], lim)
    return {"fdr_h1": h1, "fdr_h2": h2, "persistence_gap": h1 - h2,
            "n_fdr_h1": n1, "n_fdr_h2": n2}


def _ventana(far_window, n: int) -> tuple[int, int]:
    """Convierte `far_window` en (inicio, fin) validos para n muestras."""
    if isinstance(far_window, slice):
        if far_window.step not in (None, 1):
            raise ValueError("far_window no admite paso: usa una tupla (inicio, fin).")
        inicio, fin = far_window.start, far_window.stop
    else:
        try:
            inicio, fin = far_window
        except (TypeError, ValueError):
            raise ValueError(
                f"far_window = {far_window!r}: usa una tupla (inicio, fin) de indices "
                "en base cero, con fin excluido y None para llegar al extremo."
            ) from None
    inicio = 0 if inicio is None else inicio
    fin = n if fin is None else fin
    if int(inicio) != inicio or int(fin) != fin or not 0 <= inicio < fin <= n:
        raise ValueError(
            f"far_window = {far_window!r} no es una ventana valida para {n} muestras: "
            f"hace falta 0 <= inicio < fin <= {n}."
        )
    return int(inicio), int(fin)


def evaluate_run(
    statistic: np.ndarray,
    limit: float,
    onset: int | None,
    k: int = 3,
    sample_minutes: float = 3.0,
    min_fdr: float = 0.10,
    delay_convention: str = "appearance",
    far_window=None,
) -> dict:
    """Evalua una simulacion completa y devuelve FAR, FDR, retardo y FDR estratificada.

    Con onset = None se entiende que la simulacion entera es de operacion
    normal: solo se calculan la FAR y las falsas alarmas por hora, sobre toda
    la serie o sobre la ventana `far_window`.

    `far_window` (opcional, solo con onset = None) fija la ventana de la
    simulacion normal en la que se miden la FAR, `n_far` y las falsas alarmas
    por hora: una tupla (inicio, fin) de indices en base cero, con fin excluido
    y None para llegar al extremo (tambien vale un `slice` sin paso). Por
    ejemplo, (160, None) en d00_te son las muestras 161 a 960. Sin ella (por
    defecto), la serie entera, como hasta ahora. Sirve para comparar con un
    trabajo que mide la FAR en otra ventana, como la verificacion frente a Yin
    et al. (2012): la ventana se declara antes de medir, y `n_far` deja
    constancia de cuantas muestras entraron. En los ficheros con fallo no se
    aplica, porque la FAR es la del tramo anterior al inicio: pasarla con un
    `onset` es un error.

    `detected_at_chance` marca (no filtra) las confirmaciones sospechosas: una
    racha de k alarmas puede aparecer por azar en 800 muestras autocorrelacionadas
    aunque el detector no vea el fallo. Si `detected` es True pero la FDR queda
    por debajo de `min_fdr`, el retardo no debe leerse como una deteccion. El
    0,10 es una convencion declarada en el protocolo, no un resultado.
    `aggregate` y `summary` ya excluyen esos retardos.

    Columnas que anade la v0.3 (las de antes no cambian de nombre ni de
    significado, salvo que los NaN del estadistico ya no cuentan como muestras
    sin alarma):

    - `n_far`, `n_fdr`, `n_fdr_h1`, `n_fdr_h2`: muestras evaluadas (sin NaN) en
      cada tramo. Con DPCA, `n_far` es menor que `onset`.
    - `false_alarm_samples_per_hour` y `false_alarm_episodes_per_hour`: falsas
      alarmas por hora en el tramo sano, en sus dos unidades (ver esas
      funciones). En Rieth, en los ficheros con fallo ese tramo es copia de la
      simulacion normal gemela, igual que la columna `far`.

    `delay_convention` se pasa a `detection_delay` ("appearance" por defecto,
    o "confirmation"). Como `k`, es configuracion: se registra junto a los
    resultados, no en cada fila.
    """
    stat = np.asarray(statistic, dtype=float)
    lim = _limite(limit)

    if onset is None:
        if far_window is None:
            sano = stat
        else:
            inicio, fin = _ventana(far_window, stat.size)
            sano = stat[inicio:fin]
        resultado = {
            "far": np.nan, "fdr": np.nan,
            "delay_samples": np.nan, "delay_minutes": np.nan,
            "detected": False, "detected_at_chance": False,
            "fdr_h1": np.nan, "fdr_h2": np.nan, "persistence_gap": np.nan,
        }
        resultado["far"], n_far = _tasa_de_alarma(sano, lim)
        n_fdr, n_fdr_h1, n_fdr_h2 = 0, 0, 0
    else:
        if far_window is not None:
            raise ValueError(
                "far_window solo se aplica a simulaciones normales (onset=None): con "
                "un fallo, la FAR es la del tramo anterior al inicio y no cambia."
            )
        onset = _comprobar_onset(onset, stat.size)
        sano = stat[:onset]
        far_, n_far = _tasa_de_alarma(sano, lim)
        fdr_, n_fdr = _tasa_de_alarma(stat[onset:], lim)
        resultado = {"far": far_, "fdr": fdr_}
        resultado.update(detection_delay(stat, lim, onset, k, sample_minutes,
                                         convention=delay_convention))
        resultado["detected_at_chance"] = bool(resultado["detected"] and resultado["fdr"] < min_fdr)
        estratos = stratified_fdr(stat, lim, onset)
        n_fdr_h1, n_fdr_h2 = estratos.pop("n_fdr_h1"), estratos.pop("n_fdr_h2")
        resultado.update(estratos)

    resultado.update({
        "n_far": n_far,
        "n_fdr": n_fdr,
        "n_fdr_h1": n_fdr_h1,
        "n_fdr_h2": n_fdr_h2,
        "false_alarm_samples_per_hour": false_alarm_samples_per_hour(sano, lim, sample_minutes),
        "false_alarm_episodes_per_hour": false_alarm_episodes_per_hour(sano, lim, k, sample_minutes),
    })
    return resultado


def pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area bajo la curva precision-recall (precision media) por muestra.

    El OE3 de la memoria la incluye entre sus metricas, como secundaria. No
    depende del limite de control: mide si la puntuacion ordena las muestras
    con fallo por encima de las sanas, a cualquier umbral. Por construccion,
    dos configuraciones que solo difieren en el umbral (como los brazos del
    factor 1 con el mismo modelo) dan la misma PR-AUC; por eso no sustituye a
    FAR, FDR y retardo.

    Se calcula como la precision media no interpolada (average precision), la
    misma definicion que `sklearn.metrics.average_precision_score`: la suma,
    sobre los umbrales distintos, de (R_n - R_(n-1)) * P_n, con las muestras
    empatadas en puntuacion en el mismo umbral.

    Cuidado al leerla: su nivel del azar es la prevalencia (la proporcion de
    muestras positivas), no 0,5 como en la ROC-AUC. En un fichero de prueba del
    TEP 800 de las 960 muestras tienen fallo, asi que una puntuacion sin
    informacion ya da una PR-AUC del orden de 800/960. Hay que informar de la
    prevalencia al lado y decidir antes de medir que muestras forman el
    conjunto (una simulacion, o el pool de simulaciones normales y con fallo).

    Parametros
    ----------
    scores : array
        Puntuacion continua por muestra (el estadistico, el error de
        reconstruccion, la probabilidad de fallo): mayor = mas anomala. Un NaN
        es una muestra no evaluada y se excluye.
    labels : array de 0/1 o bool
        1 (True) = muestra con fallo, la clase positiva.

    Devuelve
    --------
    float. NaN si no queda ninguna muestra positiva evaluada (la PR-AUC no
    esta definida).
    """
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores y labels tienen longitudes distintas: {s.shape} y {y.shape}")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("labels debe ser 0/1 o bool (1 = muestra con fallo).")
    evaluadas = ~np.isnan(s)
    s, y = s[evaluadas], y[evaluadas].astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return np.nan

    orden = np.argsort(-s, kind="mergesort")
    s, y = s[orden], y[orden]
    # Un umbral por cada valor distinto de la puntuacion (los empates van juntos).
    ultimo_de_cada_umbral = np.concatenate((np.flatnonzero(s[1:] != s[:-1]), [s.size - 1]))
    tp = np.cumsum(y)[ultimo_de_cada_umbral]
    fp = ultimo_de_cada_umbral + 1 - tp
    precision = tp / (tp + fp)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


#: Deteccion por encima de la FAR (el suelo del criterio de no superioridad).
#: Vive en tfmfdd.stats desde la segunda parte de la v0.3, para que el suelo y
#: la hipotesis se decidan con el mismo Wilcoxon exacto; este nombre se
#: conserva por compatibilidad y es la misma funcion (ver su docstring).
detects_above_far = _contrastes.detects_above_far


#: Cotas de cada metrica por simulacion. Se usan para recortar los intervalos
#: de confianza de `aggregate`, que con la t de Student pueden salirse del rango.
_COTAS = {
    "far": (0.0, 1.0),
    "fdr": (0.0, 1.0),
    "delay_samples": (0.0, np.inf),
    "delay_minutes": (0.0, np.inf),
    "fdr_h1": (0.0, 1.0),
    "fdr_h2": (0.0, 1.0),
    "persistence_gap": (-1.0, 1.0),
    "false_alarm_samples_per_hour": (0.0, np.inf),
    "false_alarm_episodes_per_hour": (0.0, np.inf),
    "pr_auc": (0.0, 1.0),
}

#: Metricas de retardo: solo se agregan las detecciones validas.
_RETARDOS = ("delay_samples", "delay_minutes")


def aggregate(df: pd.DataFrame, by: str | list[str] = "faultNumber", ci: float = 0.95) -> pd.DataFrame:
    """Agrega resultados por simulacion en media, desviacion e intervalo de confianza.

    Recibe un DataFrame con una fila por simulacion (la salida de evaluate_run
    mas las columnas identificativas) y devuelve una fila por fallo.

    El intervalo de confianza sobre muchas simulaciones es lo que distingue
    nuestros resultados de la mayoria de la literatura, que informa de una sola
    corrida y no permite saber si una diferencia entre metodos es real o ruido.

    Reglas de la v0.3 (MET-4, I1):

    - Cada metrica lleva su n (`<metrica>_n`): el numero de simulaciones que
      entran en su media, que no siempre es `n_runs`.
    - El retardo solo se agrega sobre las detecciones validas: `detected` y no
      `detected_at_chance`. Una racha al azar no es una deteccion, y su retardo
      no se promedia. `valid_detection_ratio` es la proporcion de simulaciones
      con deteccion valida, y hay que leerla al lado del retardo: un metodo que
      solo confirma en las simulaciones faciles parece mas rapido.
    - Para el retardo se dan dos medianas (la del OE3): `<retardo>_median`,
      sobre las mismas detecciones validas que la media, y
      `<retardo>_median_censored`, sobre todas las simulaciones con tramo de
      fallo evaluado, contando las que no tienen deteccion valida como
      censuradas al final del tramo (retardo mayor que cualquiera observado).
      Si la mitad o mas no tienen deteccion valida, la censurada es infinita:
      la mediana no se alcanza dentro de la ventana. Cual se reporta lo decide
      el equipo (DIS-12).
    - Los intervalos (t de Student) se recortan al rango de cada metrica:
      [0, 1] para FAR, FDR y sus mitades, [-1, 1] para persistence_gap y
      [0, inf) para retardos y falsas alarmas por hora. Como la media verdadera
      esta dentro del rango, recortar no cambia la cobertura del intervalo.
    - Una fila con NaN en una columna de agrupacion es un error (antes se
      descartaba sin avisar).
    """
    if not 0 < ci < 1:
        raise ValueError(f"ci debe estar en (0, 1) (es {ci}).")
    claves = [by] if isinstance(by, str) else list(by)
    if df[claves].isna().any().any():
        raise ValueError(
            f"Hay filas con NaN en {claves}: groupby las descartaria sin avisar. "
            "Corrige o quita esas filas antes de agregar."
        )
    metricas = [c for c in _COTAS if c in df]

    tiene_det = "detected" in df
    tiene_azar = "detected_at_chance" in df
    if any(m in df for m in _RETARDOS) and not (tiene_det and tiene_azar):
        warnings.warn(
            "Sin las columnas 'detected' y 'detected_at_chance' no se pueden "
            "excluir las rachas al azar: el retardo se agrega sobre todos los "
            "valores no NaN.",
            stacklevel=2,
        )

    filas = []
    for clave, g in df.groupby(by):
        fila = {by: clave} if isinstance(by, str) else dict(zip(by, clave))
        fila["n_runs"] = len(g)
        fila["detection_ratio"] = g["detected"].mean() if tiene_det else np.nan
        fila["at_chance_ratio"] = g["detected_at_chance"].mean() if tiene_azar else np.nan
        if tiene_det and tiene_azar:
            valida = (g["detected"].astype(bool) & ~g["detected_at_chance"].astype(bool)).to_numpy()
        elif tiene_det:
            valida = g["detected"].astype(bool).to_numpy()
        else:
            valida = None
        hay_valida = valida is not None and valida.size > 0
        fila["valid_detection_ratio"] = float(valida.mean()) if hay_valida else np.nan

        for m in metricas:
            serie = g[m]
            if m in _RETARDOS and valida is not None:
                serie = serie[valida]
            v = serie.dropna().to_numpy(dtype=float)
            fila[f"{m}_mean"] = v.mean() if v.size else np.nan
            fila[f"{m}_std"] = v.std(ddof=1) if v.size > 1 else np.nan

            if v.size > 1:
                sem = stats.sem(v)
                margen = sem * stats.t.ppf((1 + ci) / 2.0, v.size - 1)
                bajo, alto = _COTAS[m]
                fila[f"{m}_ci_low"] = float(np.clip(v.mean() - margen, bajo, alto))
                fila[f"{m}_ci_high"] = float(np.clip(v.mean() + margen, bajo, alto))
            else:
                fila[f"{m}_ci_low"] = np.nan
                fila[f"{m}_ci_high"] = np.nan
            fila[f"{m}_n"] = int(v.size)

            if m in _RETARDOS:
                fila[f"{m}_median"] = float(np.median(v)) if v.size else np.nan
                fila[f"{m}_median_censored"] = _mediana_censurada(g, m, valida)

        filas.append(fila)

    return pd.DataFrame(filas)


def _mediana_censurada(g: pd.DataFrame, m: str, valida: np.ndarray | None) -> float:
    """Mediana del retardo con las simulaciones sin deteccion valida censuradas.

    Entran las simulaciones con tramo de fallo evaluado (`fdr` no NaN). Las que
    no tienen deteccion valida (sin racha, o con racha al azar) se censuran al
    final del tramo: cuentan como un retardo mayor que cualquiera observado
    (infinito) y la mediana es la muestral de siempre (`np.median`). Asi
    coincide con `<retardo>_median` cuando todas detectan. Si la mitad o mas
    estan censuradas sale infinita: la mediana no se alcanza dentro de la
    ventana (con n par y el corte justo en la mitad tambien, por prudencia).
    """
    if valida is None or "fdr" not in g:
        return np.nan
    con_tramo = g["fdr"].notna().to_numpy()
    if not con_tramo.any():
        return np.nan
    valores = np.where(valida, g[m].to_numpy(dtype=float), np.inf)[con_tramo]
    valores = np.where(np.isnan(valores), np.inf, valores)
    return float(np.median(valores))


#: Fallos del TEP que la literatura reporta como dificiles de detectar.
#: IDV(3) cambio escalon en la temperatura de alimentacion D, IDV(9) variacion
#: aleatoria de esa misma temperatura, IDV(15) agarrotamiento de la valvula del
#: condensador. Se reportan SIEMPRE por separado (regla 7) por dos motivos:
#: promediados con el resto, su comportamiento queda escondido detras de los
#: fallos faciles, y es en ellos donde la literatura publica las cifras mas
#: discutidas (unos trabajos los excluyen y otros publican tasas altas).
#: Separarlos no presupone ningun resultado, ni en un sentido ni en el otro.
HARD_FAULTS = (3, 9, 15)


#: Claves de `summary` retiradas en la v0.3, con lo que hay que usar en su lugar.
_CLAVES_RETIRADAS = {
    "far_medio": (
        "mezclaba la FAR de las simulaciones normales completas con la FAR antes "
        "del fallo de cada fichero. Usa 'far_normal' (la FAR del metodo, sobre las "
        "simulaciones normales) y, aparte, 'far_prefallo_medio'"
    ),
    "fallos_nunca_detectados": (
        "no media lo que decia: eran los fallos con racha de k alarmas en menos de "
        "la mitad de las simulaciones, aunque fuera al azar. Usa 'fallos_sin_racha' "
        "(esa misma definicion) o 'fallos_bajo_umbral' (FDR media por debajo de min_fdr)"
    ),
}


class _Resumen(dict):
    """dict de `summary`: si se pide una clave retirada, dice cual usar."""

    def __missing__(self, clave):
        if clave in _CLAVES_RETIRADAS:
            raise KeyError(f"'{clave}' se retiro en la v0.3: {_CLAVES_RETIRADAS[clave]}.")
        raise KeyError(clave)


def summary(agg: pd.DataFrame, min_fdr: float = 0.10) -> dict:
    """Resumen global de una tabla agregada, con y sin los fallos dificiles.

    Recibe la salida de `aggregate` para UN metodo (una fila por fallo, con la
    fila del fallo 0 para las simulaciones normales) y devuelve un dict. Cada
    media lleva su n en `<clave>_n`: cuantos fallos entran (en `far_normal_n`,
    cuantas simulaciones normales).

    Cambios de la v0.3 (I1, M5):

    - `retardo_medio_muestras` ya no incluye rachas al azar: sale de
      `aggregate`, que solo agrega detecciones validas. Es la media, entre
      fallos, del retardo medio de cada fallo; `retardo_mediano_muestras` es la
      mediana, entre fallos, de su mediana. Los fallos sin ninguna deteccion
      valida no entran (su n lo dice).
    - La FAR del metodo es la de las simulaciones normales (fila del fallo 0:
      d00_te en el clasico, test_fault00 en Rieth), en la ventana con que se
      evaluaron (enteras, salvo que se pasara `far_window` a `evaluate_run`):
      `far_normal`. La
      FAR antes del fallo de los ficheros con fallo va aparte, en
      `far_prefallo_medio`, y no es una estimacion independiente: en Rieth
      esas 160 muestras son copia de la simulacion normal gemela, y
      promediarlas cuenta el mismo tramo una vez por fichero.
    - `fallos_sin_racha`: fallos en los que menos de la mitad de las
      simulaciones tienen una racha de k alarmas (la definicion de la antigua
      `fallos_nunca_detectados`, que no media eso: una racha al azar cuenta).
      `fallos_bajo_umbral`: fallos con FDR media por debajo de `min_fdr` (el
      0,10 del protocolo), donde el retardo no se interpreta.

    `far_medio` y `fallos_nunca_detectados` ya no existen: pedirlas da un
    KeyError que dice cual usar.
    """
    if agg["faultNumber"].duplicated().any():
        raise ValueError(
            "summary espera una fila por fallo (un solo metodo o estadistico): "
            "filtra la tabla agregada antes de llamar."
        )
    con_fallo = agg[agg["faultNumber"] > 0]
    dificil = con_fallo["faultNumber"].isin(HARD_FAULTS)
    normal = agg[agg["faultNumber"] == 0]

    def media(serie: pd.Series) -> tuple[float, int]:
        v = serie.dropna()
        return (float(v.mean()) if v.size else np.nan), int(v.size)

    def mediana(serie: pd.Series) -> tuple[float, int]:
        v = serie.dropna()
        return (float(v.median()) if v.size else np.nan), int(v.size)

    def columna(tabla: pd.DataFrame, nombre: str) -> pd.Series:
        return tabla[nombre] if nombre in tabla else pd.Series(np.nan, index=tabla.index)

    r = _Resumen()
    r["fdr_medio_todos"], r["fdr_medio_todos_n"] = media(con_fallo["fdr_mean"])
    r["fdr_medio_sin_dificiles"], r["fdr_medio_sin_dificiles_n"] = media(
        con_fallo.loc[~dificil, "fdr_mean"])
    r["fdr_medio_dificiles"], r["fdr_medio_dificiles_n"] = media(con_fallo.loc[dificil, "fdr_mean"])

    if normal.empty:
        r["far_normal"], r["far_normal_n"] = np.nan, 0
    else:
        r["far_normal"] = float(normal["far_mean"].iloc[0])
        n_col = "far_n" if "far_n" in normal else "n_runs"
        r["far_normal_n"] = int(normal[n_col].iloc[0])
    r["far_prefallo_medio"], r["far_prefallo_medio_n"] = media(con_fallo["far_mean"])

    r["retardo_medio_muestras"], r["retardo_medio_muestras_n"] = media(con_fallo["delay_samples_mean"])
    r["retardo_mediano_muestras"], r["retardo_mediano_muestras_n"] = mediana(
        columna(con_fallo, "delay_samples_median"))

    r["fallos_sin_racha"] = sorted(
        con_fallo.loc[con_fallo["detection_ratio"] < 0.5, "faultNumber"].tolist())
    r["fallos_bajo_umbral"] = sorted(
        con_fallo.loc[con_fallo["fdr_mean"] < min_fdr, "faultNumber"].tolist())
    return r


def diagnosis_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, include_normal: bool,
                      normal_label=0, labels=None, hard_faults=HARD_FAULTS) -> dict:
    """Precision, recall y F1 por clase sobre toda la prueba, y macro-F1.

    La precision de una clase necesita los falsos positivos que llegan desde
    las demas clases, asi que el F1 por clase se calcula sobre TODA la prueba:
    el conjunto (pool) de simulaciones de prueba de todas las clases (regla 5).
    La macro-F1 es la media del F1 de cada clase, cada una calculada por
    separado como media armonica de su precision y su recall: pesa lo mismo
    cada tipo de fallo, y no cada muestra (memoria, 2.2.4). La exactitud
    equivale a la F1 "micro": cada error cuenta una vez como falso positivo y
    otra como falso negativo.

    Por que estaba mal el "f1" antiguo (cuaderno 02, celda [13], y el
    `results/summary/B1_clasificadores_clasico.csv` retirado en la v0.3): se
    calculaba con
    `f1_score(y_te[m], y_p[m], labels=[fault])` y `m = (y_te == fault)`, es
    decir, solo con las muestras cuya clase real es `fault`. Ahi ninguna
    muestra de otra clase puede predecirse como `fault`: no hay falsos
    positivos, la precision vale 1 y el resultado es 2 * 1 * r / (1 + r) =
    2r/(1+r), una funcion del recall r que nunca es menor que el F1 de verdad
    (son iguales solo si ninguna muestra de otra clase se clasifica como
    `fault`). El inflado es mayor cuanto mas baja es la precision de la clase,
    es decir, cuantas mas muestras de otras clases se le asignan (B3, D2).

    Los conteos usan todas las muestras: una muestra cuya clase real o predicha
    no esta en `labels` sigue contando como falso positivo o falso negativo de
    las clases que si estan, como hace scikit-learn con `labels`.

    Parametros
    ----------
    y_true, y_pred : array
        Clase real y predicha de cada muestra de toda la prueba.
    include_normal : bool
        Si la clase normal entra en el promedio de la macro-F1. Sin valor por
        defecto a proposito: es una decision pendiente del equipo y cambia la
        cifra. El F1 de la clase normal se calcula y se devuelve siempre.
    normal_label : int
        Etiqueta de la clase normal (0 en el TEP).
    labels : lista, opcional
        Clases que se informan y se promedian. Por defecto, todas las que
        aparecen en y_true o en y_pred (como scikit-learn). Una clase de
        `labels` que no aparece en ninguno de los dos es un error: su F1 no
        esta definido.
    hard_faults : tupla
        Fallos que se promedian aparte (regla 7). Por defecto, HARD_FAULTS.

    Devuelve
    --------
    dict con:
        per_class : DataFrame con una fila por clase y las columnas
            faultNumber, support, n_pred, tp, fp, fn, precision, recall, f1 e
            in_macro (si entra en la macro-F1). La precision es NaN si la clase
            no se predice nunca y el recall, si no tiene muestras reales; el F1
            se calcula como 2tp / (2tp + fp + fn), que vale 0 en esos casos.
        macro_f1, macro_f1_n : media del F1 de las clases con in_macro, y
            cuantas son.
        macro_f1_rest, macro_f1_rest_n : lo mismo sin los fallos dificiles.
        macro_f1_hard, macro_f1_hard_n : lo mismo solo con los fallos dificiles.
        include_normal : el valor usado, para guardarlo junto a la cifra.
        accuracy : exactitud sobre todas las muestras.
        n_samples : numero de muestras.

    Para el formato comun (regla 11), `per_class` va a un fichero aparte; el
    recall por (fallo, simulacion) lo da `diagnosis_recall_by_run`, y una
    macro-F1 por replica, para contrastarla, `macro_f1_by_replicate`.
    """
    if not isinstance(include_normal, (bool, np.bool_)):
        raise TypeError("include_normal debe ser True o False: es una decision del equipo.")
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"y_true e y_pred tienen longitudes distintas: {yt.shape} y {yp.shape}")
    if yt.size == 0:
        raise ValueError("No hay muestras.")
    etiquetas = np.union1d(yt, yp) if labels is None else np.asarray(labels).ravel()
    if pd.Series(etiquetas).duplicated().any():
        raise ValueError(f"labels tiene clases repetidas: {list(etiquetas)}")

    filas = []
    for c in etiquetas:
        real, predicha = yt == c, yp == c
        tp = int(np.sum(real & predicha))
        fp = int(np.sum(~real & predicha))
        fn = int(np.sum(real & ~predicha))
        if tp + fp + fn == 0:
            raise ValueError(
                f"La clase {c} no aparece ni en y_true ni en y_pred: su F1 no esta "
                "definido. Quitala de labels."
            )
        filas.append({
            "faultNumber": c.item() if hasattr(c, "item") else c,
            "support": tp + fn,
            "n_pred": tp + fp,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn),
            "in_macro": bool(include_normal or c != normal_label),
        })
    por_clase = pd.DataFrame(filas)

    en_macro = por_clase["in_macro"]
    es_dificil = por_clase["faultNumber"].isin(hard_faults)

    def media(mascara: pd.Series) -> tuple[float, int]:
        v = por_clase.loc[mascara, "f1"]
        return (float(v.mean()) if v.size else np.nan), int(v.size)

    resultado = {"per_class": por_clase}
    resultado["macro_f1"], resultado["macro_f1_n"] = media(en_macro)
    resultado["macro_f1_rest"], resultado["macro_f1_rest_n"] = media(en_macro & ~es_dificil)
    resultado["macro_f1_hard"], resultado["macro_f1_hard_n"] = media(en_macro & es_dificil)
    resultado["include_normal"] = bool(include_normal)
    resultado["accuracy"] = float(np.mean(yt == yp))
    resultado["n_samples"] = int(yt.size)
    return resultado


def macro_f1_by_replicate(y_true: np.ndarray, y_pred: np.ndarray, replicates: np.ndarray, *,
                          include_normal: bool, normal_label=0, labels=None,
                          hard_faults=HARD_FAULTS, replicate_col: str = "replicate") -> pd.DataFrame:
    """Macro-F1 por replica, para que el contraste del diagnostico tenga pares.

    La macro-F1 necesita toda la prueba (la precision de una clase depende de
    lo que se predice en las demas; ver diagnosis_metrics), asi que una
    particion da un solo numero, y con un numero por brazo no hay Wilcoxon
    posible. El OE2 contrasta la diferencia de macro-F1 entre validar los
    clasificadores por simulacion y validarlos de forma aleatoria por muestra
    con Wilcoxon y Holm (memoria, OE2), asi que hacen falta varias macro-F1 por
    brazo, emparejadas. Esta funcion da una por replica: cada replica es un
    conjunto de prueba completo, con todas las clases, y su macro-F1 es la de
    diagnosis_metrics sobre sus muestras.

    Que es una replica lo decide quien llama (la unidad es un parametro, y la
    decision es del equipo), y cambia lo que mide el contraste:

    - Una particion externa: `replicates` es el id de la particion de cada
      muestra de prueba; cada brazo se reentrena (y elige hiperparametros) en
      cada particion y se evalua en su prueba. La variabilidad incluye el
      entrenamiento. Las particiones deben usar simulaciones disjuntas; los
      pliegues de una misma validacion cruzada no son replicas independientes.
    - Un grupo de simulationRun comun a todas las clases: en Rieth, la
      simulacion r de cada fichero comparte semilla con la normal r, asi que
      las muestras de todas las clases con el mismo id (o con ids de un mismo
      bloque, por ejemplo runs // 5) forman un conjunto de prueba completo.
      Con un solo modelo por brazo, la variabilidad es solo la de la prueba: el
      contraste es condicional a esos modelos.

    Los dos brazos tienen que evaluarse sobre las mismas replicas (mismos ids)
    para que paired_test tenga pares. Cada replica tiene que tener muestras
    reales de todas las clases de `labels`: si no, su macro-F1 promediaria
    otras clases y no seria comparable con la de las demas (es un error).

    Parametros
    ----------
    y_true, y_pred : array
        Clase real y predicha de cada muestra de prueba.
    replicates : array
        Replica de cada muestra (id de la particion o del grupo de
        simulaciones). Sin NaN.
    include_normal : bool
        Si la clase normal entra en la macro-F1 (ver diagnosis_metrics). Sin
        valor por defecto: es una decision pendiente del equipo.
    normal_label, hard_faults : como en diagnosis_metrics.
    labels : lista, opcional
        Clases que se promedian, las mismas en todas las replicas. Por defecto,
        todas las que aparecen en y_true o en y_pred en toda la prueba.
    replicate_col : str
        Nombre de la columna de la replica en la salida (el `run_col` que se
        pasara a stats.compare_methods o a stats.compare_arms).

    Devuelve
    --------
    DataFrame con una fila por replica, ordenadas, y las columnas
    replicate_col, macro_f1, macro_f1_n, macro_f1_rest, macro_f1_rest_n,
    macro_f1_hard, macro_f1_hard_n, accuracy, n_samples e include_normal. Sin
    columna `method`: la anade quien llama, como en diagnosis_recall_by_run.
    El contraste, con las filas de los dos brazos juntas, por ejemplo:
    stats.compare_methods(tabla, metric="macro_f1", run_col="replicate",
    fault_col=None, baseline="por_simulacion").
    """
    if not isinstance(include_normal, (bool, np.bool_)):
        raise TypeError("include_normal debe ser True o False: es una decision del equipo.")
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    rr = pd.Series(np.asarray(replicates).ravel())
    if not (yt.shape == yp.shape == rr.shape):
        raise ValueError("y_true, y_pred y replicates deben tener la misma longitud.")
    if yt.size == 0:
        raise ValueError("No hay muestras.")
    if rr.isna().any():
        raise ValueError("Hay NaN en replicates: esas muestras no tendrian replica.")
    if pd.api.types.is_integer_dtype(rr):
        rr = rr.astype("int64")   # Rieth trae los ids en int16 (DAT-5)
    etiquetas = np.union1d(yt, yp) if labels is None else np.asarray(labels).ravel()

    filas = []
    for r, idx in sorted(rr.groupby(rr, sort=True).indices.items()):
        yt_r, yp_r = yt[idx], yp[idx]
        faltan = [c.item() if hasattr(c, "item") else c
                  for c in etiquetas if not np.any(yt_r == c)]
        if faltan:
            raise ValueError(
                f"La replica {r} no tiene muestras reales de las clases {faltan}: su "
                "macro-F1 no seria comparable con la de las demas. Cada replica tiene que "
                "ser un conjunto de prueba con todas las clases."
            )
        dm = diagnosis_metrics(yt_r, yp_r, include_normal=include_normal,
                               normal_label=normal_label, labels=etiquetas,
                               hard_faults=hard_faults)
        filas.append({
            replicate_col: r.item() if hasattr(r, "item") else r,
            **{k: dm[k] for k in ("macro_f1", "macro_f1_n", "macro_f1_rest",
                                  "macro_f1_rest_n", "macro_f1_hard", "macro_f1_hard_n",
                                  "accuracy", "n_samples", "include_normal")},
        })
    return pd.DataFrame(filas)


def diagnosis_recall_by_run(y_true: np.ndarray, y_pred: np.ndarray, runs: np.ndarray,
                            faults: np.ndarray | None = None) -> pd.DataFrame:
    """Recall del diagnostico por (fallo, simulacion), en el formato comun.

    La regla 11 pide en diagnostico una fila por (method, faultNumber,
    simulationRun) con el recall. El recall si esta bien definido por
    simulacion: solo necesita las muestras de esa simulacion. La precision y el
    F1 no, porque necesitan lo que se predice en las demas clases, y por eso
    van aparte, con `diagnosis_metrics` sobre todo el pool.

    Para una simulacion de la clase normal, el recall es la proporcion de sus
    muestras que se clasifican como normales.

    Parametros
    ----------
    y_true, y_pred : array
        Clase real y predicha de cada muestra.
    runs : array
        simulationRun de cada muestra.
    faults : array, opcional
        faultNumber del fichero (la trayectoria) de cada muestra. Por defecto,
        y_true. Hace falta pasarlo si se conservan, con etiqueta normal, las
        muestras previas al inicio de los ficheros con fallo (decision
        pendiente, regla 6): esas muestras no entran en el recall de su
        trayectoria y se cuentan en `n_other_label`.

    Devuelve
    --------
    DataFrame con una fila por (faultNumber, simulationRun) y las columnas
    recall (proporcion de las muestras de la trayectoria con etiqueta real
    igual a su fallo que se clasifican bien), n_samples (cuantas entran) y
    n_other_label (cuantas de la trayectoria tienen otra etiqueta real y no
    entran). Sin columna `method`: la anade quien llama, como con
    `evaluate_run`.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    rr = np.asarray(runs).ravel()
    ff = yt if faults is None else np.asarray(faults).ravel()
    if not (yt.shape == yp.shape == rr.shape == ff.shape):
        raise ValueError("y_true, y_pred, runs y faults deben tener la misma longitud.")
    tabla = pd.DataFrame({"faultNumber": ff, "simulationRun": rr})
    if tabla.isna().any().any():
        raise ValueError("Hay NaN en runs o faults: esas muestras no tendrian simulacion.")
    for col in ("faultNumber", "simulationRun"):
        if pd.api.types.is_integer_dtype(tabla[col]):
            tabla[col] = tabla[col].astype("int64")   # Rieth trae int16 (DAT-5)
    tabla["propia"] = yt == ff
    tabla["acierto"] = (yt == yp) & tabla["propia"].to_numpy()

    g = tabla.groupby(["faultNumber", "simulationRun"], sort=True)
    salida = pd.DataFrame({
        "n_samples": g["propia"].sum().astype("int64"),
        "aciertos": g["acierto"].sum(),
        "n_total": g.size(),
    })
    salida["recall"] = (salida["aciertos"] / salida["n_samples"]).where(salida["n_samples"] > 0)
    salida["n_other_label"] = (salida["n_total"] - salida["n_samples"]).astype("int64")
    return salida.reset_index()[["faultNumber", "simulationRun", "recall", "n_samples", "n_other_label"]]
