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
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def alarms(statistic: np.ndarray, limit: float) -> np.ndarray:
    """Vector booleano: True donde el estadistico supera el limite."""
    return np.asarray(statistic, dtype=float) > float(limit)


def far(statistic: np.ndarray, limit: float) -> float:
    """Tasa de falsas alarmas sobre muestras que se saben sanas.

    Se calcula sobre las muestras ANTERIORES al inicio del fallo, o sobre una
    simulacion completa de operacion normal.
    """
    a = alarms(statistic, limit)
    return float(a.mean()) if a.size else np.nan


def fdr(statistic: np.ndarray, limit: float) -> float:
    """Tasa de deteccion sobre muestras que se saben defectuosas."""
    a = alarms(statistic, limit)
    return float(a.mean()) if a.size else np.nan


def detection_delay(
    statistic: np.ndarray,
    limit: float,
    onset: int,
    k: int = 3,
    sample_minutes: float = 3.0,
) -> dict:
    """Retardo hasta la primera racha de k alarmas consecutivas tras el fallo.

    Parametros
    ----------
    statistic : array
        Estadistico para la simulacion completa, en orden temporal.
    limit : float
        Limite de control.
    onset : int
        Numero de muestras normales al principio. En los ficheros de prueba del
        TEP son 160: el fallo afecta a partir de la muestra 161, es decir al
        indice 160 en base cero.
    k : int
        Alarmas consecutivas necesarias para confirmar la deteccion.
    sample_minutes : float
        Minutos entre muestras. En el TEP son 3.

    Devuelve
    --------
    dict con 'delay_samples', 'delay_minutes' y 'detected'. Si el fallo nunca se
    confirma, el retardo es NaN y 'detected' es False.
    """
    a = alarms(statistic, limit)[onset:]
    if a.size < k:
        return {"delay_samples": np.nan, "delay_minutes": np.nan, "detected": False}

    # Ventana deslizante: posicion i vale si desde ahi hay k Trues seguidos.
    ventanas = np.lib.stride_tricks.sliding_window_view(a, k).all(axis=1)
    idx = np.flatnonzero(ventanas)

    if idx.size == 0:
        return {"delay_samples": np.nan, "delay_minutes": np.nan, "detected": False}

    d = int(idx[0])
    return {
        "delay_samples": d,
        "delay_minutes": d * float(sample_minutes),
        "detected": True,
    }


def stratified_fdr(statistic: np.ndarray, limit: float, onset: int) -> dict:
    """FDR partida en dos mitades del periodo con fallo, con regla fijada a priori.

    Un unico numero de FDR promedia regimenes distintos. En un fallo que el
    control compensa (por ejemplo IDV 5) el detector acierta casi siempre al
    principio y casi nunca despues, y la media no describe ninguno de los dos.
    La particion es la MITAD del periodo con fallo, la unica que se puede
    justificar sin mirar los datos. No se ajusta fallo a fallo.

    Devuelve fdr_h1 (primera mitad), fdr_h2 (segunda mitad) y persistence_gap,
    que es fdr_h1 - fdr_h2: cerca de 0 si el fallo se ve igual todo el tiempo,
    grande si se detecta y luego se pierde.
    """
    a = alarms(statistic, limit)[onset:]
    if a.size < 2:
        return {"fdr_h1": np.nan, "fdr_h2": np.nan, "persistence_gap": np.nan}
    mitad = a.size // 2
    h1, h2 = float(a[:mitad].mean()), float(a[mitad:].mean())
    return {"fdr_h1": h1, "fdr_h2": h2, "persistence_gap": h1 - h2}


def evaluate_run(
    statistic: np.ndarray,
    limit: float,
    onset: int | None,
    k: int = 3,
    sample_minutes: float = 3.0,
    min_fdr: float = 0.10,
) -> dict:
    """Evalua una simulacion completa y devuelve FAR, FDR, retardo y FDR estratificada.

    Con onset = None se entiende que la simulacion entera es de operacion
    normal: solo se calcula FAR.

    `detected_at_chance` marca (no filtra) las confirmaciones sospechosas: una
    racha de k alarmas puede aparecer por azar en 800 muestras autocorrelacionadas
    aunque el fallo sea invisible (IDV 3 en el TEP clasico: FDR 3,75 % y racha en
    la muestra 537). Si `detected` es True pero la FDR queda por debajo de
    `min_fdr`, el retardo no debe leerse como una deteccion. El 0,10 es una
    convencion declarada en el protocolo, no un resultado.
    """
    stat = np.asarray(statistic, dtype=float)

    if onset is None:
        return {
            "far": far(stat, limit),
            "fdr": np.nan,
            "delay_samples": np.nan,
            "delay_minutes": np.nan,
            "detected": False,
            "detected_at_chance": False,
            "fdr_h1": np.nan,
            "fdr_h2": np.nan,
            "persistence_gap": np.nan,
        }

    resultado = {
        "far": far(stat[:onset], limit),
        "fdr": fdr(stat[onset:], limit),
    }
    resultado.update(detection_delay(stat, limit, onset, k, sample_minutes))
    resultado["detected_at_chance"] = bool(resultado["detected"] and resultado["fdr"] < min_fdr)
    resultado.update(stratified_fdr(stat, limit, onset))
    return resultado


def aggregate(df: pd.DataFrame, by: str | list[str] = "faultNumber", ci: float = 0.95) -> pd.DataFrame:
    """Agrega resultados por simulacion en media, desviacion e intervalo de confianza.

    Recibe un DataFrame con una fila por simulacion (la salida de evaluate_run
    mas las columnas identificativas) y devuelve una fila por fallo.

    El intervalo de confianza sobre muchas simulaciones es lo que distingue
    nuestros resultados de la mayoria de la literatura, que informa de una sola
    corrida y no permite saber si una diferencia entre metodos es real o ruido.
    """
    metricas = [c for c in ("far", "fdr", "delay_samples", "delay_minutes",
                            "fdr_h1", "fdr_h2", "persistence_gap") if c in df]
    grupos = df.groupby(by)

    filas = []
    for clave, g in grupos:
        fila = {by: clave} if isinstance(by, str) else dict(zip(by, clave))
        fila["n_runs"] = len(g)
        fila["detection_ratio"] = g["detected"].mean() if "detected" in g else np.nan
        fila["at_chance_ratio"] = g["detected_at_chance"].mean() if "detected_at_chance" in g else np.nan

        for m in metricas:
            v = g[m].dropna().to_numpy()
            fila[f"{m}_mean"] = v.mean() if v.size else np.nan
            fila[f"{m}_std"] = v.std(ddof=1) if v.size > 1 else np.nan

            if v.size > 1:
                sem = stats.sem(v)
                margen = sem * stats.t.ppf((1 + ci) / 2.0, v.size - 1)
                fila[f"{m}_ci_low"] = v.mean() - margen
                fila[f"{m}_ci_high"] = v.mean() + margen
            else:
                fila[f"{m}_ci_low"] = np.nan
                fila[f"{m}_ci_high"] = np.nan

        filas.append(fila)

    return pd.DataFrame(filas)


#: Fallos del TEP que la literatura reporta como dificiles de detectar.
#: IDV(3) cambio escalon en la temperatura de alimentacion D, IDV(9) variacion
#: aleatoria de esa misma temperatura, IDV(15) agarrotamiento de la valvula del
#: condensador. Hay que reportarlos SIEMPRE por separado: promediarlos con el
#: resto esconde el unico sitio donde los metodos se diferencian de verdad.
HARD_FAULTS = (3, 9, 15)


def summary(agg: pd.DataFrame) -> dict:
    """Resumen global de una tabla agregada, con y sin los fallos dificiles."""
    con_fallo = agg[agg["faultNumber"] > 0]
    faciles = con_fallo[~con_fallo["faultNumber"].isin(HARD_FAULTS)]

    return {
        "fdr_medio_todos": con_fallo["fdr_mean"].mean(),
        "fdr_medio_sin_dificiles": faciles["fdr_mean"].mean(),
        "fdr_medio_dificiles": con_fallo[
            con_fallo["faultNumber"].isin(HARD_FAULTS)
        ]["fdr_mean"].mean(),
        "far_medio": agg["far_mean"].mean(),
        "retardo_medio_muestras": con_fallo["delay_samples_mean"].mean(),
        "fallos_nunca_detectados": sorted(
            con_fallo.loc[con_fallo["detection_ratio"] < 0.5, "faultNumber"].tolist()
        ),
    }
