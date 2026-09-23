"""Pruebas anadidas el 22-09-2026 tras la revision del paquete.

Cada una protege una correccion concreta:
- el subconjunto de 33 variables de Yin y la seleccion por nombre,
- la FDR estratificada por mitades y la marca de confirmaciones espurias,
- el experimento de calibracion sin confundido (mismo modelo, distinto limite),
- el contraste pareado con varios fallos en la misma tabla.
"""

import numpy as np
import pandas as pd
import pytest

from tfmfdd import data, metrics, stats
from tfmfdd.pca import PCAMonitor


def _tabla(n=200, m=52, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, m)), columns=data.VAR_NAMES)
    df.insert(0, "sample", np.arange(1, n + 1))
    df.insert(0, "simulationRun", 1)
    df.insert(0, "faultNumber", 0)
    return df


# ---------------------------------------------------------------- datos

def test_values_por_defecto_52_y_yin_33():
    df = _tabla()
    assert data.values(df).shape == (200, 52)
    assert data.values(df, columns=data.YIN_VARS).shape == (200, 33)
    assert data.YIN_VARS[0] == "xmeas_1" and data.YIN_VARS[21] == "xmeas_22"
    assert data.YIN_VARS[-1] == "xmv_11"


def test_values_falla_si_falta_una_columna():
    """Seleccion por nombre: una columna ausente revienta, no devuelve basura."""
    df = _tabla().drop(columns=["xmeas_7"])
    with pytest.raises(KeyError):
        data.values(df)


def test_onset_entrenamiento_clasico():
    assert data.FAULT_ONSET_TRAIN_CLASSIC == 20


# ---------------------------------------------------------------- metricas

def test_fdr_estratificada_separa_dos_regimenes():
    """Detecta la primera mitad y pierde la segunda: FDR 0,5 esconde 1,0 y 0,0."""
    onset = 10
    stat = np.zeros(onset + 100)
    stat[onset:onset + 50] = 10.0            # primera mitad por encima del limite
    r = metrics.evaluate_run(stat, limit=1.0, onset=onset)
    assert r["fdr"] == pytest.approx(0.5)
    assert r["fdr_h1"] == pytest.approx(1.0)
    assert r["fdr_h2"] == pytest.approx(0.0)
    assert r["persistence_gap"] == pytest.approx(1.0)


def test_fdr_estratificada_plana_si_regimen_constante():
    stat = np.full(210, 10.0)
    r = metrics.evaluate_run(stat, limit=1.0, onset=10)
    assert r["persistence_gap"] == pytest.approx(0.0)


def test_confirmacion_espuria_queda_marcada():
    """Tres alarmas seguidas en un fallo casi invisible: detected=True pero al azar."""
    onset = 10
    stat = np.zeros(onset + 800)
    stat[onset + 500: onset + 503] = 10.0    # racha de 3 sobre 800 -> FDR 0,4 %
    r = metrics.evaluate_run(stat, limit=1.0, onset=onset)
    assert r["detected"] is True
    assert r["fdr"] < 0.01
    assert r["detected_at_chance"] is True


def test_deteccion_real_no_se_marca():
    stat = np.zeros(810); stat[10:] = 10.0
    r = metrics.evaluate_run(stat, limit=1.0, onset=10)
    assert r["detected"] and not r["detected_at_chance"]


def test_agregacion_incluye_nuevas_columnas():
    filas = []
    for run in range(1, 6):
        stat = np.zeros(210); stat[10:110] = 10.0
        filas.append({"faultNumber": 1, "simulationRun": run,
                      **metrics.evaluate_run(stat, 1.0, onset=10)})
    agg = metrics.aggregate(pd.DataFrame(filas))
    for col in ("fdr_h1_mean", "fdr_h2_mean", "persistence_gap_mean", "at_chance_ratio"):
        assert col in agg
    assert agg.loc[0, "at_chance_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------- calibracion

def test_calibracion_compara_el_mismo_modelo():
    """A y B deben compartir escalador y cargas; solo cambia el limite del SPE."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(8, 52))
    X = rng.normal(size=(500, 8)) @ A + 0.3 * rng.normal(size=(500, 52))
    X_fit, X_cal = X[:350], X[350:]
    a = PCAMonitor(n_components=8, alpha=0.99).fit(X_fit)
    b = PCAMonitor(n_components=8, alpha=0.99).fit(X_fit, X_cal=X_cal)
    assert np.allclose(a.loadings_, b.loadings_)
    assert np.allclose(a.scaler_.mean_, b.scaler_.mean_)
    assert a.t2_limit_ == pytest.approx(b.t2_limit_)     # T2 no depende de X_cal
    assert b.spe_limit_ > a.spe_limit_                    # el reservado es mas ancho


# ---------------------------------------------------------------- contrastes

def test_compare_methods_con_varios_fallos_empareja_por_fallo():
    rng = np.random.default_rng(2)
    filas = []
    for fault in (1, 2):
        for run in range(1, 11):
            base = rng.uniform(0.5, 0.9)
            filas.append({"faultNumber": fault, "simulationRun": run, "method": "pca", "fdr": base})
            filas.append({"faultNumber": fault, "simulationRun": run, "method": "ae", "fdr": base + 0.05})
    tabla = stats.compare_methods(pd.DataFrame(filas), metric="fdr")
    assert set(tabla["faultNumber"]) == {1, 2}
    assert len(tabla) == 2
    assert (tabla["n"] == 10).all()


def test_compare_methods_rechaza_indices_duplicados_sin_fallo():
    filas = [{"simulationRun": 1, "method": m, "fdr": 0.5} for m in ("a", "b")] * 2
    with pytest.raises(ValueError, match="varias filas"):
        stats.compare_methods(pd.DataFrame(filas), metric="fdr", fault_col=None)
