"""Pruebas del paquete tfmfdd.

Ejecutar con:  pytest -q

No son un tramite. Cada una protege contra un error que ya se ha cometido o que
es facil cometer, y que NO lanza ninguna excepcion: simplemente produce numeros
equivocados con aspecto de resultado.
"""

import numpy as np
import pandas as pd
import pytest

from tfmfdd import data, limits, metrics
from tfmfdd.preprocessing import Scaler, lag_matrix
from tfmfdd.pca import PCAMonitor


# ---------------------------------------------------------------- datos

def test_d00_transpuesto_se_corrige():
    """d00.dat llega como (52, 500). Si no se transpone, todo lo demas es basura."""
    arr = np.random.rand(52, 500)
    with pytest.warns(UserWarning, match="transpuesta"):
        out = data._as_samples_by_vars(arr, "d00.dat")
    assert out.shape == (500, 52)


def test_matriz_normal_no_se_toca():
    arr = np.random.rand(960, 52)
    assert data._as_samples_by_vars(arr, "d01_te.dat").shape == (960, 52)


def test_matriz_incoherente_falla():
    with pytest.raises(ValueError, match="ninguna dimension"):
        data._as_samples_by_vars(np.random.rand(100, 33), "raro.dat")


def test_particion_sin_solapamiento():
    """Una simulacion no puede estar en dos conjuntos: seria fuga de datos."""
    tr, va, te = data.split_runs(100, seed=1)
    assert len(tr) + len(va) + len(te) == 100
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_particion_reproducible():
    assert np.array_equal(data.split_runs(50, seed=7)[0], data.split_runs(50, seed=7)[0])


# --------------------------------------------------------- preprocesado

def test_escalado_solo_con_entrenamiento():
    """El escalador NO debe reajustarse con los datos de prueba."""
    Xtr = np.random.randn(200, 5) * 3 + 10
    Xte = np.random.randn(50, 5) * 3 + 50   # media muy distinta, a proposito

    s = Scaler().fit(Xtr)
    Zte = s.transform(Xte)
    # Si se hubiera reajustado, la media de Zte seria ~0. Debe estar lejos.
    assert abs(Zte.mean()) > 1.0


def test_variable_constante_no_rompe():
    X = np.ones((100, 3))
    assert np.isfinite(Scaler().fit_transform(X)).all()


def test_lag_matrix_forma():
    X = np.arange(100).reshape(20, 5)
    assert lag_matrix(X, lags=2).shape == (18, 15)


def test_lag_matrix_contenido():
    """La primera fila debe apilar las muestras 2, 1 y 0 en ese orden."""
    X = np.arange(30).reshape(10, 3)
    L = lag_matrix(X, lags=2)
    assert np.array_equal(L[0], np.concatenate([X[2], X[1], X[0]]))


# -------------------------------------------------------------- limites

def test_t2_crece_con_la_confianza():
    assert limits.t2_limit(10, 500, 0.99) > limits.t2_limit(10, 500, 0.95)


def test_t2_exige_muestras_suficientes():
    with pytest.raises(ValueError):
        limits.t2_limit(600, 500)


def test_limites_spe_coherentes_entre_si():
    """Box, KDE y empirico deben dar valores del mismo orden."""
    rng = np.random.default_rng(0)
    spe = rng.chisquare(df=8, size=2000)
    box = limits.spe_limit_box(spe, 0.99)
    emp = limits.limit_empirical(spe, 0.99)
    kde = limits.limit_kde(spe, 0.99)
    assert 0.7 < box / emp < 1.4
    assert 0.7 < kde / emp < 1.4


# ------------------------------------------------------------- metricas

def test_far_y_fdr_basicos():
    stat = np.array([1.0, 1.0, 5.0, 5.0])
    assert metrics.far(stat, 3.0) == 0.5
    assert metrics.fdr(stat, 3.0) == 0.5


def test_retardo_exige_k_consecutivas():
    """Una sola exceedencia es ruido y no debe contar como deteccion."""
    stat = np.zeros(200)
    stat[100] = 99          # pico aislado tras el inicio del fallo
    stat[150:160] = 99      # racha real
    r = metrics.detection_delay(stat, limit=1.0, onset=50, k=3)
    assert r["detected"]
    assert r["delay_samples"] == 100   # 150 - 50, no 50


def test_retardo_nan_si_no_detecta():
    r = metrics.detection_delay(np.zeros(200), limit=1.0, onset=50, k=3)
    assert not r["detected"] and np.isnan(r["delay_samples"])


def test_retardo_en_minutos():
    stat = np.concatenate([np.zeros(60), np.full(10, 99.0)])
    r = metrics.detection_delay(stat, 1.0, onset=50, k=3, sample_minutes=3.0)
    assert r["delay_samples"] == 10 and r["delay_minutes"] == 30.0


def test_agregacion_calcula_intervalos():
    df = pd.DataFrame({
        "faultNumber": [1] * 10,
        "far": np.random.rand(10) * 0.02,
        "fdr": np.random.rand(10) * 0.1 + 0.8,
        "delay_samples": np.random.randint(5, 20, 10).astype(float),
        "delay_minutes": np.random.randint(15, 60, 10).astype(float),
        "detected": [True] * 10,
    })
    agg = metrics.aggregate(df)
    assert len(agg) == 1
    assert agg["fdr_ci_low"].iloc[0] < agg["fdr_mean"].iloc[0] < agg["fdr_ci_high"].iloc[0]


# ----------------------------------------------------------------- pca

def test_pca_detecta_un_desplazamiento_evidente():
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((500, 10))
    m = PCAMonitor(n_components=5, alpha=0.99).fit(normal[:350], X_cal=normal[350:])

    fallo = rng.standard_normal((200, 10))
    fallo[100:, 0] += 12          # desplazamiento grosero a mitad de corrida
    t2, _ = m.score(fallo)

    assert metrics.far(t2[:100], m.t2_limit_) < 0.15
    assert metrics.fdr(t2[100:], m.t2_limit_) > 0.80


def test_calibracion_reservada_da_limite_mayor():
    """Calibrar con datos reservados debe ensanchar el limite, no estrecharlo."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((600, 8))
    sin = PCAMonitor(n_components=4).fit(X[:400])
    con = PCAMonitor(n_components=4).fit(X[:400], X_cal=X[400:])
    assert con.spe_limit_ > sin.spe_limit_
    assert con.calibrated_on_holdout_ and not sin.calibrated_on_holdout_


def test_score_antes_de_fit_falla():
    with pytest.raises(RuntimeError):
        PCAMonitor().score(np.random.rand(10, 5))
