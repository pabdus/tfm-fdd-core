"""Pruebas del monitor PCA/DPCA para la v0.3 (limits, pca, preprocessing).

Cada bloque protege un cambio concreto:
- limites del T2 de Fase I (Beta), empirico y KDE, con Monte Carlo (M3, DIS-5),
- percentil empirico conforme (MET-3),
- calibracion honesta: de donde sale cada limite, X_cal vacio, limites no
  finitos, aviso al calibrar en muestra sin decirlo (M2, M11),
- indice combinado T2 o SPE de Yin et al. (2012),
- DPCA alineado con las muestras originales y lag_by_run por trayectoria
  contigua (M4, MET-1, MET-2),
- Scaler dentro de una Pipeline y gancho del escalador externo (M6, DIS-7),
- valores por defecto iguales a los de configs/baseline.yaml (M9, REP-4).

Todas usan datos sinteticos con semilla fija.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from tfmfdd import limits, metrics
from tfmfdd.pca import PCAMonitor
from tfmfdd.preprocessing import Scaler, lag_by_run

RAIZ = Path(__file__).resolve().parents[1]


def _normales(n, p, rng, autovalores=None):
    """Normales multivariantes con autovalores dados y direcciones al azar."""
    lam = np.ones(p) if autovalores is None else np.asarray(autovalores, dtype=float)
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return (rng.standard_normal((n, p)) * np.sqrt(lam)) @ Q.T


def _trayectoria(rng, n, phi=0.8):
    """Trayectoria con dinamica: dos factores AR(1) vistos en 6 variables."""
    A = np.array([[1.0, 0.8, 0.6, 0.0, 0.0, 0.3], [0.0, 0.5, -0.4, 1.0, 0.7, 0.2]])
    f = np.zeros((n, 2))
    e = rng.standard_normal((n, 2))
    for t in range(1, n):
        f[t] = phi * f[t - 1] + e[t]
    return f @ A + 0.3 * rng.standard_normal((n, 6))


def _hotelling_mc(p, m, replicas, n_nuevas, seed):
    """T2 de Hotelling con media y covarianza estimadas, en muestra y sobre datos nuevos."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((replicas, m, p))
    media = X.mean(axis=1, keepdims=True)
    D = X - media
    S_inv = np.linalg.inv(np.einsum("rij,rik->rjk", D, D) / (m - 1))
    t2_muestra = np.einsum("rij,rjk,rik->ri", D, S_inv, D)
    Y = rng.standard_normal((replicas, n_nuevas, p)) - media
    t2_nuevas = np.einsum("rij,rjk,rik->ri", Y, S_inv, Y)
    return t2_muestra, t2_nuevas


# ------------------------------------------------------------ limites del T2

def test_t2_fase1_mas_estrecho_que_fase2():
    assert limits.t2_limit_beta(27, 350, 0.99) < limits.t2_limit(27, 350, 0.99)


def test_t2_fase1_exige_muestras_suficientes():
    with pytest.raises(ValueError, match="Fase I"):
        limits.t2_limit_beta(10, 11, 0.99)


def test_t2_beta_controla_la_far_en_muestra():
    """Monte Carlo: el limite Beta deja fuera un 1 % de las propias muestras del ajuste."""
    p, m = 5, 50
    t2_muestra, _ = _hotelling_mc(p, m, replicas=3000, n_nuevas=1, seed=11)
    far = (t2_muestra > limits.t2_limit_beta(p, m, 0.99)).mean()
    assert far == pytest.approx(0.01, abs=0.002)


def test_t2_f_controla_la_far_en_datos_nuevos():
    """Monte Carlo: el limite F (Fase II) da un 1 % de falsas alarmas en datos nuevos."""
    p, m = 5, 50
    _, t2_nuevas = _hotelling_mc(p, m, replicas=3000, n_nuevas=40, seed=12)
    far = (t2_nuevas > limits.t2_limit(p, m, 0.99)).mean()
    assert far == pytest.approx(0.01, abs=0.002)


def test_t2_beta_en_datos_nuevos_da_la_far_que_predice_la_teoria():
    """Con normales independientes, el limite de Fase I sobre datos nuevos da la FAR que predice la F."""
    p, m = 5, 50
    _, t2_nuevas = _hotelling_mc(p, m, replicas=3000, n_nuevas=40, seed=13)
    lim_beta = limits.t2_limit_beta(p, m, 0.99)
    teorica = stats.f.sf(lim_beta * m * (m - p) / (p * (m**2 - 1)), p, m - p)
    far = (t2_nuevas > lim_beta).mean()
    assert teorica > 0.02
    assert far == pytest.approx(teorica, abs=0.004)


def test_pca_monitor_t2_beta_y_f_en_monte_carlo():
    """Con PCAMonitor (3 de 6 componentes): Beta en muestra y F en datos nuevos, cerca del 1 %."""
    rng = np.random.default_rng(14)
    autovalores = [8, 4, 2, 0.3, 0.2, 0.1]
    far_beta_muestra, far_f_nuevas = [], []
    for _ in range(300):
        Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        X = (rng.standard_normal((100, 6)) * np.sqrt(autovalores)) @ Q.T
        Y = (rng.standard_normal((100, 6)) * np.sqrt(autovalores)) @ Q.T
        mb = PCAMonitor(n_components=3, t2_method="beta").fit(X, calibration="in_sample")
        mf = PCAMonitor(n_components=3, t2_method="f").fit(X, calibration="in_sample")
        far_beta_muestra.append((mb.score(X)[0] > mb.t2_limit_).mean())
        far_f_nuevas.append((mf.score(Y)[0] > mf.t2_limit_).mean())
    assert np.mean(far_beta_muestra) == pytest.approx(0.01, abs=0.003)
    assert np.mean(far_f_nuevas) == pytest.approx(0.01, abs=0.003)


def test_t2_empirico_y_kde_se_calibran_con_x_cal():
    rng = np.random.default_rng(15)
    X = _normales(600, 6, rng, [8, 4, 2, 0.3, 0.2, 0.1])
    for metodo, funcion in (("empirical", limits.limit_empirical), ("kde", limits.limit_kde)):
        m = PCAMonitor(n_components=3, t2_method=metodo).fit(X[:400], X_cal=X[400:])
        t2_cal, _ = m.score(X[400:])
        assert m.t2_limit_ == pytest.approx(funcion(t2_cal, 0.99))
        assert m.limit_source_["t2"] == "holdout"


def test_t2_method_desconocido_falla():
    with pytest.raises(ValueError, match="t2_method"):
        PCAMonitor(t2_method="chi2").fit(np.random.default_rng(0).normal(size=(50, 4)))


# -------------------------------------------------- percentil empirico conforme

def test_percentil_conforme_controla_la_far_y_el_lineal_no():
    """Con n = 150 y datos intercambiables: el conforme queda en <= 1 %, el lineal por encima (MET-3)."""
    rng = np.random.default_rng(16)
    n, alpha = 150, 0.99
    U = rng.uniform(size=(4000, n))   # con uniformes, la FAR de un umbral u es 1 - u
    far_conforme = np.mean([1 - limits.limit_empirical(u, alpha, method="conformal") for u in U])
    far_lineal = np.mean([1 - limits.limit_empirical(u, alpha) for u in U])
    assert far_conforme <= 0.01 < far_lineal
    assert far_conforme == pytest.approx(1 / (n + 1), abs=0.0007)
    assert far_lineal == pytest.approx((n - alpha * (n - 1)) / (n + 1), abs=0.0007)


def test_percentil_conforme_falla_con_pocas_muestras():
    with pytest.raises(ValueError, match="conformal"):
        limits.limit_empirical(np.arange(50.0), 0.99, method="conformal")


def test_percentil_lineal_es_el_de_siempre():
    x = np.random.default_rng(17).chisquare(5, size=300)
    assert limits.limit_empirical(x, 0.99) == np.percentile(x, 99)


# ------------------------------------------------------ calibracion honesta

def test_jackson_con_x_cal_no_dice_reservado():
    """M2: Jackson usa los autovalores del ajuste; no puede figurar como reservado."""
    rng = np.random.default_rng(20)
    X = _normales(500, 8, rng)
    with pytest.warns(UserWarning, match="X_cal no interviene"):
        m = PCAMonitor(n_components=4, spe_method="jackson").fit(X[:350], X_cal=X[350:])
    assert m.limit_source_ == {"t2": "parametric_fit", "spe": "parametric_fit"}
    assert m.calibrated_on_holdout_ is False
    assert m.n_cal_ is None


def test_box_con_x_cal_dice_reservado():
    rng = np.random.default_rng(21)
    X = _normales(500, 8, rng)
    m = PCAMonitor(n_components=4, spe_method="box").fit(X[:350], X_cal=X[350:])
    assert m.limit_source_ == {"t2": "parametric_fit", "spe": "holdout"}
    assert m.calibrated_on_holdout_ is True
    assert m.n_fit_ == 350 and m.n_cal_ == 150


def test_sin_x_cal_avisa_que_calibra_en_muestra():
    X = _normales(300, 6, np.random.default_rng(22))
    with pytest.warns(UserWarning, match="en muestra"):
        m = PCAMonitor(n_components=3).fit(X)
    assert m.limit_source_["spe"] == "in_sample"
    assert m.calibrated_on_holdout_ is False


def test_en_muestra_a_proposito_no_avisa():
    X = _normales(300, 6, np.random.default_rng(23))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = PCAMonitor(n_components=3).fit(X, calibration="in_sample")
    assert m.limit_source_ == {"t2": "parametric_fit", "spe": "in_sample"}


def test_en_muestra_con_un_subconjunto_iguala_n_cal():
    """DIS-4: el brazo en muestra puede calibrarse con tantas filas del ajuste como el reservado."""
    X = _normales(500, 6, np.random.default_rng(24))
    m = PCAMonitor(n_components=3, spe_method="empirical").fit(
        X[:350], X_cal=X[200:350], calibration="in_sample"
    )
    assert m.limit_source_["spe"] == "in_sample" and m.n_cal_ == 150


def test_x_cal_que_comparte_filas_con_x_no_es_reservado():
    X = _normales(400, 6, np.random.default_rng(25))
    with pytest.raises(ValueError, match="comparte"):
        PCAMonitor(n_components=3).fit(X, X_cal=X[-100:])


def test_x_cal_vacio_falla():
    """M11: con --cal-fraction 0 el limite salia NaN en silencio."""
    X = _normales(300, 6, np.random.default_rng(26))
    with pytest.raises(ValueError, match="no tiene filas"):
        PCAMonitor(n_components=3).fit(X, X_cal=X[300:])


def test_alpha_fuera_de_rango_falla():
    X = _normales(300, 6, np.random.default_rng(27))
    with pytest.raises(ValueError, match="alpha"):
        PCAMonitor(n_components=3, alpha=99).fit(X, calibration="in_sample")


def test_limites_ignoran_nan_y_rechazan_calibracion_vacia():
    spe = np.random.default_rng(28).chisquare(6, size=500)
    con_nan = np.concatenate([[np.nan, np.nan], spe])
    assert limits.spe_limit_box(con_nan) == pytest.approx(limits.spe_limit_box(spe))
    with pytest.raises(ValueError, match="evaluados"):
        limits.spe_limit_box(np.full(10, np.nan))
    with pytest.raises(ValueError, match="infinitos"):
        limits.limit_empirical(np.array([1.0, np.inf, 2.0]))


def test_jackson_con_h0_negativo_falla():
    """MET-7: con h0 < 0 la formula devolvia un cuantil inferior sin avisar."""
    autovalores = np.concatenate([[1.0], np.full(50, 0.1)])
    with pytest.raises(ValueError, match="h0"):
        limits.spe_limit_jackson(autovalores, 0, 0.99)


# ------------------------------------------------------- indice combinado

def test_indice_combinado_es_la_disyuncion_t2_o_spe():
    datos = _normales(1000, 8, np.random.default_rng(30), [6, 4, 3, 1, 0.5, 0.3, 0.2, 0.1])
    X, prueba = datos[:600], datos[600:].copy()
    m = PCAMonitor(n_components=3).fit(X[:400], X_cal=X[400:])
    prueba[200:, 0] += 3.0
    t2, spe = m.score(prueba)
    indice = m.combined_index(t2, spe)
    o_logico = metrics.alarms(t2, m.t2_limit_) | metrics.alarms(spe, m.spe_limit_)
    assert np.array_equal(metrics.alarms(indice, 1.0), o_logico)
    r = metrics.evaluate_run(indice, 1.0, onset=200)
    assert r["fdr"] == pytest.approx(o_logico[200:].mean())


def test_indice_combinado_propaga_nan():
    m = PCAMonitor(n_components=2).fit(_normales(200, 4, np.random.default_rng(31)), calibration="in_sample")
    indice = m.combined_index(np.array([np.nan, 1.0]), np.array([0.5, np.nan]))
    assert np.isnan(indice).all()


# --------------------------------------------------------------------- DPCA

def test_dpca_score_devuelve_una_fila_por_muestra_con_nan_al_inicio():
    rng = np.random.default_rng(40)
    X_fit, X_cal = _trayectoria(rng, 500), _trayectoria(rng, 300)
    m = PCAMonitor(n_components=2, lags=2).fit(X_fit, np.ones(500), X_cal=X_cal, runs_cal=np.ones(300))
    t2, spe = m.score(_trayectoria(rng, 120), runs=np.ones(120))
    assert len(t2) == len(spe) == 120
    assert np.isnan(t2[:2]).all() and np.isnan(spe[:2]).all()
    assert np.isfinite(t2[2:]).all() and np.isfinite(spe[2:]).all()


def test_dpca_mismo_onset_y_retardo_sin_adelanto():
    """MET-1: fallo visible en el indice 170 y onset = 160; el retardo es 10 con lags 0, 1 y 2."""
    rng = np.random.default_rng(41)
    X_fit = np.vstack([_trayectoria(rng, 500) for _ in range(4)])
    runs_fit = np.repeat([1, 2, 3, 4], 500)
    X_cal = np.vstack([_trayectoria(rng, 500) for _ in range(2)])
    runs_cal = np.repeat([5, 6], 500)
    prueba = _trayectoria(rng, 400)
    prueba[170:, 0] += 8.0
    for lags in (0, 1, 2):
        m = PCAMonitor(n_components=2, lags=lags).fit(X_fit, runs_fit, X_cal=X_cal, runs_cal=runs_cal)
        _, spe = m.score(prueba, runs=np.ones(len(prueba)))
        r = metrics.evaluate_run(spe, m.spe_limit_, onset=160)
        assert r["delay_samples"] == 10, f"lags = {lags}"


def test_lag_by_run_con_dos_clases_que_comparten_ids():
    """MET-2: dos clases con ids 1 y 2 cada una; se corta por bloque contiguo y no se reordena."""
    clase = np.repeat([1, 1, 2, 2], 3)                 # fallo 1 (sim 1, 2) y fallo 2 (sim 1, 2)
    runs = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
    X = np.column_stack([clase * 100 + np.arange(12), clase])
    L, ids, pos = lag_by_run(X, runs, lags=1, return_index=True)
    assert L.shape == (8, 4)
    assert np.array_equal(ids, [1, 1, 2, 2, 1, 1, 2, 2])
    assert np.array_equal(pos, [1, 2, 4, 5, 7, 8, 10, 11])
    assert np.array_equal(L[:, 1], L[:, 3])            # ninguna fila mezcla clases


def test_lag_by_run_con_dos_columnas_separa_trayectorias_seguidas_con_el_mismo_id():
    X = np.arange(12.0).reshape(6, 2)
    runs = np.array([[1, 1], [1, 1], [1, 1], [2, 1], [2, 1], [2, 1]])   # (fallo, simulacion)
    L, _ = lag_by_run(X, runs, lags=1)
    assert L.shape == (4, 4)


def test_dpca_score_con_dos_clases_que_comparten_ids():
    """Puntuar dos clases juntas da lo mismo que puntuar cada trayectoria por separado."""
    rng = np.random.default_rng(42)
    m = PCAMonitor(n_components=2, lags=1).fit(_trayectoria(rng, 500), np.ones(500), X_cal=_trayectoria(rng, 300), runs_cal=np.ones(300))
    trayectorias = [_trayectoria(rng, 50) for _ in range(4)]
    runs = np.repeat([1, 2, 1, 2], 50)                 # fallo 1 (sim 1, 2) y fallo 2 (sim 1, 2)
    t2, spe = m.score(np.vstack(trayectorias), runs=runs)
    por_separado = np.concatenate([m.score(T, runs=np.ones(50))[1] for T in trayectorias])
    np.testing.assert_allclose(spe, por_separado)
    assert np.flatnonzero(np.isnan(spe)).tolist() == [0, 50, 100, 150]


def test_dpca_sin_runs_trata_x_como_una_trayectoria_y_avisa():
    """A1 --lags 1 fallaba con ValueError porque no pasa runs."""
    rng = np.random.default_rng(43)
    X = _trayectoria(rng, 400)
    with pytest.warns(UserWarning, match="UNA sola"):
        m = PCAMonitor(n_components=2, lags=1).fit(X[:300], X_cal=X[300:])
    with pytest.warns(UserWarning, match="UNA sola"):
        t2, _ = m.score(X[300:])
    assert len(t2) == 100 and np.isnan(t2[0])


# ------------------------------------------------------------------ escalado

def test_scaler_dentro_de_pipeline_y_gridsearch():
    """M6: el Scaler fallaba en todos los pliegues con TypeError."""
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC

    rng = np.random.default_rng(50)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    grupos = np.repeat(np.arange(6), 20)
    busqueda = GridSearchCV(
        Pipeline([("esc", Scaler()), ("svm", SVC())]),
        {"svm__C": [1, 10]},
        cv=GroupKFold(3),
    ).fit(X, y, groups=grupos)
    assert np.isfinite(busqueda.best_score_)
    escalador = busqueda.best_estimator_.named_steps["esc"]
    assert escalador.normal_label == 0
    assert escalador.n_samples_fit_ == (y == 0).sum()   # el reajuste final, solo con las normales


def test_scaler_con_etiquetas_se_ajusta_solo_con_las_normales_por_defecto():
    """Memoria E2, parr. 367: la estandarizacion, solo con las normales, tambien en el diagnostico."""
    rng = np.random.default_rng(51)
    X = np.vstack([rng.normal(0, 1, (100, 3)), rng.normal(50, 5, (60, 3))])
    y = np.repeat([0, 4], [100, 60])
    s = Scaler().fit(X, y)
    np.testing.assert_allclose(s.mean_, X[:100].mean(axis=0))
    np.testing.assert_allclose(s.std_, X[:100].std(axis=0, ddof=1))
    assert s.transform(X).shape == X.shape and s.n_samples_fit_ == 100


def test_scaler_sin_etiquetas_hace_lo_de_siempre():
    X = np.random.default_rng(52).normal(3, 2, (80, 4))
    for s in (Scaler().fit(X), Scaler(normal_label=None).fit(X), Scaler().fit(X, np.zeros(80))):
        np.testing.assert_allclose(s.mean_, X.mean(axis=0))
        np.testing.assert_allclose(s.std_, X.std(axis=0, ddof=1))


def test_scaler_con_todas_las_clases_hay_que_pedirlo():
    """normal_label=None ignora las etiquetas: se aparta del protocolo y es explicito."""
    rng = np.random.default_rng(56)
    X = np.vstack([rng.normal(0, 1, (100, 3)), rng.normal(50, 5, (60, 3))])
    y = np.repeat([0, 4], [100, 60])
    s = Scaler(normal_label=None).fit(X, y)
    np.testing.assert_allclose(s.mean_, X.mean(axis=0))
    assert s.n_samples_fit_ == 160


def test_scaler_sin_filas_normales_falla():
    with pytest.raises(ValueError, match="No hay filas"):
        Scaler().fit(np.ones((10, 2)), np.full(10, 3))


def test_gancho_del_escalador_limpio_reproduce_el_interno():
    rng = np.random.default_rng(53)
    X = _normales(500, 6, rng) * 3 + 10
    interno = PCAMonitor(n_components=3).fit(X[:350], X_cal=X[350:])
    externo = PCAMonitor(n_components=3).fit(X[:350], X_cal=X[350:], scaler=Scaler().fit(X[:350]))
    assert externo.scaler_source_ == "external"
    np.testing.assert_allclose(externo.spe_limit_, interno.spe_limit_)
    np.testing.assert_allclose(externo.score(X[350:])[1], interno.score(X[350:])[1])


def test_gancho_del_escalador_con_la_prueba_deja_de_ser_nulo():
    """DIS-7: escalar con ajuste y prueba juntos ya cambia el modelo (antes, nada)."""
    datos = _normales(800, 6, np.random.default_rng(54), [5, 3, 1, 0.5, 0.3, 0.2])
    X, prueba = datos[:500], datos[500:].copy()
    prueba[:, :2] += 4.0
    limpio = PCAMonitor(n_components=3).fit(X[:350], X_cal=X[350:])
    sucio = PCAMonitor(n_components=3).fit(
        X[:350], X_cal=X[350:], scaler=Scaler().fit(np.vstack([X[:350], prueba]))
    )
    assert not np.allclose(sucio.score(prueba)[1], limpio.score(prueba)[1])
    assert not np.isclose(sucio.spe_limit_, limpio.spe_limit_)


def test_gancho_del_escalador_con_dpca_escala_antes_de_los_retardos():
    rng = np.random.default_rng(55)
    X_fit = _trayectoria(rng, 400)
    m = PCAMonitor(n_components=2, lags=1).fit(
        X_fit, np.ones(400), X_cal=_trayectoria(rng, 200), runs_cal=np.ones(200),
        scaler=Scaler().fit(X_fit),
    )
    t2, _ = m.score(_trayectoria(rng, 60), runs=np.ones(60))
    assert len(t2) == 60 and np.isnan(t2[0]) and np.isfinite(t2[1:]).all()


# ------------------------------------------------------ valores por defecto

def test_valores_por_defecto_coinciden_con_el_yaml():
    """M9: configs/baseline.yaml es la unica fuente de los valores por defecto."""
    yaml = pytest.importorskip("yaml")
    ruta = RAIZ / "configs" / "baseline.yaml"
    if not ruta.exists():
        pytest.skip("no esta configs/baseline.yaml")
    cfg = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    m = PCAMonitor()
    assert m.variance == cfg["pca"]["variance"]
    assert m.spe_method == cfg["pca"]["spe_method"]
    assert m.lags == cfg["pca"]["lags"]
    assert m.alpha == cfg["protocolo"]["alpha"]
