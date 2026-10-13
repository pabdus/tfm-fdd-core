"""Pruebas de las correcciones del 12/10 en el monitor (limits, pca, preprocessing).

Cada una cubre un hallazgo de la verificacion independiente:
- V-01: el escalador externo se guarda copiado y se sabe con cuantas filas se ajusto,
- V-02: aviso cuando una trayectoria es mas larga que cualquier simulacion del TEP,
- V-09: con error_score="raise", un pliegue sin la clase normal hace fallar la busqueda,
- V-10: `samples` detecta muestras no consecutivas antes de apilar retardos,
- I5: limit_kde da el cuantil de verdad de la KDE, tambien con estadisticos negativos,
- M6: spe_limit_jackson exige los autovalores en orden decreciente.

Datos sinteticos con semilla fija.
"""

import warnings

import numpy as np
import pytest
from scipy import integrate, stats

from tfmfdd import limits
from tfmfdd.pca import PCAMonitor
from tfmfdd.preprocessing import TEP_MAX_RUN_LENGTH, Scaler, lag_by_run


def _datos(n, p, rng, desplazamiento=0.0):
    A = np.random.default_rng(100).normal(size=(p, p))
    return rng.normal(size=(n, p)) @ A + desplazamiento


def _cuantil_kde_integrado(x, alpha):
    """Referencia independiente: integra la densidad de la KDE en una rejilla fina."""
    kde = stats.gaussian_kde(x)
    h = float(np.sqrt(kde.covariance[0, 0]))
    lo, hi = x.min() - 12 * h, x.max() + 12 * h
    rejilla = np.linspace(lo, hi, int(np.ceil((hi - lo) / (h / 50))) + 1)
    acumulada = integrate.cumulative_trapezoid(kde.pdf(rejilla), rejilla, initial=0.0)
    return float(np.interp(alpha, acumulada, rejilla))


# ------------------------------------------------------------------- V-01

def test_reajustar_el_escalador_externo_no_cambia_el_modelo():
    """El mismo Scaler reutilizado para el brazo S1 dejaba el limpio sin deteccion."""
    rng = np.random.default_rng(1)
    X_fit, X_cal, X_te = _datos(300, 5, rng), _datos(100, 5, rng), _datos(200, 5, rng, 3.0)
    sc = Scaler()
    limpio = PCAMonitor(n_components=2).fit(X_fit, X_cal=X_cal, scaler=sc.fit(X_fit))
    antes, _ = limpio.score(X_te)
    PCAMonitor(n_components=2).fit(X_fit, X_cal=X_cal, scaler=sc.fit(np.vstack([X_fit, X_te])))
    despues, _ = limpio.score(X_te)
    assert limpio.scaler_ is not sc
    np.testing.assert_array_equal(antes, despues)


def test_el_modelo_dice_con_cuantas_filas_se_ajusto_el_escalador():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(2)
    X_fit, X_cal, X_te = _datos(300, 4, rng), _datos(100, 4, rng), _datos(150, 4, rng)
    interno = PCAMonitor(n_components=2).fit(X_fit, X_cal=X_cal)
    limpio = PCAMonitor(n_components=2).fit(X_fit, X_cal=X_cal, scaler=Scaler().fit(X_fit))
    s1 = PCAMonitor(n_components=2).fit(X_fit, X_cal=X_cal, scaler=Scaler().fit(np.vstack([X_fit, X_te])))
    sk = PCAMonitor(n_components=2).fit(X_fit, X_cal=X_cal, scaler=StandardScaler().fit(X_fit))
    assert interno.scaler_n_samples_fit_ == interno.n_fit_ == 300
    assert limpio.scaler_n_samples_fit_ == 300
    assert s1.scaler_n_samples_fit_ == 450
    assert sk.scaler_n_samples_fit_ == 300


def test_escalador_que_no_dice_sus_filas_deja_none():
    class SoloTransform:
        def __init__(self, media, desviacion):
            self.media, self.desviacion = media, desviacion

        def transform(self, X):
            return (np.asarray(X, dtype=float) - self.media) / self.desviacion

    X = _datos(300, 4, np.random.default_rng(3))
    m = PCAMonitor(n_components=2).fit(
        X, calibration="in_sample", scaler=SoloTransform(X.mean(axis=0), X.std(axis=0, ddof=1))
    )
    assert m.scaler_source_ == "external" and m.scaler_n_samples_fit_ is None


# ------------------------------------------------------------------- V-02

def test_dos_ficheros_con_el_mismo_simulationrun_avisan():
    """En el clasico simulationRun vale 1 en todos los ficheros: con runs 1D se funden."""
    X = np.random.default_rng(4).normal(size=(2 * 960, 3))
    with pytest.warns(UserWarning, match="1920 filas seguidas con el mismo identificador"):
        lag_by_run(X, np.ones(len(X)), lags=1)


def test_una_simulacion_entera_o_dos_columnas_no_avisan():
    X = np.random.default_rng(5).normal(size=(2 * 960, 3))
    runs_2d = np.column_stack([np.repeat([1, 2], 960), np.ones(2 * 960)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lag_by_run(X[:TEP_MAX_RUN_LENGTH], np.ones(TEP_MAX_RUN_LENGTH), lags=1)
        L, _ = lag_by_run(X, runs_2d, lags=1)
    assert len(L) == 2 * 959


def test_una_clave_2d_repetida_tambien_avisa():
    """Entrenamiento y prueba de la misma clase seguidos: (0, 1) en 500 + 960 filas."""
    X = np.random.default_rng(6).normal(size=(1460, 3))
    runs = np.column_stack([np.zeros(1460), np.ones(1460)])
    with pytest.warns(UserWarning, match="misma clave"):
        lag_by_run(X, runs, lags=1)


def test_el_aviso_senala_la_linea_de_quien_llama():
    rng = np.random.default_rng(7)
    m = PCAMonitor(n_components=2, lags=1).fit(
        rng.normal(size=(300, 3)), np.ones(300), X_cal=rng.normal(size=(100, 3)), runs_cal=np.ones(100)
    )
    with pytest.warns(UserWarning, match="mas que cualquier simulacion del TEP") as registro:
        m.score(rng.normal(size=(1500, 3)), runs=np.ones(1500))
    assert any(aviso.filename == __file__ for aviso in registro)


# ------------------------------------------------------------------- V-09

def test_gridsearch_con_error_score_raise_falla_si_un_pliegue_no_tiene_normales():
    from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    rng = np.random.default_rng(8)
    X, y, g = [], [], []
    for clase in (1, 2):
        for sim in (1, 2, 3):
            X.append(rng.normal(loc=clase, size=(20, 3))); y.append(np.full(20, clase)); g.append(np.full(20, sim))
    X.append(rng.normal(size=(20, 3))); y.append(np.zeros(20, int)); g.append(np.full(20, 1))
    X, y, g = np.vstack(X), np.concatenate(y), np.concatenate(g)
    busqueda = GridSearchCV(
        Pipeline([("esc", Scaler()), ("knn", KNeighborsClassifier())]),
        {"knn__n_neighbors": [1, 3]},
        cv=LeaveOneGroupOut(),
        error_score="raise",
    )
    with pytest.raises(ValueError, match="No hay filas con etiqueta 0"):
        busqueda.fit(X, y, groups=g)


# ------------------------------------------------------------------- V-10

def test_samples_detecta_muestras_no_consecutivas():
    """Tras repartir por muestra, apilar retardos uniria muestras que no son vecinas."""
    X = np.random.default_rng(9).normal(size=(10, 2))
    sample = np.array([1, 2, 3, 5, 6, 7, 8, 9, 10, 11])
    with pytest.raises(ValueError, match="no son consecutivas"):
        lag_by_run(X, np.ones(10), lags=1, samples=sample)


def test_samples_consecutivos_dan_lo_mismo_que_sin_samples():
    X = np.random.default_rng(10).normal(size=(12, 2))
    runs = np.repeat([1, 2], 6)
    sample = np.tile(np.arange(1, 7), 2)
    con, ids_con = lag_by_run(X, runs, lags=2, samples=sample)
    sin, ids_sin = lag_by_run(X, runs, lags=2)
    np.testing.assert_array_equal(con, sin)
    np.testing.assert_array_equal(ids_con, ids_sin)


def test_pca_monitor_comprueba_samples_en_fit_y_en_score():
    rng = np.random.default_rng(11)
    X_fit, X_cal, X_te = rng.normal(size=(200, 3)), rng.normal(size=(100, 3)), rng.normal(size=(50, 3))
    seguidas = lambda n: np.arange(1, n + 1)
    con_salto = lambda n: np.concatenate([np.arange(1, n // 2 + 1), np.arange(n // 2 + 2, n + 2)])
    m = PCAMonitor(n_components=2, lags=1).fit(
        X_fit, np.ones(200), X_cal=X_cal, runs_cal=np.ones(100), samples=seguidas(200), samples_cal=seguidas(100)
    )
    np.testing.assert_array_equal(m.score(X_te, np.ones(50), samples=seguidas(50))[0], m.score(X_te, np.ones(50))[0])
    with pytest.raises(ValueError, match="no son consecutivas"):
        m.score(X_te, np.ones(50), samples=con_salto(50))
    with pytest.raises(ValueError, match="no son consecutivas"):
        PCAMonitor(n_components=2, lags=1).fit(X_fit, np.ones(200), X_cal=X_cal, runs_cal=np.ones(100), samples=con_salto(200))
    with pytest.raises(ValueError, match="no son consecutivas"):
        PCAMonitor(n_components=2, lags=1).fit(X_fit, np.ones(200), X_cal=X_cal, runs_cal=np.ones(100), samples_cal=con_salto(100))


def test_sin_retardos_samples_no_se_usa():
    X = np.random.default_rng(12).normal(size=(100, 3))
    m = PCAMonitor(n_components=2).fit(X, calibration="in_sample", samples=np.zeros(100))
    assert m.n_fit_ == 100


# --------------------------------------------------------------------- I5

def test_kde_da_el_cuantil_de_la_kde_con_un_estadistico_negativo():
    """Antes, con N(-10, 1) el limite caia en el centro de la distribucion."""
    x = np.random.default_rng(13).normal(-10, 1, 2000)
    limite = limits.limit_kde(x, 0.99)
    assert stats.gaussian_kde(x).integrate_box_1d(-np.inf, limite) == pytest.approx(0.99, abs=1e-9)
    assert limite == pytest.approx(_cuantil_kde_integrado(x, 0.99), abs=1e-3)


@pytest.mark.parametrize("muestra", ["chi2_3", "lognormal", "chi2_30"])
def test_kde_coincide_con_la_integracion_numerica(muestra):
    rng = np.random.default_rng(14)
    x = {"chi2_3": rng.chisquare(3, 500), "lognormal": rng.lognormal(0, 1, 500), "chi2_30": rng.chisquare(30, 500)}[muestra]
    for alpha in (0.95, 0.99):
        assert limits.limit_kde(x, alpha) == pytest.approx(_cuantil_kde_integrado(x, alpha), rel=1e-4)


def test_kde_con_valores_iguales_falla():
    with pytest.raises(ValueError, match="iguales"):
        limits.limit_kde(np.full(50, 3.0), 0.99)


# --------------------------------------------------------------------- M6

def test_jackson_exige_autovalores_decrecientes():
    autovalores = np.array([5.0, 3.0, 1.0, 0.5, 0.2, 0.1])
    limits.spe_limit_jackson(autovalores, 2, 0.99)
    with pytest.raises(ValueError, match="orden decreciente"):
        limits.spe_limit_jackson(autovalores[::-1], 2, 0.99)


def test_jackson_acepta_los_autovalores_de_pca_monitor():
    X = _datos(400, 6, np.random.default_rng(15))
    m = PCAMonitor(n_components=2, spe_method="jackson").fit(X, calibration="in_sample")
    assert limits.spe_limit_jackson(m.eigenvalues_, 2, 0.99) == m.spe_limit_
