"""Pruebas de la v0.3: coste computacional (tfmfdd.cost) e indice de robustez (tfmfdd.robustness).

Cada prueba protege una idea:
- el coste devuelve las columnas esperadas, en minusculas, con tiempos
  positivos y finitos, para un modelo de numpy y otro de scikit-learn;
- el tiempo por muestra es el tiempo total entre las muestras (con un reloj
  falso, de forma exacta) y apenas cambia con el numero de muestras;
- el pico de memoria ve una asignacion grande hecha a proposito, en el proceso
  y en la traza, y sabe cuando su cifra es exacta;
- el registro de la maquina no importa torch;
- las decisiones del equipo no tienen valor por defecto;
- el indice reproduce las Tablas 1 a 3 del articulo y un caso con intervalo
  calculado a mano con sus ecuaciones;
- el ruido es reproducible con la semilla, tiene el nivel pedido y, por grupos,
  no depende del orden de las filas.
"""

import inspect
import math
import platform
import sys
import time

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tfmfdd import cost, robustness


class _PCAMinimo:
    """PCA de numpy para las pruebas, sin depender de los cambios de PCAMonitor."""

    def fit(self, X):
        self.media_ = X.mean(axis=0)
        _, _, vt = np.linalg.svd(X - self.media_, full_matrices=False)
        self.P_ = vt[:3].T
        return self

    def score(self, X):
        Z = X - self.media_
        residuo = Z - (Z @ self.P_) @ self.P_.T
        return np.sum(residuo**2, axis=1)


# ---------------------------------------------------------------- coste

@pytest.mark.filterwarnings("ignore:.*pico de memoria:RuntimeWarning")
def test_measure_cost_devuelve_claves_y_valores_finitos():
    rng = np.random.default_rng(0)
    X_fit = rng.normal(size=(400, 10))
    X_inf = rng.normal(size=(300, 10))
    fila = cost.measure_cost(
        fit=lambda: _PCAMinimo().fit(X_fit),
        predict=lambda modelo, xb: modelo.score(xb),
        X_infer=X_inf, method="pca_prueba", batch_size=10,
        n_repeats_fit=3, n_warmup_fit=1, n_repeats_infer=3, n_warmup_infer=2,
    )
    for k in ("fit_time_s_median", "fit_time_s_min", "fit_time_s_max",
              "infer_time_per_sample_s_median", "infer_latency_s_median",
              "infer_latency_s_p95"):
        assert np.isfinite(fila[k]) and fila[k] > 0, k
    # El reloj de CPU de Windows avanza a saltos: puede valer 0 en tramos cortos.
    for k in ("fit_cpu_s_median", "infer_cpu_per_sample_s_median",
              "fit_peak_process_bytes", "infer_peak_process_bytes"):
        assert np.isfinite(fila[k]) and fila[k] >= 0, k
    assert fila["process_peak_lifetime_bytes"] > 0
    assert fila["fit_time_s_min"] <= fila["fit_time_s_median"] <= fila["fit_time_s_max"]
    assert fila["method"] == "pca_prueba"
    assert fila["batch_size"] == 10 and fila["n_infer_samples"] == 300
    assert fila["n_repeats_fit"] == 3 and fila["n_warmup_infer"] == 2
    assert fila["infer_sample_shape"] == "(10,)"
    assert math.isnan(fila["fit_peak_traced_bytes"])          # sin traza pedida
    assert fila["python_version"] == platform.python_version()
    assert all(k == k.lower() for k in fila)                   # formato comun
    assert len(pd.DataFrame([fila])) == 1


@pytest.mark.filterwarnings("ignore:.*pico de memoria:RuntimeWarning")
def test_measure_cost_con_scikit_learn_y_traza():
    vecinos = pytest.importorskip("sklearn.neighbors")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = rng.integers(0, 3, 300)
    fila = cost.measure_cost(
        fit=lambda: vecinos.KNeighborsClassifier(n_neighbors=3).fit(X, y),
        predict=lambda modelo, xb: modelo.predict(xb),
        X_infer=X[:60], method="knn", batch_size=1,
        n_repeats_fit=2, n_warmup_fit=0, n_repeats_infer=2, n_warmup_infer=1,
        traced_memory=True,
    )
    assert fila["infer_time_per_sample_s_median"] > 0
    assert np.isfinite(fila["fit_peak_traced_bytes"]) and fila["fit_peak_traced_bytes"] >= 0
    assert fila["sklearn_version"] is not None


def test_tiempo_por_muestra_con_reloj_falso(monkeypatch):
    """Cada muestra 'tarda' 1 ms exacto: por muestra sale 1 ms y por lote, 10 ms."""
    reloj = {"t": 0.0}
    monkeypatch.setattr(cost, "_reloj", lambda: reloj["t"])
    monkeypatch.setattr(cost, "_cpu", lambda: reloj["t"])

    def predict(xb):
        reloj["t"] += 0.001 * len(xb)

    for n in (100, 400, 105):          # 105: el ultimo lote (5) no entra en la latencia
        r = cost.time_inference(predict, np.zeros((n, 3)), batch_size=10,
                                n_repeats=3, n_warmup=2)
        assert r["infer_time_per_sample_s_median"] == pytest.approx(0.001)
        assert r["infer_cpu_per_sample_s_median"] == pytest.approx(0.001)
        assert r["infer_latency_s_median"] == pytest.approx(0.010)
        assert r["infer_latency_s_p95"] == pytest.approx(0.010)
        assert r["n_infer_samples"] == n


def test_inferencia_por_muestra_escala_con_el_numero_de_muestras():
    """Con el reloj real: 4 veces mas muestras, mas tiempo total y casi el mismo por muestra."""
    rng = np.random.default_rng(2)
    W = rng.normal(size=(52, 52))
    X = rng.normal(size=(16000, 52))

    def predict(xb):
        return np.tanh(xb @ W) @ W.T

    r1 = cost.time_inference(predict, X[:4000], batch_size=200, n_repeats=5, n_warmup=2)
    r4 = cost.time_inference(predict, X, batch_size=200, n_repeats=5, n_warmup=2)
    total1 = r1["infer_time_per_sample_s_median"] * 4000
    total4 = r4["infer_time_per_sample_s_median"] * 16000
    assert total4 > total1
    cociente = r4["infer_time_per_sample_s_median"] / r1["infer_time_per_sample_s_median"]
    assert 0.25 < cociente < 4.0


def test_time_fit_ajusta_un_modelo_nuevo_en_cada_llamada():
    llamadas = []

    def fit():
        llamadas.append(1)
        return object()

    r, modelo = cost.time_fit(fit, n_repeats=4, n_warmup=2)
    assert len(llamadas) == 6
    assert len(r["fit_wall_s"]) == 4 and len(r["fit_cpu_s"]) == 4
    assert modelo is not None
    assert r["n_repeats_fit"] == 4 and r["n_warmup_fit"] == 2


def test_measure_ve_una_asignacion_grande_en_el_proceso():
    tam = 80_000_000                     # bytes
    with cost.measure(process_memory=True, traced_memory=False, poll_interval_s=0.001) as m:
        a = np.ones(tam // 8)            # np.ones escribe todas las paginas
        time.sleep(0.05)                 # para que el muestreo la vea si no hay pico nuevo
        del a
    assert np.isfinite(m["peak_process_bytes"])
    assert m["peak_process_bytes"] >= 0.9 * tam
    assert m["wall_s"] > 0
    assert m["memory_poll_samples"] >= 2


def test_measure_ve_una_asignacion_grande_en_la_traza():
    tam = 40_000_000
    with cost.measure(process_memory=False, traced_memory=True) as m:
        a = np.ones(tam // 8)
        del a
    assert m["peak_traced_bytes"] >= 0.99 * tam
    assert m["retained_traced_bytes"] < 0.01 * tam
    assert m["timing_with_tracemalloc"] is True


def test_pico_del_tramo_dice_si_es_exacto():
    # El pico de vida subio: el maximo del tramo es ese pico nuevo (exacto).
    assert cost._pico_del_tramo(100, 150, 180, None) == (80.0, True)
    # No subio: se usa el muestreo, que es una cota inferior.
    assert cost._pico_del_tramo(100, 150, 150, 130) == (30.0, False)
    assert cost._pico_del_tramo(100, 150, 150, 90) == (0.0, False)
    # No subio y no hubo muestreo: no se sabe.
    valor, exacto = cost._pico_del_tramo(100, 150, 150, None)
    assert math.isnan(valor) and exacto is False


def test_machine_info_no_importa_torch_y_registra_el_entorno():
    torch_antes = "torch" in sys.modules
    info = cost.machine_info()
    assert ("torch" in sys.modules) == torch_antes
    assert info["torch_loaded"] == torch_antes
    assert info["python_version"] == platform.python_version()
    assert info["numpy_version"] == np.__version__
    assert info["cpu_count_logical"] >= 1
    assert isinstance(info["threadpools"], list)
    assert "blas_threads" in info and "process_time_tick_s" in info
    assert all(k == k.lower() for k in info)


def test_measure_cost_rechaza_configuraciones_imposibles():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        cost.time_inference(lambda xb: None, X, batch_size=6, n_repeats=1, n_warmup=0)
    with pytest.raises(ValueError):
        cost.time_fit(lambda: None, n_repeats=0, n_warmup=0)
    with pytest.raises(TypeError):
        cost.time_fit(lambda: None, n_repeats=2.5, n_warmup=0)


def _sin_valor_por_defecto(funcion, nombres):
    parametros = inspect.signature(funcion).parameters
    for nombre in nombres:
        assert parametros[nombre].default is inspect.Parameter.empty, (
            f"{funcion.__name__}({nombre}=...) esconderia una decision del equipo"
        )


def test_las_decisiones_del_equipo_no_tienen_valor_por_defecto():
    _sin_valor_por_defecto(cost.measure_cost, [
        "method", "batch_size", "n_repeats_fit", "n_warmup_fit",
        "n_repeats_infer", "n_warmup_infer"])
    _sin_valor_por_defecto(cost.time_fit, ["n_repeats", "n_warmup"])
    _sin_valor_por_defecto(cost.time_inference, ["batch_size", "n_repeats", "n_warmup"])
    _sin_valor_por_defecto(robustness.add_noise, [
        "eta", "sigma_s", "noise_type", "seed", "variables"])
    _sin_valor_por_defecto(robustness.noisy_versions, [
        "levels", "sigma_s", "noise_type", "seed", "n_repeats", "variables",
        "common_random_numbers"])
    _sin_valor_por_defecto(robustness.error_rate, ["metric"])
    _sin_valor_por_defecto(robustness.robustness_curve, ["levels", "errors", "confidence"])
    _sin_valor_por_defecto(robustness.robustness_index, ["levels", "errors", "confidence"])


# ---------------------------------------------------------------- indice de robustez

#: Bernal-de Lazaro et al. (2018), Tablas 1 a 3, p. 643 (p. 8 del PDF): dos
#: niveles de ruido (eta_0 = 1, eta_1 = 2), E = 1 - acierto y Delta_I omitido.
#: (ejemplo, clasificador, A_0%, A_x%, I_yy publicado, I_yy + D_p publicado)
TABLAS_1_A_3 = [
    ("I", "F1", 1.00, 1.00, 0.00, 0.00),
    ("I", "F2", 0.80, 0.80, 0.50, 0.50),
    ("II", "F1", 0.70, 0.70, 0.75, 0.75),
    ("II", "F2", 0.60, 0.60, 1.00, 1.00),
    ("III", "F1", 0.75, 0.25, 1.62, 2.12),
    ("III", "F2", 0.90, 0.10, 1.85, 2.65),
    ("IV", "F1", 0.50, 0.50, 1.25, 1.25),
    ("IV", "F2", 0.75, 0.25, 1.62, 2.12),
    ("V", "F1", 0.60, 0.50, 1.20, 1.30),
    ("V", "F2", 0.70, 0.50, 1.15, 1.35),
    ("VI", "F1", 0.50, 0.50, 1.25, 1.25),
    ("VI", "F2", 0.70, 0.50, 1.15, 1.35),
]


@pytest.mark.parametrize("ejemplo, clasificador, a0, ax, iyy_pub, j_pub", TABLAS_1_A_3)
def test_indice_reproduce_las_tablas_1_a_3_del_articulo(ejemplo, clasificador, a0, ax, iyy_pub, j_pub):
    r = robustness.robustness_index([1.0, 2.0], [1 - a0, 1 - ax], confidence=0.99)
    # Las tablas redondean a dos decimales.
    assert abs(r["i_yy"] - iyy_pub) <= 0.005 + 1e-9
    assert abs(r["i_yy"] + r["d_p"] - j_pub) <= 0.005 + 1e-9
    # Con un solo error por nivel no hay intervalo: Delta_I y J quedan en NaN.
    assert math.isnan(r["delta_i"]) and math.isnan(r["j_ril"])


def test_dp_cuenta_la_pendiente_del_primer_tramo():
    """Ejemplo III, F1: si D_p saltara el tramo eta_0 -> eta_1, valdria 0 y no 0,5."""
    r = robustness.robustness_index([1.0, 2.0], [0.25, 0.75], confidence=0.99)
    assert r["i_yy"] == pytest.approx(0.5 * (1.0**2 * 0.25 + 2.0**2 * 0.75))   # ec. 4
    assert r["d_p"] == pytest.approx((0.75 - 0.25) / (2.0 - 1.0))              # ec. 7


def test_indice_caso_a_mano_con_intervalo():
    """Tres niveles con paso 0,5 y tres errores por nivel, con las ecs. 4 a 7 escritas tal cual."""
    niveles = [1.0, 1.5, 2.0]
    errores = [[0.10, 0.12, 0.14], [0.20, 0.24, 0.28], [0.30, 0.36, 0.42]]
    medias = [0.12, 0.24, 0.36]
    desv = [0.02, 0.04, 0.06]                     # desviaciones muestrales (ddof = 1)
    t = stats.t.ppf(1 - 0.05 / 2, 3 - 1)          # 95 %, n_e - 1 = 2 grados de libertad
    assert t == pytest.approx(4.3027, abs=1e-4)   # valor de tabla
    margen = [t * s / math.sqrt(3) for s in desv]
    d_eta = 0.5

    i_yy = d_eta / 2 * (1.0**2 * medias[0] + 2 * 1.5**2 * medias[1] + 2.0**2 * medias[2])
    d_p = (medias[1] - medias[0]) / 0.5 + (medias[2] - medias[1]) / 0.5
    i0_max = d_eta / 2 * ((medias[0] + margen[0]) + 2 * (medias[1] + margen[1]) + (medias[2] + margen[2]))
    i0_min = d_eta / 2 * ((medias[0] - margen[0]) + 2 * (medias[1] - margen[1]) + (medias[2] - margen[2]))

    r = robustness.robustness_index(niveles, errores, confidence=0.95)
    assert r["i_yy"] == pytest.approx(i_yy)
    assert r["d_p"] == pytest.approx(d_p)
    assert r["i0_max"] == pytest.approx(i0_max)
    assert r["i0_min"] == pytest.approx(i0_min)
    assert r["delta_i"] == pytest.approx(i0_max - i0_min)
    assert r["j_ril"] == pytest.approx(i_yy + d_p + i0_max - i0_min)
    assert r["grid_uniform"] is True and r["delta_eta"] == pytest.approx(0.5)
    assert r["n_e_min"] == r["n_e_max"] == 3

    curva = robustness.robustness_curve(niveles, errores, confidence=0.95)
    assert list(curva.columns) == ["level", "n_e", "mean_error", "std_error",
                                   "t_quantile", "e_min", "e_max"]
    assert curva["std_error"].to_numpy() == pytest.approx(desv)


def test_dp_con_paso_uniforme_es_el_salto_total_entre_el_paso():
    """La propiedad del informe 06 (seccion 6.2): con paso uniforme D_p no depende de la forma."""
    niveles = np.linspace(1.0, 3.0, 21)
    rng = np.random.default_rng(3)
    errores = np.clip(0.1 + 0.2 * (niveles - 1) + rng.normal(0, 0.02, niveles.size), 0, 1)
    r = robustness.robustness_index(niveles, errores, confidence=0.95)
    assert r["d_p"] == pytest.approx((errores[-1] - errores[0]) / 0.1)


def test_rejilla_no_uniforme_se_marca():
    r = robustness.robustness_index([1.0, 1.1, 1.5, 3.0], [0.1, 0.2, 0.3, 0.4], confidence=0.95)
    assert r["grid_uniform"] is False and math.isnan(r["delta_eta"])
    # Trapecio general: suma de (eta_i - eta_(i-1)) * (f_(i-1) + f_i) / 2 con f = eta^2 * E.
    f = np.array([1.0, 1.1, 1.5, 3.0]) ** 2 * np.array([0.1, 0.2, 0.3, 0.4])
    esperado = 0.1 * (f[0] + f[1]) / 2 + 0.4 * (f[1] + f[2]) / 2 + 1.5 * (f[2] + f[3]) / 2
    assert r["i_yy"] == pytest.approx(esperado)


def test_rejilla_y_errores_invalidos_fallan_con_mensaje():
    with pytest.raises(ValueError):
        robustness.robustness_index([1.5, 2.0], [0.1, 0.2], confidence=0.95)    # no empieza en 1
    with pytest.raises(ValueError):
        robustness.robustness_index([1.0, 1.0, 2.0], [0.1, 0.1, 0.2], confidence=0.95)
    with pytest.raises(ValueError):
        robustness.robustness_index([1.0], [0.1], confidence=0.95)
    with pytest.raises(ValueError):
        robustness.robustness_index([1.0, 2.0], [0.1], confidence=0.95)        # un nivel sin errores
    with pytest.raises(ValueError):
        robustness.robustness_index([1.0, 2.0], [10.0, 20.0], confidence=0.95)  # porcentajes
    with pytest.raises(ValueError):
        robustness.robustness_index([1.0, 2.0], [0.1, 0.2], confidence=95)


# ---------------------------------------------------------------- metrica de error

def test_error_eq3_en_deteccion_es_la_media_de_far_y_no_deteccion():
    y_true = np.r_[np.zeros(100, bool), np.ones(100, bool)]
    y_pred = np.r_[np.zeros(90, bool), np.ones(10, bool), np.ones(80, bool), np.zeros(20, bool)]
    assert robustness.error_rate(y_true, y_pred, metric="far") == pytest.approx(0.10)
    assert robustness.error_rate(y_true, y_pred, metric="missed_detection") == pytest.approx(0.20)
    assert robustness.error_rate(y_true, y_pred, metric="eq3") == pytest.approx((0.10 + 0.20) / 2)


def test_error_eq3_y_macro_f1_coinciden_con_scikit_learn():
    metricas = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y_true = rng.integers(0, 4, 500)
    y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 4, 500))
    assert robustness.error_rate(y_true, y_pred, metric="eq3") == pytest.approx(
        1 - metricas.balanced_accuracy_score(y_true, y_pred))
    assert robustness.error_rate(y_true, y_pred, metric="one_minus_macro_f1") == pytest.approx(
        1 - metricas.f1_score(y_true, y_pred, average="macro"))
    # Una clase que nunca se predice tiene F1 = 0.
    y_pred2 = np.where(y_pred == 3, 0, y_pred)
    assert robustness.error_rate(y_true, y_pred2, metric="one_minus_macro_f1",
                                 labels=[0, 1, 2, 3]) == pytest.approx(
        1 - metricas.f1_score(y_true, y_pred2, average="macro", labels=[0, 1, 2, 3],
                              zero_division=0))


def test_error_rechaza_nan_y_metricas_desconocidas():
    with pytest.raises(ValueError):
        robustness.error_rate(np.array([0.0, 1.0]), np.array([np.nan, 1.0]), metric="eq3")
    with pytest.raises(ValueError):
        robustness.error_rate([0, 1], [0, 1], metric="accuracy")
    with pytest.raises(ValueError):
        robustness.error_rate([0, 0], [0, 1], metric="eq3", labels=[0, 1])   # clase 1 sin muestras
    with pytest.raises(ValueError):
        robustness.error_rate([True, True], [True, False], metric="far")       # sin muestras sanas


# ---------------------------------------------------------------- ruido

def test_ruido_reproducible_con_la_semilla():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 4))
    s = np.ones(4)
    a = robustness.add_noise(X, 3.0, s, noise_type="gaussian", seed=42, variables="all")
    b = robustness.add_noise(X, 3.0, s, noise_type="gaussian", seed=42, variables="all")
    c = robustness.add_noise(X, 3.0, s, noise_type="gaussian", seed=43, variables="all")
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)
    v1 = [x for *_, x in robustness.noisy_versions(
        X, [1.0, 2.0], s, noise_type="uniform", seed=(7, 1), n_repeats=2,
        variables="all", common_random_numbers=False)]
    v2 = [x for *_, x in robustness.noisy_versions(
        X, [1.0, 2.0], s, noise_type="uniform", seed=(7, 1), n_repeats=2,
        variables="all", common_random_numbers=False)]
    assert all(np.array_equal(p, q) for p, q in zip(v1, v2))


@pytest.mark.parametrize("tipo", ["gaussian", "uniform"])
def test_ruido_tiene_el_nivel_pedido(tipo):
    """Lema 1: desviacion del ruido raiz(eta - 1) * sigma_s; ecs. 8 y 9: severidad eta."""
    rng = np.random.default_rng(6)
    medias = np.array([10.0, 50.0, 5.0, 100.0])
    desv = np.array([1.0, 2.0, 0.5, 3.0])
    X = medias + desv * rng.standard_normal((200_000, 4))
    sigma_s = X.std(axis=0, ddof=1)
    eta = 4.0
    Xr = robustness.add_noise(X, eta, sigma_s, noise_type=tipo, seed=7, variables="all")
    ruido = Xr - X
    assert ruido.std(axis=0, ddof=1) / sigma_s == pytest.approx(np.full(4, math.sqrt(eta - 1)), rel=0.01)
    assert np.abs(ruido.mean(axis=0) / sigma_s).max() < 0.02
    assert robustness.severity_level(Xr, X) == pytest.approx(eta, rel=0.02)
    if tipo == "uniform":
        assert (np.abs(ruido) <= math.sqrt(3) * math.sqrt(eta - 1) * sigma_s + 1e-12).all()


def test_ruido_eta_1_no_cambia_nada_y_solo_toca_las_variables_pedidas():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(50, 4))
    s = np.ones(4)
    sin = robustness.add_noise(X, 1.0, s, noise_type="gaussian", seed=1, variables="all")
    assert np.array_equal(sin, X) and sin is not X
    parcial = robustness.add_noise(X, 5.0, s, noise_type="gaussian", seed=1, variables=[0, 2])
    assert np.array_equal(parcial[:, [1, 3]], X[:, [1, 3]])
    assert not np.allclose(parcial[:, [0, 2]], X[:, [0, 2]])


def test_ruido_por_grupos_no_depende_del_orden_de_las_filas():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(60, 3))
    grupos = np.repeat([5, 9, 2], 20)
    kw = dict(noise_type="gaussian", seed=11, variables="all")
    a = robustness.add_noise(X, 2.0, np.ones(3), groups=grupos, **kw)
    orden = np.r_[40:60, 0:20, 20:40]                      # otro orden de las simulaciones
    b = robustness.add_noise(X[orden], 2.0, np.ones(3), groups=grupos[orden], **kw)
    assert np.array_equal(b, a[orden])
    solo9 = grupos == 9                                    # una simulacion suelta
    c = robustness.add_noise(X[solo9], 2.0, np.ones(3), groups=grupos[solo9], **kw)
    assert np.array_equal(c, a[solo9])
    # Claves compuestas (fallo, simulacion): (1, 5) y (2, 5) reciben ruidos distintos.
    claves = np.c_[np.repeat([1, 2], 30), np.full(60, 5)]
    d = robustness.add_noise(X, 2.0, np.ones(3), groups=claves, **kw)
    assert not np.allclose(d[:30] - X[:30], d[30:] - X[30:])


def test_numeros_aleatorios_comunes_reescalan_la_misma_realizacion():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(100, 3))
    kw = dict(noise_type="gaussian", seed=3, n_repeats=2, variables="all")
    comunes = {(i, r): Xr - X for i, _, r, Xr in robustness.noisy_versions(
        X, [1.0, 2.0, 5.0], np.ones(3), common_random_numbers=True, **kw)}
    assert np.allclose(comunes[(0, 0)], 0.0)
    assert np.allclose(comunes[(2, 0)], comunes[(1, 0)] * math.sqrt((5 - 1) / (2 - 1)))
    assert not np.allclose(comunes[(1, 0)], comunes[(1, 1)])
    indep = {(i, r): Xr - X for i, _, r, Xr in robustness.noisy_versions(
        X, [1.0, 2.0, 5.0], np.ones(3), common_random_numbers=False, **kw)}
    assert not np.allclose(indep[(2, 0)], indep[(1, 0)] * 2.0)


def test_ruido_rechaza_entradas_peligrosas():
    df = pd.DataFrame({"faultNumber": [0, 0], "xmeas_1": [1.0, 2.0]})
    with pytest.raises(TypeError):
        robustness.add_noise(df, 2.0, np.ones(2), noise_type="gaussian", seed=1, variables="all")
    X = np.zeros((4, 2))
    with pytest.raises(ValueError):
        robustness.add_noise(X, 0.5, np.ones(2), noise_type="gaussian", seed=1, variables="all")
    with pytest.raises(TypeError):
        robustness.add_noise(X, 2.0, np.ones(2), noise_type="gaussian", seed=None, variables="all")
    with pytest.raises(ValueError):
        robustness.add_noise(X, 2.0, np.ones(2), noise_type="rosa", seed=1, variables="all")
    with pytest.raises(ValueError):
        robustness.add_noise(X, 2.0, np.ones(3), noise_type="gaussian", seed=1, variables="all")
    with pytest.raises(ValueError):                       # grupos negativos, aunque eta = 1
        robustness.add_noise(X, 1.0, np.ones(2), noise_type="gaussian", seed=1, variables="all",
                             groups=np.array([0, 0, -1, 1]))
    # La rejilla se comprueba al llamar, no al empezar a iterar.
    with pytest.raises(ValueError):
        robustness.noisy_versions(X, [2.0, 3.0], np.ones(2), noise_type="gaussian", seed=1,
                                  n_repeats=1, variables="all", common_random_numbers=True)
