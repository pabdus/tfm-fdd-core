"""Pruebas de las metricas de la v0.3 (12-10-2026).

Una idea por prueba y datos sinteticos. Cubren:
1. NaN = muestra no evaluada (contrato entre el monitor y las metricas).
2. Diagnostico: F1 por clase sobre toda la prueba y macro-F1 (D2, regla 5).
3. aggregate y summary sin rachas al azar, con su n, la mediana del retardo
   e intervalos dentro del rango (M5, I1, MET-4).
4. Falsas alarmas por hora, en sus dos unidades (OE3).
5. PR-AUC (OE3).
6. Deteccion por encima de la FAR (criterio de no superioridad).
7. Convencion del retardo.
8. Ventana declarada de la FAR en las simulaciones normales (far_window).
"""

import numpy as np
import pandas as pd
import pytest

from tfmfdd import metrics

ONSET = 160


# ------------------------------------------------- 1. NaN = no evaluada

def test_nan_no_es_alarma():
    a = metrics.alarms(np.array([np.nan, 5.0, 0.0]), 1.0)
    assert a.tolist() == [False, True, False]


def test_far_y_fdr_excluyen_nan_del_denominador():
    """Antes, 2 NaN de 4 muestras contaban como sanas y la FAR salia 0,25."""
    stat = np.array([np.nan, np.nan, 0.0, 5.0])
    assert metrics.far(stat, 1.0) == pytest.approx(0.5)
    assert metrics.fdr(stat, 1.0) == pytest.approx(0.5)


def test_tramo_sin_muestras_evaluadas_da_nan():
    stat = np.concatenate([np.zeros(ONSET), np.full(10, np.nan)])
    r = metrics.evaluate_run(stat, 1.0, onset=ONSET)
    assert np.isnan(r["fdr"]) and r["n_fdr"] == 0 and not r["detected"]
    assert np.isnan(metrics.far(np.full(10, np.nan), 1.0))


def test_limite_no_finito_es_un_error():
    for limite in (np.nan, np.inf):
        with pytest.raises(ValueError, match="no es finito"):
            metrics.evaluate_run(np.zeros(20), limite, onset=10)


def test_racha_no_atraviesa_un_nan():
    stat = np.zeros(ONSET + 20)
    stat[ONSET:ONSET + 2] = 9.0
    stat[ONSET + 2] = np.nan              # muestra no evaluada en mitad de la racha
    stat[ONSET + 3:ONSET + 6] = 9.0
    r = metrics.detection_delay(stat, 1.0, onset=ONSET, k=3)
    assert r["delay_samples"] == 3        # la racha que confirma empieza tras el NaN


def test_retardo_en_la_escala_original_con_nan_iniciales():
    """Como en DPCA: las lags primeras muestras sin evaluar no desplazan el retardo."""
    for lags in (0, 1, 2):
        stat = np.zeros(ONSET + 100)
        stat[:lags] = np.nan
        stat[ONSET + 10:] = 9.0
        r = metrics.evaluate_run(stat, 1.0, onset=ONSET)
        assert r["delay_samples"] == 10
        assert (r["n_far"], r["n_fdr"]) == (ONSET - lags, 100)


def test_evaluate_run_cuenta_muestras_por_tramo():
    stat = np.zeros(ONSET + 100)
    stat[ONSET + 10] = np.nan             # cae en la primera mitad del tramo con fallo
    r = metrics.evaluate_run(stat, 1.0, onset=ONSET)
    assert (r["n_far"], r["n_fdr"], r["n_fdr_h1"], r["n_fdr_h2"]) == (ONSET, 99, 49, 50)


def test_evaluate_run_mismas_claves_con_y_sin_inicio():
    stat = np.zeros(ONSET + 100)
    assert list(metrics.evaluate_run(stat, 1.0, None)) == list(metrics.evaluate_run(stat, 1.0, ONSET))


def test_onset_fuera_de_rango_es_un_error():
    with pytest.raises(ValueError, match="onset"):
        metrics.evaluate_run(np.zeros(100), 1.0, onset=-5)


# ------------------------------------------------- 2. diagnostico

# Matriz de confusion conocida (filas: clase real; columnas: predicha) con la
# clase normal (0), un fallo facil (1) y uno dificil (3):
#             pred 0  pred 1  pred 3
#   real 0       8       1       1
#   real 1       2       6       2
#   real 3       4       0       6
CLASES = [0, 1, 3]
MATRIZ = np.array([[8, 1, 1], [2, 6, 2], [4, 0, 6]])


def _desde_matriz(matriz, clases):
    y_true, y_pred = [], []
    for i, real in enumerate(clases):
        for j, pred in enumerate(clases):
            y_true += [real] * int(matriz[i, j])
            y_pred += [pred] * int(matriz[i, j])
    return np.array(y_true), np.array(y_pred)


def test_diagnostico_con_matriz_conocida():
    """A mano: tp en la diagonal, fp = columna - tp, fn = fila - tp, F1 = 2tp/(2tp+fp+fn)."""
    y_true, y_pred = _desde_matriz(MATRIZ, CLASES)
    r = metrics.diagnosis_metrics(y_true, y_pred, include_normal=True)
    t = r["per_class"].set_index("faultNumber")
    assert t["tp"].tolist() == [8, 6, 6]
    assert t["fp"].tolist() == [14 - 8, 7 - 6, 9 - 6]
    assert t["fn"].tolist() == [10 - 8, 10 - 6, 10 - 6]
    assert t["precision"].tolist() == pytest.approx([8 / 14, 6 / 7, 6 / 9])
    assert t["recall"].tolist() == pytest.approx([0.8, 0.6, 0.6])
    assert t["f1"].tolist() == pytest.approx([16 / 24, 12 / 17, 12 / 19])
    assert r["macro_f1"] == pytest.approx((16 / 24 + 12 / 17 + 12 / 19) / 3)
    assert r["accuracy"] == pytest.approx(20 / 30)


def test_diagnostico_coincide_con_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_azar = rng.integers(0, 6, 500)
    casos = [_desde_matriz(MATRIZ, CLASES),
             (y_azar, np.where(rng.random(500) < 0.6, y_azar, rng.integers(0, 6, 500)))]
    for y_true, y_pred in casos:
        clases = sorted(set(y_true) | set(y_pred))
        r = metrics.diagnosis_metrics(y_true, y_pred, include_normal=False)
        t = r["per_class"].set_index("faultNumber").loc[clases]
        p, rec, f1, soporte = skm.precision_recall_fscore_support(y_true, y_pred, labels=clases)
        np.testing.assert_allclose(t["precision"], p)
        np.testing.assert_allclose(t["recall"], rec)
        np.testing.assert_allclose(t["f1"], f1)
        np.testing.assert_array_equal(t["support"], soporte)
        sin_normal = [c for c in clases if c != 0]
        assert r["macro_f1"] == pytest.approx(
            skm.f1_score(y_true, y_pred, labels=sin_normal, average="macro"))
        assert r["accuracy"] == pytest.approx(skm.accuracy_score(y_true, y_pred))


def test_diagnostico_sin_la_clase_normal_en_el_promedio():
    y_true, y_pred = _desde_matriz(MATRIZ, CLASES)
    r = metrics.diagnosis_metrics(y_true, y_pred, include_normal=False)
    assert r["macro_f1"] == pytest.approx((12 / 17 + 12 / 19) / 2)
    assert r["macro_f1_n"] == 2 and r["include_normal"] is False
    normal = r["per_class"].set_index("faultNumber").loc[0]
    assert normal["f1"] == pytest.approx(16 / 24)      # se sigue calculando
    assert not normal["in_macro"]


def test_diagnostico_con_los_dificiles_aparte():
    y_true, y_pred = _desde_matriz(MATRIZ, CLASES)
    r = metrics.diagnosis_metrics(y_true, y_pred, include_normal=True)
    assert r["macro_f1_hard"] == pytest.approx(12 / 19)
    assert r["macro_f1_rest"] == pytest.approx((16 / 24 + 12 / 17) / 2)
    assert (r["macro_f1_hard_n"], r["macro_f1_rest_n"]) == (1, 2)


def test_include_normal_es_obligatorio():
    y_true, y_pred = _desde_matriz(MATRIZ, CLASES)
    with pytest.raises(TypeError):
        metrics.diagnosis_metrics(y_true, y_pred)


def test_f1_antiguo_restringido_a_la_clase_real_infla():
    """Solo con las muestras de la clase real no hay falsos positivos: sale 2r/(1+r)."""
    y_true, y_pred = _desde_matriz(MATRIZ, CLASES)
    m = y_true == 3
    viejo = metrics.diagnosis_metrics(y_true[m], y_pred[m], include_normal=True, labels=[3])
    real = metrics.diagnosis_metrics(y_true, y_pred, include_normal=True)
    r = 0.6
    f1_viejo = viejo["per_class"]["f1"].iloc[0]
    assert f1_viejo == pytest.approx(2 * r / (1 + r))
    assert f1_viejo > real["per_class"].set_index("faultNumber").loc[3, "f1"]


def test_clase_de_labels_ausente_es_un_error():
    y_true, y_pred = _desde_matriz(MATRIZ, CLASES)
    with pytest.raises(ValueError, match="no aparece"):
        metrics.diagnosis_metrics(y_true, y_pred, include_normal=True, labels=[0, 1, 3, 21])


def test_recall_por_fallo_y_simulacion():
    # Pool: clase 0 (simulaciones 1 y 2) y clase 1 (simulaciones 1 y 2), 10 muestras cada una.
    y_true = np.repeat([0, 0, 1, 1], 10)
    runs = np.tile(np.repeat([1, 2], 10), 2)
    y_pred = y_true.copy()
    y_pred[0:2] = 1                       # normal, simulacion 1: recall 0,8
    y_pred[30:35] = 0                     # fallo 1, simulacion 2: recall 0,5
    t = metrics.diagnosis_recall_by_run(y_true, y_pred, runs)
    assert t[["faultNumber", "simulationRun"]].values.tolist() == [[0, 1], [0, 2], [1, 1], [1, 2]]
    assert t["recall"].tolist() == pytest.approx([0.8, 1.0, 1.0, 0.5])
    assert t["n_samples"].tolist() == [10] * 4 and t["n_other_label"].tolist() == [0] * 4


def test_recall_por_simulacion_aparta_otras_etiquetas():
    """Muestras previas al fallo conservadas con etiqueta normal: no entran en el recall del fallo."""
    y_true = np.concatenate([np.zeros(4, int), np.ones(6, int)])
    y_pred = np.concatenate([np.zeros(4, int), np.ones(3, int), np.zeros(3, int)])
    t = metrics.diagnosis_recall_by_run(y_true, y_pred, runs=np.ones(10, int),
                                        faults=np.ones(10, int))
    assert len(t) == 1 and t.loc[0, "recall"] == pytest.approx(0.5)
    assert (t.loc[0, "n_samples"], t.loc[0, "n_other_label"]) == (6, 4)


# ------------------------------------------------- 3. aggregate y summary

def _simulacion(retardo=None, racha_al_azar_en=None):
    """Estadistico de una simulacion de prueba: 160 sanas y 800 con fallo."""
    stat = np.zeros(ONSET + 800)
    if retardo is not None:
        stat[ONSET + retardo:] = 9.0      # deteccion real: FDR alta
    if racha_al_azar_en is not None:
        i = ONSET + racha_al_azar_en
        stat[i:i + 3] = 9.0               # una sola racha de 3: FDR 3/800
    return stat


def _filas(fallo, casos):
    return [{"faultNumber": fallo, "simulationRun": run,
             **metrics.evaluate_run(_simulacion(*caso), 1.0, ONSET)}
            for run, caso in enumerate(casos, start=1)]


def test_aggregate_retardo_excluye_rachas_al_azar():
    """Con la racha al azar, la media antigua era (10 + 20 + 30 + 500) / 4."""
    df = pd.DataFrame(_filas(1, [(10, None), (20, None), (30, None), (None, 500)]))
    agg = metrics.aggregate(df).iloc[0]
    assert agg["delay_samples_mean"] == pytest.approx(20.0)
    assert agg["delay_samples_n"] == 3
    assert agg["delay_samples_median"] == pytest.approx(20.0)
    assert agg["valid_detection_ratio"] == pytest.approx(0.75)
    assert agg["detection_ratio"] == pytest.approx(1.0)


def test_aggregate_mediana_censurada():
    """Las simulaciones sin deteccion valida cuentan como retardo mayor que cualquiera."""
    mitad_sin = pd.DataFrame(_filas(1, [(10, None), (12, None), (None, 500), (None, None)]))
    agg = metrics.aggregate(mitad_sin).iloc[0]
    assert agg["delay_samples_median"] == pytest.approx(11.0)       # solo las validas
    assert np.isinf(agg["delay_samples_median_censored"])           # no se alcanza
    mayoria = pd.DataFrame(_filas(1, [(10, None), (12, None), (14, None), (None, None), (None, None)]))
    assert metrics.aggregate(mayoria).iloc[0]["delay_samples_median_censored"] == pytest.approx(14.0)


def test_aggregate_intervalos_no_se_salen_del_rango():
    """Con la t de Student salian [-0,0044; 0,0094] para la FAR y [0,962; 1,018] para la FDR."""
    df = pd.DataFrame({"faultNumber": 1, "simulationRun": range(1, 6),
                       "far": [0, 0, 0, 0, 0.0125], "fdr": [1, 1, 1, 1, 0.95],
                       "detected": True, "detected_at_chance": False})
    agg = metrics.aggregate(df).iloc[0]
    assert agg["far_ci_low"] == 0.0 and agg["fdr_ci_high"] == 1.0
    assert agg["far_ci_low"] <= agg["far_mean"] <= agg["far_ci_high"]
    assert agg["fdr_ci_low"] <= agg["fdr_mean"] <= agg["fdr_ci_high"]


def test_aggregate_da_el_n_de_cada_metrica():
    df = pd.DataFrame(_filas(1, [(10, None), (None, None), (None, None)]))
    agg = metrics.aggregate(df).iloc[0]
    assert (agg["n_runs"], agg["far_n"], agg["fdr_n"], agg["delay_samples_n"]) == (3, 3, 3, 1)


def test_aggregate_clave_con_nan_es_un_error():
    df = pd.DataFrame(_filas(1, [(10, None), (20, None)]))
    df.loc[1, "faultNumber"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        metrics.aggregate(df)


def _agg_sintetica():
    """Tabla agregada a mano: fila 0 normal, dos fallos faciles y el 3 casi invisible."""
    return pd.DataFrame({
        "faultNumber": [0, 1, 2, 3],
        "n_runs": [10, 10, 10, 10],
        "detection_ratio": [0.0, 1.0, 1.0, 1.0],     # el 3 tiene racha en todas, al azar
        "far_mean": [0.01, 0.5, 0.5, 0.5],
        "far_n": [10, 10, 10, 10],
        "fdr_mean": [np.nan, 0.9, 0.8, 0.05],
        "delay_samples_mean": [np.nan, 10.0, 20.0, np.nan],
        "delay_samples_median": [np.nan, 9.0, 19.0, np.nan],
    })


def test_summary_separa_far_normal_y_prefallo():
    """La FAR antigua mezclaba la fila normal con las prefallo: (0,01 + 3 * 0,5) / 4."""
    s = metrics.summary(_agg_sintetica())
    assert (s["far_normal"], s["far_normal_n"]) == (pytest.approx(0.01), 10)
    assert (s["far_prefallo_medio"], s["far_prefallo_medio_n"]) == (pytest.approx(0.5), 3)


def test_summary_fallos_bajo_umbral_y_sin_racha():
    """El caso de A1: racha (al azar) en todas las simulaciones del 3 y FDR de 0,05."""
    s = metrics.summary(_agg_sintetica())
    assert s["fallos_sin_racha"] == []
    assert s["fallos_bajo_umbral"] == [3]


def test_summary_lleva_el_n_de_cada_media():
    s = metrics.summary(_agg_sintetica())
    assert (s["fdr_medio_todos"], s["fdr_medio_todos_n"]) == (pytest.approx(1.75 / 3), 3)
    assert (s["fdr_medio_sin_dificiles"], s["fdr_medio_sin_dificiles_n"]) == (pytest.approx(0.85), 2)
    assert (s["fdr_medio_dificiles"], s["fdr_medio_dificiles_n"]) == (pytest.approx(0.05), 1)
    assert (s["retardo_medio_muestras"], s["retardo_medio_muestras_n"]) == (pytest.approx(15.0), 2)
    assert (s["retardo_mediano_muestras"], s["retardo_mediano_muestras_n"]) == (pytest.approx(14.0), 2)


def test_summary_retardo_sin_rachas_al_azar():
    normal = [{"faultNumber": 0, "simulationRun": 1, **metrics.evaluate_run(np.zeros(960), 1.0, None)}]
    df = pd.DataFrame(normal + _filas(1, [(10, None), (20, None), (30, None), (None, 500)]))
    s = metrics.summary(metrics.aggregate(df))
    assert s["retardo_medio_muestras"] == pytest.approx(20.0)


def test_summary_clave_retirada_dice_cual_usar():
    s = metrics.summary(_agg_sintetica())
    with pytest.raises(KeyError, match="far_normal"):
        s["far_medio"]
    with pytest.raises(KeyError, match="fallos_sin_racha"):
        s["fallos_nunca_detectados"]


def test_summary_exige_una_fila_por_fallo():
    agg = pd.concat([_agg_sintetica(), _agg_sintetica()], ignore_index=True)
    with pytest.raises(ValueError, match="una fila por fallo"):
        metrics.summary(agg)


# ------------------------------------------------- 4. falsas alarmas por hora

def test_muestras_en_alarma_por_hora():
    stat = np.zeros(40)                   # 40 muestras de 3 minutos = 2 horas
    stat[[5, 20]] = 9.0
    assert metrics.false_alarm_samples_per_hour(stat, 1.0) == pytest.approx(1.0)
    assert metrics.false_alarm_samples_per_hour(stat, 1.0) == pytest.approx(metrics.far(stat, 1.0) * 20)


def test_episodios_por_hora_cuentan_rachas_de_k():
    stat = np.zeros(40)
    stat[0:3] = 9.0                       # un episodio
    stat[10:12] = 9.0                     # racha de 2: no confirma
    stat[20:26] = 9.0                     # racha de 6: un solo episodio
    assert metrics.false_alarm_episodes_per_hour(stat, 1.0, k=3) == pytest.approx(2 / 2)


def test_episodios_no_atraviesan_nan():
    stat = np.zeros(40)
    stat[0:2] = 9.0
    stat[2] = np.nan
    stat[3:5] = 9.0
    assert metrics.false_alarm_episodes_per_hour(stat, 1.0, k=3) == 0.0


def test_por_hora_usa_sample_minutes():
    stat = np.zeros(60)                   # 60 muestras de 1 minuto = 1 hora
    stat[0] = 9.0
    assert metrics.false_alarm_samples_per_hour(stat, 1.0, sample_minutes=1.0) == pytest.approx(1.0)


def test_evaluate_run_da_las_dos_unidades():
    stat = np.zeros(ONSET + 100)
    stat[0:3] = 9.0                       # un episodio de 3 muestras en 160 * 3 min = 8 h
    r = metrics.evaluate_run(stat, 1.0, ONSET)
    assert r["false_alarm_samples_per_hour"] == pytest.approx(3 / 8)
    assert r["false_alarm_episodes_per_hour"] == pytest.approx(1 / 8)


# ------------------------------------------------- 5. PR-AUC

def test_pr_auc_valor_conocido():
    """Orden: positivo, negativo, positivo, negativo -> 1/2 * 1 + 1/2 * 2/3."""
    assert metrics.pr_auc([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0]) == pytest.approx(0.5 + 1 / 3)


def test_pr_auc_empates_entran_en_el_mismo_umbral():
    assert metrics.pr_auc([0.5, 0.5, 0.5], [1, 1, 0]) == pytest.approx(2 / 3)


def test_pr_auc_coincide_con_sklearn():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.normal(size=300) + y, 1)     # redondeo: muchos empates
    assert metrics.pr_auc(s, y) == pytest.approx(skm.average_precision_score(y, s))


def test_pr_auc_excluye_nan_y_sin_positivos_es_nan():
    assert metrics.pr_auc([np.nan, 0.9, 0.1], [0, 1, 0]) == pytest.approx(1.0)
    assert np.isnan(metrics.pr_auc([0.9, 0.1], [0, 0]))


# ------------------------------------------------- 6. deteccion por encima de la FAR

def test_detecta_por_encima_de_su_far():
    rng = np.random.default_rng(2)
    far = rng.uniform(0.0, 0.02, 20)
    fdr = far + 0.30 + rng.normal(0, 0.02, 20)
    r = metrics.detects_above_far(fdr, far, margin=0.0)
    assert r["detects_above_far"] and r["n"] == 20


def test_fdr_del_orden_de_la_far_no_cuenta_como_deteccion():
    far = np.full(8, 0.01)
    fdr = far + np.array([0.01, -0.01, 0.02, -0.02, 0.005, -0.005, 0.015, -0.015])
    assert not metrics.detects_above_far(fdr, far, margin=0.0)["detects_above_far"]


def test_el_margen_exige_un_exceso_relevante():
    far = np.full(20, 0.01)
    fdr = far + 0.01 + np.linspace(-0.001, 0.001, 20)   # exceso estable de 1 punto
    assert metrics.detects_above_far(fdr, far, margin=0.0)["detects_above_far"]
    assert not metrics.detects_above_far(fdr, far, margin=0.02)["detects_above_far"]


def test_pocas_simulaciones_no_establecen_deteccion():
    r = metrics.detects_above_far([0.9] * 4, [0.01] * 4, margin=0.0)
    assert np.isnan(r["p_value"]) and not r["detects_above_far"]


def test_el_margen_es_obligatorio():
    with pytest.raises(TypeError):
        metrics.detects_above_far([0.5] * 10, [0.01] * 10)


# ------------------------------------------------- 7. convencion del retardo

def test_retardo_cero_si_la_racha_empieza_con_el_fallo():
    stat = np.concatenate([np.zeros(ONSET), np.full(50, 9.0)])
    assert metrics.detection_delay(stat, 1.0, ONSET, k=3)["delay_samples"] == 0
    r = metrics.detection_delay(stat, 1.0, ONSET, k=3, convention="confirmation")
    assert r["delay_samples"] == 2 and r["delay_minutes"] == 6.0


def test_confirmacion_es_aparicion_mas_k_menos_1():
    stat = np.zeros(ONSET + 100)
    stat[ONSET + 7:] = 9.0
    for k in (1, 3, 6):
        aparicion = metrics.detection_delay(stat, 1.0, ONSET, k=k)["delay_samples"]
        confirmacion = metrics.detection_delay(stat, 1.0, ONSET, k=k,
                                               convention="confirmation")["delay_samples"]
        assert aparicion == 7 and confirmacion == aparicion + k - 1


def test_convencion_por_defecto_es_la_de_siempre():
    stat = np.zeros(ONSET + 100)
    stat[ONSET + 7:] = 9.0
    assert metrics.evaluate_run(stat, 1.0, ONSET)["delay_samples"] == 7
    assert metrics.evaluate_run(stat, 1.0, ONSET, delay_convention="confirmation")["delay_samples"] == 9


def test_convencion_desconocida_es_un_error():
    with pytest.raises(ValueError, match="appearance"):
        metrics.detection_delay(np.zeros(200), 1.0, ONSET, convention="aparicion")


# ------------------------------------------------- 8. ventana de la FAR (simulaciones normales)

def _normal_con_alarmas():
    """Simulacion normal de 960 muestras: 16 alarmas sueltas en las 160 primeras y 8 despues."""
    stat = np.zeros(960)
    stat[0:160:10] = 9.0
    stat[160:960:100] = 9.0
    return stat


def test_far_normal_sin_ventana_usa_toda_la_serie():
    r = metrics.evaluate_run(_normal_con_alarmas(), 1.0, None)
    assert r["far"] == pytest.approx(24 / 960) and r["n_far"] == 960


def test_far_normal_en_una_ventana_declarada():
    """(160, None) son las muestras 161 a 960: 8 alarmas en 800 muestras, 40 horas."""
    r = metrics.evaluate_run(_normal_con_alarmas(), 1.0, None, far_window=(160, None))
    assert r["far"] == pytest.approx(8 / 800) and r["n_far"] == 800
    assert r["false_alarm_samples_per_hour"] == pytest.approx(8 / 40)


def test_far_window_acepta_slice():
    claves = ("far", "n_far", "false_alarm_samples_per_hour", "false_alarm_episodes_per_hour")
    a = metrics.evaluate_run(_normal_con_alarmas(), 1.0, None, far_window=(160, None))
    b = metrics.evaluate_run(_normal_con_alarmas(), 1.0, None, far_window=slice(160, None))
    assert [a[c] for c in claves] == [b[c] for c in claves]


def test_far_window_con_fallo_es_un_error():
    """En los ficheros con fallo la FAR sigue siendo la del tramo anterior al inicio."""
    with pytest.raises(ValueError, match="far_window"):
        metrics.evaluate_run(np.zeros(960), 1.0, onset=ONSET, far_window=(160, None))


def test_far_window_invalida_es_un_error():
    for ventana in ((500, 100), (-1, None), (0, 2000), (160,), "160"):
        with pytest.raises(ValueError, match="far_window"):
            metrics.evaluate_run(np.zeros(960), 1.0, None, far_window=ventana)
