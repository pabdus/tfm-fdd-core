"""Pruebas de las correcciones del 12/10 en stats.py y metrics.py.

Cada bloque cubre un hallazgo de la revision independiente de la parte
estadistica (verif_estadistica: I1 a I4, M1 a M5, M7, M8 y M10) o la
simplificacion de su seccion 3. Datos sinteticos y semilla fija. Donde hay
inferencia, una comprobacion de nivel por Monte Carlo pequena: la tasa simulada
no pasa de su nivel nominal mas tres errores estandar.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

from tfmfdd import metrics, stats
from tfmfdd.stats import LowPowerWarning

ALPHA = 0.05


def _cota_mc(nivel, reps):
    return nivel + 3 * np.sqrt(nivel * (1 - nivel) / reps)


@contextmanager
def _sin_avisos():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowPowerWarning)
        yield


# ------------------------------------------------ I1. p exacto por no nulas

def test_p_exacto_con_muchos_pares_y_pocas_no_nulas():
    """600 pares, 597 nulos y 3 positivos: el p exacto es 1/8 (antes, la normal daba 0,042)."""
    a = np.r_[np.zeros(597), [0.01, 0.02, 0.03]] + 0.5
    with _sin_avisos():
        r = stats.paired_test(a, np.full(600, 0.5), alternative="greater")
    assert r["p_method"] == "exact"
    assert r["p_value"] == pytest.approx(1 / 8)


def test_no_superioridad_con_600_simulaciones_casi_todas_en_delta():
    ventaja = np.r_[np.full(597, 0.02), [0.05, 0.06, 0.07]]
    with _sin_avisos():
        r = stats.non_superiority_test(ventaja, delta=0.02)
    assert r["p_method"] == "exact" and r["outcome"] == stats.OUTCOME_INCONCLUSIVE


def test_rama_normal_solo_con_muchas_no_nulas():
    rng = np.random.default_rng(1)
    d = rng.normal(0.01, 1, 700)
    assert stats.paired_test(d, np.zeros(700))["p_method"] == "normal"
    assert stats.paired_test(d[:500], np.zeros(500))["p_method"] == "exact"


# ------------------------------------------------ I2. la ventaja publicada

def _familia(ventaja, faults=(3, 9, 15), bases=("pca", "dpca")):
    return [{"fault": f, "method": "ae", "baseline": b, "advantage": ventaja,
             "detects_method": True, "detects_baseline": True} for f in faults for b in bases]


_CONTROL = [{"fault": 10, "method": "ae", "baseline": "pca",
             "detects_method": True, "detects_baseline": True}]


def test_veredicto_dice_que_no_evalua_la_ventaja_publicada():
    ventaja = 0.012 + np.linspace(-0.004, 0.004, 20)
    out = stats.hypothesis_verdict(_familia(ventaja), delta=0.02, controls=_CONTROL)
    assert out["verdict"] == stats.VERDICT_SUPPORTS
    assert out["published_condition"] == stats.PUBLISHED_NOT_EVALUATED
    assert out["published"] is None
    assert "parrafo 349): no evaluada" in out["explanation"]


def test_veredicto_muestra_la_ventaja_publicada_sin_combinarla():
    """La reproduccion de I2: el veredicto 'apoya' y la publicada 'igual o mayor'."""
    ventaja = 0.012 + np.linspace(-0.004, 0.004, 20)
    fila = {"article": "Articulo inventado (2000)", "metric": "fdr", "method": "ae",
            "baseline": "pca", "value": 0.005, "page": "p. 1",
            "comparison": stats.COMPARISON_WITHIN}
    publicada = stats.published_advantage_test(ventaja, fila, metric="fdr")
    assert publicada["outcome"] == stats.PUBLISHED_NOT_LESS
    out = stats.hypothesis_verdict(_familia(ventaja), delta=0.02, controls=_CONTROL,
                                   published=[publicada])
    assert out["verdict"] == stats.VERDICT_SUPPORTS                 # no se combina (D9)
    assert out["published_condition"] == stats.PUBLISHED_NOT_EVALUATED
    assert out["published"]["outcome"].tolist() == [stats.PUBLISHED_NOT_LESS]
    assert "igual o mayor que la publicada" in out["explanation"]


def test_published_mal_formado_es_un_error():
    with pytest.raises(ValueError, match="published_advantage_test"):
        stats.hypothesis_verdict(_familia(np.zeros(20)), delta=0.02, controls=_CONTROL,
                                 published=[{"outcome": "menor"}])


# ------------------------------------------------ I3. supuesto de simetria

@pytest.mark.parametrize("funcion", [
    stats.paired_test, stats.non_superiority_test, stats.hypothesis_verdict,
    stats.effect_test, stats.compare_arms, stats.detects_above_far,
    stats.published_advantage_test,
])
def test_el_supuesto_de_simetria_esta_donde_se_decide(funcion):
    assert "simetri" in funcion.__doc__


def test_los_analisis_de_sensibilidad_estan_documentados():
    doc = stats.non_superiority_test.__doc__
    assert "binomtest" in doc and "ttest_1samp" in doc and "Bootstrap" in doc


# ------------------------------------------------ I4. unidad con replicas

def _cruzado(seed, reps, R, N, su, sv, se=0.002):
    """Modelo cruzado d_rj = u_r + v_j + e_rj sin efecto medio; cada repeticion es un 'fallo'."""
    rng = np.random.default_rng(seed)
    trozos = []
    for k in range(reps):
        d = (rng.normal(0, su, R)[:, None] + rng.normal(0, sv, N)[None, :]
             + rng.normal(0, se, (R, N)))
        limpio = pd.DataFrame({"faultNumber": k, "replica": np.repeat(np.arange(R), N),
                               "simulationRun": np.tile(np.arange(N), R), "arm": "limpio",
                               "far": np.full(R * N, 0.01)})
        trozos += [limpio, limpio.assign(arm="sucio", far=0.01 + d.ravel())]
    return pd.concat(trozos, ignore_index=True)


def test_replicate_col_exige_declarar_la_unidad():
    df = _cruzado(40, 1, 4, 6, 0.002, 0.002)
    with pytest.raises(ValueError, match="unidad"):
        stats.compare_methods(df, metric="far", method_col="arm", baseline="limpio",
                              replicate_col="replica")
    with pytest.raises(ValueError, match="unidad"):
        stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, family=(),
                           replicate_col="replica")
    with pytest.raises(ValueError, match="replicate_col"):
        stats.compare_methods(df, metric="far", method_col="arm", baseline="limpio",
                              unit="simulation")
    with pytest.raises(ValueError, match="unit"):
        stats.compare_methods(df, metric="far", method_col="arm", baseline="limpio",
                              replicate_col="replica", unit="replica")


def test_unidad_simulacion_promedia_las_replicas_de_cada_simulacion():
    df = _cruzado(41, 1, 4, 12, 0.002, 0.004)
    t = stats.compare_methods(df, metric="far", method_col="arm", baseline="limpio",
                              replicate_col="replica", unit="simulation")
    medias = df.pivot_table(index="simulationRun", columns="arm", values="far", aggfunc="mean")
    esperado = stats.paired_test(medias["sucio"].to_numpy(), medias["limpio"].to_numpy())
    fila = t.iloc[0]
    assert (fila["n"], fila["n_cells"], fila["unit"]) == (12, 48, "simulationRun")
    assert fila["p_value"] == pytest.approx(esperado["p_value"])


def test_nivel_de_cada_unidad_en_el_modelo_cruzado_monte_carlo():
    """Cada unidad controla el error frente a su poblacion y no frente a la otra (I4):
    si el efecto varia entre simulaciones de prueba, solo unit='simulation' guarda el
    nivel; si varia entre replicas, solo unit='replicate'."""
    reps = 100
    for su, sv, valida, otra in ((0.0005, 0.01, "simulation", "replicate"),
                                 (0.01, 0.0005, "replicate", "simulation")):
        df = _cruzado(42, reps, 8, 30, su, sv)
        tasa = {}
        with _sin_avisos():
            for unidad in (valida, otra):
                t = stats.compare_methods(df, metric="far", method_col="arm",
                                          baseline="limpio", replicate_col="replica",
                                          unit=unidad)
                tasa[unidad] = float((t["p_value"] <= ALPHA).mean())
        assert tasa[valida] <= _cota_mc(ALPHA, reps)
        assert tasa[otra] > 3 * ALPHA


def test_compare_arms_con_unidad_simulacion():
    df = _cruzado(43, 1, 4, 12, 0.002, 0.004)
    t = stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, family=(),
                           replicate_col="replica", unit="simulation")
    assert (t["n"].iloc[0], t["unit"].iloc[0]) == (12, "simulationRun")


# ------------------------------------------------ M1. familia declarada

def _tabla_con_no_contrastable():
    rng = np.random.default_rng(0)
    filas = []
    for f in (1, 2):
        for m in ("A", "B", "C"):
            n = 3 if (f == 2 and m == "C") else 12
            for r in range(1, n + 1):
                filas.append({"faultNumber": f, "method": m, "simulationRun": r,
                              "fdr": rng.normal(0.5, 0.05)})
    return pd.DataFrame(filas)


def test_familia_declarada_cuenta_las_no_contrastables_con_p_1():
    with _sin_avisos(), pytest.warns(UserWarning, match="solo estan en uno"):
        t = stats.compare_methods(_tabla_con_no_contrastable(), baseline="A", family=())
    esperado = stats.holm_correction(t["p_value"].fillna(1.0).to_numpy())["p_adjusted"]
    assert np.allclose(t["p_adjusted"].to_numpy(), esperado.to_numpy())
    assert (t["family_m"] == 4).all()                       # el m que usa Holm
    assert t["p_adjusted"].iloc[3] == 1.0 and not t["significant"].iloc[3]


def test_sin_familia_declarada_la_no_contrastable_sigue_fuera():
    with _sin_avisos(), pytest.warns(UserWarning, match="solo estan en uno"):
        t = stats.compare_methods(_tabla_con_no_contrastable(), baseline="A")
    assert np.isnan(t["p_adjusted"].iloc[3])


# ------------------------------------------------ M2. simulaciones sin pareja

def test_simulaciones_sin_pareja_avisan():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"method": ["A"] * 10 + ["B"] * 10,
                       "simulationRun": list(range(1, 11)) + list(range(5, 15)),
                       "fdr": rng.normal(0.5, 0.05, 20)})
    with pytest.warns(UserWarning, match="solo estan en uno"):
        t = stats.compare_methods(df, fault_col=None)
    assert t["n"].iloc[0] == 6
    brazos = df.rename(columns={"method": "arm"})
    with pytest.warns(UserWarning, match="solo estan en uno"):
        stats.compare_arms(brazos, "fdr", "arm", "A", margin=0.01, family=(), by=())


def test_ids_de_tipos_distintos_lo_dicen():
    df = pd.DataFrame({"method": ["A"] * 6 + ["B"] * 6,
                       "simulationRun": list(range(1, 7)) + [str(i) for i in range(1, 7)],
                       "fdr": np.linspace(0.1, 0.6, 12)})
    with pytest.warns(UserWarning, match="tipos distintos"):
        with _sin_avisos():
            stats.compare_methods(df, fault_col=None)


# ------------------------------------------------ M3. familia y controles

def test_familia_de_cinco_no_apoya():
    ventaja = 0.012 + np.linspace(-0.004, 0.004, 20)
    with pytest.warns(UserWarning, match="no es la declarada"):
        out = stats.hypothesis_verdict(_familia(ventaja)[:5], delta=0.02, controls=_CONTROL)
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE and not out["declared_family"]
    assert "no es la declarada" in out["explanation"]


def test_familia_con_otros_fallos_no_apoya():
    ventaja = 0.012 + np.linspace(-0.004, 0.004, 20)
    with pytest.warns(UserWarning, match="no es la declarada"):
        out = stats.hypothesis_verdict(_familia(ventaja, faults=(3, 9, 16)), delta=0.02,
                                       controls=_CONTROL)
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE


def test_sin_controles_no_apoya_y_lo_explica():
    ventaja = 0.012 + np.linspace(-0.004, 0.004, 20)
    with pytest.warns(UserWarning, match="sin fallos de control"):
        out = stats.hypothesis_verdict(_familia(ventaja), delta=0.02)
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE
    assert "no hay fallos de control" in out["explanation"]
    assert out["declared_family"]


def test_familia_no_declarada_puede_refutar_con_aviso():
    fam = _familia(0.2 + np.linspace(-0.01, 0.01, 20))[:5]
    with pytest.warns(UserWarning, match="no es la declarada"):
        out = stats.hypothesis_verdict(fam, delta=0.02, controls=_CONTROL)
    assert out["verdict"] == stats.VERDICT_REFUTES
    assert "no es la declarada" in out["explanation"]


# ------------------------------------------------ M4. tabla del veredicto

def test_ventaja_none_es_no_contrastable():
    fam = _familia(0.012 + np.linspace(-0.004, 0.004, 20))
    fam[5]["advantage"] = None
    with _sin_avisos():
        out = stats.hypothesis_verdict(fam, delta=0.02, controls=_CONTROL)
    fila = out["table"].iloc[5]
    assert fila["outcome"] == stats.OUTCOME_UNTESTABLE and fila["n"] == 0
    assert fila["p_above_delta_holm"] == 1.0
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE and out["m"] == 6


def test_el_desenlace_de_los_controles_va_aparte():
    ventaja = 0.012 + np.linspace(-0.004, 0.004, 20)
    control = [dict(_CONTROL[0], advantage=0.5 + np.linspace(-0.01, 0.01, 20))]
    out = stats.hypothesis_verdict(_familia(ventaja), delta=0.02, controls=control)
    ctrl = out["table"][out["table"]["role"] == "control"].iloc[0]
    assert pd.isna(ctrl["outcome"])
    assert ctrl["control_outcome"] == stats.OUTCOME_RELEVANT
    assert out["verdict"] == stats.VERDICT_SUPPORTS


# ------------------------------------------------ M5. Holm en el umbral

def test_holm_en_el_umbral_exacto():
    h = stats.holm_correction([0.05 / 11] + [0.9] * 10)
    assert h["p_adjusted"].iloc[0] > 0.05                  # el resto de coma flotante
    assert bool(h["significant"].iloc[0])                  # pero el paso a paso rechaza


def test_holm_por_pasos_con_p_en_los_umbrales():
    rng = np.random.default_rng(5)
    for _ in range(300):
        m = int(rng.integers(2, 12))
        p = np.where(rng.random(m) < 0.5, ALPHA / rng.integers(1, m + 1, m), rng.random(m))
        orden = np.argsort(p, kind="stable")
        esperado = np.zeros(m, dtype=bool)
        for i, j in enumerate(orden):
            if p[j] <= ALPHA / (m - i):
                esperado[j] = True
            else:
                break
        assert np.array_equal(stats.holm_correction(p)["significant"].to_numpy(), esperado)


# ------------------------------------------------ M7. infinitos

def test_infinitos_dan_un_error_claro():
    a = np.r_[np.inf, np.arange(1.0, 8.0)]
    b = np.r_[np.inf, np.zeros(7)]
    with pytest.raises(ValueError, match="infinitos"):
        stats.paired_test(a, b)
    with pytest.raises(ValueError, match="infinitos"):
        stats.non_superiority_test(a, delta=0.0)
    with pytest.raises(ValueError, match="infinitos"):
        stats.effect_test(a, margin=0.0)
    with pytest.raises(ValueError, match="infinitos"):
        stats.detects_above_far(a, np.zeros(8), margin=0.0)


# ------------------------------------------------ M8. redondeo con escalas diminutas

def test_diferencias_diminutas_no_dan_nan():
    with _sin_avisos():
        r = stats.paired_test(np.arange(1, 9) * 1e-300, np.zeros(8))
    assert r["p_value"] == pytest.approx(2 / 2 ** 8)
    assert not np.isnan(stats._round_noise(np.array([1e-300, 2e-300]), 2e-300)).any()


# ------------------------------------------------ M10. sin resultados a mano

def test_evaluate_run_no_remite_a_un_csv_de_resultados():
    assert "A1_pca_baseline_runs.csv" not in metrics.evaluate_run.__doc__


# ------------------------------------------------ seccion 3: nucleo comun

def test_ventaja_publicada_y_no_superioridad_dan_lo_mismo():
    rng = np.random.default_rng(6)
    for _ in range(60):
        n = int(rng.integers(5, 40))
        d = rng.integers(-10, 40, n) / 800
        valor = int(rng.integers(0, 30)) / 800
        fila = {"article": "x", "metric": "fdr", "method": "ae", "baseline": "pca",
                "value": valor, "page": "1", "comparison": stats.COMPARISON_WITHIN}
        with _sin_avisos():
            ns = stats.non_superiority_test(d, delta=valor)
        pub = stats.published_advantage_test(d, fila, metric="fdr")
        for a, b in (("p_below_published", "p_below_delta"),
                     ("p_above_published", "p_above_delta"),
                     ("hl_estimate", "hl_estimate"), ("ci_low", "ci_low"),
                     ("ci_high", "ci_high"), ("n_nonzero", "n_nonzero")):
            assert pub[a] == ns[b]


def test_delta_booleano_es_un_error():
    with pytest.raises(ValueError, match="delta"):
        stats.non_superiority_test(np.zeros(10), delta=True)
    with pytest.raises(ValueError, match="delta"):
        stats.hypothesis_verdict(_familia(np.zeros(20)), delta=True, controls=_CONTROL)
