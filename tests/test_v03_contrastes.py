"""Pruebas de los contrastes de la v0.3 (tfmfdd/stats.py).

Cada prueba protege una pieza de la regla con la que se contrasta la hipotesis
del TFM o un defecto ya encontrado (M8, MET-8, MET-9, I2). Todas usan datos
sinteticos y semilla fija.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from tfmfdd import stats
from tfmfdd.stats import LowPowerWarning


def _p_minimo_exacto_scipy(n, alternative):
    """p exacto de scipy con n diferencias positivas y distintas (el minimo posible)."""
    d = np.arange(1, n + 1, dtype=float)
    alt = "greater" if alternative == "greater" else "two-sided"
    return sps.wilcoxon(d, alternative=alt, method="exact").pvalue


@contextmanager
def _sin_avisos_de_potencia():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowPowerWarning)
        yield


# ------------------------------------------------- estimador e intervalo

def test_hodges_lehmann_a_mano():
    """[1, 2, 4]: Walsh = 1, 1.5, 2, 2.5, 3, 4 -> mediana 2.25 (ni la media ni la mediana)."""
    r = stats.hodges_lehmann([1.0, 2.0, 4.0])
    assert r["hl_estimate"] == pytest.approx(2.25)


def test_intervalo_walsh_a_mano():
    """n = 6 al 90 %: P(W+ <= 2) = 3/64 <= 0,05 < 5/64, asi que c = 3 -> [W_(3), W_(19)].

    Los 21 promedios de Walsh de [1, 2, 4, 7, 11, 16], ordenados, empiezan
    1, 1.5, 2 y terminan 11, 11.5, 13.5, 16; el undecimo es 6.5.
    """
    r = stats.hodges_lehmann([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], ci_level=0.90)
    assert r["hl_estimate"] == pytest.approx(6.5)
    assert r["ci_low"] == pytest.approx(2.0)
    assert r["ci_high"] == pytest.approx(11.5)
    # Al 95 %, con 6 pares, c = 1: el intervalo va del menor al mayor promedio.
    r95 = stats.hodges_lehmann([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], ci_level=0.95)
    assert (r95["ci_low"], r95["ci_high"]) == (1.0, 16.0)


def test_intervalo_infinito_si_no_se_puede_rechazar():
    """Con 5 pares el bilateral al 95 % no rechaza nada: el intervalo es toda la recta."""
    r = stats.hodges_lehmann([0.1, 0.2, 0.3, 0.4, 0.5], ci_level=0.95)
    assert r["ci_low"] == -np.inf and r["ci_high"] == np.inf


def test_cobertura_del_intervalo_monte_carlo():
    """Sin empates, la cobertura es la exacta 1 - 2 P(W+ <= c - 1), que se calcula aqui enumerando."""
    n, theta, reps = 12, 0.3, 400
    # Distribucion exacta de W+ con n = 12 por enumeracion de los 2^12 signos.
    sumas = np.array([sum(i + 1 for i in range(n) if (k >> i) & 1) for k in range(2 ** n)])
    cdf = np.array([(sumas <= t).mean() for t in range(n * (n + 1) // 2 + 1)])
    c = int(np.flatnonzero(cdf <= 0.025)[-1]) + 1
    cobertura_exacta = 1 - 2 * cdf[c - 1]

    rng = np.random.default_rng(20261011)
    cubre = 0
    for _ in range(reps):
        d = theta + rng.standard_t(df=3, size=n)          # simetrica alrededor de theta
        r = stats.hodges_lehmann(d, ci_level=0.95)
        cubre += r["ci_low"] <= theta <= r["ci_high"]
    cobertura = cubre / reps
    ee = np.sqrt(cobertura_exacta * (1 - cobertura_exacta) / reps)
    assert cobertura_exacta >= 0.95
    assert abs(cobertura - cobertura_exacta) <= 3 * ee


# ------------------------------------------------------------ p exacto

def test_p_exacto_coincide_con_scipy_sin_empates():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = int(rng.integers(5, 30))
        a, b = rng.normal(0.3, 1, n), rng.normal(0, 1, n)
        for alt in ("two-sided", "less", "greater"):
            with _sin_avisos_de_potencia():
                r = stats.paired_test(a, b, alternative=alt)
            sp = sps.wilcoxon(a, b, alternative=alt, zero_method="zsplit", method="exact")
            assert r["p_value"] == pytest.approx(sp.pvalue, rel=1e-9, abs=1e-15)
            assert r["statistic"] == pytest.approx(sp.statistic)


@pytest.mark.skipif(not hasattr(sps, "PermutationMethod"),
                    reason="esta version de scipy no tiene PermutationMethod")
def test_p_exacto_coincide_con_permutacion_con_empates_y_ceros():
    """Con empates y n <= 13, scipy enumera todos los signos: debe dar lo mismo."""
    rng = np.random.default_rng(2)
    for _ in range(15):
        n = int(rng.integers(5, 14))
        d = rng.integers(-3, 4, size=n) / 800.0
        if np.all(d == 0):
            continue
        for alt in ("two-sided", "less", "greater"):
            with _sin_avisos_de_potencia():
                r = stats.paired_test(0.5 + d, np.full(n, 0.5), alternative=alt)
            sp = sps.wilcoxon(d, alternative=alt, zero_method="zsplit",
                              method=sps.PermutationMethod())
            assert r["p_value"] == pytest.approx(sp.pvalue, rel=1e-9, abs=1e-15)


def test_aproximacion_normal_por_encima_del_maximo_exacto():
    """Con mas de EXACT_MAX_N pares, la normal con correccion de continuidad (como scipy)."""
    rng = np.random.default_rng(3)
    n = stats.EXACT_MAX_N + 50
    a, b = rng.normal(0.05, 1, n), rng.normal(0, 1, n)
    r = stats.paired_test(a, b, alternative="greater")
    try:
        sp = sps.wilcoxon(a, b, alternative="greater", zero_method="zsplit",
                          method="asymptotic", correction=True)
    except (TypeError, ValueError):
        pytest.skip("esta version de scipy no acepta method='asymptotic'")
    assert r["p_method"] == "normal"
    assert r["p_value"] == pytest.approx(sp.pvalue, rel=1e-9)


def test_ruido_de_coma_flotante_no_crea_signos():
    """(0,03 - 0,01) - 0,02 da -3,5e-18: sin redondear, diez "empates exactos con delta" apoyarian."""
    ventaja = np.full(10, 0.03) - np.full(10, 0.01)
    assert (ventaja - 0.02)[0] != 0.0                      # el resto existe
    with pytest.warns(LowPowerWarning):                    # 0 diferencias no nulas
        r = stats.non_superiority_test(ventaja, delta=0.02)
    assert r["n_nonzero"] == 0
    assert r["outcome"] == stats.OUTCOME_INCONCLUSIVE


# ----------------------------------------------------- paired_test

def test_paired_test_conserva_las_claves_antiguas():
    rng = np.random.default_rng(4)
    r = stats.paired_test(rng.normal(size=20), rng.normal(size=20))
    for k in ("n", "statistic", "p_value", "effect_size", "median_difference"):
        assert k in r
    for k in ("hl_estimate", "ci_low", "ci_high", "ci_level", "n_nonzero", "p_method"):
        assert k in r


def test_mediana_nula_con_p_significativo_m8():
    """30 empates en 0 y 20 diferencias de +0,02: la mediana es 0, Hodges-Lehmann no (M8)."""
    a = np.r_[np.zeros(30), np.full(20, 0.02)] + 0.5
    b = np.full(50, 0.5)
    r = stats.paired_test(a, b)
    assert r["median_difference"] == 0.0
    assert r["p_value"] < 0.05
    assert r["hl_estimate"] > 0
    # Con tantos empates, el extremo inferior es el promedio de Walsh 0 aunque el
    # contraste rechace 0: el intervalo no decide, decide el p (docstring de hodges_lehmann).
    assert r["ci_low"] == 0.0


def test_efecto_con_signo_y_sin_depender_de_la_alternativa():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.5, 1, 30), rng.normal(0, 1, 30)
    r = stats.paired_test(a, b)
    r_inv = stats.paired_test(b, a)
    assert r["effect_size"] > 0
    assert r_inv["effect_size"] == pytest.approx(-r["effect_size"])
    for alt in ("less", "greater"):
        assert stats.paired_test(a, b, alternative=alt)["effect_size"] == pytest.approx(r["effect_size"])
    # Todas peores: -1 exacto (antes salia +1,0000000000002).
    assert stats.paired_test(np.zeros(20), np.ones(20))["effect_size"] == -1.0


def test_diferencias_simetricas_dan_efecto_cero_no_nan_met9():
    d = np.array([0.01, -0.01, 0.02, -0.02, 0.03, -0.03])
    r = stats.paired_test(0.5 + d, np.full(6, 0.5))
    assert r["p_value"] == 1.0
    assert r["effect_size"] == 0.0


def test_todas_las_diferencias_nulas_met9():
    """Sin evidencia en ningun sentido: p = 1, efecto 0, estadistico n(n+1)/4 (no el 0 de maxima evidencia)."""
    with pytest.warns(LowPowerWarning):
        r = stats.paired_test(np.full(20, 0.3), np.full(20, 0.3))
    assert r["p_value"] == 1.0 and r["effect_size"] == 0.0
    assert r["statistic"] == 20 * 21 / 4
    assert (r["hl_estimate"], r["ci_low"], r["ci_high"]) == (0.0, 0.0, 0.0)


def test_menos_de_cinco_pares_no_se_contrasta():
    r = stats.paired_test([0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0])
    assert r["n"] == 4 and np.isnan(r["p_value"]) and np.isnan(r["hl_estimate"])


def test_alpha_del_limite_no_se_confunde_con_el_de_los_contrastes():
    with pytest.raises(ValueError, match="nivel de confianza"):
        stats.paired_test(np.ones(10), np.zeros(10), alpha=0.99)
    with pytest.raises(ValueError, match="nivel de confianza"):
        stats.holm_correction([0.01, 0.02], alpha=0.99)


# ------------------------------------------------------------- potencia

def test_n_minimo_con_el_calculo_exacto():
    """n_min es el primer n cuyo p minimo exacto (scipy, sin empates) llega a alpha/m."""
    for m in (1, 2, 3, 4, 6, 8, 12, 20):
        for alt in ("two-sided", "greater"):
            n = stats.min_runs_for_significance(0.05, m, alt)
            assert _p_minimo_exacto_scipy(n, alt) <= 0.05 / m
            assert _p_minimo_exacto_scipy(n - 1, alt) > 0.05 / m


def test_cinco_replicas_nunca_significativas_al_5_por_ciento():
    assert stats.min_runs_for_significance(0.05, 1, "two-sided") == 6
    with pytest.warns(LowPowerWarning):
        r = stats.paired_test(np.arange(1, 6) + 1.0, np.zeros(5))   # 5 de 5 mejoran
    assert r["p_value"] > 0.05


def test_sin_aviso_cuando_hay_simulaciones_suficientes():
    with warnings.catch_warnings():
        warnings.simplefilter("error", LowPowerWarning)
        stats.paired_test(np.arange(1, 7) + 1.0, np.zeros(6))


def test_los_ceros_no_cuentan_para_la_potencia():
    """20 simulaciones con 17 empates: solo 3 diferencias no nulas, y el aviso lo dice."""
    a = np.r_[np.zeros(17), [0.1, 0.2, 0.3]]
    with pytest.warns(LowPowerWarning, match="3 diferencias no nulas"):
        stats.paired_test(a, np.zeros(20))


# --------------------------------------------------------------- Holm

def _holm_por_pasos(p, alpha):
    orden = np.argsort(p, kind="stable")
    sig = np.zeros(len(p), dtype=bool)
    for i, j in enumerate(orden):
        if p[j] <= alpha / (len(p) - i):
            sig[j] = True
        else:
            break
    return sig


def test_holm_p_ajustados_a_mano():
    t = stats.holm_correction([0.01, 0.04, 0.03])
    assert t["p_adjusted"].tolist() == pytest.approx([0.03, 0.06, 0.06])
    assert t["significant"].tolist() == [True, False, False]


def test_holm_significativo_igual_al_procedimiento_por_pasos():
    rng = np.random.default_rng(6)
    for _ in range(200):
        p = rng.uniform(0, 0.08, size=int(rng.integers(1, 9)))
        t = stats.holm_correction(p)
        assert np.array_equal(t["significant"].to_numpy(), _holm_por_pasos(p, 0.05))


def test_holm_deja_fuera_los_nan_met8():
    t = stats.holm_correction([0.02, 0.03, np.nan])
    assert t["significant"].tolist() == [True, True, False]
    assert np.isnan(t["p_adjusted"].iloc[2])


# ------------------------------------------------------- compare_methods

def _tabla_tres_metodos(seed=7):
    rng = np.random.default_rng(seed)
    filas = []
    for fault in (1, 2):
        for run in range(1, 21):
            base = rng.uniform(0.4, 0.6)
            filas += [
                {"faultNumber": fault, "simulationRun": run, "method": "pca", "fdr": base},
                {"faultNumber": fault, "simulationRun": run, "method": "ae",
                 "fdr": base + 0.05 + rng.normal(0, 0.01)},
                {"faultNumber": fault, "simulationRun": run, "method": "knn",
                 "fdr": base + rng.normal(0, 0.01)},
            ]
    return pd.DataFrame(filas)


def test_compare_methods_con_baseline_hace_k_menos_1_por_fallo():
    tabla = stats.compare_methods(_tabla_tres_metodos(), metric="fdr", baseline="pca")
    assert len(tabla) == 4                                    # 2 fallos x (3 - 1)
    assert set(tabla["method_b"]) == {"pca"}
    ae = tabla[tabla["method_a"] == "ae"]
    assert (ae["hl_estimate"] > 0.04).all() and (ae["effect_size"] > 0).all()   # metodo - baseline
    # Holm dentro de cada fallo sobre 2 comparaciones:
    for _, g in tabla.groupby("faultNumber"):
        esperado = stats.holm_correction(g["p_value"].to_numpy())["p_adjusted"].to_numpy()
        assert np.allclose(g["p_adjusted"].to_numpy(), esperado)


def test_compare_methods_sin_baseline_sigue_siendo_todos_contra_todos():
    tabla = stats.compare_methods(_tabla_tres_metodos(), metric="fdr")
    assert len(tabla) == 6                                    # 2 fallos x 3 pares
    assert (tabla["method_a"] < tabla["method_b"]).all()      # orden alfabetico, como antes


def test_compare_methods_baseline_inexistente_falla():
    with pytest.raises(ValueError, match="baseline"):
        stats.compare_methods(_tabla_tres_metodos(), metric="fdr", baseline="dpca")


# --------------------------------------------------- ventaja y no superioridad

def test_paired_advantage_orienta_el_signo():
    res = pd.DataFrame({
        "faultNumber": 3, "simulationRun": [1, 2, 1, 2],
        "method": ["ae", "ae", "pca", "pca"], "far": [0.01, 0.02, 0.03, 0.03],
    })
    v = stats.paired_advantage(res, "far", "ae", "pca", higher_is_better=False, fault=3)
    assert v.tolist() == pytest.approx([0.02, 0.01])          # menos FAR es ventaja
    with pytest.raises(TypeError):
        stats.paired_advantage(res, "far", "ae", "pca", higher_is_better=None, fault=3)


def test_paired_advantage_avisa_de_simulaciones_sin_pareja():
    res = pd.DataFrame({
        "faultNumber": 3, "simulationRun": [1, 2, 3, 1, 2],
        "method": ["ae", "ae", "ae", "pca", "pca"], "fdr": [0.1, 0.2, 0.3, 0.1, 0.1],
    })
    with pytest.warns(UserWarning, match="un solo metodo"):
        v = stats.paired_advantage(res, "fdr", "ae", "pca", higher_is_better=True, fault=3)
    assert list(v.index) == [1, 2]


def test_no_superioridad_apoya():
    rng = np.random.default_rng(8)
    r = stats.non_superiority_test(rng.normal(0.0, 0.01, 20), delta=0.05)
    assert r["outcome"] == stats.OUTCOME_SUPPORTS
    assert r["ci_high"] < 0.05 and r["ci_level"] == pytest.approx(0.90)


def test_no_superioridad_ventaja_relevante():
    rng = np.random.default_rng(9)
    r = stats.non_superiority_test(0.2 + rng.normal(0, 0.01, 20), delta=0.05)
    assert r["outcome"] == stats.OUTCOME_RELEVANT
    assert r["ci_low"] > 0.05


def test_no_superioridad_inconclusa():
    """Mucha dispersion: compatible con una ventaja menor y con una mayor que delta."""
    rng = np.random.default_rng(10)
    r = stats.non_superiority_test(0.05 + rng.normal(0, 0.3, 20), delta=0.05)
    assert r["outcome"] == stats.OUTCOME_INCONCLUSIVE
    assert r["ci_low"] < 0.05 < r["ci_high"]


def test_no_superioridad_p_e_intervalo_coinciden_sin_empates():
    """Sin empates, apoya <=> ci_high <= delta y ventaja relevante <=> ci_low > delta."""
    rng = np.random.default_rng(11)
    for _ in range(60):
        n = int(rng.integers(7, 40))
        d = rng.normal(rng.uniform(-0.05, 0.15), rng.uniform(0.01, 0.1), n)
        r = stats.non_superiority_test(d, delta=0.05)
        assert (r["outcome"] == stats.OUTCOME_SUPPORTS) == (r["ci_high"] <= 0.05)
        assert (r["outcome"] == stats.OUTCOME_RELEVANT) == (r["ci_low"] > 0.05)


def test_delta_es_obligatorio_y_no_negativo():
    with pytest.raises(TypeError):
        stats.non_superiority_test(np.zeros(10))              # sin delta
    with pytest.raises(ValueError):
        stats.non_superiority_test(np.zeros(10), delta=-0.01)


# ------------------------------------------------------- veredicto global

def _familia(ventajas, detecta=True):
    claves = [(f, b) for f in (3, 9, 15) for b in ("pca", "dpca")]
    return [{"fault": f, "method": "ae", "baseline": b, "advantage": v,
             "detects_method": detecta, "detects_baseline": False}
            for (f, b), v in zip(claves, ventajas)]


def _controles(ok=True):
    return [{"fault": 10, "method": "ae", "baseline": "pca",
             "detects_method": True, "detects_baseline": ok}]


def _sin_ventaja(rng, n=20):
    return rng.normal(0.0, 0.01, n)


def test_veredicto_apoya_si_las_seis_apoyan():
    rng = np.random.default_rng(12)
    out = stats.hypothesis_verdict(_familia([_sin_ventaja(rng) for _ in range(6)]),
                                   delta=0.05, controls=_controles())
    assert out["verdict"] == stats.VERDICT_SUPPORTS
    assert out["m"] == 6
    assert out["table"]["counts_as_support"].iloc[:6].all()


def test_veredicto_refuta_con_una_ventaja_relevante_tras_holm():
    rng = np.random.default_rng(13)
    v = [_sin_ventaja(rng) for _ in range(6)]
    v[3] = 0.2 + rng.normal(0, 0.01, 20)                      # fallo 9 frente a dpca
    out = stats.hypothesis_verdict(_familia(v), delta=0.05, controls=_controles())
    assert out["verdict"] == stats.VERDICT_REFUTES
    t = out["table"]
    assert t.loc[3, "outcome"] == stats.OUTCOME_RELEVANT
    assert t.loc[3, "p_above_delta_holm"] <= 0.05
    assert "fallo 9" in out["explanation"]


def test_holm_puede_impedir_refutar():
    """6 simulaciones por encima de delta: p = 1/64 <= 0,05 sola, pero > 0,05/6 en la familia."""
    rng = np.random.default_rng(14)
    v = [_sin_ventaja(rng) for _ in range(6)]
    v[0] = np.array([0.20, 0.21, 0.22, 0.23, 0.24, 0.25])
    assert stats.non_superiority_test(v[0], delta=0.05)["outcome"] == stats.OUTCOME_RELEVANT
    with pytest.warns(LowPowerWarning, match="refutar exige al menos 7"):
        out = stats.hypothesis_verdict(_familia(v), delta=0.05, controls=_controles())
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE
    assert out["table"].loc[0, "outcome"] == stats.OUTCOME_INCONCLUSIVE


def test_veredicto_inconcluso_si_una_no_apoya():
    rng = np.random.default_rng(15)
    v = [_sin_ventaja(rng) for _ in range(6)]
    v[5] = 0.05 + rng.normal(0, 0.3, 20)                      # inconclusa
    out = stats.hypothesis_verdict(_familia(v), delta=0.05, controls=_controles())
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE
    assert "fallo 15, ae frente a dpca: inconclusa" in out["explanation"]


def test_suelo_de_deteccion_no_cuenta_como_apoyo():
    rng = np.random.default_rng(16)
    fam = _familia([_sin_ventaja(rng) for _ in range(6)])
    fam[0]["detects_method"] = False                          # ninguno detecta el fallo 3
    out = stats.hypothesis_verdict(fam, delta=0.05, controls=_controles())
    t = out["table"]
    assert t.loc[0, "outcome"] == stats.OUTCOME_SUPPORTS and bool(t.loc[0, "floor"])
    assert not bool(t.loc[0, "counts_as_support"])
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE
    assert "suelo" in out["explanation"]


def test_control_fallido_impide_apoyar():
    rng = np.random.default_rng(17)
    out = stats.hypothesis_verdict(_familia([_sin_ventaja(rng) for _ in range(6)]),
                                   delta=0.05, controls=_controles(ok=False))
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE
    assert "control" in out["explanation"]


def test_veredicto_sin_controles_avisa():
    # Correcciones del 12/10 (M3): sin controles el veredicto ya no puede ser "apoya".
    rng = np.random.default_rng(18)
    with pytest.warns(UserWarning, match="sin fallos de control"):
        out = stats.hypothesis_verdict(_familia([_sin_ventaja(rng) for _ in range(6)]),
                                       delta=0.05)
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE


def test_veredicto_exige_saber_si_detectan():
    rng = np.random.default_rng(19)
    fam = _familia([_sin_ventaja(rng) for _ in range(6)])
    for c in fam:
        del c["detects_method"]
    with pytest.raises(ValueError, match="suelo"):
        stats.hypothesis_verdict(fam, delta=0.05, controls=_controles())
    # Con una funcion detects(metodo, fallo) -> bool, vale.
    out = stats.hypothesis_verdict(fam, delta=0.05, controls=_controles(),
                                   detects=lambda metodo, fallo: metodo == "ae")
    assert out["verdict"] == stats.VERDICT_SUPPORTS


def test_comparacion_no_contrastable_cuenta_en_holm_y_no_apoya():
    rng = np.random.default_rng(20)
    v = [_sin_ventaja(rng) for _ in range(6)]
    v[2] = np.array([0.0, 0.01, -0.01])                       # 3 simulaciones
    with pytest.warns(LowPowerWarning):
        out = stats.hypothesis_verdict(_familia(v), delta=0.05, controls=_controles())
    t = out["table"]
    assert t.loc[2, "outcome"] == stats.OUTCOME_UNTESTABLE
    assert out["verdict"] == stats.VERDICT_INCONCLUSIVE
    assert out["m"] == 6
