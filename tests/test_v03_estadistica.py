"""Pruebas de la segunda parte de la v0.3 en stats.py y metrics.py (12-10-2026).

Una idea por prueba, con datos sinteticos y semilla fija. Cubren:
1. El suelo de deteccion con el mismo Wilcoxon exacto que la hipotesis
   (detects_above_far vive en stats y metrics la reexporta) y su paso directo
   a hypothesis_verdict.
2. Replicas de ajuste y familias de Holm en compare_methods.
3. Efecto frente a un margen de equivalencia (effect_test, compare_arms).
4. Macro-F1 por replica (metrics.macro_f1_by_replicate).
5. Ventaja medida frente a la publicada (published_advantage_test).
6. Mecanismo del mejor compromiso (pareto_front, best_compromise).
Donde hay inferencia, una comprobacion de nivel por Monte Carlo pequena: la
tasa de error simulada no pasa de su nivel nominal mas tres errores estandar.
Las tablas de ventajas publicadas de las pruebas son inventadas.
"""

import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from tfmfdd import metrics, stats
from tfmfdd.stats import LowPowerWarning

ALPHA = 0.05


@contextmanager
def _sin_avisos_de_potencia():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowPowerWarning)
        yield


def _cota_mc(nivel, reps):
    """Nivel nominal mas tres errores estandar de una proporcion simulada."""
    return nivel + 3 * np.sqrt(nivel * (1 - nivel) / reps)


def _p_mayor_por_enumeracion(x):
    """p exacto de H1: theta > 0 enumerando los 2^n signos (x sin ceros, n pequeno)."""
    x = np.asarray(x, dtype=float)
    rangos = sps.rankdata(np.abs(x))
    w_obs = rangos[x > 0].sum()
    n = x.size
    bits = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    return float(np.mean(bits @ rangos >= w_obs - 1e-9))


# ------------------------------------------- 1. suelo de deteccion e hipotesis

def test_detects_above_far_es_la_misma_funcion_en_metrics():
    assert metrics.detects_above_far is stats.detects_above_far


def test_suelo_y_no_superioridad_dan_el_mismo_p():
    """El suelo es la 'ventaja relevante' de non_superiority_test sobre fdr - far frente al margen."""
    rng = np.random.default_rng(31)
    for _ in range(150):
        n = int(rng.integers(5, 60))
        far = rng.integers(0, 24, n) / 800                 # rejilla de Rieth: empates y ceros
        fdr = far + rng.integers(-8, 40, n) / 800
        margen = int(rng.integers(0, 16)) / 800
        r = stats.detects_above_far(fdr, far, margin=margen)
        with _sin_avisos_de_potencia():
            ns = stats.non_superiority_test(fdr - far, delta=margen)
        assert r["p_value"] == ns["p_above_delta"]
        assert r["statistic"] == ns["statistic"]
        assert r["detects_above_far"] == (ns["p_above_delta"] <= ALPHA)


def test_suelo_usa_el_p_exacto_con_empates():
    """16 simulaciones con empates: el p es el de enumerar los 2^16 signos."""
    k = np.array([3, 3, 2, 2, 2, 1, 1, -1, 4, 4, 5, 2, 3, -2, 1, 3])
    far = np.full(16, 0.01)
    r = stats.detects_above_far(far + k / 800, far, margin=0.0)
    assert r["p_method"] == "exact"
    assert r["p_value"] == pytest.approx(_p_mayor_por_enumeracion(k), rel=1e-12)


def test_suelo_no_cuenta_restos_de_coma_flotante():
    """0,005 - 0,00375 - 0,00125 da +2,2e-19: sin redondear, diez excesos nulos 'detectarian'."""
    fdr, far, margen = np.full(10, 0.005), np.full(10, 0.00375), 0.00125
    assert (fdr - far - margen)[0] > 0                     # el resto existe
    r = stats.detects_above_far(fdr, far, margin=margen)
    assert r["n_nonzero"] == 0 and r["p_value"] == 1.0
    assert not r["detects_above_far"]


def test_suelo_margen_no_negativo_y_alpha_de_contrastes():
    with pytest.raises(ValueError, match="margin"):
        stats.detects_above_far([0.5] * 10, [0.01] * 10, margin=-0.01)
    with pytest.raises(ValueError, match="nivel de confianza"):
        stats.detects_above_far([0.5] * 10, [0.01] * 10, margin=0.0, alpha=0.99)


def test_nivel_del_suelo_monte_carlo():
    """Con el exceso centrado en el margen (H0 en el borde), detecta en falso como mucho alpha."""
    rng = np.random.default_rng(32)
    reps, n, margen = 400, 20, 4 / 800
    falsas = 0
    for _ in range(reps):
        far = rng.integers(0, 16, n) / 800
        exceso = margen + rng.integers(-6, 7, n) / 800       # simetrico en el margen, con empates
        falsas += stats.detects_above_far(far + exceso, far, margin=margen)["detects_above_far"]
    assert falsas / reps <= _cota_mc(ALPHA, reps)


def _familia_con_suelo(seed, como_dict):
    """Seis comparaciones sin ventaja; el AE detecta en 9 y 15 y no en 3; los baselines, nunca."""
    rng = np.random.default_rng(seed)
    comps = []
    for f in (3, 9, 15):
        for b in ("pca", "dpca"):
            far = rng.integers(0, 16, 20) / 800
            det_ae = stats.detects_above_far(
                far + (0.0 if f == 3 else 0.2) + rng.integers(-4, 5, 20) / 800, far, margin=0.01)
            det_b = stats.detects_above_far(far + rng.integers(-4, 5, 20) / 800, far, margin=0.01)
            comps.append({
                "fault": f, "method": "ae", "baseline": b,
                "advantage": rng.normal(0.0, 0.01, 20),
                "detects_method": det_ae if como_dict else det_ae["detects_above_far"],
                "detects_baseline": det_b if como_dict else det_b["detects_above_far"],
            })
    return comps


_CONTROL = [{"fault": 10, "method": "ae", "baseline": "pca",
             "detects_method": True, "detects_baseline": True}]


def test_veredicto_acepta_el_dict_del_suelo_tal_cual():
    a = stats.hypothesis_verdict(_familia_con_suelo(33, True), delta=0.05, controls=_CONTROL)
    b = stats.hypothesis_verdict(_familia_con_suelo(33, False), delta=0.05, controls=_CONTROL)
    assert a["verdict"] == b["verdict"] == stats.VERDICT_INCONCLUSIVE    # el fallo 3, en el suelo
    comunes = [c for c in b["table"].columns if not c.startswith("p_detects")]
    pd.testing.assert_frame_equal(a["table"][comunes], b["table"][comunes])
    conf = a["table"]["role"] == "confirmatoria"
    assert a["table"].loc[conf, "p_detects_method"].notna().all()
    assert b["table"]["p_detects_method"].isna().all()
    assert bool(a["table"].loc[0, "floor"])


def test_detects_puede_devolver_el_dict():
    rng = np.random.default_rng(34)
    far = rng.integers(0, 16, 20) / 800
    ruido = rng.integers(-4, 5, 20) / 800

    def detecta(metodo, fallo):
        return stats.detects_above_far(far + (0.2 if metodo == "ae" else 0.0) + ruido, far,
                                       margin=0.01)

    comps = [{"fault": f, "method": "ae", "baseline": b, "advantage": rng.normal(0, 0.01, 20)}
             for f in (3, 9, 15) for b in ("pca", "dpca")]
    out = stats.hypothesis_verdict(comps, delta=0.05, controls=_CONTROL, detects=detecta)
    assert out["verdict"] == stats.VERDICT_SUPPORTS
    assert (out["table"]["p_detects_method"].iloc[:6] <= ALPHA).all()


def test_dict_sin_la_decision_sigue_siendo_un_error():
    comps = _familia_con_suelo(35, False)
    comps[0]["detects_method"] = {"p_value": 0.01}
    with pytest.raises(ValueError, match="detects_above_far"):
        stats.hypothesis_verdict(comps, delta=0.05, controls=_CONTROL)


# -------------------------------------- 2. replicas y familias en compare_methods

_COLUMNAS_DE_SIEMPRE = [
    "faultNumber", "method_a", "method_b", "n", "statistic", "p_value", "effect_size",
    "median_difference", "hl_estimate", "ci_low", "ci_high", "ci_level", "n_nonzero",
    "p_method", "alternative", "alpha", "p_adjusted", "significant",
]


def _tabla_fallos(seed):
    rng = np.random.default_rng(seed)
    filas = []
    for fault in (1, 2):
        for run in range(1, 21):
            base = rng.integers(300, 500) / 800
            for m, extra in (("pca", 0), ("ae", 40), ("knn", 2)):
                filas.append({"faultNumber": fault, "simulationRun": run, "method": m,
                              "fdr": base + (extra + rng.integers(-6, 7)) / 800})
    return pd.DataFrame(filas)


def _tabla_replicas(seed, R=8, N=10, efecto=0.0, sd_replica=0.01):
    """Brazos limpio y sucio, R replicas de ajuste por N simulaciones de prueba.

    La diferencia sucio - limpio tiene un efecto comun a toda la replica (sd_replica)
    mas ruido de celda: las celdas de una replica no son independientes.
    """
    rng = np.random.default_rng(seed)
    rep = np.repeat(np.arange(1, R + 1), N)
    run = np.tile(np.arange(1, N + 1), R)
    base = rng.uniform(0.01, 0.03, R * N)
    dif = efecto + np.repeat(rng.normal(0, sd_replica, R), N) + rng.normal(0, 0.002, R * N)
    limpio = pd.DataFrame({"replica": rep, "simulationRun": run, "arm": "limpio", "far": base})
    return pd.concat([limpio, limpio.assign(arm="sucio", far=base + dif)], ignore_index=True)


def test_compare_methods_por_defecto_no_cambia():
    """Mismas columnas que antes, el p de paired_test y Holm dentro de cada fallo."""
    df = _tabla_fallos(41)
    t = stats.compare_methods(df, metric="fdr", baseline="pca")
    assert list(t.columns) == _COLUMNAS_DE_SIEMPRE
    for f, g in t.groupby("faultNumber"):
        sub = df[df["faultNumber"] == f]
        base = sub[sub["method"] == "pca"].set_index("simulationRun")["fdr"]
        for _, fila in g.iterrows():
            otro = sub[sub["method"] == fila["method_a"]].set_index("simulationRun")["fdr"]
            p = stats.paired_test(otro.to_numpy(), base.loc[otro.index].to_numpy())["p_value"]
            assert fila["p_value"] == p
        esperado = stats.holm_correction(g["p_value"].to_numpy())["p_adjusted"].to_numpy()
        assert np.array_equal(g["p_adjusted"].to_numpy(), esperado)


def test_replicas_contrastan_la_media_de_cada_replica():
    df = _tabla_replicas(42)
    t = stats.compare_methods(df, metric="far", method_col="arm", fault_col=None,
                              baseline="limpio", replicate_col="replica", unit="replicate")
    medias = df.pivot_table(index="replica", columns="arm", values="far", aggfunc="mean")
    esperado = stats.paired_test(medias["sucio"].to_numpy(), medias["limpio"].to_numpy())
    fila = t.iloc[0]
    assert (fila["n"], fila["n_cells"], fila["unit"]) == (8, 80, "replica")
    assert fila["p_value"] == pytest.approx(esperado["p_value"])
    assert fila["hl_estimate"] == pytest.approx(esperado["hl_estimate"])


def test_replicas_quitan_las_celdas_con_nan():
    df = _tabla_replicas(43)
    quitar = (df["arm"] == "sucio") & (df["replica"] == 1) & (df["simulationRun"] <= 3)
    df.loc[quitar, "far"] = np.nan
    t = stats.compare_methods(df, metric="far", method_col="arm", fault_col=None,
                              baseline="limpio", replicate_col="replica", unit="replicate")
    assert t["n_cells"].iloc[0] == 77 and t["n"].iloc[0] == 8


def test_replicas_evitan_la_pseudorreplicacion_monte_carlo():
    """Con efecto de replica y sin efecto medio, por replica se rechaza como mucho alpha;
    tratando las R x N celdas como independientes, mucho mas."""
    rng = np.random.default_rng(44)
    reps = 80
    df = pd.concat([_tabla_replicas(int(rng.integers(1 << 30)), R=10, N=20, sd_replica=0.02)
                    .assign(faultNumber=k) for k in range(reps)], ignore_index=True)
    por_replica = stats.compare_methods(df, metric="far", method_col="arm", baseline="limpio",
                                        replicate_col="replica", unit="replicate")
    assert (por_replica["p_value"] <= ALPHA).mean() <= _cota_mc(ALPHA, reps)
    celdas = df.assign(simulationRun=df["replica"] * 1000 + df["simulationRun"])
    ingenuo = stats.compare_methods(celdas, metric="far", method_col="arm", baseline="limpio")
    assert (ingenuo["p_value"] <= ALPHA).mean() > 3 * ALPHA


def test_familia_unica_con_todos_los_fallos():
    t = stats.compare_methods(_tabla_fallos(45), metric="fdr", baseline="pca", family=())
    esperado = stats.holm_correction(t["p_value"].to_numpy())["p_adjusted"].to_numpy()
    assert np.array_equal(t["p_adjusted"].to_numpy(), esperado)
    assert (t["family_m"] == 4).all()


def test_familia_por_comparacion_a_traves_de_los_fallos():
    t = stats.compare_methods(_tabla_fallos(46), metric="fdr", baseline="pca",
                              family=["method_a"])
    for _, g in t.groupby("method_a"):
        esperado = stats.holm_correction(g["p_value"].to_numpy())["p_adjusted"].to_numpy()
        assert np.array_equal(g["p_adjusted"].to_numpy(), esperado)
        assert (g["family_m"] == 2).all()


def test_familia_por_defecto_es_cada_fallo():
    df = _tabla_fallos(47)
    a = stats.compare_methods(df, metric="fdr", baseline="pca")
    b = stats.compare_methods(df, metric="fdr", baseline="pca", family=["faultNumber"])
    assert np.array_equal(a["p_adjusted"].to_numpy(), b["p_adjusted"].to_numpy())
    assert "family_m" not in a and (b["family_m"] == 2).all()


def test_familia_con_columna_desconocida_es_un_error():
    with pytest.raises(ValueError, match="family"):
        stats.compare_methods(_tabla_fallos(48), metric="fdr", baseline="pca",
                              family=["metric"])


# ---------------------------------------- 3. efecto frente a un margen de equivalencia

def test_efecto_con_margen_cero_es_el_wilcoxon_bilateral():
    rng = np.random.default_rng(51)
    for _ in range(100):
        n = int(rng.integers(5, 50))
        d = rng.integers(-6, 9, n) / 800
        r = stats.effect_test(d, margin=0.0)
        with _sin_avisos_de_potencia():
            p = stats.paired_test(0.5 + d, np.full(n, 0.5))["p_value"]
        assert r["p_relevant"] == p == r["p_value"]
        assert r["outcome"] != stats.EFFECT_EQUIVALENT


def test_equivalencia_es_la_de_dos_no_superioridades():
    """p_equivalent = el mayor de 'apoya' frente a +delta (d) y frente a -delta (-d)."""
    rng = np.random.default_rng(52)
    for _ in range(100):
        n = int(rng.integers(5, 50))
        d = rng.integers(-10, 11, n) / 800
        delta = int(rng.integers(0, 12)) / 800
        r = stats.effect_test(d, margin=delta)
        with _sin_avisos_de_potencia():
            arriba = stats.non_superiority_test(d, delta=delta)["p_below_delta"]
            abajo = stats.non_superiority_test(-d, delta=delta)["p_below_delta"]
        assert r["p_equivalent"] == max(arriba, abajo)


def test_tres_desenlaces_y_el_signo():
    rng = np.random.default_rng(53)
    sube = stats.effect_test(0.05 + rng.normal(0, 0.005, 30), margin=0.01)
    baja = stats.effect_test(-0.05 + rng.normal(0, 0.005, 30), margin=0.01)
    nulo = stats.effect_test(rng.normal(0, 0.002, 30), margin=0.01)
    dudoso = stats.effect_test(0.01 + rng.normal(0, 0.05, 30), margin=0.01)
    assert sube["outcome"] == baja["outcome"] == stats.EFFECT_RELEVANT
    assert sube["hl_estimate"] > 0 > baja["hl_estimate"]          # el signo no se presupone
    assert nulo["outcome"] == stats.EFFECT_EQUIVALENT
    assert dudoso["outcome"] == stats.EFFECT_INCONCLUSIVE


def test_efecto_real_y_equivalente_a_cero_a_la_vez():
    """Con muchas unidades, un efecto pequeno es significativo y aun asi equivalente a cero."""
    rng = np.random.default_rng(54)
    r = stats.effect_test(0.002 + rng.normal(0, 0.002, 200), margin=0.01)
    assert r["p_value"] <= ALPHA
    assert r["outcome"] == stats.EFFECT_EQUIVALENT


def test_efecto_decision_e_intervalos_coinciden_sin_empates():
    """Sin empates: relevante <=> intervalo al 1 - alpha fuera de [-delta, delta];
    equivalente <=> intervalo al 1 - 2 alpha dentro de (-delta, delta)."""
    rng = np.random.default_rng(56)
    for _ in range(150):
        n = int(rng.integers(6, 40))
        d = rng.normal(rng.uniform(-0.1, 0.1), rng.uniform(0.005, 0.08), n)
        delta = rng.uniform(0.0, 0.06)
        r = stats.effect_test(d, margin=delta)
        relevante = (r["ci_low"] > delta) or (r["ci_high"] < -delta)
        equivalente = (r["eq_ci_low"] > -delta) and (r["eq_ci_high"] < delta)
        assert (r["outcome"] == stats.EFFECT_RELEVANT) == relevante
        assert (r["outcome"] == stats.EFFECT_EQUIVALENT) == equivalente


def test_efecto_margen_obligatorio():
    with pytest.raises(TypeError):
        stats.effect_test(np.zeros(10))
    for malo in (-0.01, None, np.nan):
        with pytest.raises(ValueError, match="margin"):
            stats.effect_test(np.zeros(10), margin=malo)


def test_efecto_pocas_unidades_no_contrastable():
    r = stats.effect_test([0.1, 0.2, 0.3, 0.4], margin=0.01)
    assert r["outcome"] == stats.OUTCOME_UNTESTABLE and np.isnan(r["p_relevant"])


def test_efecto_nivel_en_el_borde_monte_carlo():
    """Con theta = delta: 'efecto relevante' en falso como mucho alpha/2 (un lado del
    bilateral) y 'equivalente a cero' en falso como mucho alpha."""
    rng = np.random.default_rng(55)
    reps, n, delta = 300, 20, 0.5
    relevantes = equivalentes = 0
    for _ in range(reps):
        r = stats.effect_test(delta + rng.standard_t(5, n), margin=delta)
        relevantes += r["outcome"] == stats.EFFECT_RELEVANT
        equivalentes += r["outcome"] == stats.EFFECT_EQUIVALENT
    assert relevantes / reps <= _cota_mc(ALPHA / 2, reps)
    assert equivalentes / reps <= _cota_mc(ALPHA, reps)


def _tabla_brazos(seed, n=20, efectos=None):
    """Dos estadisticos x tres brazos (limpio, s1, s2) sobre las mismas simulaciones.

    efectos: {(estadistico, brazo): theta} en la FAR; la FDR no cambia entre brazos.
    """
    efectos = efectos or {}
    rng = np.random.default_rng(seed)
    trozos = []
    for est in ("t2", "spe"):
        far = rng.uniform(0.005, 0.02, n)
        fdr = rng.uniform(0.6, 0.9, n)
        for brazo in ("limpio", "s1", "s2"):
            if brazo == "limpio":
                far_b, fdr_b = far, fdr
            else:
                far_b = far + efectos.get((est, brazo), 0.0) + rng.normal(0, 0.001, n)
                fdr_b = fdr + rng.normal(0, 0.001, n)
            trozos.append(pd.DataFrame({"statistic": est, "arm": brazo,
                                        "simulationRun": np.arange(1, n + 1),
                                        "far": far_b, "fdr": fdr_b}))
    return pd.concat(trozos, ignore_index=True)


def test_compare_arms_una_comparacion_es_effect_test():
    df = _tabla_brazos(61, efectos={("t2", "s1"): 0.02})
    sub = df[(df["statistic"] == "t2") & df["arm"].isin(["limpio", "s1"])]
    t = stats.compare_arms(sub, "far", "arm", "limpio", margin=0.005, family=(), by=())
    d = sub.loc[sub["arm"] == "s1", "far"].to_numpy() - sub.loc[sub["arm"] == "limpio", "far"].to_numpy()
    r = stats.effect_test(d, margin=0.005)
    assert len(t) == 1 and t["family_m"].iloc[0] == 1
    for k in ("p_value", "p_relevant", "p_equivalent", "hl_estimate", "ci_low", "eq_ci_high"):
        assert t[k].iloc[0] == pytest.approx(r[k])
    assert t["outcome"].iloc[0] == r["outcome"] == stats.EFFECT_RELEVANT


def test_compare_arms_holm_sobre_la_familia_declarada():
    df = _tabla_brazos(62, efectos={("t2", "s1"): 0.02, ("spe", "s2"): -0.02})
    t = stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, family=(),
                           by=("statistic",))
    assert len(t) == 4 and (t["family_m"] == 4).all()      # 2 estadisticos x 2 brazos sucios
    for p, ajustado in (("p_value", "p_adjusted"), ("p_relevant", "p_relevant_holm"),
                        ("p_equivalent", "p_equivalent_holm")):
        esperado = stats.holm_correction(t[p].to_numpy())["p_adjusted"].to_numpy()
        assert np.allclose(t[ajustado].to_numpy(), esperado)
    fila = t[(t["statistic"] == "spe") & (t["arm"] == "s2")].iloc[0]
    assert fila["hl_estimate"] < 0 and fila["outcome"] == stats.EFFECT_RELEVANT
    por_estadistico = stats.compare_arms(df, "far", "arm", "limpio", margin=0.005,
                                         family=["statistic"], by=("statistic",))
    assert (por_estadistico["family_m"] == 2).all()


def test_compare_arms_varias_metricas_cada_una_con_su_margen():
    df = _tabla_brazos(63, efectos={("t2", "s1"): 0.02})
    t = stats.compare_arms(df, ["far", "fdr"], "arm", "limpio",
                           margin={"far": 0.005, "fdr": 0.02}, family=["statistic"],
                           by=("statistic",))
    assert set(t["metric"]) == {"far", "fdr"} and (t["family_m"] == 4).all()
    assert set(t.loc[t["metric"] == "fdr", "margin"]) == {0.02}
    with pytest.raises(ValueError, match="dict"):
        stats.compare_arms(df, ["far", "fdr"], "arm", "limpio", margin=0.005, family=(),
                           by=("statistic",))


def test_compare_arms_familia_y_margen_sin_valor_por_defecto():
    df = _tabla_brazos(64)
    with pytest.raises(TypeError):
        stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, by=("statistic",))
    with pytest.raises(TypeError):
        stats.compare_arms(df, "far", "arm", "limpio", family=(), by=("statistic",))
    with pytest.raises(ValueError, match="family"):
        stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, family=None,
                           by=("statistic",))


def test_compare_arms_nan_en_los_grupos_es_un_error():
    df = _tabla_brazos(67)
    df.loc[0, "statistic"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, family=(),
                           by=("statistic",))


def test_compare_arms_con_replicas_de_ajuste():
    df = _tabla_replicas(65, R=8, N=10, efecto=0.02, sd_replica=0.003)
    t = stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, family=(), by=(),
                           replicate_col="replica", unit="replicate")
    fila = t.iloc[0]
    assert (fila["n"], fila["n_cells"], fila["unit"]) == (8, 80, "replica")
    assert fila["outcome"] == stats.EFFECT_RELEVANT


def test_compare_arms_no_contrastable_cuenta_en_la_familia():
    df = _tabla_brazos(66, efectos={("t2", "s1"): 0.02})
    df = df[(df["statistic"] == "t2") | (df["simulationRun"] <= 3)]   # spe con 3 simulaciones
    with pytest.warns(LowPowerWarning):
        t = stats.compare_arms(df, "far", "arm", "limpio", margin=0.005, family=(),
                               by=("statistic",))
    assert (t.loc[t["statistic"] == "spe", "outcome"] == stats.OUTCOME_UNTESTABLE).all()
    assert (t["family_m"] == 4).all()
    t2 = t[t["statistic"] == "t2"]
    esperado = stats.holm_correction(t2["p_relevant"].tolist() + [1.0, 1.0])["p_adjusted"]
    assert np.allclose(t2["p_relevant_holm"].to_numpy(), esperado.to_numpy()[:2])


# ------------------------------------------------------ 4. macro-F1 por replica

def _diagnostico_por_replica(seed, R=4, clases=(0, 1, 3, 5), por_clase=30, acierto=0.8):
    rng = np.random.default_rng(seed)
    yt = np.tile(np.repeat(clases, por_clase), R)
    rep = np.repeat(np.arange(1, R + 1), len(clases) * por_clase)
    yp = np.where(rng.uniform(size=yt.size) < acierto, yt, rng.choice(clases, yt.size))
    return yt, yp, rep


def test_macro_f1_por_replica_es_la_de_cada_replica():
    yt, yp, rep = _diagnostico_por_replica(71)
    t = metrics.macro_f1_by_replicate(yt, yp, rep, include_normal=False)
    assert t["replicate"].tolist() == [1, 2, 3, 4]
    for _, fila in t.iterrows():
        m = rep == fila["replicate"]
        dm = metrics.diagnosis_metrics(yt[m], yp[m], include_normal=False, labels=np.unique(yt))
        assert fila["macro_f1"] == dm["macro_f1"]
        assert fila["macro_f1_hard"] == dm["macro_f1_hard"]
        assert fila["n_samples"] == m.sum() and not fila["include_normal"]


def test_macro_f1_por_replica_exige_todas_las_clases_en_cada_replica():
    yt, yp, rep = _diagnostico_por_replica(72)
    quitar = (rep == 2) & (yt == 5)
    with pytest.raises(ValueError, match="replica 2"):
        metrics.macro_f1_by_replicate(yt[~quitar], yp[~quitar], rep[~quitar],
                                      include_normal=False)


def test_macro_f1_por_replica_da_pares_al_contraste():
    """Dos brazos evaluados en las mismas 8 replicas: compare_methods tiene 8 pares."""
    tablas = []
    for semilla, brazo, acierto in ((73, "por_simulacion", 0.7), (74, "por_muestra", 0.9)):
        yt, yp, rep = _diagnostico_por_replica(semilla, R=8, acierto=acierto)
        tablas.append(metrics.macro_f1_by_replicate(yt, yp, rep, include_normal=True)
                      .assign(method=brazo))
    t = stats.compare_methods(pd.concat(tablas, ignore_index=True), metric="macro_f1",
                              run_col="replicate", fault_col=None, baseline="por_simulacion")
    assert t["n"].iloc[0] == 8 and t["hl_estimate"].iloc[0] > 0


def test_macro_f1_por_replica_include_normal_obligatorio():
    yt, yp, rep = _diagnostico_por_replica(75)
    with pytest.raises(TypeError):
        metrics.macro_f1_by_replicate(yt, yp, rep)


def test_macro_f1_por_replica_ids_int16():
    yt, yp, rep = _diagnostico_por_replica(76)
    t = metrics.macro_f1_by_replicate(yt, yp, rep.astype(np.int16), include_normal=True)
    assert t["replicate"].dtype == np.int64


# ------------------------------------------------------- 5. ventaja publicada

def _fila_publicada(valor=0.10, comparacion=stats.COMPARISON_WITHIN, metrica="fdr"):
    return {"article": "Articulo inventado (2000)", "metric": metrica, "method": "ae",
            "baseline": "pca", "value": valor, "page": "p. 1, Tabla 1",
            "comparison": comparacion}


def test_ventaja_menor_que_la_publicada():
    rng = np.random.default_rng(81)
    r = stats.published_advantage_test(0.02 + rng.normal(0, 0.01, 30), _fila_publicada(),
                                       metric="fdr")
    assert r["outcome"] == stats.PUBLISHED_LESS and r["ci_high"] < 0.10
    assert r["published_minus_measured"] == pytest.approx(0.10 - r["hl_estimate"])
    assert (r["gap_ci_low"], r["gap_ci_high"]) == pytest.approx((0.10 - r["ci_high"],
                                                                  0.10 - r["ci_low"]))


def test_ventaja_igual_o_mayor_que_la_publicada():
    rng = np.random.default_rng(82)
    r = stats.published_advantage_test(0.20 + rng.normal(0, 0.01, 30), _fila_publicada(),
                                       metric="fdr")
    assert r["outcome"] == stats.PUBLISHED_NOT_LESS and r["ci_low"] > 0.10


def test_ventaja_frente_a_la_publicada_inconclusa():
    rng = np.random.default_rng(83)
    r = stats.published_advantage_test(0.10 + rng.normal(0, 0.3, 30), _fila_publicada(),
                                       metric="fdr")
    assert r["outcome"] == stats.PUBLISHED_INCONCLUSIVE


def test_ventaja_publicada_es_la_no_superioridad_con_delta_la_cifra():
    rng = np.random.default_rng(84)
    for _ in range(60):
        n = int(rng.integers(5, 50))
        d = rng.integers(-10, 40, n) / 800
        valor = int(rng.integers(0, 30)) / 800
        r = stats.published_advantage_test(d, _fila_publicada(valor), metric="fdr")
        with _sin_avisos_de_potencia():
            ns = stats.non_superiority_test(d, delta=valor)
        assert r["p_below_published"] == ns["p_below_delta"]
        assert r["p_above_published"] == ns["p_above_delta"]


def test_ventaja_publicada_negativa_se_admite():
    rng = np.random.default_rng(85)
    r = stats.published_advantage_test(-0.10 + rng.normal(0, 0.005, 20),
                                       _fila_publicada(valor=-0.05), metric="fdr")
    assert r["outcome"] == stats.PUBLISHED_LESS


def test_comparacion_entre_articulos_se_declara():
    rng = np.random.default_rng(86)
    r = stats.published_advantage_test(rng.normal(0, 0.01, 20),
                                       _fila_publicada(comparacion=stats.COMPARISON_BETWEEN),
                                       metric="fdr")
    assert r["comparison"] == stats.COMPARISON_BETWEEN and "entre articulos" in r["note"]
    with pytest.raises(ValueError, match="comparison"):
        stats.published_advantage_test(rng.normal(0, 0.01, 20),
                                       _fila_publicada(comparacion="otro"), metric="fdr")


def test_ventaja_publicada_en_otra_metrica_es_un_error():
    with pytest.raises(ValueError, match="misma metrica"):
        stats.published_advantage_test(np.zeros(20), _fila_publicada(metrica="accuracy"),
                                       metric="fdr")


def test_tabla_de_ventajas_sin_columnas_obligatorias_es_un_error():
    fila = _fila_publicada()
    del fila["page"]
    with pytest.raises(ValueError, match="page"):
        stats.published_advantage_test(np.zeros(20), fila, metric="fdr")
    assert set(stats.PUBLISHED_TABLE_COLUMNS) == set(_fila_publicada())


def test_nivel_frente_a_la_publicada_monte_carlo():
    """Con la ventaja centrada en la cifra publicada, cada desenlace en falso como mucho alpha."""
    rng = np.random.default_rng(87)
    reps, n, valor = 400, 20, 24 / 800
    menores = mayores = 0
    for _ in range(reps):
        d = valor + rng.integers(-8, 9, n) / 800           # simetrica en la cifra, con empates
        r = stats.published_advantage_test(d, _fila_publicada(valor), metric="fdr")
        menores += r["outcome"] == stats.PUBLISHED_LESS
        mayores += r["outcome"] == stats.PUBLISHED_NOT_LESS
    assert menores / reps <= _cota_mc(ALPHA, reps)
    assert mayores / reps <= _cota_mc(ALPHA, reps)


# ------------------------------------------------------- 6. mejor compromiso

def _candidatos():
    return pd.DataFrame({"far": [0.010, 0.012, 0.030, 0.011],
                         "fdr": [0.70, 0.71, 0.90, 0.60],
                         "delay": [12.0, 8.0, 5.0, 20.0]},
                        index=["pca", "dpca", "ae", "knn"])


def test_frente_de_pareto_construido():
    t = stats.pareto_front(_candidatos(), {"far": "min", "fdr": "max"})
    assert t["pareto"].to_dict() == {"pca": True, "dpca": True, "ae": True, "knn": False}
    assert t.loc["knn", "dominated_by"] == ["pca"]


def test_pareto_con_tolerancia_no_saca_a_nadie_por_diferencias_menores():
    tabla = pd.DataFrame({"far": [0.010, 0.0102, 0.030, 0.011], "fdr": [0.70, 0.695, 0.90, 0.60]},
                         index=["pca", "pca_bis", "ae", "knn"])
    estricto = stats.pareto_front(tabla, {"far": "min", "fdr": "max"})
    assert not estricto.loc["pca_bis", "pareto"]
    tolerante = stats.pareto_front(tabla, {"far": "min", "fdr": "max"},
                                   tolerance={"far": 0.005, "fdr": 0.02})
    assert tolerante["pareto"].to_dict() == {"pca": True, "pca_bis": True, "ae": True,
                                             "knn": False}


def test_pareto_direccion_explicita():
    with pytest.raises(TypeError):
        stats.pareto_front(_candidatos())
    with pytest.raises(ValueError, match="direccion"):
        stats.pareto_front(_candidatos(), {"far": "menos"})
    with pytest.raises(ValueError):
        stats.pareto_front(_candidatos(), ["far", "fdr"])


def test_pareto_nan_es_un_error():
    tabla = _candidatos()
    tabla.loc["ae", "far"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        stats.pareto_front(tabla, {"far": "min", "fdr": "max"})


def test_compromiso_tope_de_far_orden_y_empates():
    out = stats.best_compromise(_candidatos(), constraints=[("far", "<=", 0.02)],
                                order=[("fdr", "max"), ("delay", "min")],
                                ties={"fdr": 0.02, "delay": 0.0})
    assert out["selected"] == ["dpca"]
    t = out["table"]
    assert not t.loc["ae", "meets_constraints"] and "far" in t.loc["ae", "eliminated_at"]
    assert t.loc["knn", "eliminated_at"].startswith("paso 1")
    assert t.loc["pca", "eliminated_at"].startswith("paso 2")
    assert t["selected"].tolist() == [False, True, False, False]


def test_compromiso_compartido():
    out = stats.best_compromise(_candidatos(), constraints=[("far", "<=", 0.02)],
                                order=[("fdr", "max"), ("delay", "min")],
                                ties={"fdr": 0.02, "delay": 5.0})
    assert sorted(out["selected"]) == ["dpca", "pca"]
    assert "compartido" in out["explanation"]


def test_compromiso_sin_criterio_por_defecto():
    with pytest.raises(TypeError):
        stats.best_compromise(_candidatos())
    with pytest.raises(TypeError):
        stats.best_compromise(_candidatos(), constraints=[], order=[("fdr", "max")])
    with pytest.raises(ValueError, match="ties"):
        stats.best_compromise(_candidatos(), constraints=[],
                              order=[("fdr", "max"), ("delay", "min")], ties={"fdr": 0.0})


def test_compromiso_con_regla_de_empate_que_es_una_funcion():
    llamadas = []

    def empate(columna, mejor, otro):
        llamadas.append((columna, mejor, otro))
        return otro == "knn"

    out = stats.best_compromise(_candidatos(), constraints=[], order=[("fdr", "max")],
                                ties={"fdr": empate})
    assert out["selected"] == ["ae", "knn"]
    assert ("fdr", "ae", "pca") in llamadas


def test_compromiso_nan_solo_importa_si_llega_al_paso():
    tabla = _candidatos()
    tabla.loc["ae", "delay"] = np.nan                      # ae no pasa el tope de FAR
    out = stats.best_compromise(tabla, constraints=[("far", "<=", 0.02)],
                                order=[("fdr", "max"), ("delay", "min")],
                                ties={"fdr": 0.02, "delay": 0.0})
    assert out["selected"] == ["dpca"]
    with pytest.raises(ValueError, match="NaN"):
        stats.best_compromise(tabla, constraints=[], order=[("delay", "min")],
                              ties={"delay": 0.0})


def test_compromiso_sin_candidatos_que_cumplan():
    out = stats.best_compromise(_candidatos(), constraints=[("far", "<", 0.001)],
                                order=[("fdr", "max")], ties={"fdr": 0.0})
    assert out["selected"] == [] and "Ningun" in out["explanation"]
