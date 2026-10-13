"""Pruebas de los mecanismos de las ablaciones (tfmfdd/ablation.py) y del brazo S2 en PCAMonitor.score.

Cada bloque protege un mecanismo que las tres familias tienen que construir igual:
- factor 3: reparto limpio por simulacion y sucio por muestra o por bloques, con
  las mismas filas y los mismos tamanos (H3 y H10 de la trazabilidad; DIS-8),
- factor 2, brazo S2: cada simulacion de prueba estandarizada con sus propios
  estadisticos, con el mismo modelo que el brazo limpio (H5),
- factor 4: bootstrap por bloques de una simulacion, cobertura, conclusiones
  contrarias y submuestreo de simulaciones de ajuste (H4; DIS-9),
- ventanas por trayectoria y su alineado con las muestras originales (H11).

Todas usan datos sinteticos con semilla fija, salvo una que lee de Rieth las
simulaciones 1 y 2 de dos ficheros (ya miradas, regla 3) y se salta si no estan.
Los umbrales de las pruebas Monte Carlo se declaran en su docstring; no son
resultados del TFM.
"""

import numpy as np
import pytest

from tfmfdd import ablation, data, metrics
from tfmfdd.pca import PCAMonitor
from tfmfdd.preprocessing import Scaler, lag_by_run


# ------------------------------------------------------------- utilidades

def _ids(n_tray, n):
    """simulationRun de n_tray trayectorias de n muestras seguidas (ids 1..n_tray)."""
    return np.repeat(np.arange(1, n_tray + 1), n)


def _ar1(rng, n_tray, n, phi):
    """n_tray series AR(1) estacionarias de n muestras, media 0 (una por fila)."""
    e = rng.standard_normal((n_tray, n))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0] / np.sqrt(1.0 - phi**2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x


def _trayectorias(rng, n_tray, n, phi=0.8):
    """Datos de proceso con dinamica: dos factores AR(1) vistos en 6 variables, apilados."""
    A = np.array([[1.0, 0.8, 0.6, 0.0, 0.0, 0.3], [0.0, 0.5, -0.4, 1.0, 0.7, 0.2]])
    partes = []
    for _ in range(n_tray):
        f = np.zeros((n, 2))
        e = rng.standard_normal((n, 2))
        for t in range(1, n):
            f[t] = phi * f[t - 1] + e[t]
        partes.append(f @ A + 0.3 * rng.standard_normal((n, 6)))
    return np.vstack(partes)


def _misma_tray(runs):
    """True en t si las filas t y t + 1 son de la misma trayectoria."""
    runs = np.asarray(runs)
    if runs.ndim == 1:
        return runs[1:] == runs[:-1]
    return (runs[1:] == runs[:-1]).all(axis=1)


def _vecinas_entre(runs, a, b):
    """Pares de muestras consecutivas de una misma trayectoria con una en a y la otra en b."""
    cruza = (a[:-1] & b[1:]) | (b[:-1] & a[1:])
    return int((_misma_tray(runs) & cruza).sum())


def _parte(r):
    """0, 1 o 2 por fila segun el RowSplit; -1 si no esta en ninguna parte."""
    return np.select([r.fit, r.cal, r.eval], [0, 1, 2], -1)


# ------------------------------------------------- factor 3: el brazo limpio

def test_reparto_por_simulacion_no_parte_ninguna_simulacion():
    runs = _ids(10, 50)
    r = ablation.split_by_run(runs, (0.6, 0.2, 0.2), seed=3)
    parte = _parte(r)
    assert (parte >= 0).all()
    assert (r.fit.astype(int) + r.cal + r.eval == 1).all()
    for i in np.unique(runs):
        assert np.unique(parte[runs == i]).size == 1
    assert _vecinas_entre(runs, r.fit, r.eval) == 0
    assert _vecinas_entre(runs, r.fit, r.cal) == 0


def test_reparto_por_simulacion_igual_que_split_runs():
    """Con los mismos ids, las mismas fracciones y la misma semilla, el reparto del protocolo."""
    for n, fr, semilla in [(10, (0.6, 0.2, 0.2), 42), (37, (0.5, 0.3, 0.2), 7), (20, (0.7, 0.3, 0.0), 1)]:
        runs = _ids(n, 5)
        r = ablation.split_by_run(runs, fr, semilla)
        for mascara, ids in zip(r, data.split_runs(n, fr, semilla)):
            assert np.array_equal(np.unique(runs[mascara]), ids)
    # Con ids que no empiezan en 1: los que quedan tras exclude.
    fuera = [1, 2, 3, 10, 11]
    runs = np.repeat(np.setdiff1d(np.arange(1, 31), fuera), 4)
    r = ablation.split_by_run(runs, (0.6, 0.2, 0.2), 5)
    for mascara, ids in zip(r, data.split_runs(30, (0.6, 0.2, 0.2), 5, exclude=fuera)):
        assert np.array_equal(np.unique(runs[mascara]), ids)


def test_reparto_limpio_agrupa_solo_por_simulationrun():
    runs = _ids(6, 10)
    fallo = np.zeros_like(runs)
    with pytest.raises(ValueError, match="gemelas"):
        ablation.split_by_run(np.column_stack([fallo, runs]), (0.6, 0.2, 0.2), 1)
    # Las fracciones las valida data.split_runs (correcciones del 12/10, V-08).
    with pytest.raises(ValueError, match="fraccion"):
        ablation.split_by_run(runs, (0.6, 0.2), 1)
    with pytest.raises(ValueError, match="fraccion"):
        ablation.split_by_run(runs, (0.6, 0.3, 0.3), 1)


# -------------------------------------------------- factor 3: el brazo sucio

def test_reparto_por_muestra_pone_vecinas_en_ajuste_y_evaluacion():
    """El sucio deja muestras vecinas de una misma simulacion a los dos lados; el limpio, ninguna.

    Con el 60 % de las filas en el ajuste, una muestra de evaluacion tiene alguna
    de sus dos vecinas en el ajuste con probabilidad 1 - 0,4^2 = 0,84 (sin contar
    los extremos de cada trayectoria); se exige mas de la mitad.
    """
    runs = _ids(10, 100)
    limpio = ablation.split_by_run(runs, (0.6, 0.2, 0.2), 5)
    sucio = ablation.split_by_sample(runs, limpio.sizes, 5, block_length=1)
    assert _vecinas_entre(runs, limpio.fit, limpio.eval) == 0
    assert _vecinas_entre(runs, sucio.fit, sucio.eval) > 0

    ev, ajuste = np.flatnonzero(sucio.eval), np.flatnonzero(sucio.fit)
    antes = np.isin(ev - 1, ajuste) & np.r_[False, _misma_tray(runs)][ev]
    despues = np.isin(ev + 1, ajuste) & np.r_[_misma_tray(runs), False][ev]
    assert (antes | despues).mean() > 0.5


def test_reparto_por_muestra_mismas_filas_y_tamanos_que_el_limpio():
    runs = _ids(12, 80)
    limpio = ablation.split_by_run(runs, (0.5, 0.25, 0.25), 9)
    sucio = ablation.split_by_sample(runs, limpio.sizes, 9, block_length=1)
    assert sucio.sizes == limpio.sizes
    assert (_parte(sucio) >= 0).all()
    assert (sucio.fit.astype(int) + sucio.cal + sucio.eval == 1).all()
    # Las mismas simulaciones, repartidas por dentro: todas tienen filas en el ajuste.
    assert set(np.unique(runs[sucio.fit])) == set(np.unique(runs))


def test_reparto_por_bloques_dosis_respuesta():
    """Con bloques mas largos, menos pares de vecinas quedan entre ajuste y evaluacion.

    Esperado (60 % y 20 %): con L = 1 cruza cada par consecutivo con probabilidad
    2 x 0,6 x 0,2; con L > 1 solo pueden cruzar las fronteras entre bloques, que
    son unas L veces menos. Con L igual a la trayectoria no cruza ninguno.
    """
    runs = _ids(20, 200)
    limpio = ablation.split_by_run(runs, (0.6, 0.2, 0.2), 11)
    cruces = []
    for largo in (1, 10, 50, 200):
        s = ablation.split_by_sample(runs, limpio.sizes, 11, block_length=largo)
        assert s.sizes == limpio.sizes
        cruces.append(_vecinas_entre(runs, s.fit, s.eval))
    assert cruces[0] > cruces[1] > cruces[2] > cruces[3] == 0


def test_reparto_por_bloques_corta_como_mucho_dos_bloques():
    runs = _ids(7, 103)                      # 103 no es multiplo de 10: bloques finales mas cortos
    limpio = ablation.split_by_run(runs, (0.5, 0.3, 0.2), 2)
    bloques = ablation.block_ids(runs, 10)
    parte = _parte(ablation.split_by_sample(runs, limpio.sizes, 2, block_length=10))
    partes_por_bloque = np.array([np.unique(parte[bloques == b]).size for b in np.unique(bloques)])
    assert (partes_por_bloque <= 2).all()
    assert (partes_por_bloque > 1).sum() <= 2


def test_bloques_no_cruzan_trayectorias_con_ids_compartidos():
    """Dos clases con ids compartidos, una trayectoria detras de otra con el mismo simulationRun.

    Con las dos columnas, ningun bloque mezcla clases; con solo simulationRun,
    las dos trayectorias seguidas con id 1 forman una sola (por eso el modulo
    pide faultNumber y simulationRun en Rieth).
    """
    fallo = np.repeat([0, 3, 0, 3], 25)
    sim = np.repeat([1, 1, 2, 2], 25)
    runs2 = np.column_stack([fallo, sim])
    bloques = ablation.block_ids(runs2, 7)
    for b in np.unique(bloques):
        filas = np.flatnonzero(bloques == b)
        assert len(filas) <= 7
        assert (np.diff(filas) == 1).all()
        assert len(np.unique(runs2[filas], axis=0)) == 1
    bloques_1d = ablation.block_ids(sim, 7)
    assert any(np.unique(fallo[bloques_1d == b]).size > 1 for b in np.unique(bloques_1d))


def test_bloque_de_trayectoria_entera_reparte_trayectorias():
    """Con L >= la trayectoria y tamanos multiplos, cada trayectoria cae entera en una parte."""
    runs = _ids(10, 60)
    limpio = ablation.split_by_run(runs, (0.6, 0.2, 0.2), 8)
    s = ablation.split_by_sample(runs, limpio.sizes, 8, block_length=1000)
    parte = _parte(s)
    for i in np.unique(runs):
        assert np.unique(parte[runs == i]).size == 1


def test_reparto_por_muestra_reproducible_con_semilla():
    runs = _ids(5, 40)
    tamanos = (120, 40, 40)
    a = ablation.split_by_sample(runs, tamanos, 21, block_length=4)
    b = ablation.split_by_sample(runs, tamanos, 21, block_length=4)
    c = ablation.split_by_sample(runs, tamanos, 22, block_length=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a.fit, c.fit)


def test_reparto_por_muestra_valida_entradas():
    runs = _ids(4, 25)
    limpio = ablation.split_by_run(runs, (0.5, 0.25, 0.25), 0)
    with pytest.raises(ValueError, match="split_by_run"):
        ablation.split_by_sample(runs, (50, 25, 20), 0, block_length=1)
    with pytest.raises(TypeError, match="sizes"):
        ablation.split_by_sample(runs, limpio, 0, block_length=1)
    for malo in (0, 1.5, True):
        with pytest.raises(ValueError, match="block_length"):
            ablation.split_by_sample(runs, limpio.sizes, 0, block_length=malo)
    with pytest.raises(TypeError):
        ablation.split_by_sample(runs, limpio.sizes, 0)        # sin block_length declarado


def test_bloques_como_grupos_de_validacion_cruzada():
    """La receta del OE2: GroupKFold con los bloques (sucio) o con simulationRun (limpio)."""
    from sklearn.model_selection import GroupKFold

    runs = _ids(6, 40)
    X = np.zeros((len(runs), 1))
    cv = GroupKFold(n_splits=4, shuffle=True, random_state=0)

    def vecinas_partidas(grupos):
        total = 0
        for tr, te in cv.split(X, groups=grupos):
            assert not set(grupos[tr]) & set(grupos[te])
            en_tr = np.zeros(len(runs), dtype=bool)
            en_tr[tr] = True
            total += _vecinas_entre(runs, en_tr, ~en_tr)
        return total

    assert vecinas_partidas(ablation.block_ids(runs, 5)) > 0
    assert vecinas_partidas(runs) == 0


def test_dpca_con_retardos_fuera_es_el_mismo_modelo():
    """La receta del brazo sucio con DPCA: apilar con lag_by_run y ajustar PCAMonitor(lags=0).

    Sin gancho del escalador, es el mismo modelo que PCAMonitor(lags=2): mismas
    cargas, mismos limites y, alineado con align_to_samples, los mismos
    estadisticos sobre la prueba.
    """
    rng = np.random.default_rng(4)
    X_fit, X_cal, X_te = (_trayectorias(rng, 3, 120) for _ in range(3))
    r_fit = r_cal = r_te = _ids(3, 120)
    kw = dict(n_components=3, t2_method="empirical", spe_method="empirical")
    dpca = PCAMonitor(lags=2, **kw).fit(X_fit, r_fit, X_cal=X_cal, runs_cal=r_cal)
    Xl, _, _ = lag_by_run(X_fit, r_fit, 2, return_index=True)
    Xlc, _, _ = lag_by_run(X_cal, r_cal, 2, return_index=True)
    pca = PCAMonitor(lags=0, **kw).fit(Xl, X_cal=Xlc)
    assert np.allclose(dpca.loadings_, pca.loadings_)
    assert dpca.t2_limit_ == pytest.approx(pca.t2_limit_)
    assert dpca.spe_limit_ == pytest.approx(pca.spe_limit_)
    assert dpca.n_fit_ == pca.n_fit_

    t2_d, spe_d = dpca.score(X_te, r_te)
    Xlt, _, pos = lag_by_run(X_te, r_te, 2, return_index=True)
    t2_p, spe_p = pca.score(Xlt)
    assert np.allclose(ablation.align_to_samples(t2_p, pos, len(X_te)), t2_d, equal_nan=True)
    assert np.allclose(ablation.align_to_samples(spe_p, pos, len(X_te)), spe_d, equal_nan=True)


# ------------------------------------------------ factor 2: el brazo S2

def _modelo_y_prueba(rng, lags=0, gancho=False):
    X_fit = _trayectorias(rng, 4, 150)
    X_cal = _trayectorias(rng, 2, 150)
    r_fit, r_cal = _ids(4, 150), _ids(2, 150)
    esc = Scaler().fit(X_fit) if gancho else None
    m = PCAMonitor(n_components=3, lags=lags, t2_method="empirical", spe_method="empirical").fit(
        X_fit, r_fit, X_cal=X_cal, runs_cal=r_cal, scaler=esc)
    X_te = _trayectorias(rng, 3, 200)
    return m, X_te, _ids(3, 200)


def test_score_sin_escalador_es_el_de_siempre():
    """score(X) no cambia, y pasarle su propio escalador da lo mismo (PCA sin gancho)."""
    m, X_te, _ = _modelo_y_prueba(np.random.default_rng(1))
    t2, spe = m.score(X_te)
    t2_b, spe_b = m.score(X_te, scaler=m.scaler_)
    assert np.array_equal(t2, t2_b) and np.array_equal(spe, spe_b)


def test_score_con_su_escalador_reproduce_el_dpca_con_gancho():
    m, X_te, r_te = _modelo_y_prueba(np.random.default_rng(2), lags=2, gancho=True)
    t2, spe = m.score(X_te, r_te)
    t2_b, spe_b = m.score(X_te, r_te, scaler=m.scaler_)
    assert np.allclose(t2, t2_b, equal_nan=True) and np.allclose(spe, spe_b, equal_nan=True)


def test_score_con_otro_escalador_no_cambia_el_modelo():
    """La sustitucion vale solo para esa llamada: cargas, limites y escalador siguen igual."""
    m, X_te, _ = _modelo_y_prueba(np.random.default_rng(3))
    escalador, cargas = m.scaler_, m.loadings_.copy()
    limites = (m.t2_limit_, m.spe_limit_)
    t2_antes, _ = m.score(X_te)
    t2_otro, _ = m.score(X_te, scaler=Scaler().fit(X_te))
    assert not np.allclose(t2_antes, t2_otro)
    assert m.scaler_ is escalador
    assert np.array_equal(m.loadings_, cargas)
    assert (m.t2_limit_, m.spe_limit_) == limites
    assert np.array_equal(m.score(X_te)[0], t2_antes)


def test_s2_cambia_los_estadisticos_y_resta_un_escalon_de_media():
    """El S2 usa el mismo modelo que el limpio y cambia los estadisticos de la prueba.

    Caso construido (no es un resultado sobre el TEP): un escalon de media de
    cuatro desviaciones en una variable, desde la muestra 160 hasta el final
    (800 de 960 muestras). Estandarizar la simulacion con su propia media resta
    casi todo el escalon, y la FDR del T2 baja. El signo depende del caso: por
    eso las ablaciones lo miden y no lo presuponen.
    """
    rng = np.random.default_rng(5)
    X_fit = _trayectorias(rng, 4, 300)
    X_cal = _trayectorias(rng, 2, 300)
    m = PCAMonitor(n_components=3, t2_method="empirical", spe_method="empirical").fit(
        X_fit, _ids(4, 300), X_cal=X_cal, runs_cal=_ids(2, 300))
    X_te = _trayectorias(rng, 2, 960)
    r_te = _ids(2, 960)
    con_fallo = (r_te == 2) & (np.tile(np.arange(960), 2) >= 160)
    X_te[con_fallo, 0] += 4.0 * X_fit[:, 0].std()

    t2_l, spe_l = m.score(X_te, r_te)
    t2_s, spe_s = ablation.score_standardized_by_run(m, X_te, r_te)
    assert not np.allclose(t2_l, t2_s)
    assert not np.allclose(spe_l[r_te == 1], spe_s[r_te == 1])     # tambien en la simulacion normal
    fdr_limpio = metrics.fdr(t2_l[con_fallo], m.t2_limit_)
    fdr_s2 = metrics.fdr(t2_s[con_fallo], m.t2_limit_)
    assert fdr_s2 < fdr_limpio


def test_estandarizar_por_simulacion_media_cero_y_desviacion_uno():
    rng = np.random.default_rng(6)
    fallo = np.repeat([0, 3, 0, 3], 40)
    sim = np.repeat([1, 1, 2, 2], 40)
    runs2 = np.column_stack([fallo, sim])
    X = rng.standard_normal((160, 4)) * [1.0, 5.0, 0.1, 1.0] + np.repeat([0.0, 10.0, -3.0, 2.0], 40)[:, None]
    X[:, 3] = 7.0                                     # variable constante: no divide por cero
    Z = ablation.standardize_by_run(X, runs2)
    for ini in range(0, 160, 40):
        bloque = Z[ini : ini + 40]
        assert np.allclose(bloque.mean(axis=0), 0.0, atol=1e-12)
        assert np.allclose(bloque[:, :3].std(axis=0, ddof=1), 1.0)
        assert np.allclose(bloque[:, 3], 0.0)
        assert np.allclose(bloque, Scaler().fit(X[ini : ini + 40]).transform(X[ini : ini + 40]))
    # Con solo simulationRun, las dos trayectorias seguidas con id 1 se estandarizan juntas.
    Z1 = ablation.standardize_by_run(X, sim)
    assert not np.allclose(Z1[:40].mean(axis=0)[:3], 0.0)
    # float32 (como llegan las variables de Rieth) se conserva.
    Z32 = ablation.standardize_by_run(X.astype(np.float32), runs2)
    assert Z32.dtype == np.float32
    assert np.allclose(Z32, Z, atol=1e-5)


def test_s2_por_trayectoria_igual_que_a_mano():
    m, X_te, r_te = _modelo_y_prueba(np.random.default_rng(7))
    t2, spe = ablation.score_standardized_by_run(m, X_te, r_te)
    for i in np.unique(r_te):
        filas = r_te == i
        t2_i, spe_i = m.score(X_te[filas], scaler=Scaler().fit(X_te[filas]))
        assert np.allclose(t2[filas], t2_i) and np.allclose(spe[filas], spe_i)


def test_s2_con_dpca_alinea_y_exige_el_gancho():
    m, X_te, r_te = _modelo_y_prueba(np.random.default_rng(8), lags=2, gancho=True)
    t2, spe = ablation.score_standardized_by_run(m, X_te, r_te)
    nan_esperado = np.zeros(len(X_te), dtype=bool)
    for ini in (0, 200, 400):
        nan_esperado[ini : ini + 2] = True
    assert np.array_equal(np.isnan(t2), nan_esperado)
    assert np.array_equal(np.isnan(spe), nan_esperado)

    sin_gancho, X_te2, r_te2 = _modelo_y_prueba(np.random.default_rng(8), lags=2, gancho=False)
    with pytest.raises(ValueError, match="gancho"):
        ablation.score_standardized_by_run(sin_gancho, X_te2, r_te2)


def test_score_con_escalador_invalido():
    m, X_te, r_te = _modelo_y_prueba(np.random.default_rng(9))
    with pytest.raises(TypeError, match="transform"):
        m.score(X_te, scaler=object())
    with pytest.raises(ValueError):
        m.score(X_te, scaler=Scaler().fit(X_te[:, :4]))
    with pytest.raises(ValueError, match="runs"):
        ablation.score_standardized_by_run(m, X_te, None)
    with pytest.raises(ValueError, match="trayectoria"):
        ablation.standardize_by_run(X_te[:3], [1, 2, 3])


# ------------------------------------------------ factor 4: una frente a muchas

def test_bootstrap_por_bloques_coincide_con_su_definicion():
    """Con las mismas extracciones, la media de concatenar ceil(n/L) bloques circulares y cortar a n.

    Bloques circulares desde las correcciones del 12/10 (I6): el inicio va de 0
    a n - 1 y el bloque da la vuelta al final de la serie.
    """
    x = np.random.default_rng(10).standard_normal(103)          # 103 no es multiplo de 10
    largo, B = 10, 60
    res = ablation.block_bootstrap_ci(x, block_length=largo, n_boot=B, seed=7, ci_level=0.9)
    k = -(-x.size // largo)
    inicios = np.random.default_rng(7).integers(0, x.size, size=(B, k))
    medias = np.array([np.concatenate([x[(i + np.arange(largo)) % x.size] for i in fila])[: x.size].mean()
                       for fila in inicios])
    bajo, alto = np.quantile(medias, [0.05, 0.95])
    assert res["estimate"] == pytest.approx(x.mean())
    assert res["ci_low"] == pytest.approx(bajo)
    assert res["ci_high"] == pytest.approx(alto)
    assert res["se"] == pytest.approx(medias.std(ddof=1))
    assert (res["n"], res["block_length"], res["n_boot"], res["ci_level"]) == (103, 10, 60, 0.9)


def test_cobertura_ar1_bloques_largos_cerca_de_la_nominal_y_bloques_de_1_corta():
    """Cobertura de la media de las N simulaciones con el intervalo de una sola.

    AR(1) con phi = 0,8, 100 simulaciones de 800 muestras (las de un tramo con
    fallo de Rieth), 200 remuestreos, nivel 0,95. Con bloques de 1 el bootstrap
    trata las muestras como independientes: la varianza de la media de un AR(1)
    es (1 + phi) / (1 - phi) = 9 veces la de muestras independientes, asi que
    el intervalo sale tres veces mas estrecho y cubre en torno a
    P(|Z| < 1,96 / 3), aproximadamente 0,49. Con bloques de 50 se acerca a la
    nominal (el bootstrap de bloques se queda algo corto con pocos bloques).
    Umbrales: >= 0,80 con 50, <= 0,65 con 1.
    """
    x = _ar1(np.random.default_rng(0), 100, 800, 0.8)
    cobertura = {}
    for largo in (1, 50):
        gen = np.random.default_rng(1000)
        res = [ablation.block_bootstrap_ci(serie, block_length=largo, n_boot=200, seed=gen) for serie in x]
        media_n = np.mean([r["estimate"] for r in res])
        cobertura[largo] = ablation.interval_coverage(
            [r["ci_low"] for r in res], [r["ci_high"] for r in res], media_n)["coverage"]
    assert cobertura[50] >= 0.80
    assert cobertura[1] <= 0.65
    assert cobertura[50] - cobertura[1] >= 0.25


def test_bootstrap_quita_los_nan_del_principio():
    """Como el DPCA: las primeras muestras no evaluadas no cuentan y no cambian el intervalo."""
    x = np.random.default_rng(11).standard_normal(200)
    con_nan = np.r_[np.nan, np.nan, x]
    a = ablation.block_bootstrap_ci(con_nan, block_length=5, n_boot=100, seed=3)
    b = ablation.block_bootstrap_ci(x, block_length=5, n_boot=100, seed=3)
    assert a == b
    assert a["n"] == 200 and a["estimate"] == pytest.approx(x.mean())


def test_bootstrap_de_la_serie_de_alarmas_estima_la_far_de_metrics():
    """La serie de alarmas con los NaN conservados da la misma FAR que metrics (DPCA incluido)."""
    rng = np.random.default_rng(15)
    stat = rng.chisquare(3, 960)
    stat[:2] = np.nan                                   # las lags primeras del DPCA
    limite = 7.8
    x = np.where(np.isnan(stat), np.nan, stat > limite)
    res = ablation.block_bootstrap_ci(x, block_length=20, n_boot=100, seed=0)
    assert res["estimate"] == pytest.approx(metrics.far(stat, limite))
    assert res["n"] == metrics.evaluate_run(stat, limite, onset=None)["n_far"]


def test_bootstrap_reproducible_y_encadenable():
    x = np.random.default_rng(12).standard_normal(300)
    kw = dict(block_length=10, n_boot=50)
    assert ablation.block_bootstrap_ci(x, seed=4, **kw) == ablation.block_bootstrap_ci(x, seed=4, **kw)
    gen = np.random.default_rng(4)
    primera = ablation.block_bootstrap_ci(x, seed=gen, **kw)
    segunda = ablation.block_bootstrap_ci(x, seed=gen, **kw)
    assert primera == ablation.block_bootstrap_ci(x, seed=4, **kw)
    assert primera != segunda


def test_bootstrap_valida_entradas():
    x = np.random.default_rng(13).standard_normal(50)
    with pytest.raises(ValueError, match="block_length"):
        ablation.block_bootstrap_ci(x, block_length=0, n_boot=10, seed=0)
    with pytest.raises(ValueError, match="mayor"):
        ablation.block_bootstrap_ci(x, block_length=51, n_boot=10, seed=0)
    with pytest.raises(ValueError, match="n_boot"):
        ablation.block_bootstrap_ci(x, block_length=5, n_boot=0, seed=0)
    with pytest.raises(ValueError, match="ci_level"):
        ablation.block_bootstrap_ci(x, block_length=5, n_boot=10, seed=0, ci_level=95)
    with pytest.raises(ValueError, match="infinitos"):
        ablation.block_bootstrap_ci(np.r_[x, np.inf], block_length=5, n_boot=10, seed=0)
    with pytest.raises(ValueError, match="NaN"):
        ablation.block_bootstrap_ci(np.full(5, np.nan), block_length=1, n_boot=10, seed=0)
    with pytest.raises(ValueError, match="1D"):
        ablation.block_bootstrap_ci(x.reshape(5, 10), block_length=5, n_boot=10, seed=0)
    with pytest.raises(TypeError):
        ablation.block_bootstrap_ci(x, block_length=5, n_boot=10)        # sin semilla


def test_cobertura_cuenta_por_debajo_y_por_encima():
    bajo = [0.0, 1.0, -1.0, np.nan, 2.0]
    alto = [2.0, 3.0, 0.5, 1.0, 4.0]
    res = ablation.interval_coverage(bajo, alto, 1.5)
    assert res == {"coverage": 0.5, "n": 4, "n_below": 1, "n_above": 1}
    assert ablation.interval_coverage(bajo, alto, 2.0)["coverage"] == 0.75   # extremos incluidos
    with pytest.raises(ValueError, match="ci_low > ci_high"):
        ablation.interval_coverage([1.0], [0.0], 0.5)
    assert np.isnan(ablation.interval_coverage([np.nan], [1.0], 0.5)["coverage"])


def test_conclusiones_contrarias_a_mano():
    single = [0.1, -0.2, 0.0, 0.3, np.nan, -0.1]
    pos = ablation.contrary_conclusion_rate(single, 0.05)
    assert pos == {"rate": 0.4, "n": 5, "n_contrary": 2, "n_same": 2, "n_zero": 1, "reference_sign": 1}
    neg = ablation.contrary_conclusion_rate(single, -1.0)
    assert (neg["n_contrary"], neg["n_same"], neg["reference_sign"]) == (2, 2, -1)


def test_conclusiones_contrarias_con_el_intervalo_de_una_simulacion():
    """Quien concluye solo si su intervalo excluye el 0: las inconclusas no son contrarias."""
    lo = np.array([0.1, -0.5, -0.3, 0.2])
    hi = np.array([0.4, -0.1, 0.2, 0.6])
    conclusion = np.where(lo > 0, 1, np.where(hi < 0, -1, 0))
    res = ablation.contrary_conclusion_rate(conclusion, 0.15)
    assert (res["n_same"], res["n_contrary"], res["n_zero"]) == (2, 1, 1)
    assert res["rate"] == pytest.approx(0.25)


def test_conclusiones_contrarias_sin_signo_de_referencia():
    res = ablation.contrary_conclusion_rate([0.1, -0.1, 0.0], 0.0)
    assert np.isnan(res["rate"]) and res["reference_sign"] == 0 and res["n_zero"] == 1
    with pytest.raises(ValueError, match="NaN"):
        ablation.contrary_conclusion_rate([0.1], np.nan)


def test_submuestreo_disjunto_sin_repetidos():
    candidatos = np.arange(41, 81)
    rep = ablation.subsample_runs(candidatos, 5, 8, 3, disjoint=True)
    assert rep.shape == (8, 5) and rep.dtype == np.int64
    assert np.unique(rep).size == 40
    assert np.isin(rep, candidatos).all()
    assert (np.diff(rep, axis=1) > 0).all()
    assert np.array_equal(rep, ablation.subsample_runs(candidatos, 5, 8, 3, disjoint=True))


def test_submuestreo_no_disjunto_y_errores():
    candidatos = list(range(1, 11))
    rep = ablation.subsample_runs(candidatos, 4, 6, 0, disjoint=False)
    assert rep.shape == (6, 4)
    assert all(np.unique(fila).size == 4 for fila in rep)
    with pytest.raises(ValueError, match="disjoint=False"):
        ablation.subsample_runs(candidatos, 4, 3, 0, disjoint=True)
    with pytest.raises(TypeError):
        ablation.subsample_runs(candidatos, 2, 2, 0)                     # sin declarar disjoint
    with pytest.raises(TypeError, match="disjoint"):
        ablation.subsample_runs(candidatos, 2, 2, 0, disjoint="si")
    with pytest.raises(ValueError, match="repetidos"):
        ablation.subsample_runs([1, 1, 2], 1, 1, 0, disjoint=True)
    with pytest.raises(ValueError, match="candidatos"):
        ablation.subsample_runs(candidatos, 11, 1, 0, disjoint=False)


# ---------------------------------------------------- ventanas por trayectoria

def test_ventanas_no_cruzan_trayectorias():
    """Con faultNumber y simulationRun, ninguna ventana mezcla dos trayectorias, ni dos clases
    con el mismo id seguidas; con solo simulationRun, alguna si las mezclaria."""
    fallo = np.repeat([0, 3, 0, 3], 30)
    sim = np.repeat([1, 1, 2, 2], 30)
    runs2 = np.column_stack([fallo, sim])
    largo = 6
    fines = ablation.window_ends(runs2, largo, 1)
    assert len(fines) == 4 * (30 - largo + 1)
    for e in fines:
        assert len(np.unique(runs2[e - largo + 1 : e + 1], axis=0)) == 1
    fines_1d = ablation.window_ends(sim, largo, 1)
    assert any(np.unique(fallo[e - largo + 1 : e + 1]).size > 1 for e in fines_1d)


def test_ventanas_igual_que_lag_by_run():
    """Con length = lags + 1 y paso 1: las mismas posiciones y las mismas filas que el DPCA."""
    rng = np.random.default_rng(14)
    X = rng.standard_normal((3 * 25, 4))
    runs = _ids(3, 25)
    matriz, _, pos = lag_by_run(X, runs, 2, return_index=True)
    fines = ablation.window_ends(runs, 3, 1)
    assert np.array_equal(fines, pos)
    V = ablation.stack_windows(X, fines, 3)
    assert V.shape == (len(fines), 3, 4)
    assert np.array_equal(V[:, -1, :], X[fines])                       # la ultima es la que etiqueta
    assert np.allclose(V[:, ::-1, :].reshape(len(fines), -1), matriz)  # lag_matrix: actual primero


def test_ventanas_con_paso():
    runs = np.r_[np.full(20, 1), np.full(17, 2)]
    fines = ablation.window_ends(runs, 5, 3)
    assert np.array_equal(fines, np.r_[np.arange(4, 20, 3), np.arange(24, 37, 3)])
    assert len(fines) == (20 - 5) // 3 + 1 + (17 - 5) // 3 + 1


def test_alineado_nan_al_principio_y_retardo_en_la_escala_de_las_muestras():
    """Puntuacion por ventana llevada a su ultima muestra: mismo onset que PCA y DPCA.

    Estadistico por muestra: 0 antes del indice 170 y 1 desde ahi (fallo
    visible 10 muestras despues del inicio, 160). Puntuacion de una ventana de
    10: la media de la ventana. Con limite 0,5, la primera ventana por encima
    es la que termina en 175, asi que el retardo con onset = 160 es 15. Si se
    alineara por la primera muestra de la ventana saldria 6: adelantado L - 1.
    """
    n, largo = 960, 10
    runs = _ids(2, n)
    s = np.tile((np.arange(n) >= 170).astype(float), 2)
    fines = ablation.window_ends(runs, largo, 1)
    puntuacion = ablation.stack_windows(s[:, None], fines, largo)[:, :, 0].mean(axis=1)
    stat = ablation.align_to_samples(puntuacion, fines, len(s))
    for ini in (0, n):
        assert np.isnan(stat[ini : ini + largo - 1]).all()
        assert not np.isnan(stat[ini + largo - 1 : ini + n]).any()
    res = metrics.evaluate_run(stat[:n], 0.5, onset=160)
    assert res["delay_samples"] == 15
    assert res["n_far"] == 160 - (largo - 1)
    mal_alineado = ablation.align_to_samples(puntuacion, fines - (largo - 1), len(s))
    assert metrics.evaluate_run(mal_alineado[:n], 0.5, onset=160)["delay_samples"] == 6


def test_ventanas_solapadas_al_azar_comparten_filas_entre_ajuste_y_evaluacion():
    """Brazo sucio con ventanas (Shamim et al., 2026, protocolo B) frente al reparto por simulacion."""
    runs = _ids(8, 60)
    largo = 10
    fines = ablation.window_ends(runs, largo, 1)
    g = runs[fines]
    limpio = ablation.split_by_run(g, (0.5, 0.25, 0.25), 4)
    sucio = ablation.split_by_sample(g, limpio.sizes, 4, block_length=1)

    def filas(mascara):
        e = fines[mascara]
        return set((e[:, None] + np.arange(1 - largo, 1)).ravel().tolist())

    assert not filas(limpio.fit) & filas(limpio.eval)
    assert filas(sucio.fit) & filas(sucio.eval)


def test_ventanas_validan_entradas():
    with pytest.raises(ValueError, match="ventanas de 10"):
        ablation.window_ends(np.r_[np.full(20, 1), np.full(5, 2)], 10, 1)
    with pytest.raises(ValueError, match="length"):
        ablation.window_ends(_ids(2, 20), 0, 1)
    with pytest.raises(ValueError, match="step"):
        ablation.window_ends(_ids(2, 20), 5, 0)
    X32 = np.ones((20, 3), dtype=np.float32)
    assert ablation.stack_windows(X32, ablation.window_ends(_ids(1, 20), 5, 1), 5).dtype == np.float32
    with pytest.raises(ValueError, match="fuera de rango"):
        ablation.stack_windows(X32, [2], 5)
    with pytest.raises(ValueError, match="repetidas"):
        ablation.align_to_samples([1.0, 2.0], [4, 4], 10)
    with pytest.raises(ValueError, match="una fila por ventana"):
        ablation.align_to_samples([1.0], [4, 5], 10)
    alineado = ablation.align_to_samples(np.ones((2, 3)), [4, 9], 10)
    assert alineado.shape == (10, 3) and np.isnan(alineado[0]).all()


# ------------------------------------------------------------ datos reales

def _hay_rieth(*ficheros):
    try:
        import pyarrow.parquet  # noqa: F401
    except Exception:
        return False
    carpeta = data._carpeta_rieth(None)[0]
    return all((carpeta / f).exists() for f in ficheros)


@pytest.mark.skipif(not _hay_rieth("test_fault00.parquet", "test_fault03.parquet"),
                    reason="faltan los Parquet de Rieth o pyarrow no puede leer Parquet (define TFM_RIETH_DIR)")
def test_rieth_ventanas_y_s2_por_trayectoria_con_ids_compartidos():
    """Rieth, simulaciones 1 y 2 de test_fault00 y test_fault03 (ya miradas, regla 3).

    Ordenadas por simulationRun, la normal 1 va seguida de su gemela con fallo 3:
    con faultNumber y simulationRun son cuatro trayectorias y ninguna ventana ni
    estandarizacion las mezcla. No se calcula ninguna metrica.
    """
    import pandas as pd

    df = pd.concat([data.load_rieth(f, "test", runs=[1, 2]) for f in (0, 3)], ignore_index=True)
    df = df.sort_values(["simulationRun", "faultNumber", "sample"], ignore_index=True)
    runs2 = df[["faultNumber", "simulationRun"]].to_numpy()
    n = data.N_SAMPLES_RIETH["test"]
    fines = ablation.window_ends(runs2, 40, 1)
    assert len(fines) == 4 * (n - 40 + 1)
    assert (runs2[fines - 39] == runs2[fines]).all()     # filas seguidas: basta mirar los extremos
    X = data.values(df)
    Z = ablation.standardize_by_run(X, runs2)
    for ini in range(0, 4 * n, n):
        assert np.allclose(Z[ini : ini + n].mean(axis=0), 0.0, atol=1e-8)
