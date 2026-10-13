"""Pruebas de las correcciones del 12/10 en tfmfdd/ablation.py.

- I6 (verif_estadistica): la falta de cobertura del bootstrap de bloques
  aparece aunque todas las simulaciones salgan del mismo proceso, asi que no es
  solo pseudorreplicacion; y con un retardo de deteccion el centro del
  bootstrap de bloques moviles se desplaza. Ahora los bloques son circulares
  (centro exacto) e `interval_coverage` admite una referencia por intervalo.
- V-08 (verif_codigo): `split_by_run` llama a `data.split_runs` en vez de
  repetir su algoritmo. Ninguna prueba depende de como reparte
  `data.split_runs` una fraccion 0 (lo corrige otro agente en paralelo).

Datos sinteticos con semilla fija. Los umbrales de las pruebas Monte Carlo
salen de un sondeo con seis semillas y se declaran en cada docstring, con el
margen que dejan; no son resultados del TFM.
"""

import math

import numpy as np
import pytest
from scipy import stats

from tfmfdd import ablation, data


def _alarmas(rng, n_series, n, phi):
    """Alarmas de n_series simulaciones del MISMO proceso estacionario: AR(1) de varianza 1 por
    encima de z_0,99, con FAR verdadera 0,01."""
    e = rng.standard_normal((n_series, n)) * math.sqrt(1.0 - phi**2)
    x = np.empty((n_series, n))
    x[:, 0] = rng.standard_normal(n_series)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return (x > stats.norm.ppf(0.99)).astype(float)


def _cobertura(series, largo, semilla, referencia):
    gen = np.random.default_rng(semilla)
    res = [ablation.block_bootstrap_ci(s, block_length=largo, n_boot=200, seed=gen) for s in series]
    return ablation.interval_coverage([r["ci_low"] for r in res], [r["ci_high"] for r in res], referencia)


# ------------------------------------------------------------------- I6

def test_cobertura_corta_aunque_todas_las_simulaciones_salgan_del_mismo_proceso():
    """La falta de cobertura no es solo pseudorreplicacion.

    200 simulaciones del mismo AR(1) estacionario (no hay nada que varie entre
    ellas salvo el azar de cada serie), 800 muestras, alarmas con FAR
    verdadera 0,01, intervalo al 95 % con bloques de 50. Con phi = 0,9 la
    cobertura de la FAR verdadera queda lejos de 0,95 (de 0,700 a 0,770 en el
    sondeo, siempre con los fallos por debajo: simulaciones con pocas alarmas o
    ninguna); umbral, < 0,85. Control: con alarmas independientes (phi = 0) y
    bloques de 1, el mismo bootstrap cubre (de 0,910 a 0,985 en el sondeo);
    umbral, >= 0,85.
    """
    autocorrelada = _cobertura(_alarmas(np.random.default_rng(0), 200, 800, 0.9), 50, 100, 0.01)
    assert autocorrelada["coverage"] < 0.85
    assert autocorrelada["n_below"] > autocorrelada["n_above"]
    independiente = _cobertura(_alarmas(np.random.default_rng(0), 200, 800, 0.0), 1, 100, 0.01)
    assert independiente["coverage"] >= 0.85


def test_bootstrap_circular_centrado_en_la_media_con_un_retardo():
    """Con un retardo de deteccion, el centro del bootstrap es la media de la serie.

    Serie de 800 muestras: sin alarmas en las 100 primeras y despues alarmas
    independientes con probabilidad 0,84. Con bloques circulares de 100, la
    media de 20 000 replicas queda a menos de 5 errores Monte Carlo de la media
    de la serie. Con bloques moviles (su centro exacto, calculado con el peso de
    cada posicion) se desplazaria hacia el tramo final mas de 20 errores (en el
    sondeo, +0,044 frente a un error de 0,0006).
    """
    rng = np.random.default_rng(50)
    x = np.zeros(800)
    x[100:] = rng.standard_normal(700) > -1.0
    largo, B = 100, 20000
    medias = ablation._medias_circulares(x, largo, B, np.random.default_rng(1))
    ee = medias.std() / math.sqrt(B)
    assert abs(medias.mean() - x.mean()) < 5 * ee

    n, k = x.size, -(-x.size // largo)
    pos = np.arange(n)
    cubre = np.minimum(pos, n - largo) - np.maximum(pos - largo + 1, 0) + 1
    resto = n - (k - 1) * largo
    cubre_ultimo = np.clip(np.minimum(pos, n - largo) - np.maximum(pos - resto + 1, 0) + 1, 0, None)
    centro_moviles = (((k - 1) * cubre + cubre_ultimo) / (n - largo + 1) * x).sum() / n
    assert centro_moviles - x.mean() > 20 * ee


def test_bootstrap_circular_centrado_tambien_cuando_n_no_es_multiplo_de_l():
    """El centro exacto no depende de cortar el ultimo bloque: n = 103, L = 10."""
    x = np.r_[np.zeros(30), np.ones(73)]
    B = 40000
    medias = ablation._medias_circulares(x, 10, B, np.random.default_rng(2))
    assert abs(medias.mean() - x.mean()) < 5 * medias.std() / math.sqrt(B)


def test_cobertura_con_una_referencia_por_intervalo():
    """Cada intervalo frente a su propia referencia, no frente a una comun."""
    propia = ablation.interval_coverage([0.0, 2.0], [1.0, 3.0], [0.5, 2.5])
    assert (propia["coverage"], propia["n_below"], propia["n_above"]) == (1.0, 0, 0)
    comun = ablation.interval_coverage([0.0, 2.0], [1.0, 3.0], 1.5)      # un escalar, como antes
    assert (comun["coverage"], comun["n_below"], comun["n_above"]) == (0.0, 1, 1)
    # La referencia sin la propia simulacion: la media de las demas.
    est = np.array([0.010, 0.012, 0.030, 0.008])
    sin_si_misma = (est.sum() - est) / (len(est) - 1)
    res = ablation.interval_coverage(est - 0.004, est + 0.004, sin_si_misma)
    assert res["n"] == 4 and res["n_above"] == 1      # la de 0,030 queda por encima de la media de las otras


def test_cobertura_referencia_por_intervalo_nan_fuera_y_errores():
    bajo, alto = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    res = ablation.interval_coverage(bajo, alto, [0.5, np.nan, 2.0])
    assert (res["n"], res["coverage"], res["n_below"]) == (2, 0.5, 1)
    with pytest.raises(ValueError, match="por intervalo"):
        ablation.interval_coverage(bajo, alto, [0.5, 0.5])
    with pytest.raises(ValueError, match="infinitos"):
        ablation.interval_coverage(bajo, alto, [0.5, np.inf, 0.5])
    with pytest.raises(ValueError, match="finito"):
        ablation.interval_coverage(bajo, alto, np.inf)


# ------------------------------------------------------------------ V-08

def test_reparto_limpio_llama_a_data_split_runs(monkeypatch):
    """split_by_run no repite el algoritmo: llama a data.split_runs y usa lo que devuelve."""
    llamadas = []

    def falso(n_runs, fractions, seed, *, exclude=None):
        llamadas.append((n_runs, tuple(fractions), seed, None if exclude is None else sorted(exclude)))
        return np.array([7]), np.array([3, 9]), np.array([5])

    monkeypatch.setattr(data, "split_runs", falso)
    g = np.repeat([3, 5, 7, 9], 4)
    r = ablation.split_by_run(g, (0.25, 0.5, 0.25), 11)
    assert llamadas == [(9, (0.25, 0.5, 0.25), 11, [1, 2, 4, 6, 8])]
    assert np.array_equal(r.fit, g == 7)
    assert np.array_equal(r.cal, np.isin(g, [3, 9]))
    assert np.array_equal(r.eval, g == 5)


def test_reparto_limpio_sin_huecos_no_pasa_exclude(monkeypatch):
    """Con los ids 1..n seguidos, la llamada es la de siempre: split_runs(n, fractions, seed)."""
    llamadas = []
    original = data.split_runs

    def espia(n_runs, fractions, seed, *, exclude=None):
        llamadas.append(exclude)
        return original(n_runs, fractions, seed, exclude=exclude)

    monkeypatch.setattr(data, "split_runs", espia)
    ablation.split_by_run(np.repeat(np.arange(1, 11), 3), (0.6, 0.2, 0.2), 4)
    assert llamadas == [None]


def test_reparto_limpio_exige_ids_enteros_desde_1():
    with pytest.raises(ValueError, match="< 1"):
        ablation.split_by_run(np.repeat([0, 1, 2], 3), (0.6, 0.2, 0.2), 0)
    with pytest.raises(TypeError, match="enteros"):
        ablation.split_by_run(np.repeat([1.0, 2.0, 3.0], 3), (0.6, 0.2, 0.2), 0)
    with pytest.raises(TypeError, match="enteros"):
        ablation.split_by_run(np.repeat(["a", "b"], 3), (0.5, 0.5, 0.0), 0)
