"""Pruebas de la adenda de la configuracion del KPCA (04/11/2026; acta del 26/10, secciones 17 a 22).

El KPCA no tiene codigo todavia. Estas pruebas fijan el bloque kpca de los dos yaml, comprueban
que las claves a las que remite existen con el valor decidido y escriben las tres reglas que la
adenda define con palabras (el ancho, los componentes y el SPE) como referencia para el codigo
que venga. Al final, el formato del registro y del yaml.
"""
from pathlib import Path

import numpy as np

from tfmfdd import config, data
from tfmfdd.pca import PCAMonitor
from tfmfdd.preprocessing import Scaler

RAIZ = Path(__file__).resolve().parents[1]
YAMLS = (RAIZ / "configs" / "baseline.yaml", RAIZ / "tfmfdd" / "configs" / "baseline.yaml")
KPCA = {"nucleo": "rbf", "ancho": "mediana_d2", "ancho_sensibilidad": 520.0, "estadistico": "spe",
        "componentes": "pca.variance", "submuestreo": "ninguno", "limite": "limites_protocolo"}


def _cfg():
    return config.load_yaml(YAMLS[0])


def _ancho(X_ajuste):
    """kpca.ancho = mediana_d2: estandarizar con las propias muestras de ajuste y tomar la mediana de
    ||x_i - x_j||^2 sobre los pares i < j, sin factor 2."""
    Z = Scaler().fit(X_ajuste).transform(X_ajuste)
    i, j = np.triu_indices(len(Z), 1)
    return float(np.median(((Z[i] - Z[j]) ** 2).sum(1)))


def _componentes(autovalores, varianza):
    """kpca.componentes = pca.variance: ordenar de mayor a menor, los negativos numericos a 0 y el
    primer p cuya varianza acumulada llega a la fraccion, sin pasar del numero de autovalores."""
    lam = np.clip(np.sort(np.asarray(autovalores, dtype=float))[::-1], 0.0, None)
    acumulada = np.cumsum(lam) / lam.sum()
    return min(int(np.searchsorted(acumulada, varianza) + 1), lam.size)


def _spe(X_ajuste, X_nuevas, c, p):
    """kpca.estadistico = spe, calculado entero: k~(x, x) menos la suma de t_i^2 en los p primeros
    componentes del nucleo centrado, con k~(x, x) = 1 - (2/N) sum_j k(x, x_j) + media(K)."""
    def nucleo(A, B):
        return np.exp(-np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0) / c)
    K = nucleo(X_ajuste, X_ajuste)
    media_col, media = K.mean(0), K.mean()
    lam, V = np.linalg.eigh(K - media_col[None, :] - media_col[:, None] + media)
    lam, V = lam[::-1][:p], V[:, ::-1][:, :p]
    kt = nucleo(X_nuevas, X_ajuste)
    t = (kt - kt.mean(1, keepdims=True) - media_col[None, :] + media) @ (V / np.sqrt(lam))
    return 1.0 - 2.0 * kt.mean(1) + media - (t ** 2).sum(1)


def test_bloque_kpca_igual_en_los_dos_yaml():
    for ruta in YAMLS:
        assert config.get(config.load_yaml(ruta), "kpca") == KPCA, ruta


def test_lo_que_remite_el_bloque():
    """Las claves a las que remite el bloque, con los valores con que se decidio."""
    cfg = _cfg()
    assert config.get(cfg, config.get(cfg, "kpca.componentes")) == 0.85
    limite = config.get(cfg, config.get(cfg, "kpca.limite"))
    assert (limite["spe_method"], limite["quantile_method"]) == ("empirical", "linear")
    assert limite["nivel_por_estadistico"] == "sidak"
    assert config.get(cfg, "protocolo.alpha") == 0.99
    assert config.get(cfg, "particion.rieth.normales_ajuste") == 14              # 14 x 500 = 7000
    assert config.get(cfg, "particion.rieth.normales_calibracion") == 20
    assert data.N_SAMPLES_RIETH["train"] == 500
    assert config.get(cfg, "kpca.ancho_sensibilidad") == 10 * data.N_VARS        # Lee et al. (2004)
    assert {"nombre": "kpca", "familia": 1} in config.get(cfg, "compromiso.deteccion.candidatos")
    assert not [c for c in config.pending(cfg) if c.startswith("kpca")]


def test_el_ancho_es_la_mediana_de_los_pares_distintos_sin_factor_2():
    """Cuatro puntos ya estandarizados: las seis distancias al cuadrado son 1, 4, 4, 5, 9 y 13, y la
    mediana es 4,5. Con la diagonal (ceros) y los pares repetidos saldria 4; con factor 2, 9. En el
    par tipico el nucleo vale e^-1. Con datos en bruto, el ancho es el de los datos estandarizados con
    sus propias muestras, no el de los datos sin estandarizar."""
    Z = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    i, j = np.triu_indices(4, 1)
    c = float(np.median(((Z[i] - Z[j]) ** 2).sum(1)))
    assert c == 4.5
    D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(2)
    assert np.median(D) == 4.0 and 2 * c == 9.0
    assert np.exp(-4.5 / c) == np.exp(-1.0)
    rng = np.random.default_rng(20261104)
    X = rng.normal(size=(30, 3)) * np.array([1.0, 10.0, 100.0]) + 5.0
    Zx = (X - X.mean(0)) / X.std(0, ddof=1)
    ii, jj = np.triu_indices(30, 1)
    assert np.isclose(_ancho(X), float(np.median(((Zx[ii] - Zx[jj]) ** 2).sum(1))))
    assert not np.isclose(_ancho(X), float(np.median(((X[ii] - X[jj]) ** 2).sum(1))))


def test_los_componentes_son_la_regla_de_pcamonitor():
    """La regla del bloque da el mismo p que PCAMonitor con pca.variance, y una fraccion alcanzada
    justo cuenta."""
    cfg = _cfg()
    varianza = config.get(cfg, "pca.variance")
    assert _componentes([5, 3, 1, 1], varianza) == 3                 # 0,5 / 0,8 / 0,9: el tercero
    assert _componentes([4, 4, 1, 1], 0.8) == 2                      # 0,8 justo: el segundo
    assert _componentes([1, 5, 1, 3], varianza) == 3                 # desordenados: se ordenan
    assert _componentes([5, 3, 1, 1, -2.0], varianza) == 3           # el negativo cuenta como 0 (en
                                                                     # valor absoluto serian 4)
    assert _componentes([0.1] * 10, 1.0) == 10                       # la suma no llega a 1: el tope
    rng = np.random.default_rng(20261104)
    X = rng.normal(size=(600, 8)) @ rng.normal(size=(8, 8))
    monitor = PCAMonitor(variance=varianza).fit(X[:400], X_cal=X[400:])
    assert monitor.n_components_ == _componentes(monitor.eigenvalues_, varianza)


def test_el_spe_se_calcula_entero():
    """El SPE de una muestra nueva es k~(x, x) menos su proyeccion en los p componentes, no la
    aproximacion de Lee et al. (2004, ec. 20), que suma t_j^2 en los n autovalores no nulos: con
    p = n, la de Lee da 0 para cualquier muestra y la entera no. En las muestras del ajuste, con
    p = n, las dos dan 0."""
    rng = np.random.default_rng(20261104)
    X = rng.normal(size=(12, 3))
    n = len(X) - 1                                       # el centrado quita un autovalor
    lejos = np.array([[4.0, 4.0, 4.0]])
    assert np.allclose(_spe(X, X, c=6.0, p=n), 0.0, atol=1e-8)
    assert _spe(X, lejos, c=6.0, p=n)[0] > 0.5           # casi toda la norma queda fuera
    a = _spe(X, X[:1] + 0.3, c=6.0, p=3)[0]
    b = _spe(X, X[:1] + 0.3, c=6.0, p=5)[0]
    assert a > b > 0                                     # mas componentes, menos residuo


def test_formato_del_registro_y_del_yaml():
    """El registro es ASCII, con 5 barras por fila y en orden de fecha; las dos copias del yaml son
    identicas, ASCII y de 96 columnas como mucho."""
    registro = (RAIZ / "docs" / "decisiones.md").read_bytes().decode("ascii")
    filas = [l for l in registro.splitlines() if l.startswith("| 20")]
    assert filas and all(l.count("|") == 5 for l in filas)
    assert [l[2:12] for l in filas] == sorted(l[2:12] for l in filas)
    yamls = [ruta.read_bytes() for ruta in YAMLS]
    assert yamls[0] == yamls[1]
    assert max(len(l) for l in yamls[0].decode("ascii").splitlines()) <= 96

