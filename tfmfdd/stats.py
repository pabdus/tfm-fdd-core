"""Contrastes estadisticos para comparar metodos.

Una diferencia entre dos metodos no es un resultado hasta que se comprueba que
no es ruido. Como evaluamos ambos metodos sobre LAS MISMAS simulaciones, el
contraste correcto es pareado: se compara la diferencia dentro de cada
simulacion, no las medias sueltas.

Se usa Wilcoxon de rangos con signo en vez de la t de Student porque no exige
normalidad, y las metricas de FDD rara vez son normales (el FDR se acumula
contra el 1, el retardo esta acotado por abajo). Wilcoxon contrasta la
pseudomediana de las diferencias (la mediana de los promedios de Walsh); su
estimador es el de Hodges-Lehmann y su intervalo sale de invertir el mismo
contraste, de modo que contraste, estimador e intervalo hablan del mismo
parametro. El p es exacto (la distribucion del estadistico condicionada a los
valores absolutos, con empates y ceros), salvo con muchas diferencias no nulas,
donde la aproximacion normal ya es buena (EXACT_MAX_N).

No exigir normalidad no es no suponer nada: el nivel exacto de Wilcoxon supone
que la diferencia es simetrica alrededor de su pseudomediana. Que pasa sin
simetria y que analisis de sensibilidad se pueden declarar esta en
non_superiority_test ("Supuesto de simetria"), y vale para todas las reglas de
este modulo.

Que hay en el modulo (los cambios de cada version estan en CAMBIOS.md):

- paired_test, holm_correction y compare_methods: el contraste pareado, la
  correccion de Holm y la comparacion por fallo (contra un baseline, con
  replicas de ajuste y con la familia de Holm que se declare).
- hodges_lehmann y min_runs_for_significance: la diferencia con su intervalo y
  el minimo de simulaciones para poder rechazar; LowPowerWarning avisa cuando
  no se llega.
- paired_advantage, non_superiority_test, detects_above_far e
  hypothesis_verdict: la regla de la hipotesis (memoria, parrafo 368), con el
  suelo de deteccion. published_advantage_test: la otra condicion de la
  prediccion (parrafo 349), que el veredicto no evalua.
- effect_test y compare_arms: el efecto de un factor de las ablaciones frente
  a un margen de equivalencia, con tres desenlaces.
- pareto_front y best_compromise: el mecanismo del criterio de mejor
  compromiso, sin criterio por defecto.

Todas las reglas que comparan una diferencia con un umbral usan la misma pieza
(_frente_a), para que decidan con el mismo Wilcoxon exacto y el mismo redondeo.
Este modulo no importa metrics.py (metrics.py si importa este), para que no
haya importaciones circulares.

El `alpha` de este modulo es el nivel de significacion de los contrastes
(alpha_contrastes, 0,05). No es el nivel de confianza del limite de control
(protocolo.alpha = 0,99 en configs/baseline.yaml): son cosas distintas, se
nombran distinto en la memoria, y un alpha >= 0,5 se rechaza con un error.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats as sps


#: Por debajo de este numero de pares no se calcula el contraste (p = NaN). Con
#: 4 pares o menos ninguna alternativa puede dar p <= 0,05: el p minimo exacto
#: es 2/2^n bilateral y 1/2^n unilateral. Es el valor de las versiones anteriores.
MIN_PAIRS = 5

#: Hasta este numero de diferencias NO nulas el p es exacto (si ademas la
#: programacion dinamica cabe en los topes de abajo); por encima se usa la
#: aproximacion normal, con la varianza condicional y correccion de
#: continuidad, que con tantas diferencias no nulas es muy buena. Lo que fija la
#: distribucion nula, con los ceros repartidos, son las diferencias no nulas y
#: no los pares: con cientos de pares y unas pocas no nulas la normal se
#: equivoca mucho (revision del 12/10, I1), y por eso la rama se decide por
#: ellas. Con hasta 500 pares el p es siempre exacto (Rieth tiene 500
#: simulaciones de prueba por fallo). Por encima de los topes, con muchos ceros
#: la normal todavia puede dar p algo bajos (revision del 12/10, N1); la via del
#: TFM no llega ahi.
EXACT_MAX_N = 500

#: Topes de la programacion dinamica del p exacto: posiciones del vector (la
#: suma de los pesos, 2 x rango, de las diferencias no nulas; 5e6 son 40 MB) y
#: trabajo (posiciones x diferencias no nulas). 500 pares sin ceros son
#: 250500 posiciones y 1,25e8 de trabajo, y caben; los topes evitan esperas de
#: minutos o un vector de gigas con miles de pares. Fuera de ellos, la normal.
_EXACT_MAX_SIZE = 5_000_000
_EXACT_MAX_WORK = 250_000_000

#: Cifras significativas con las que se comparan las diferencias antes de
#: asignar signos y rangos. Las metricas son fracciones de muestras (la FDR de
#: Rieth es un multiplo de 1/800) y la resta en coma flotante deja restos:
#: (0,03 - 0,01) - 0,02 da -3,5e-18 en vez de 0, y 0,03 - 0,0175 no es igual a
#: 0,0275 - 0,015. Sin redondear, un cero contaria como diferencia negativa y
#: dos empates como valores distintos. Se redondea a 10 cifras relativas a la
#: mayor magnitud de los datos, muy por debajo de cualquier diferencia real.
_DIGITS = 10

#: Desenlaces de una comparacion (vocabulario del parrafo del criterio de no
#: superioridad de la memoria).
OUTCOME_SUPPORTS = "apoya"
OUTCOME_RELEVANT = "ventaja relevante"
OUTCOME_INCONCLUSIVE = "inconclusa"
OUTCOME_UNTESTABLE = "no contrastable"

#: Veredictos de la familia confirmatoria.
VERDICT_SUPPORTS = "apoya"
VERDICT_REFUTES = "refuta"
VERDICT_INCONCLUSIVE = "inconclusa"

#: Fallos de la familia confirmatoria que declara la memoria (parrafo 368: "La
#: familia confirmatoria incluye los fallos 3, 9 y 15, con el autoencoder del
#: tercer objetivo frente a PCA y DPCA, es decir, seis comparaciones").
CONFIRMATORY_FAULTS = (3, 9, 15)

#: Estado de la condicion "menor que la publicada" (parrafo 349) en la salida
#: de hypothesis_verdict: el veredicto no la evalua (ver su docstring).
PUBLISHED_NOT_EVALUATED = "no evaluada"

#: Desenlaces del efecto de un factor frente a su margen de equivalencia
#: (effect_test, compare_arms). Con menos de MIN_PAIRS unidades, el desenlace
#: es OUTCOME_UNTESTABLE ("no contrastable"), como en el resto del modulo.
EFFECT_RELEVANT = "efecto relevante"
EFFECT_EQUIVALENT = "equivalente a cero"
EFFECT_INCONCLUSIVE = "inconcluso"

#: Desenlaces de la comparacion con la ventaja publicada
#: (published_advantage_test).
PUBLISHED_LESS = "menor que la publicada"
PUBLISHED_NOT_LESS = "igual o mayor que la publicada"
PUBLISHED_INCONCLUSIVE = "inconclusa"

#: Tipo de comparacion de una ventaja publicada. La memoria (criterio de no
#: superioridad) pide declararlo: "Si el articulo no tiene baseline lineal, la
#: comparacion es entre articulos y se declara asi".
COMPARISON_WITHIN = "dentro del articulo"
COMPARISON_BETWEEN = "entre articulos"

#: Columnas obligatorias de la tabla de ventajas publicadas. El significado de
#: cada una esta en el docstring de published_advantage_test.
PUBLISHED_TABLE_COLUMNS = ("article", "metric", "method", "baseline", "value", "page",
                           "comparison")

_ALTERNATIVAS = ("two-sided", "less", "greater")
_DIRECCIONES = ("max", "min")
_OPERADORES = {"<=": np.less_equal, "<": np.less, ">=": np.greater_equal, ">": np.greater}


class LowPowerWarning(UserWarning):
    """Con estos pares el contraste no puede salir significativo al nivel pedido.

    Wilcoxon exacto con n diferencias no nulas no puede dar un p menor que
    2/2^n (bilateral) o 1/2^n (unilateral). Si ese minimo ya supera alpha, o
    alpha/m tras Holm sobre m contrastes, "no significativo" no informa de nada:
    lo decide el tamano de muestra, no los datos.
    """


# ---------------------------------------------------------------- utilidades

def _check_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if not (0.0 < alpha < 0.5):
        raise ValueError(
            f"alpha = {alpha}: aqui alpha es el nivel de significacion de los "
            "contrastes (alpha_contrastes, 0,05) y debe estar entre 0 y 0,5. Si "
            "viene de protocolo.alpha del yaml (0,99), ese es el nivel de "
            "confianza del limite de control, otra cosa."
        )
    return alpha


def _check_alternative(alternative: str) -> str:
    if alternative not in _ALTERNATIVAS:
        raise ValueError(f"alternative debe ser una de {_ALTERNATIVAS}, no {alternative!r}")
    return alternative


def _check_margin(valor, nombre: str) -> float:
    """Un margen (delta, banda del suelo) tiene que ser un numero finito >= 0."""
    if valor is None or isinstance(valor, (bool, np.bool_)):
        raise ValueError(f"{nombre} debe ser un numero finito >= 0, no {valor!r}")
    try:
        x = float(valor)
    except (TypeError, ValueError):
        raise ValueError(f"{nombre} debe ser un numero finito >= 0, no {valor!r}") from None
    if not np.isfinite(x) or x < 0:
        raise ValueError(f"{nombre} debe ser un numero finito >= 0, no {valor!r}")
    return x


def _round_noise(x: np.ndarray, scale: float) -> np.ndarray:
    """Redondea a _DIGITS cifras significativas relativas a `scale` (ver _DIGITS).

    Con escalas diminutas (por debajo de unos 1e-290) np.round necesitaria mas
    de 300 decimales y devolveria NaN; ahi no se redondea (M8).
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite(scale) or scale <= 0:
        return x
    decimales = _DIGITS - 1 - int(math.floor(math.log10(scale)))
    if decimales > 300:
        return x
    return np.round(x, decimales)


def _finitos(x: np.ndarray, que: str) -> np.ndarray:
    """Comprueba que no hay infinitos (un NaN si vale: es "no evaluada")."""
    x = np.asarray(x, dtype=float)
    if np.isinf(x).any():
        raise ValueError(
            f"{que} tiene valores infinitos en las posiciones "
            f"{np.flatnonzero(np.isinf(x))[:10].tolist()}: una metrica por simulacion "
            "tiene que ser finita, o NaN si no se evaluo (por ejemplo, un retardo sin "
            "deteccion)."
        )
    return x


def _warn_low_power(mensaje: str, casos=None, stacklevel: int = 2) -> None:
    """Emite un LowPowerWarning: un mensaje, o una cabecera y la lista de casos.

    Con `casos`, avisa solo si la lista no esta vacia y muestra los diez
    primeros (y cuantos mas quedan). `stacklevel` es el que usaria quien llama
    (2 = la linea que llamo a la funcion publica).
    """
    if casos is not None:
        if not casos:
            return
        mensaje = (mensaje + "; ".join(casos[:10])
                   + (f"; y {len(casos) - 10} mas" if len(casos) > 10 else ""))
    warnings.warn(mensaje, LowPowerWarning, stacklevel=stacklevel + 1)


def _max_abs(*arrays) -> float:
    vals = [float(np.max(np.abs(a))) for a in arrays if np.size(a)]
    return max(vals) if vals else 0.0


def _pmf_sign_sum(weights) -> tuple[np.ndarray, int]:
    """Distribucion de S = suma de B_i * w_i, con B_i ~ Bernoulli(1/2) independientes.

    Es la distribucion nula del estadistico de rangos con signo: si las
    diferencias son simetricas alrededor de 0, cada diferencia no nula es
    positiva o negativa con probabilidad 1/2, condicionado a los valores
    absolutos observados. Los pesos son los rangos multiplicados por 2 (enteros
    aunque haya rangos medios). Se calcula por programacion dinamica, sumando
    los pesos de menor a mayor para que el soporte crezca despacio; los conteos
    se dividen por 2^n al final (y cada 512 pasos, para no desbordar el float).

    Devuelve la funcion de probabilidad sobre 0, g, 2g, ..., sum(w) (indexada
    por suma / g) y el divisor comun g.
    """
    w = np.sort(np.asarray(weights, dtype=np.int64))
    if w.size == 0:
        return np.ones(1), 1
    g = int(np.gcd.reduce(w))
    w = w // g
    conteo = np.zeros(int(w.sum()) + 1)
    conteo[0] = 1.0
    alto = 0
    pendiente = 0
    for wi in w:
        wi = int(wi)
        # S nueva = S vieja (signo -) o S vieja + wi (signo +). numpy resuelve
        # bien el solapamiento de los dos tramos en la suma en sitio.
        conteo[wi:alto + wi + 1] += conteo[:alto + 1]
        alto += wi
        pendiente += 1
        if pendiente == 512:
            conteo[:alto + 1] *= 2.0 ** -512
            pendiente = 0
    conteo *= 2.0 ** -pendiente
    return conteo, g


@lru_cache(maxsize=32)
def _null_cdf_no_ties(n: int) -> np.ndarray:
    """Funcion de distribucion exacta de W+ con n diferencias sin empates ni ceros."""
    pmf, _ = _pmf_sign_sum(np.arange(1, n + 1))
    cdf = np.cumsum(pmf)
    cdf.setflags(write=False)
    return cdf


def _signed_rank(d: np.ndarray) -> dict:
    """Estadistico de rangos con signo con los ceros repartidos (zsplit) y sus p.

    `d` ya viene sin NaN y redondeado. Con zsplit, cada cero entra en la
    ordenacion y la mitad de su rango va a W+ y la otra mitad a W-, igual que
    `scipy.stats.wilcoxon(..., zero_method="zsplit")`. Los ceros no cambian de
    signo bajo H0, asi que no aportan variabilidad: la distribucion nula
    exacta solo recorre los signos de las diferencias no nulas, y por eso la
    eleccion entre el p exacto y la normal se hace por su numero (EXACT_MAX_N y
    los topes de la programacion dinamica), no por el de pares.
    """
    n = int(d.size)
    rangos = sps.rankdata(np.abs(d))
    pos, neg, cero = d > 0, d < 0, d == 0
    medio_cero = float(rangos[cero].sum()) / 2.0
    w_plus = float(rangos[pos].sum()) + medio_cero
    w_minus = float(rangos[neg].sum()) + medio_cero
    n_nonzero = int(n - cero.sum())
    pesos = np.rint(2.0 * rangos[~cero]).astype(np.int64)
    suma_pesos = int(pesos.sum())

    if n_nonzero == 0:
        # Todas las diferencias son 0: el estadistico es constante bajo
        # cualquier cambio de signo, asi que no hay evidencia en ningun sentido.
        p_less = p_greater = 1.0
        metodo = "exact"
    elif (n_nonzero <= EXACT_MAX_N and suma_pesos <= _EXACT_MAX_SIZE
          and suma_pesos * n_nonzero <= _EXACT_MAX_WORK):
        pmf, g = _pmf_sign_sum(pesos)
        s_obs = int(np.rint(2.0 * float(rangos[pos].sum()))) // g
        p_less = float(min(1.0, pmf[: s_obs + 1].sum()))
        p_greater = float(min(1.0, pmf[s_obs:].sum()))
        metodo = "exact"
    else:
        media = n * (n + 1) / 4.0
        sd = math.sqrt(float((rangos[~cero] ** 2).sum()) / 4.0)
        p_less = float(sps.norm.cdf((w_plus - media + 0.5) / sd))
        p_greater = float(sps.norm.sf((w_plus - media - 0.5) / sd))
        metodo = "normal"

    return {
        "n": n,
        "n_nonzero": n_nonzero,
        "w_plus": w_plus,
        "w_minus": w_minus,
        # Biserial de rangos para datos pareados: (W+ - W-) / (W+ + W-), con
        # W+ + W- = n(n+1)/2. Va de -1 (todas negativas) a 1 (todas positivas).
        "rank_biserial": (w_plus - w_minus) / (n * (n + 1) / 2.0) if n else np.nan,
        "p_less": p_less,
        "p_greater": p_greater,
        "p_two_sided": float(min(1.0, 2.0 * min(p_less, p_greater))),
        "p_method": metodo,
    }


def _rank_biserial(d: np.ndarray) -> float:
    """Biserial de rangos de d frente a 0, con los ceros repartidos (sin calcular el p)."""
    n = int(d.size)
    if n == 0:
        return np.nan
    rangos = sps.rankdata(np.abs(d))
    return float((rangos[d > 0].sum() - rangos[d < 0].sum()) / (n * (n + 1) / 2.0))


def _low_power(n_nonzero: int, alpha: float, m: int, alternative: str) -> int | None:
    """Devuelve el n minimo si con n_nonzero diferencias no se puede llegar a alpha/m."""
    n_min = min_runs_for_significance(alpha, m, alternative)
    return n_min if n_nonzero < n_min else None


def _frente_a(d: np.ndarray, umbral: float) -> dict:
    """Wilcoxon de las diferencias d frente a un umbral: los dos p unilaterales.

    Es la pieza comun de todas las reglas que comparan una diferencia con un
    valor de referencia: la no superioridad (umbral = delta), el suelo de
    deteccion (umbral = banda sobre la FAR), la ventaja publicada (umbral = la
    cifra publicada) y la equivalencia de las ablaciones (umbral = +-margen).
    Contrasta la pseudomediana theta de d sobre x = d - umbral, con los ceros
    repartidos y el p exacto de _signed_rank. `d` ya viene sin NaN y
    redondeado; la resta con el umbral se vuelve a redondear (ver _DIGITS),
    para que una diferencia igual al umbral cuente como cero.

    Devuelve p_below (H1: theta < umbral), p_above (H1: theta > umbral),
    w_plus (W+ de x), n_nonzero (diferencias no nulas de x) y p_method.
    """
    x = _round_noise(d - umbral, max(_max_abs(d), abs(umbral)))
    sr = _signed_rank(x)
    return {"p_below": sr["p_less"], "p_above": sr["p_greater"], "w_plus": sr["w_plus"],
            "n_nonzero": sr["n_nonzero"], "p_method": sr["p_method"]}


def _dos_unilaterales(d: np.ndarray, umbral: float, alpha: float) -> dict:
    """Los dos unilaterales de d frente a un umbral, con Hodges-Lehmann al 1 - 2 alpha.

    Es el nucleo comun de non_superiority_test (umbral = delta) y de
    published_advantage_test (umbral = la cifra publicada), que son la misma
    regla con otro umbral: asi no pueden divergir. `d` ya viene sin NaN ni
    infinitos y con al menos MIN_PAIRS valores.
    """
    d = _round_noise(d, _max_abs(d))
    hl = hodges_lehmann(d, 1.0 - 2.0 * alpha)
    fr = _frente_a(d, umbral)
    return {"hl_estimate": hl["hl_estimate"], "ci_low": hl["ci_low"],
            "ci_high": hl["ci_high"], "ci_level": hl["ci_level"],
            "p_below": fr["p_below"], "p_above": fr["p_above"], "statistic": fr["w_plus"],
            "effect_size": float(_rank_biserial(d)), "n_nonzero": fr["n_nonzero"],
            "p_method": fr["p_method"]}


# ------------------------------------------------------------- potencia

def min_runs_for_significance(alpha: float = 0.05, m: int = 1,
                              alternative: str = "two-sided") -> int:
    """Numero minimo de simulaciones para que Wilcoxon pueda salir significativo.

    Con pocas simulaciones Wilcoxon no puede rechazar nunca, haga lo que haga
    el metodo: con n diferencias no nulas, el p exacto mas pequeno posible es
    el de todas del mismo signo, 2/2^n en el bilateral y 1/2^n en el
    unilateral. Por ejemplo, con 5 replicas el minimo bilateral es 2/2^5, mayor
    que 0,05. Con Holm sobre m contrastes el primer paso exige p <= alpha/m, asi
    que hace falta todavia mas. Por debajo de este n, "no significativo" lo
    decide el tamano de muestra y no los datos, que es justo el error de diseno
    que el TFM denuncia (y una hipotesis que predice "no significativa" se
    cumpliria sola).

    Los ceros (simulaciones con la misma metrica en los dos metodos) no cuentan:
    con zero_method="zsplit" no cambian de signo bajo H0, asi que el p minimo
    depende del numero de diferencias no nulas.

    Parametros
    ----------
    alpha : float
        Nivel de significacion de los contrastes (alpha_contrastes).
    m : int
        Numero de contrastes de la familia de Holm (1 si no se corrige; 6 en la
        familia confirmatoria).
    alternative : {"two-sided", "less", "greater"}
        Los unilaterales necesitan una simulacion menos que el bilateral.

    Devuelve
    --------
    int, el menor n con p minimo <= alpha/m.
    """
    alpha = _check_alpha(alpha)
    _check_alternative(alternative)
    if int(m) != m or m < 1:
        raise ValueError(f"m debe ser un entero >= 1, no {m!r}")
    colas = 2 if alternative == "two-sided" else 1
    n = 1
    while colas * m > alpha * 2.0 ** n:   # p minimo = colas / 2^n > alpha / m
        n += 1
    return n


# ---------------------------------------------------------- estimacion

def hodges_lehmann(d, ci_level: float = 0.95) -> dict:
    """Estimador de Hodges-Lehmann de la diferencia pareada, con su intervalo.

    Wilcoxon de rangos con signo no contrasta la media ni la mediana de las
    diferencias, sino su pseudomediana: la mediana de los promedios de Walsh
    (d_i + d_j) / 2, con i <= j. El estimador de Hodges-Lehmann es justo esa
    mediana, y su intervalo es el conjunto de valores theta que el propio
    contraste no rechaza sobre d - theta: el estadistico W+ sobre d - theta es
    el numero de promedios de Walsh mayores que theta (mas la mitad de los
    iguales, con los ceros repartidos), asi que los extremos del intervalo son
    dos promedios de Walsh ordenados. Asi se reporta la diferencia en la misma
    escala en que se decide, que es lo que hace falta para compararla con una
    diferencia minima relevante. La mediana de las diferencias no sirve: con
    muchos empates en 0 vale 0 aunque el contraste sea significativo (M8).

    El intervalo usa la distribucion exacta de W+ sin empates, que es la
    habitual; con n > EXACT_MAX_N, su aproximacion normal. Sin empates su
    cobertura es exacta (al menos ci_level). Con empates (las metricas son
    fracciones de muestras) es aproximada y tiende a ser conservadora (algo
    mas ancho), y un extremo puede coincidir con el valor que se contrasta:
    con 30 ceros y 20 diferencias de +0,02 el intervalo al 95 % empieza en 0
    aunque el contraste rechace 0, porque 0 es un promedio de Walsh repetido y
    el conjunto de valores no rechazados es abierto por ese lado. Por eso las
    decisiones se toman con el p, que si es exacto con empates. Se prefiere a
    un bootstrap pareado porque no depende de una semilla, es coherente con
    Wilcoxon y con 20 simulaciones un bootstrap de la mediana es muy discreto.

    La cobertura exacta supone que las diferencias son simetricas alrededor de
    su pseudomediana, el mismo supuesto que hace exacto a Wilcoxon. Con
    asimetria fuerte, el nivel es aproximado.

    Parametros
    ----------
    d : array
        Diferencias pareadas, una por simulacion. Los NaN se descartan.
    ci_level : float
        Nivel del intervalo bilateral.

    Devuelve
    --------
    dict con 'hl_estimate', 'ci_low', 'ci_high' y 'ci_level'. Si con esos pares
    ningun valor se puede rechazar (por ejemplo, 5 pares al 95 %), el
    intervalo es (-inf, inf).
    """
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level debe estar entre 0 y 1, no {ci_level!r}")
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    n = int(d.size)
    if n == 0:
        return {"hl_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "ci_level": float(ci_level)}

    i, j = np.triu_indices(n)
    walsh = np.sort((d[i] + d[j]) / 2.0)
    total = walsh.size                       # n(n+1)/2
    estimacion = float(np.median(walsh))

    # c - 1 = mayor t con P(W+ <= t) <= (1 - ci_level)/2: los valores de W+
    # hasta ahi se rechazan en el bilateral. El intervalo es [W_(c), W_(M+1-c)].
    cola = (1.0 - ci_level) / 2.0
    if n <= EXACT_MAX_N:
        cdf = _null_cdf_no_ties(n)
        rechazados = np.flatnonzero(cdf <= cola * (1.0 + 1e-9))
        c = int(rechazados[-1]) + 1 if rechazados.size else 0
    else:
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        t = math.floor(total / 2.0 - 0.5 - sps.norm.ppf(1.0 - cola) * sd)
        c = t + 1 if t >= 0 else 0

    if c < 1:
        bajo, alto = -np.inf, np.inf
    else:
        bajo, alto = float(walsh[c - 1]), float(walsh[total - c])
    return {"hl_estimate": estimacion, "ci_low": bajo, "ci_high": alto,
            "ci_level": float(ci_level)}


# ------------------------------------------------------------ contrastes

def paired_test(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided",
                alpha: float = 0.05, ci_level: float | None = None) -> dict:
    """Wilcoxon pareado entre dos metodos sobre las mismas simulaciones.

    Contrasta las diferencias a - b, una por simulacion, con los ceros
    repartidos entre las dos colas (zero_method="zsplit"), porque las metricas
    acotadas producen muchas diferencias nulas. El p es exacto hasta
    EXACT_MAX_N diferencias no nulas, tambien con empates y ceros (ver
    _signed_rank), y no depende de la version de scipy.

    Lo que contrasta: que a - b es simetrica alrededor de 0. Su nivel es exacto
    con esa simetria; si la diferencia es asimetrica, el p no se refiere a su
    mediana ni a su media, y el nivel frente a "pseudomediana = 0" es
    aproximado (ver "Supuesto de simetria" en non_superiority_test).

    Ademas del p, reporta la diferencia en la escala del contraste: el
    estimador de Hodges-Lehmann de a - b con su intervalo (ver hodges_lehmann).
    Un p sin la diferencia al lado no dice si la ventaja importa.

    El tamano de efecto es la correlacion biserial de rangos para datos
    pareados, (W+ - W-) / (W+ + W-). Se eligio frente a r = Z / sqrt(n) porque
    lleva signo por construccion (positivo si a tiende a ser mayor que b), va
    de -1 a 1, no depende de la alternativa ni de reconstruir Z a partir del p
    (la version anterior lo hacia siempre con 1 - p/2 y fallaba en los
    unilaterales: M8), y vale 0 si todas las diferencias son nulas (MET-9).

    Parametros
    ----------
    a, b : array
        Metrica de cada metodo en las mismas simulaciones, en el mismo orden.
        Los pares con algun NaN se descartan; un infinito es un error (una
        metrica por simulacion es finita, o NaN si no se evaluo).
    alternative : {"two-sided", "less", "greater"}
        Hipotesis alternativa sobre la pseudomediana de a - b.
    alpha : float
        Nivel de significacion (alpha_contrastes). Solo se usa para el nivel
        del intervalo por defecto y para el aviso de potencia: este contraste
        no decide nada, devuelve el p.
    ci_level : float, opcional
        Nivel del intervalo. Por defecto 1 - alpha en el bilateral y 1 - 2 alpha
        en los unilaterales (cada extremo es entonces la cota unilateral al
        1 - alpha que corresponde al contraste).

    Devuelve
    --------
    dict con las claves de siempre:
      n, statistic (convencion de scipy: min(W+, W-) en el bilateral y W+ en
      los unilaterales, con los ceros repartidos), p_value, effect_size
      (biserial de rangos, con signo) y median_difference (mediana de a - b);
    y las nuevas:
      hl_estimate, ci_low, ci_high, ci_level, n_nonzero (diferencias no
      nulas), p_method ("exact" o "normal") y alternative.
    Con menos de MIN_PAIRS pares todo vale NaN salvo n.
    """
    alternative = _check_alternative(alternative)
    alpha = _check_alpha(alpha)
    if ci_level is None:
        ci_level = 1.0 - alpha if alternative == "two-sided" else 1.0 - 2.0 * alpha

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Longitudes distintas: {a.shape} y {b.shape}")

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = _finitos(a[mask], "a"), _finitos(b[mask], "b")
    n = int(a.size)

    if n < MIN_PAIRS:
        return {"n": n, "statistic": np.nan, "p_value": np.nan,
                "effect_size": np.nan, "median_difference": np.nan,
                "hl_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "ci_level": float(ci_level), "n_nonzero": np.nan,
                "p_method": None, "alternative": alternative}

    d = _round_noise(a - b, _max_abs(a, b))
    sr = _signed_rank(d)
    hl = hodges_lehmann(d, ci_level)

    if alternative == "two-sided":
        p = sr["p_two_sided"]
        estadistico = min(sr["w_plus"], sr["w_minus"])
    else:
        p = sr["p_less"] if alternative == "less" else sr["p_greater"]
        estadistico = sr["w_plus"]

    n_min = _low_power(sr["n_nonzero"], alpha, 1, alternative)
    if n_min is not None:
        _warn_low_power(
            f"paired_test con {sr['n_nonzero']} diferencias no nulas (n = {n}): "
            f"ningun resultado puede dar p <= {alpha} ({alternative}); hacen falta "
            f"al menos {n_min}. Ver min_runs_for_significance.")

    return {
        "n": n,
        "statistic": float(estadistico),
        "p_value": float(p),
        "effect_size": float(sr["rank_biserial"]),
        "median_difference": float(np.median(d)),
        "hl_estimate": hl["hl_estimate"],
        "ci_low": hl["ci_low"],
        "ci_high": hl["ci_high"],
        "ci_level": hl["ci_level"],
        "n_nonzero": sr["n_nonzero"],
        "p_method": sr["p_method"],
        "alternative": alternative,
    }


def holm_correction(p_values: list[float], alpha: float = 0.05) -> pd.DataFrame:
    """Correccion de Holm-Bonferroni para comparaciones multiples.

    Comparar cinco metodos entre si son diez contrastes; con alpha = 0.05 cabe
    esperar medio falso positivo solo por azar. Holm ajusta el umbral y es menos
    conservador que Bonferroni sin perder control del error.

    Ademas del umbral de cada paso devuelve el p ajustado,
    p_adj(i) = max_{j <= i} min(1, (m - j + 1) p(j)) con los p ordenados de
    menor a mayor, que se compara directamente con alpha. Leido en un CSV, el p
    ajustado no confunde; el umbral de un p que va detras de uno no
    significativo, si (MET-8). `significant` es el procedimiento por pasos: se
    rechaza mientras p(i) <= alpha / (m - i + 1) y se para en el primero que
    no. Coincide con p_adjusted <= alpha salvo en el umbral exacto, donde el
    producto (m - i + 1) p(i) puede quedar un ulp por encima de alpha por coma
    flotante (con m = 11 y p = 0,05/11 da 0,05000000000000001; revision del
    12/10, M5): ahi manda el procedimiento por pasos.

    Los p NaN (pares no contrastables, por ejemplo con menos de MIN_PAIRS
    simulaciones) quedan fuera de la familia: m es el numero de p validos, y
    esas filas salen con rank, threshold y p_adjusted vacios y significant
    False. Antes contaban en m y volvian a Holm mas conservador sin decirlo
    (MET-8). Si se quiere que una comparacion declarada cuente en la familia
    aunque no se pueda contrastar, se le pone p = 1 antes de llamar (es lo que
    hace hypothesis_verdict).
    """
    alpha = _check_alpha(alpha)
    p = np.asarray(p_values, dtype=float).reshape(-1)
    validos = ~np.isnan(p)
    if np.any((p[validos] < 0) | (p[validos] > 1)):
        raise ValueError("Hay valores p fuera de [0, 1].")

    n = p.size
    rango = pd.array([pd.NA] * n, dtype="Int64")
    umbral = np.full(n, np.nan)
    ajustado = np.full(n, np.nan)
    significativo = np.zeros(n, dtype=bool)

    m = int(validos.sum())
    if m:
        idx = np.flatnonzero(validos)
        orden = idx[np.argsort(p[idx], kind="stable")]
        k = np.arange(1, m + 1)
        factor = m - k + 1
        adj = np.minimum(1.0, np.maximum.accumulate(factor * p[orden]))
        rango[orden] = k
        umbral[orden] = alpha / factor
        ajustado[orden] = adj
        # Por pasos: se rechaza hasta el primer p que no pasa su umbral.
        significativo[orden] = np.minimum.accumulate(p[orden] <= alpha / factor)

    return pd.DataFrame({
        "p_value": p,
        "rank": rango,
        "threshold": umbral,
        "p_adjusted": ajustado,
        "significant": significativo,
    })


#: Unidades del contraste con replicas de ajuste (ver compare_methods).
_UNIDADES = ("replicate", "simulation")


def _check_unit(replicate_col, unit, run_col: str, columnas) -> str | None:
    """La unidad con replicas de ajuste es una decision: obligatoria y explicita."""
    if replicate_col is None:
        if unit is not None:
            raise ValueError("unit solo tiene sentido con replicate_col (replicas de ajuste).")
        return None
    if unit is None:
        raise ValueError(
            "Con replicate_col hay que declarar la unidad del contraste: "
            "unit='replicate' (la media de cada replica de ajuste; generaliza a otras "
            "replicas, condicionado a las simulaciones de prueba usadas) o "
            "unit='simulation' (la media de las replicas en cada simulacion de prueba; "
            "generaliza a otras simulaciones, condicionado a las replicas). Es una "
            "decision del registro (ablaciones.unidad_replica en configs/baseline.yaml); "
            "la memoria dice que la unidad de analisis es la simulacion (parrafo 367)."
        )
    if unit not in _UNIDADES:
        raise ValueError(f"unit debe ser una de {_UNIDADES}, no {unit!r}")
    if unit == "simulation" and run_col not in columnas:
        raise ValueError(f"unit='simulation' necesita la columna de simulacion {run_col!r}.")
    return unit


def _paired_units(g: pd.DataFrame, metric: str, method_col: str, run_col: str,
                  replicate_col: str | None, unit: str | None, m1, m2) -> tuple:
    """Valores emparejados de dos metodos (o brazos) dentro de un grupo, por unidad.

    Sin `replicate_col`, la unidad es la simulacion: se empareja por run_col,
    como siempre. Con `replicate_col`, se emparejan las celdas (replica,
    simulacion), se descartan los pares con algun NaN y se promedian a y b
    hasta la unidad declarada: dentro de cada replica (unit="replicate") o
    dentro de cada simulacion de prueba (unit="simulation"). Si la tabla no
    tiene run_col, cada replica tiene ya una fila por metodo.

    Devuelve a, b (uno por unidad, en el mismo orden), el numero de celdas
    emparejadas sin NaN (None sin replicate_col) y el de simulaciones (o
    celdas) que solo estan en uno de los dos metodos y se descartan (M2).
    """
    if replicate_col is None:
        claves, que = run_col, "simulacion"
    elif run_col in g:
        claves, que = [replicate_col, run_col], "(replica, simulacion)"
    else:
        claves, que = replicate_col, "replica"
    d1 = g[g[method_col] == m1].set_index(claves)[metric]
    d2 = g[g[method_col] == m2].set_index(claves)[metric]
    if d1.index.has_duplicates or d2.index.has_duplicates:
        raise ValueError(
            f"Hay varias filas por {que} para un mismo metodo; agrupa por la columna "
            "que las distingue (fault_col en compare_methods, by en compare_arms) o "
            "filtra la tabla."
        )
    comunes = d1.index.intersection(d2.index)
    sueltas = len(d1.index.union(d2.index)) - len(comunes)
    a = d1.loc[comunes].to_numpy(dtype=float, na_value=np.nan)
    b = d2.loc[comunes].to_numpy(dtype=float, na_value=np.nan)
    if replicate_col is None:
        return a, b, None, sueltas
    validos = ~(np.isnan(a) | np.isnan(b))
    nivel = 1 if (unit == "simulation" and isinstance(comunes, pd.MultiIndex)) else 0
    claves_unidad = comunes.get_level_values(nivel) if isinstance(comunes, pd.MultiIndex) else comunes
    medias = (pd.DataFrame({"u": np.asarray(claves_unidad)[validos], "a": a[validos],
                            "b": b[validos]})
              .groupby("u", sort=True).mean())
    return medias["a"].to_numpy(), medias["b"].to_numpy(), int(validos.sum()), sueltas


def _unpaired_note(donde: str, m1, m2, sueltas: int, n_comunes: int) -> str:
    """Texto del aviso de simulaciones sin pareja (M2)."""
    extra = " (ninguna en comun: ids de tipos distintos?)" if n_comunes == 0 else ""
    return f"{donde}{m1} frente a {m2}: {sueltas} en un solo metodo{extra}"


def _warn_unpaired(funcion: str, notas: list[str]) -> None:
    if notas:
        warnings.warn(
            f"{funcion}: se descartan simulaciones (o celdas) que solo estan en uno de los "
            "dos metodos; los dos deben evaluarse sobre los mismos ids: "
            + "; ".join(notas[:10]) + (f"; y {len(notas) - 10} mas" if len(notas) > 10 else ""),
            stacklevel=3,
        )


def _compare_group(g: pd.DataFrame, metric: str, method_col: str, run_col: str,
                   baseline: str | None, alpha: float, alternative: str,
                   replicate_col: str | None = None, unit: str | None = None,
                   donde: str = "", notas: list | None = None) -> pd.DataFrame:
    """Contrastes de un grupo (un fallo, o la tabla entera), todavia sin Holm.

    Las comparaciones con simulaciones sin pareja se anotan en `notas` (si se
    pasa), para avisar una sola vez desde la funcion publica.
    """
    metodos = sorted(g[method_col].unique())
    if baseline is None:
        pares = [(m1, m2) for i, m1 in enumerate(metodos) for m2 in metodos[i + 1:]]
    else:
        if baseline not in metodos:
            raise ValueError(
                f"El baseline {baseline!r} no esta entre los metodos {metodos}"
                " de este grupo."
            )
        pares = [(m, baseline) for m in metodos if m != baseline]

    filas = []
    for m1, m2 in pares:
        a, b, celdas, sueltas = _paired_units(g, metric, method_col, run_col, replicate_col,
                                              unit, m1, m2)
        if sueltas and notas is not None:
            notas.append(_unpaired_note(donde, m1, m2, sueltas, a.size))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LowPowerWarning)
            r = paired_test(a, b, alternative=alternative, alpha=alpha)
        fila = {"method_a": m1, "method_b": m2}
        if replicate_col is not None:
            fila.update({"unit": replicate_col if unit == "replicate" else run_col,
                         "n_cells": celdas})
        filas.append({**fila, **r})
    return pd.DataFrame(filas)


def _check_family(family, permitidas: list) -> list:
    """Columnas que definen las familias de Holm: una str o una secuencia (vacia = una sola familia)."""
    if isinstance(family, str):
        family = [family]
    try:
        columnas = list(family)
    except TypeError:
        raise ValueError(
            f"family = {family!r}: pasa una lista de columnas de {permitidas} (una familia "
            "por cada combinacion de sus valores), o () para una sola familia con todo."
        ) from None
    for c in columnas:
        if c not in permitidas:
            raise ValueError(
                f"family: {c!r} no es una de las columnas que identifican una "
                f"comparacion ({permitidas})."
            )
    if len(set(columnas)) != len(columnas):
        raise ValueError(f"family tiene columnas repetidas: {columnas}")
    return columnas


def _families(tabla: pd.DataFrame, columnas: list) -> list:
    """Indices de las filas de cada familia, en el orden de sus claves."""
    if not columnas:
        return [tabla.index]
    return [g.index for _, g in tabla.groupby(columnas, sort=True, dropna=False)]


def _holm_by_family(tabla: pd.DataFrame, columnas: list, alpha: float, p_cols: dict,
                    nan_as_one: bool = False) -> np.ndarray:
    """Holm dentro de cada familia, para cada columna de p de `p_cols`.

    `p_cols` es {columna_p: (columna_ajustada, columna_decision)}: escribe en
    `tabla` los p ajustados y la decision del procedimiento por pasos de
    holm_correction (no p_ajustado <= alpha, que difiere en el umbral exacto;
    M5). Devuelve el numero de p que entran en la familia de cada fila (el m de
    Holm). Con nan_as_one, un p NaN (comparacion no contrastable) cuenta en la
    familia con p = 1, para que m sea el declarado; sin el, queda fuera de la
    familia, como en holm_correction.
    """
    for ajustado, decision in p_cols.values():
        tabla[ajustado] = np.nan
        tabla[decision] = False
    m = np.zeros(len(tabla), dtype=np.int64)
    posicion = pd.Series(np.arange(len(tabla)), index=tabla.index)
    primera = next(iter(p_cols))
    for idx in _families(tabla, columnas):
        for col_p, (ajustado, decision) in p_cols.items():
            p = tabla.loc[idx, col_p].to_numpy(dtype=float)
            if nan_as_one:
                p = np.where(np.isnan(p), 1.0, p)
            h = holm_correction(p, alpha=alpha)
            tabla.loc[idx, ajustado] = h["p_adjusted"].to_numpy()
            tabla.loc[idx, decision] = h["significant"].to_numpy()
        validos = len(idx) if nan_as_one else int(tabla.loc[idx, primera].notna().sum())
        m[posicion.loc[idx].to_numpy()] = validos
    return m


def compare_methods(results: pd.DataFrame, metric: str = "fdr",
                    method_col: str = "method", run_col: str = "simulationRun",
                    fault_col: str | None = "faultNumber", *,
                    baseline: str | None = None, alpha: float = 0.05,
                    alternative: str = "two-sided", replicate_col: str | None = None,
                    family=None, unit: str | None = None) -> pd.DataFrame:
    """Compara metodos sobre una metrica, pareando por simulacion, con correccion de Holm.

    `results` debe tener una fila por metodo y simulacion (y fallo, si hay
    varios). Los pares se emparejan por simulacion Y por fallo: si `fault_col`
    esta en la tabla, el contraste se hace dentro de cada fallo, con su propia
    correccion de Holm entre las comparaciones de ese fallo. Antes se
    indexaba solo por simulacion y, con varios fallos en la misma tabla, los
    indices se duplicaban y los pares se desalineaban en silencio.

    Dos analisis (regla 8 y `decisiones.md`, 22/09):

    - Con `baseline` (analisis principal): las k - 1 comparaciones de cada
      metodo contra el baseline, con Holm sobre esas k - 1 dentro de cada
      fallo. `method_a` es el otro metodo y `method_b` el baseline, asi que la
      diferencia es metodo - baseline.
    - Sin `baseline` (secundario, lo de siempre): todos contra todos, con los
      metodos en orden alfabetico (`method_a` < `method_b`) y Holm sobre todos
      los pares de cada fallo.

    El signo de las diferencias es el de la metrica (a - b): en FAR y retardo,
    donde menos es mejor, una diferencia negativa favorece a `method_a`.

    Las simulaciones que solo tiene uno de los dos metodos se descartan con un
    aviso, como en paired_advantage: los metodos tienen que evaluarse sobre los
    mismos ids (regla 3).

    Como todos los contrastes del modulo, el Wilcoxon de cada comparacion
    supone que la diferencia es simetrica alrededor de su pseudomediana; con
    asimetria, el nivel es aproximado (ver "Supuesto de simetria" en
    non_superiority_test).

    Replicas de ajuste (`replicate_col` y `unit`). En la ablacion de la
    calibracion, cada brazo se ajusta R veces con simulaciones normales
    distintas (las replicas) y se evalua sobre las mismas N simulaciones de
    prueba. Las R x N celdas estan cruzadas: las N de una replica comparten el
    modelo y el limite, y las R de una simulacion comparten la simulacion de
    prueba. Un Wilcoxon sobre las R x N celdas como si fueran independientes
    estaria pseudorreplicado (informe 06, 2.1) y daria p demasiado pequenos.
    Con `replicate_col`, las celdas se emparejan por (replica, simulacion), se
    descartan las que tienen NaN (un retardo no detectado sale del promedio,
    con el sesgo de DIS-12) y se promedian hasta la unidad que se declare en
    `unit`, sin valor por defecto, porque cada una contesta otra pregunta (es
    la decision ablaciones.unidad_replica del registro):

    - unit="replicate": la media de las N simulaciones de prueba dentro de cada
      replica, y el Wilcoxon sobre las R medias. Generaliza a otras replicas de
      ajuste (otros conjuntos normales de ajuste y calibracion), pero
      condicionado a las N simulaciones de prueba usadas: si el efecto cambia
      de una simulacion de prueba a otra, las R medias comparten ese termino y
      el contraste no vale para la poblacion de simulaciones de prueba (el error
      de tipo I frente a ella sube muy por encima de alpha; revision del 12/10,
      I4).
    - unit="simulation": la media de las R replicas dentro de cada simulacion
      de prueba, y el Wilcoxon sobre las N medias. Es la unidad de analisis de
      la memoria ("La unidad de analisis sera la simulacion", parrafo 367).
      Generaliza a otras simulaciones de prueba, pero condicionado a las R
      replicas de ajuste: si el efecto cambia de una replica a otra, las N
      medias comparten ese termino, con el mismo problema en espejo.

    Si cada replica se evalua sobre sus propias simulaciones de prueba (un
    diseno anidado, no cruzado), unit="replicate" ya generaliza a las dos
    poblaciones, porque cada media de replica lleva su propia muestra de
    simulaciones (revision del 12/10, segunda pasada de I4). Con el diseno
    cruzado, generalizar a las dos a la vez exige un contraste para disenos
    cruzados (por ejemplo, una t con dos factores aleatorios y los grados de
    libertad de Satterthwaite; en la misma revision, el bootstrap que remuestrea
    a la vez replicas y simulaciones salio anticonservador con pocas replicas),
    que no esta implementado. Con pocas unidades no
    puede salir nada (min_runs_for_significance: con Holm sobre 4, hacen falta
    8).

    Familias de Holm (`family`). Por defecto (None), Holm dentro de cada fallo,
    como siempre, y una comparacion no contrastable (menos de MIN_PAIRS
    unidades) queda fuera de la familia de su fallo, como en holm_correction.
    La familia que pide una ablacion la fija el registro de decisiones y puede
    ser otra: por ejemplo, la misma comparacion en todos los fallos
    (family=["method_a"]) o todas las comparaciones de la llamada (family=()).
    `family` es la lista de columnas de la salida cuyas combinaciones de
    valores definen cada familia: de fault_col, "method_a" y "method_b". Con
    una familia declarada, las comparaciones no contrastables cuentan en ella
    con p = 1, como en compare_arms e hypothesis_verdict: la familia es la
    declarada y su tamano (family_m, el m de Holm) no depende de los datos.
    Para que la familia sean las metricas de un factor, o los estadisticos y
    los monitores a la vez, compare_arms.

    Parametros
    ----------
    results : DataFrame
        Formato comun de resultados (regla 11).
    metric : str
        Columna que se contrasta.
    method_col, run_col, fault_col : str
        Columnas de metodo, simulacion y fallo. Con fault_col=None no se agrupa.
    baseline : str, opcional
        Nombre del metodo de referencia (por ejemplo "pca").
    alpha : float
        Nivel de significacion de los contrastes (alpha_contrastes); se usa en
        Holm y en el intervalo de la diferencia (1 - alpha).
    alternative : {"two-sided", "less", "greater"}
        Alternativa de cada contraste, sobre method_a - method_b.
    replicate_col : str, opcional
        Columna de la replica de ajuste (ver arriba). Por defecto, sin
        replicas: la unidad es la simulacion.
    family : None, str o secuencia de str
        Columnas que definen las familias de Holm (ver arriba). None: dentro de
        cada fallo.
    unit : {"replicate", "simulation"}
        Obligatoria con replicate_col y sin valor por defecto: la unidad a la que
        se promedian las celdas (ver arriba). Sin replicate_col no se usa.

    Devuelve
    --------
    DataFrame con una fila por comparacion: las columnas de paired_test, mas
    method_a, method_b, alpha, p_adjusted (Holm) y significant (la decision del
    procedimiento por pasos de Holm). Con replicate_col, ademas unit (la
    columna de la unidad: la de la replica o la de la simulacion) y n_cells
    (celdas emparejadas), y `n` es el numero de unidades. Con family, ademas
    family_m (el tamano de la familia de cada fila, que es el m de Holm). Si
    alguna comparacion no tiene unidades suficientes para salir significativa
    tras Holm, emite un LowPowerWarning con la lista.
    """
    alpha = _check_alpha(alpha)
    alternative = _check_alternative(alternative)
    unit = _check_unit(replicate_col, unit, run_col, results.columns)

    agrupa = fault_col is not None and fault_col in results
    notas = []
    if agrupa:
        tablas = []
        for f, g in results.groupby(fault_col):
            t = _compare_group(g, metric, method_col, run_col, baseline, alpha, alternative,
                               replicate_col, unit, donde=f"fallo {f}, ", notas=notas)
            if t.empty:
                continue
            t.insert(0, fault_col, f)
            tablas.append(t)
        tabla = pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()
    else:
        tabla = _compare_group(results, metric, method_col, run_col, baseline, alpha,
                               alternative, replicate_col, unit, notas=notas)
    _warn_unpaired("compare_methods", notas)

    if family is None:
        columnas_familia = [fault_col] if agrupa else []
    else:
        columnas_familia = _check_family(
            family, ([fault_col] if agrupa else []) + ["method_a", "method_b"])

    if tabla.empty:
        return tabla
    tabla["alpha"] = alpha
    m_familia = _holm_by_family(tabla, columnas_familia, alpha,
                                {"p_value": ("p_adjusted", "significant")},
                                nan_as_one=family is not None)
    if family is not None:
        tabla["family_m"] = m_familia

    # Aviso de potencia: con Holm sobre m comparaciones hace falta p <= alpha/m.
    pobres = []
    for idx in _families(tabla, columnas_familia):
        m = len(idx)
        n_min = min_runs_for_significance(alpha, m, alternative)
        for _, fila in tabla.loc[idx].iterrows():
            nz = fila["n_nonzero"]
            if pd.isna(nz) or nz < n_min:
                donde = f"fallo {fila[fault_col]}, " if agrupa else ""
                cuantas = (f"menos de {MIN_PAIRS} pares" if pd.isna(nz)
                           else f"{int(nz)} diferencias no nulas")
                pobres.append(f"{donde}{fila['method_a']} frente a {fila['method_b']}: "
                              f"{cuantas} (hacen falta {n_min} con Holm sobre {m})")
    _warn_low_power("compare_methods: estas comparaciones no pueden salir significativas "
                    f"con alpha = {alpha}: ", pobres)
    return tabla


# ------------------------------------------- efecto de un factor (ablaciones)

def _effect_outcome(p_relevant: float, p_equivalent: float, alpha: float) -> str:
    if np.isnan(p_relevant):
        return OUTCOME_UNTESTABLE
    if p_relevant <= alpha:
        return EFFECT_RELEVANT
    if p_equivalent <= alpha:
        return EFFECT_EQUIVALENT
    return EFFECT_INCONCLUSIVE


def effect_test(difference, margin: float, alpha: float = 0.05) -> dict:
    """Efecto de un factor frente a un margen de equivalencia: relevante, equivalente a cero o inconcluso.

    La memoria cuantifica el efecto de cada fallo de diseno "con una regla de
    decision definida antes de ejecutar; un efecto equivalente a cero tambien
    es un resultado" (OE1; el OE2 dice lo mismo de su diferencia). Con un
    contraste de "hay diferencia o no", un efecto equivalente a cero no se
    puede afirmar nunca (no rechazar no es evidencia de ausencia), y con muchas
    simulaciones cualquier diferencia minima sale significativa. Por eso esta
    funcion decide frente a un margen de equivalencia delta, la diferencia
    minima relevante de la metrica: un efecto dentro de +-delta es "equivalente
    a cero" aunque sea distinto de cero, y uno fuera de +-delta es un "efecto
    relevante". Es la regla del intervalo frente a +-delta del informe 06
    (2.4) y del anexo de diseno (H-A), escrita con el mismo Wilcoxon exacto que
    el resto del modulo (_frente_a), para que sea igual en las cuatro
    ablaciones.

    Que se contrasta. theta es la pseudomediana de la diferencia por unidad
    (simulacion, o replica de ajuste si se agrego antes; ver compare_arms) y su
    estimador es el de Hodges-Lehmann. El signo no se presupone (memoria,
    2.2.4): el efecto se busca a los dos lados.

    - "efecto relevante": se rechaza H0: -delta <= theta <= delta. Son dos
      Wilcoxon unilaterales, sobre d - delta (H1: theta > delta) y sobre
      d + delta (H1: theta < -delta), cada uno a alpha/2; su p es
      p_relevant = min(1, 2 min(p_arriba, p_abajo)). Sin empates equivale a que
      el intervalo al 1 - alpha quede entero fuera de [-delta, delta]. Con
      delta = 0 es el Wilcoxon bilateral de siempre (el p de paired_test): la
      regla de la memoria para decidir si una diferencia es real es el caso
      delta = 0.
    - "equivalente a cero": se rechaza H0: |theta| >= delta con los dos
      unilaterales de la equivalencia, sobre d - delta (H1: theta < delta) y
      sobre d + delta (H1: theta > -delta), cada uno a alpha; p_equivalent es
      el mayor de los dos. Sin empates equivale a que el intervalo al
      1 - 2 alpha quede dentro de (-delta, delta). Con delta = 0 no puede
      pasar: un efecto no se puede probar exactamente nulo.
    - "inconcluso": ninguna de las dos. Pasa con pocas unidades, o con un
      efecto distinto de cero cuyo intervalo cruza delta (real, pero sin saber
      si es relevante).
    - "no contrastable": menos de MIN_PAIRS unidades.

    Los dos desenlaces positivos son incompatibles: si p_relevant <= alpha, uno
    de los unilaterales de efecto vale como mucho alpha/2 y su opuesto, que es
    uno de los de la equivalencia, al menos 1 - alpha/2.

    Al lado se devuelve el p del Wilcoxon bilateral de theta = 0 (p_value),
    que es la regla con que la memoria decide si una diferencia es real: un
    efecto puede ser real y a la vez equivalente a cero (pequeno frente a
    delta), y la tabla debe dejarlo ver.

    Supuesto: los niveles de los cuatro unilaterales son exactos si la
    diferencia es simetrica alrededor de su pseudomediana; si no, son
    aproximados (ver "Supuesto de simetria" en non_superiority_test, con los
    analisis de sensibilidad que se pueden declarar).

    Esta funcion decide una sola comparacion, al nivel alpha. Para una familia
    de comparaciones (las metricas de un factor, los fallos, los
    estadisticos), con Holm sobre la familia declarada, compare_arms.

    Parametros
    ----------
    difference : array
        Diferencia pareada por unidad: brazo sucio - brazo limpio (o lo que se
        compare), en las unidades de la metrica. Los NaN se descartan; un
        infinito es un error.
    margin : float
        Margen de equivalencia delta >= 0, en las unidades de la metrica
        (fraccion: 0,005 es medio punto de FAR; en el retardo, muestras o
        minutos, segun la columna). Sin valor por defecto: lo fija el registro
        de decisiones antes de ejecutar, metrica a metrica.
    alpha : float
        Nivel de significacion (alpha_contrastes).

    Devuelve
    --------
    dict con n, margin, alpha; hl_estimate, ci_low, ci_high y ci_level (el
    intervalo al 1 - alpha, el de la decision de efecto); eq_ci_low, eq_ci_high
    y eq_ci_level (al 1 - 2 alpha, el de la equivalencia); effect_size
    (biserial de rangos, con signo), n_nonzero (diferencias no nulas), p_value
    (bilateral, theta = 0), p_relevant, p_equivalent, p_method y outcome.
    """
    alpha = _check_alpha(alpha)
    margin = _check_margin(margin, "margin")
    d = np.asarray(difference, dtype=float).reshape(-1)
    d = _finitos(d[~np.isnan(d)], "difference")
    n = int(d.size)
    resultado = {"n": n, "margin": margin, "alpha": alpha,
                 "hl_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                 "ci_level": 1.0 - alpha, "eq_ci_low": np.nan, "eq_ci_high": np.nan,
                 "eq_ci_level": 1.0 - 2.0 * alpha, "effect_size": np.nan,
                 "n_nonzero": np.nan, "p_value": np.nan, "p_relevant": np.nan,
                 "p_equivalent": np.nan, "p_method": None, "outcome": OUTCOME_UNTESTABLE}
    if n < MIN_PAIRS:
        return resultado

    d = _round_noise(d, _max_abs(d))
    sr = _signed_rank(d)
    hl = hodges_lehmann(d, 1.0 - alpha)
    hl_eq = hodges_lehmann(d, 1.0 - 2.0 * alpha)
    arriba = _frente_a(d, margin)       # d - delta
    abajo = _frente_a(d, -margin)       # d + delta
    p_relevante = min(1.0, 2.0 * min(arriba["p_above"], abajo["p_below"]))
    p_equivalente = max(arriba["p_below"], abajo["p_above"])
    resultado.update({
        "hl_estimate": hl["hl_estimate"], "ci_low": hl["ci_low"], "ci_high": hl["ci_high"],
        "eq_ci_low": hl_eq["ci_low"], "eq_ci_high": hl_eq["ci_high"],
        "effect_size": float(sr["rank_biserial"]), "n_nonzero": sr["n_nonzero"],
        "p_value": sr["p_two_sided"], "p_relevant": p_relevante,
        "p_equivalent": p_equivalente, "p_method": sr["p_method"],
        "outcome": _effect_outcome(p_relevante, p_equivalente, alpha),
    })
    return resultado


def _margins_per_metric(margin, metricas: list) -> dict:
    if isinstance(margin, Mapping):
        faltan = [m for m in metricas if m not in margin]
        if faltan:
            raise ValueError(f"margin no tiene margen para las metricas {faltan}.")
        return {m: _check_margin(margin[m], f"margin[{m!r}]") for m in metricas}
    if len(metricas) > 1:
        raise ValueError(
            "Con varias metricas, margin tiene que ser un dict {metrica: margen}: cada "
            "metrica tiene sus unidades y su diferencia minima relevante."
        )
    return {metricas[0]: _check_margin(margin, "margin")}


def compare_arms(results: pd.DataFrame, metric, arm_col: str, clean, *, margin, family,
                 by=("faultNumber",), run_col: str = "simulationRun",
                 replicate_col: str | None = None, alpha: float = 0.05,
                 unit: str | None = None) -> pd.DataFrame:
    """Efecto de cada brazo de una ablacion frente al brazo limpio, con la familia de Holm declarada.

    Es la regla de decision de las ablaciones (memoria, OE1; Figura 2: "Wilcoxon
    pareado con correccion de Holm para el efecto de cada ablacion"), la misma
    para los cuatro factores y para la diferencia de validacion del OE2. Cada
    fila compara un brazo sucio con el limpio (el protocolo) en una metrica y
    un grupo (un fallo, un estadistico, un monitor...), con los mismos datos de
    prueba: la diferencia es brazo - limpio, con su signo, que no se presupone.

    Lo que hace, en orden:

    1. Empareja por unidad dentro de cada grupo de `by`: por simulacion, o, con
       `replicate_col`, por (replica, simulacion), promediando hasta la unidad
       declarada en `unit` (las R x N celdas estan cruzadas; que poblacion
       cubre cada unidad esta en compare_methods). Las simulaciones que solo
       tiene uno de los dos brazos se descartan con un aviso. Los agregados
       entre fallos, si se quieren (por ejemplo, los 17 fallos no dificiles),
       se calculan antes, dentro de cada simulacion, y se pasan como un grupo
       mas (regla 1: las gemelas de Rieth).
    2. Decide cada comparacion con effect_test frente a su margen.
    3. Aplica Holm dentro de cada familia declarada, por separado a los tres p:
       p_value (theta = 0, si la diferencia es real), p_relevant y
       p_equivalent. El desenlace sale de la decision de Holm por pasos:
       "efecto relevante" si rechaza con p_relevant, "equivalente a cero" si
       rechaza con p_equivalent y si no "inconcluso". Asi cada tipo de
       afirmacion (efecto relevante, equivalencia) tiene su error de familia
       acotado por alpha, y con margen 0 la regla es el Wilcoxon con Holm de
       siempre. Las comparaciones no contrastables cuentan en la familia con
       p = 1: la familia es la declarada antes de ejecutar y su tamano no
       depende de los datos.

    Los niveles suponen diferencias simetricas alrededor de su pseudomediana
    (ver effect_test y "Supuesto de simetria" en non_superiority_test).

    La familia y el margen de cada metrica son decisiones del registro (el
    informe 06, 2.4, propone por ejemplo una metrica principal por factor y
    dos grupos de fallos): no tienen valor por defecto.

    Parametros
    ----------
    results : DataFrame
        Una fila por (brazo, grupo, simulacion), o por (brazo, grupo, replica,
        simulacion) con replicate_col, con las metricas en columnas.
    metric : str o lista de str
        Metrica o metricas que se contrastan. Con varias, la salida tiene una
        fila por metrica, y "metric" puede entrar en la familia.
    arm_col : str
        Columna con el nombre del brazo.
    clean : valor de arm_col
        El brazo limpio (el protocolo). Cada otro brazo del grupo se compara
        con el.
    margin : float o dict {metrica: float}
        Margen de equivalencia de cada metrica, >= 0, en sus unidades (ver
        effect_test). Con varias metricas, un dict.
    family : str o secuencia de str
        Columnas de la salida cuyas combinaciones definen cada familia de Holm:
        de `by`, "metric" y "arm". () = una sola familia con todas las filas.
        Por ejemplo, family=() con una metrica, dos estadisticos en `by` y dos
        brazos sucios da una familia de 4 (la opcion 1 del informe 06, 2.4).
    by : secuencia de str
        Columnas que definen cada grupo de comparacion (por defecto, el
        fallo). () = sin grupos.
    run_col : str
        Columna de la simulacion (la unidad, salvo con replicate_col).
    replicate_col : str, opcional
        Columna de la replica de ajuste (el factor de calibracion).
    alpha : float
        Nivel de significacion (alpha_contrastes).
    unit : {"replicate", "simulation"}
        Obligatoria con replicate_col y sin valor por defecto: la unidad a la que
        se promedian las celdas (ver compare_methods). Sin replicate_col no se
        usa.

    Devuelve
    --------
    DataFrame con una fila por (grupo, metrica, brazo): las columnas de `by`,
    metric, arm, clean, unit (la columna de la unidad), n (unidades), n_cells
    (solo con replicate_col), hl_estimate, ci_low, ci_high, ci_level,
    eq_ci_low, eq_ci_high, eq_ci_level, effect_size, n_nonzero, p_value,
    p_relevant, p_equivalent, p_method, margin, alpha, family_m (tamano de la
    familia), p_adjusted y significant (Holm de p_value), p_relevant_holm,
    p_equivalent_holm y outcome. Avisa con LowPowerWarning de las filas que,
    con sus unidades y el tamano de su familia, no pueden llegar a alguno de
    los dos desenlaces positivos. Una fila con NaN en una columna de `by` es un
    error (groupby la descartaria sin avisar).
    """
    alpha = _check_alpha(alpha)
    metricas = [metric] if isinstance(metric, str) else list(metric)
    if not metricas:
        raise ValueError("metric esta vacia.")
    margenes = _margins_per_metric(margin, metricas)
    if isinstance(by, str):
        by = [by]
    by = list(by)
    if family is None:
        raise ValueError(
            "family es obligatoria: la familia de Holm de cada ablacion la fija el "
            "registro de decisiones. Pasa () para una sola familia con todas las filas."
        )
    familia = _check_family(family, by + ["metric", "arm"])
    necesarias = by + [arm_col] + metricas + ([replicate_col] if replicate_col else [run_col])
    faltan = [c for c in necesarias if c not in results]
    if faltan:
        raise ValueError(f"Faltan columnas en results: {faltan}")
    unit = _check_unit(replicate_col, unit, run_col, results.columns)
    if by and results[by].isna().any().any():
        raise ValueError(f"Hay filas con NaN en {by}: groupby las descartaria sin avisar.")

    grupos = results.groupby(by, sort=True) if by else [((), results)]
    filas, notas = [], []
    for clave, g in grupos:
        clave = clave if isinstance(clave, tuple) else (clave,)
        brazos = sorted(g[arm_col].unique())
        if clean not in brazos:
            raise ValueError(
                f"El brazo limpio {clean!r} no esta en el grupo {dict(zip(by, clave))} "
                f"(brazos: {brazos})."
            )
        donde = "".join(f"{c} {v}, " for c, v in zip(by, clave))
        for m in metricas:
            for brazo in brazos:
                if brazo == clean:
                    continue
                a, b, celdas, sueltas = _paired_units(g, m, arm_col, run_col, replicate_col,
                                                      unit, brazo, clean)
                if sueltas:
                    notas.append(_unpaired_note(f"{donde}{m}, ", brazo, clean, sueltas, a.size))
                validos = ~(np.isnan(a) | np.isnan(b))
                a, b = a[validos], b[validos]
                d = _round_noise(a - b, _max_abs(a, b))
                r = effect_test(d, margenes[m], alpha)
                fila = dict(zip(by, clave))
                fila.update({"metric": m, "arm": brazo, "clean": clean,
                             "unit": replicate_col if unit == "replicate" else run_col,
                             "n": r["n"]})
                if replicate_col:
                    fila["n_cells"] = celdas
                fila.update({k: r[k] for k in (
                    "hl_estimate", "ci_low", "ci_high", "ci_level", "eq_ci_low", "eq_ci_high",
                    "eq_ci_level", "effect_size", "n_nonzero", "p_value", "p_relevant",
                    "p_equivalent", "p_method", "margin", "alpha")})
                filas.append(fila)
    _warn_unpaired("compare_arms", notas)
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla

    tabla["family_m"] = _holm_by_family(
        tabla, familia, alpha,
        {"p_value": ("p_adjusted", "significant"),
         "p_relevant": ("p_relevant_holm", "_relevante"),
         "p_equivalent": ("p_equivalent_holm", "_equivalente")},
        nan_as_one=True)
    tabla["outcome"] = [
        OUTCOME_UNTESTABLE if np.isnan(p)
        else EFFECT_RELEVANT if rel else EFFECT_EQUIVALENT if eq else EFFECT_INCONCLUSIVE
        for p, rel, eq in zip(tabla["p_relevant"], tabla["_relevante"], tabla["_equivalente"])
    ]
    finales = ["family_m", "p_adjusted", "significant", "p_relevant_holm",
               "p_equivalent_holm", "outcome"]
    tabla = tabla[[c for c in tabla.columns
                   if c not in finales and c not in ("_relevante", "_equivalente")] + finales]

    # Aviso de potencia: el efecto relevante es bilateral y la equivalencia,
    # dos unilaterales; con Holm sobre m, cada uno necesita su minimo.
    pobres = []
    for _, fila in tabla.iterrows():
        m_f = int(fila["family_m"])
        n_rel = min_runs_for_significance(alpha, m_f, "two-sided")
        n_eq = min_runs_for_significance(alpha, m_f, "greater")
        nz = fila["n_nonzero"]
        if pd.isna(nz) or nz < n_rel or fila["n"] < n_eq:
            donde = ", ".join(f"{c} {fila[c]}" for c in by)
            pobres.append(
                f"{donde + ', ' if donde else ''}{fila['metric']}, {fila['arm']} frente a "
                f"{clean}: {fila['n']} unidades ({'sin contraste' if pd.isna(nz) else int(nz)}"
                f" no nulas; el efecto relevante necesita {n_rel} no nulas y la "
                f"equivalencia {n_eq} unidades, con Holm sobre {m_f})")
    _warn_low_power("compare_arms: estas comparaciones no pueden llegar a todos los "
                    f"desenlaces con alpha = {alpha}: ", pobres)
    return tabla


# ----------------------------------------------- hipotesis del TFM

def paired_advantage(results: pd.DataFrame, metric: str, method: str, baseline: str,
                     higher_is_better: bool, fault=None, method_col: str = "method",
                     run_col: str = "simulationRun",
                     fault_col: str = "faultNumber") -> pd.Series:
    """Ventaja de un metodo sobre un baseline, simulacion a simulacion, con el signo de "mejor".

    La regla de la hipotesis aplica Wilcoxon a "la ventaja menos delta". Eso
    solo tiene sentido si la ventaja esta orientada de modo que positivo
    signifique que el metodo es mejor: en FDR, metodo - baseline; en FAR o en
    retardo, donde menos es mejor, baseline - metodo. Equivocar el signo
    convierte una ventaja en una desventaja y da la vuelta a la conclusion, sin
    ningun error. Por eso `higher_is_better` no tiene valor por defecto.

    Las simulaciones se emparejan por su identificador. Si un metodo tiene
    simulaciones que el otro no, se avisa: los tres bloques tienen que evaluar
    sobre los mismos ids de prueba (regla 3). Los pares con NaN se descartan con
    un aviso: con el retardo, un NaN es "no detectado", y descartarlo favorece
    al metodo que falla en las simulaciones dificiles (DIS-12).

    Parametros
    ----------
    results : DataFrame
        Formato comun de resultados (regla 11).
    metric : str
        Columna de la metrica (por ejemplo "fdr", o una columna de exceso de la
        FDR sobre la FAR que haya calculado quien llama).
    method, baseline : str
        Metodo evaluado (por ejemplo "ae") y baseline lineal ("pca", "dpca").
    higher_is_better : bool
        True si en la metrica mas es mejor (FDR), False si menos es mejor (FAR,
        retardo).
    fault : opcional
        Fallo que se selecciona. Si la tabla tiene varios fallos es obligatorio.

    Devuelve
    --------
    Series indexada por simulacion, ordenada, con la ventaja (positivo = el
    metodo es mejor que el baseline).
    """
    if not isinstance(higher_is_better, (bool, np.bool_)):
        raise TypeError("higher_is_better debe ser True o False, explicito.")
    df = results
    if fault is not None:
        df = df[df[fault_col] == fault]
    elif fault_col in df and df[fault_col].nunique() > 1:
        raise ValueError("La tabla tiene varios fallos: pasa fault=.")

    a = df[df[method_col] == method].set_index(run_col)[metric]
    b = df[df[method_col] == baseline].set_index(run_col)[metric]
    if a.empty or b.empty:
        raise ValueError(f"No hay filas de {method!r} o de {baseline!r} con fault={fault!r}.")
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("Hay varias filas por simulacion para un mismo metodo.")

    comunes = a.index.intersection(b.index).sort_values()
    sueltas = len(a.index.union(b.index)) - len(comunes)
    if sueltas:
        warnings.warn(
            f"paired_advantage: {sueltas} simulaciones estan en un solo metodo y se "
            f"descartan ({method} frente a {baseline}, fallo {fault}). Los dos "
            "metodos deben evaluarse sobre los mismos ids.", stacklevel=2,
        )
    ventaja = a.loc[comunes] - b.loc[comunes]
    if not higher_is_better:
        ventaja = -ventaja
    n_nan = int(ventaja.isna().sum())
    if n_nan:
        warnings.warn(
            f"paired_advantage: se descartan {n_nan} pares con NaN ({method} frente "
            f"a {baseline}, fallo {fault}). Con el retardo, descartar los no "
            "detectados sesga el contraste (DIS-12).", stacklevel=2,
        )
        ventaja = ventaja.dropna()
    ventaja.name = f"advantage_{method}_vs_{baseline}"
    return ventaja


def non_superiority_test(advantage, delta: float, alpha: float = 0.05) -> dict:
    """Contraste de no superioridad de una comparacion frente a la diferencia minima relevante.

    La hipotesis predice que la ventaja de los metodos no lineales y profundos
    sobre los lineales "no sera significativa" en los fallos 3, 9 y 15. Leida
    al pie de la letra (que un contraste no rechace), esa prediccion se cumple
    sola con pocas simulaciones y se refuta sola con muchas, por diferencias
    irrelevantes (informe 06, 1a). Por eso la memoria la contrasta con un
    criterio de no superioridad: el Wilcoxon se aplica a la ventaja menos una
    diferencia minima relevante delta, y cada comparacion puede apoyar la
    hipotesis, mostrar una ventaja relevante o quedar inconclusa.

    Que se contrasta. Sea theta la pseudomediana de la ventaja (el parametro
    de Wilcoxon; su estimador es el de Hodges-Lehmann). Sobre x = ventaja - delta
    se hacen dos contrastes de Wilcoxon unilaterales, cada uno al nivel alpha:

    - "apoya": H0: theta >= delta frente a H1: theta < delta (alternativa
      "less" sobre x). Rechazar H0 es evidencia de que la ventaja es menor que
      la diferencia minima relevante. Su p es p_below_delta.
    - "ventaja relevante": H0: theta <= delta frente a H1: theta > delta
      (alternativa "greater" sobre x). Rechazar H0 es evidencia de una ventaja
      mayor que delta. Su p es p_above_delta.
    - "inconclusa": no se rechaza ninguna de las dos. Pasa cuando los datos son
      compatibles a la vez con una ventaja menor y con una mayor que delta, por
      ejemplo por falta de simulaciones; no es un "apoyo debil".

    Los dos desenlaces positivos son incompatibles: los dos p unilaterales
    suman al menos 1, asi que no pueden ser ambos <= alpha < 0,5.

    Aqui "ventaja relevante" se decide al nivel alpha sin correccion, porque es
    una sola comparacion. En la familia confirmatoria, la refutacion lleva
    Holm (hypothesis_verdict).

    El intervalo que se devuelve es el de Hodges-Lehmann al nivel 1 - 2 alpha
    (90 % con alpha = 0,05): cada extremo es la cota unilateral al 1 - alpha
    del contraste correspondiente. Sin empates, "apoya" equivale a
    ci_high <= delta y "ventaja relevante" a ci_low > delta, de modo que la
    decision se puede presentar como la posicion del intervalo frente a delta.
    Con empates la equivalencia es aproximada y manda el p, que es lo que dice
    la memoria ("El contraste de Wilcoxon se aplica a la ventaja menos una
    diferencia minima relevante").

    El signo importa: `advantage` tiene que estar orientada de modo que
    positivo signifique "el metodo es mejor que el baseline" (en FAR o retardo,
    baseline - metodo). paired_advantage la construye con el signo explicito.

    Supuesto de simetria. El Wilcoxon de rangos con signo contrasta, en rigor,
    que la ventaja es simetrica alrededor de delta, no que su pseudomediana es
    delta. Con una ventaja simetrica las dos cosas coinciden y el nivel de cada
    unilateral es exacto. Si es asimetrica (lo plausible en los fallos
    dificiles: casi todas las simulaciones con ventaja cerca de 0 y unas pocas
    en las que un metodo si detecta, una cola larga), el error de uno de los
    dos unilaterales puede pasar de alpha aunque la pseudomediana sea
    exactamente delta: con la cola hacia arriba, el de "apoya"; hacia abajo, el
    de "ventaja relevante" (lo midio por simulacion la revision del codigo del
    12/10, I3). En la interseccion-union del veredicto, el apoyo en falso
    queda acotado por el peor de los niveles de las comparaciones, no por
    alpha. Lo mismo vale para todas las reglas del modulo que deciden con
    Wilcoxon: paired_test, detects_above_far, effect_test, compare_arms,
    published_advantage_test e hypothesis_verdict.

    Que parametro contrasta la hipotesis (pseudomediana, mediana o media) lo
    decide el registro, que puede declarar de antemano un analisis de
    sensibilidad con contrastes que no suponen simetria, sobre
    x = ventaja - delta. No estan implementados aqui porque cada uno es una
    linea de scipy y contesta por otro parametro, asi que no sustituyen a este:
    si discrepan, se informa.
    - Prueba de signos (la mediana): con k diferencias positivas de las n' no
      nulas, scipy.stats.binomtest(k, n', 0.5, alternative="less") para
      "apoya" y alternative="greater" para "ventaja relevante".
    - t pareada (la media): scipy.stats.ttest_1samp(x, 0, alternative=...),
      aproximada con muchas simulaciones aunque la ventaja no sea normal.
    - Bootstrap de la media por simulaciones (percentil o BCa), con semilla.

    Parametros
    ----------
    advantage : array
        Ventaja pareada, una por simulacion de prueba (las mismas simulaciones
        para los dos metodos). Los NaN se descartan; un infinito es un error.
    delta : float
        Diferencia minima relevante, en las unidades de la ventaja (fraccion,
        no puntos porcentuales: 2 puntos de FDR son 0,02), sobre la
        pseudomediana. No tiene valor por defecto: la fija Pablo en el registro
        de decisiones antes de los experimentos confirmatorios, y conviene
        leerla de configs/baseline.yaml (contrastes.delta) con sus unidades: un
        2 en lugar de 0,02 no da error (en el retardo, delta puede ser mayor
        que 1) y cambia la conclusion. Debe ser >= 0.
    alpha : float
        Nivel de significacion de cada contraste unilateral (alpha_contrastes).

    Devuelve
    --------
    dict con n, delta, alpha, hl_estimate, ci_low, ci_high, ci_level,
    p_below_delta, p_above_delta, statistic (W+ de ventaja - delta, con los
    ceros repartidos), effect_size (biserial de rangos de la ventaja frente a
    0), n_nonzero (diferencias no nulas de ventaja - delta), p_method y outcome
    ("apoya", "ventaja relevante", "inconclusa", o "no contrastable" con menos
    de MIN_PAIRS simulaciones).
    """
    alpha = _check_alpha(alpha)
    delta = _check_margin(delta, "delta")

    d = np.asarray(advantage, dtype=float).reshape(-1)
    d = _finitos(d[~np.isnan(d)], "advantage")
    n = int(d.size)
    base = {"n": n, "delta": delta, "alpha": alpha}

    if n < MIN_PAIRS:
        return {**base, "hl_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "ci_level": 1.0 - 2.0 * alpha, "p_below_delta": np.nan,
                "p_above_delta": np.nan, "statistic": np.nan, "effect_size": np.nan,
                "n_nonzero": np.nan, "p_method": None, "outcome": OUTCOME_UNTESTABLE}

    r = _dos_unilaterales(d, delta, alpha)    # el mismo nucleo que la ventaja publicada

    n_min = _low_power(r["n_nonzero"], alpha, 1, "less")
    if n_min is not None:
        _warn_low_power(
            f"non_superiority_test con {r['n_nonzero']} diferencias no nulas: "
            f"ningun resultado puede dar p <= {alpha} en un contraste unilateral; "
            f"hacen falta al menos {n_min}.")

    if r["p_below"] <= alpha:
        desenlace = OUTCOME_SUPPORTS
    elif r["p_above"] <= alpha:
        desenlace = OUTCOME_RELEVANT
    else:
        desenlace = OUTCOME_INCONCLUSIVE

    return {**base, **{k: r[k] for k in ("hl_estimate", "ci_low", "ci_high", "ci_level")},
            "p_below_delta": r["p_below"], "p_above_delta": r["p_above"],
            "statistic": r["statistic"], "effect_size": r["effect_size"],
            "n_nonzero": r["n_nonzero"], "p_method": r["p_method"],
            "outcome": desenlace}


def detects_above_far(fdr, far, *, margin: float, alpha: float = 0.05) -> dict:
    """Decide si un metodo detecta un fallo por encima de su tasa de falsas alarmas.

    El criterio de no superioridad de la memoria dice que una comparacion en la
    que ninguno de los dos metodos detecta el fallo por encima de su tasa de
    falsas alarmas no cuenta como apoyo. Hace falta porque una FDR del orden de
    la FAR no es deteccion (I6): un detector que alarma al azar en el 1 % de
    las muestras tambien "detecta" el 1 % de las muestras con fallo. Esta
    funcion da la regla para un metodo y un fallo; combinar los dos metodos de
    cada comparacion es cosa de hypothesis_verdict, que acepta el dict que
    devuelve tal cual (en 'detects_method' y 'detects_baseline', o desde la
    funcion `detects`).

    Por que vive en stats.py. La escribio la v0.3 en metrics.py llamando a
    scipy con el metodo automatico, que con empates y entre 14 y 50
    simulaciones usa la aproximacion normal, y sin redondear la resta (los
    restos de coma flotante cambian signos y empates; ver _DIGITS). Aqui usa el
    mismo Wilcoxon exacto que la hipotesis: su p es el p_above_delta de
    non_superiority_test(fdr - far, delta=margin). metrics.detects_above_far
    sigue existiendo y es esta misma funcion.

    Criterio: la unidad es la simulacion (DIS-14), y por cada una se toma la
    diferencia pareada d = fdr - far. El metodo detecta por encima de su FAR si
    la pseudomediana de d supera `margin` de forma significativa: Wilcoxon de
    rangos con signo unilateral (H1: theta > margin) sobre d - margin, con los
    ceros repartidos y p exacto, y p <= alpha. Con menos de MIN_PAIRS
    simulaciones no se contrasta (p = NaN) y la respuesta es False.

    False significa "no se ha podido establecer que detecte por encima de su
    FAR", no "se ha demostrado que no detecta": la regla solo acepta como apoyo
    una deteccion establecida. Su nivel supone que el exceso fdr - far es
    simetrico alrededor de su pseudomediana (ver "Supuesto de simetria" en
    non_superiority_test); un exceso con cola larga hacia arriba (pocas
    simulaciones que si detectan) es justo el caso en que ese supuesto falla.

    Que FAR se empareja con cada FDR lo decide el equipo; las opciones son:
    - en Rieth, la de la simulacion normal gemela (el mismo simulationRun en
      test_fault00) en el mismo tramo que la FDR, de la muestra 161 a la 960:
      mismo ruido y misma ventana, el contrafactual exacto de "sin fallo"; se
      obtiene con `metrics.far(estadistico_normal[160:], limite)` o con
      `metrics.evaluate_run(estadistico_normal, limite, None,
      far_window=(160, None))`;
    - la FAR del propio fichero antes del fallo (columna `far` de
      `evaluate_run`): menos muestras y otra ventana (en Rieth, las 160
      primeras de la gemela).
    En el TEP clasico hay una simulacion por fallo: no hay contraste posible.

    Parametros
    ----------
    fdr, far : array
        FDR y FAR por simulacion, emparejadas por posicion (la misma
        simulacion en la misma posicion). Los pares con algun NaN se descartan;
        un infinito es un error.
    margin : float
        Exceso minimo de la FDR sobre la FAR, en proporcion (0,02 son 2
        puntos), >= 0. Sin valor por defecto a proposito: con 0 basta un exceso
        distinto del azar, y con muchas simulaciones cualquier exceso minimo
        sale significativo (DIS-1); un margen positivo exige un exceso
        relevante. Es una decision del equipo, como la diferencia minima
        relevante.
    alpha : float
        Nivel del contraste (alpha_contrastes, el 0,05 del protocolo).

    Devuelve
    --------
    dict con n, mean_excess (media de fdr - far), median_excess, hl_excess
    (Hodges-Lehmann de fdr - far, la escala en que se decide), statistic (W+
    de fdr - far - margin), p_value, n_nonzero (diferencias no nulas de
    fdr - far - margin), p_method, margin, alpha y detects_above_far (bool).
    """
    a = np.asarray(fdr, dtype=float).ravel()
    b = np.asarray(far, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"fdr y far tienen longitudes distintas: {a.shape} y {b.shape}")
    alpha = _check_alpha(alpha)
    margin = _check_margin(margin, "margin")
    validos = ~(np.isnan(a) | np.isnan(b))
    a, b = _finitos(a[validos], "fdr"), _finitos(b[validos], "far")
    d = _round_noise(a - b, _max_abs(a, b))
    n = int(d.size)
    resultado = {
        "n": n,
        "mean_excess": float(d.mean()) if n else np.nan,
        "median_excess": float(np.median(d)) if n else np.nan,
        "hl_excess": np.nan,
        "statistic": np.nan,
        "p_value": np.nan,
        "n_nonzero": np.nan,
        "p_method": None,
        "margin": margin,
        "alpha": alpha,
        "detects_above_far": False,
    }
    if n < MIN_PAIRS:
        return resultado
    fr = _frente_a(d, margin)
    resultado.update({
        "hl_excess": hodges_lehmann(d)["hl_estimate"],
        "statistic": fr["w_plus"],
        "p_value": fr["p_above"],
        "n_nonzero": fr["n_nonzero"],
        "p_method": fr["p_method"],
        "detects_above_far": bool(fr["p_above"] <= alpha),
    })
    return resultado


def _as_bool(valor, que: str) -> tuple[bool, float]:
    """Valor de deteccion: un bool, o el dict de detects_above_far (bool y su p)."""
    if isinstance(valor, (bool, np.bool_)):
        return bool(valor), np.nan
    if isinstance(valor, Mapping) and "detects_above_far" in valor:
        decision = valor["detects_above_far"]
        if isinstance(decision, (bool, np.bool_)):
            p = valor.get("p_value", np.nan)
            return bool(decision), (np.nan if p is None else float(p))
    raise ValueError(
        f"{que} debe ser True o False, o el dict que devuelve detects_above_far, "
        f"no {valor!r}."
    )


def _normalize_entries(entradas, detects, rol: str, con_ventaja: bool) -> list[dict]:
    if entradas is None:
        return []
    if isinstance(entradas, pd.DataFrame):
        entradas = entradas.to_dict("records")
    salida = []
    for e in entradas:
        e = dict(e)
        for clave in ("fault", "method", "baseline"):
            if clave not in e:
                raise ValueError(f"Cada {rol} necesita la clave {clave!r}.")
        if con_ventaja and "advantage" not in e:
            raise ValueError(f"Cada {rol} necesita la clave 'advantage'.")
        for clave, quien in (("detects_method", e["method"]), ("detects_baseline", e["baseline"])):
            if clave not in e or e[clave] is None:
                if detects is None:
                    raise ValueError(
                        f"Falta {clave!r} en la {rol} (fallo {e['fault']}, "
                        f"{e['method']} frente a {e['baseline']}) y no se paso "
                        "`detects`: sin ese dato no se puede aplicar el suelo de deteccion."
                    )
                e[clave] = detects(quien, e["fault"])
            e[clave], e["p_" + clave] = _as_bool(e[clave], f"{clave} (fallo {e['fault']})")
        salida.append(e)
    return salida


def _declared_family_problems(comps: list[dict]) -> list[str]:
    """En que se aparta la familia recibida de la que declara la memoria (parrafo 368)."""
    problemas = []
    fallos = {c["fault"] for c in comps}
    metodos = {c["method"] for c in comps}
    bases = {c["baseline"] for c in comps}
    if fallos != set(CONFIRMATORY_FAULTS):
        problemas.append(f"los fallos son {sorted(fallos, key=str)}, no 3, 9 y 15")
    if len(metodos) != 1:
        problemas.append(f"hay {len(metodos)} metodos, no uno (el autoencoder del OE3)")
    if len(bases) != 2:
        problemas.append(f"hay {len(bases)} baselines, no dos (PCA y DPCA)")
    pares = {(c["fault"], c["baseline"]) for c in comps}
    if len(comps) != len(CONFIRMATORY_FAULTS) * 2 or len(pares) != len(fallos) * len(bases):
        problemas.append(f"hay {len(comps)} comparaciones, no las seis (cada fallo frente a "
                         "cada baseline)")
    return problemas


def hypothesis_verdict(comparisons, delta: float, alpha: float = 0.05,
                       controls=None, detects=None, published=None) -> dict:
    """Veredicto de la familia confirmatoria con el criterio de no superioridad de la memoria.

    Es la regla con la que la memoria contrasta la parte de la prediccion que
    se refiere a los fallos 3, 9 y 15 (capitulo 3, parrafo del criterio de no
    superioridad), traducida a codigo. No evalua la otra condicion de la
    prediccion, la ventaja publicada (punto 7):

    1. Cada comparacion es la ventaja, simulacion a simulacion, de un metodo no
       lineal o profundo sobre un baseline lineal en un fallo, y se contrasta
       con non_superiority_test frente a delta.
    2. La familia confirmatoria son los fallos 3, 9 y 15 con el autoencoder del
       OE3 frente a PCA y DPCA: seis comparaciones.
    3. La hipotesis se refuta si alguna comparacion muestra una ventaja
       relevante tras aplicar la correccion de Holm a la familia.
    4. Se apoya solo si las seis la apoyan.
    5. Una comparacion en la que ninguno de los dos metodos detecta el fallo
       por encima de su tasa de falsas alarmas no cuenta como apoyo (suelo de
       deteccion): que la ventaja sea menor que delta cuando ninguno ve nada
       no informa de nada.
    6. Se incluyen fallos de control en los que si se espera deteccion. La
       memoria no dice que pasa si un control falla; esta implementacion lo
       interpreta asi, a falta de que lo fije el registro de decisiones: un
       control se supera si los dos metodos detectan el fallo por encima de su
       FAR, y si alguno falla el veredicto no puede ser "apoya" (un "no hay
       ventaja relevante" de unos metodos que no detectan ni lo detectable no
       dice nada de los metodos). Los controles no entran en Holm ni pueden
       refutar. Si traen ventaja, su desenlace (sin Holm, solo informativo) va
       en la columna control_outcome, no en outcome, para que la tabla no
       diga "ventaja relevante" en un veredicto que apoya.
    7. La prediccion se refuta tambien "si la ventaja resulta igual o mayor que
       la publicada" (parrafo 349). Este veredicto NO evalua esa condicion: la
       salida lo dice en 'published_condition' ("no evaluada") y en una linea
       de la explicacion. Si se pasan los resultados de
       published_advantage_test (`published`), se muestran (tabla 'published'
       y lineas de la explicacion) pero no cambian el veredicto: como se
       combinan las dos condiciones lo decide el registro de decisiones
       (hipotesis.tabla_ventajas_publicadas y su regla). Hasta entonces,
       "apoya" solo dice que se apoya la no superioridad en 3, 9 y 15.
    8. La familia y los controles son los declarados. "Se apoya solo si las
       seis la apoyan" y "se incluyen fallos de control" son, para la memoria,
       condiciones de la regla: si la familia recibida no tiene la forma
       declarada (los fallos 3, 9 y 15 de CONFIRMATORY_FAULTS, un metodo
       frente a dos baselines, seis comparaciones) o no hay controles, el
       veredicto no puede ser "apoya" y sale "inconclusa", con el motivo y un
       aviso. Un aviso se perderia en un registro; el veredicto es lo que se
       copia a la tabla de resultados. "Refuta" sigue siendo posible (los
       controles no refutan, y una ventaja relevante tras Holm es evidencia
       contra la prediccion), con un aviso si la familia no es la declarada.
    En el resto de casos el veredicto es "inconclusa", con la lista de por que.

    Por que la combinacion es correcta. Para cada comparacion j hay dos
    contrastes unilaterales sobre x_j = ventaja_j - delta (ver
    non_superiority_test): el de apoyo, H0: theta_j >= delta, y el de ventaja
    relevante, H0: theta_j <= delta.

    - Apoyo. La afirmacion "en las seis la ventaja es menor que delta" es la
      interseccion de las seis alternativas; su nula es la union de las seis
      nulas (alguna ventaja >= delta). Rechazar esa union exige rechazar cada
      una de sus partes, y cada contraste al nivel alpha basta: para apoyar en
      falso hay que rechazar en falso al menos una nula verdadera, y eso tiene
      probabilidad <= alpha. Es un contraste de interseccion-union (Berger, 1982
      [VERIFICAR]). Corregir aqui por multiplicidad (por ejemplo, cada uno a
      alpha/6) no hace falta para controlar el error y solo haria mas dificil
      apoyar. Por eso cada uno va sin corregir.
    - Refutacion. "Alguna ventaja es relevante" es la union de las seis
      alternativas; su nula es la interseccion (todas las ventajas <= delta), y
      basta un rechazo para afirmarla. Sin correccion, la probabilidad de
      refutar en falso creceria con el numero de comparaciones; Holm la
      mantiene <= alpha con cualquier dependencia entre ellas (aqui hay mucha:
      el autoencoder es el mismo frente a PCA y frente a DPCA, y en Rieth la
      simulacion r de cada fallo comparte el ruido con la r de los demas,
      DIS-2). Para la decision global, "alguna tras Holm" equivale al primer
      paso, p minimo <= alpha/m; Holm decide ademas cuales se marcan.
    - Las dos cosas no pueden pasar a la vez en una comparacion (sus dos p
      suman al menos 1), y el apoyo exige ademas que ninguna refute.

    Una comparacion con menos de MIN_PAIRS simulaciones (o con 'advantage'
    None) sale "no contrastable": no apoya, y entra en Holm con p = 1, de modo
    que la familia sigue siendo la declarada (m no depende de los datos).

    Los niveles de los dos razonamientos suponen ventajas simetricas alrededor
    de su pseudomediana; con asimetria, el apoyo en falso queda acotado por el
    peor de los niveles de las comparaciones (ver "Supuesto de simetria" en
    non_superiority_test, con los analisis de sensibilidad que se pueden
    declarar).

    Parametros
    ----------
    comparisons : lista de dict (o DataFrame), una entrada por comparacion de
        la familia, con las claves:
        'fault', 'method', 'baseline';
        'advantage': array por simulacion, positivo = el metodo es mejor
        (paired_advantage la construye con el signo explicito); None, si no
        hay datos (la comparacion sale no contrastable);
        'detects_method' y 'detects_baseline': si cada metodo detecta el
        fallo por encima de su FAR, como bool o como el dict que devuelve
        detects_above_far, tal cual (entonces su p queda en la tabla, en
        p_detects_method y p_detects_baseline). Se calculan fuera, con
        detects_above_far y la banda que fije el registro, o se obtienen de
        `detects`.
    delta : float
        Diferencia minima relevante, sin valor por defecto (ver
        non_superiority_test).
    alpha : float
        Nivel de significacion de los contrastes (alpha_contrastes).
    controls : lista de dict, opcional
        Fallos de control, con 'fault', 'method', 'baseline', y
        'detects_method' y 'detects_baseline' (o `detects`). Si traen
        'advantage', se contrastan tambien, solo para informar
        (control_outcome). Sin controles el veredicto no puede ser "apoya"
        (punto 8): la memoria los incluye en la regla.
    detects : callable(metodo, fallo) -> bool o dict de detects_above_far, opcional
        Se usa para las entradas que no traen 'detects_method' o
        'detects_baseline'.
    published : lista de dict (o DataFrame), opcional
        Resultados de published_advantage_test (uno por comparacion con la
        ventaja publicada). Se muestran, no se combinan con el veredicto
        (punto 7). No se comprueba que sean de la misma metrica ni de los
        mismos metodos que la familia (los nombres de un articulo y los
        nuestros pueden diferir con razon): eso lo garantiza quien llama.

    Devuelve
    --------
    dict con:
      'verdict': "apoya", "refuta" o "inconclusa" (de la regla del parrafo
      368, sin la condicion de la ventaja publicada);
      'explanation': texto con el porque;
      'table': DataFrame, una fila por comparacion y por control (columna
      'role'), con el contraste, el p ajustado por Holm de la ventaja relevante,
      el desenlace, el suelo (con el p de cada deteccion si se paso el dict de
      detects_above_far), si cuenta como apoyo y, en los controles, si se
      superan y el desenlace informativo de su ventaja (control_outcome);
      'alpha', 'delta', 'm' (tamano de la familia), 'n_min_refute' (simulaciones
      no nulas necesarias para poder refutar tras Holm) y 'n_min_support' (para
      poder apoyar);
      'declared_family': True si la familia tiene la forma declarada (punto 8);
      'published_condition': "no evaluada" (PUBLISHED_NOT_EVALUATED), siempre:
      el veredicto no incluye la condicion del parrafo 349;
      'published': DataFrame con los resultados de published_advantage_test
      que se pasaron, o None.

    Ejemplo (esquema; METRICA, DELTA, BANDA, ALPHA_CONTRASTES y CONTROLES
    los fija el registro de decisiones; fdr_y_far(metodo, fallo) es de quien
    llama y devuelve la FDR y la FAR emparejada de cada simulacion)::

        def detecta(metodo, fallo):
            fdr, far = fdr_y_far(metodo, fallo)
            return detects_above_far(fdr, far, margin=BANDA, alpha=ALPHA_CONTRASTES)

        comps = []
        for f in (3, 9, 15):
            for base in ("pca", "dpca"):
                adv = paired_advantage(res, METRICA, "ae", base,
                                       higher_is_better=True, fault=f)
                comps.append({"fault": f, "method": "ae", "baseline": base,
                              "advantage": adv})
        ctrl = [{"fault": f, "method": "ae", "baseline": b}
                for f in CONTROLES for b in ("pca", "dpca")]
        out = hypothesis_verdict(comps, delta=DELTA, alpha=ALPHA_CONTRASTES,
                                 controls=ctrl, detects=detecta)
        print(out["verdict"]); print(out["explanation"])
    """
    alpha = _check_alpha(alpha)
    delta = _check_margin(delta, "delta")

    comps = _normalize_entries(comparisons, detects, "comparacion", con_ventaja=True)
    ctrls = _normalize_entries(controls, detects, "control", con_ventaja=False)
    m = len(comps)
    if m == 0:
        raise ValueError("La familia confirmatoria esta vacia.")
    claves = [(c["fault"], c["method"], c["baseline"]) for c in comps]
    if len(set(claves)) != m:
        raise ValueError("Hay comparaciones repetidas (mismo fallo, metodo y baseline).")
    problemas_familia = _declared_family_problems(comps)
    publicadas = _published_results(published)

    filas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LowPowerWarning)
        for rol, entradas in (("confirmatoria", comps), ("control", ctrls)):
            for e in entradas:
                fila = {"role": rol, "fault": e["fault"], "method": e["method"],
                        "baseline": e["baseline"]}
                ventaja = e.get("advantage")
                if ventaja is not None or rol == "confirmatoria":
                    # Sin ventaja, una comparacion confirmatoria no es contrastable (M4).
                    r = non_superiority_test([] if ventaja is None else ventaja, delta, alpha)
                    fila.update({k: r[k] for k in (
                        "n", "hl_estimate", "ci_low", "ci_high", "ci_level",
                        "p_below_delta", "p_above_delta", "n_nonzero", "p_method",
                        "outcome")})
                fila["detects_method"] = e["detects_method"]
                fila["detects_baseline"] = e["detects_baseline"]
                fila["p_detects_method"] = e["p_detects_method"]
                fila["p_detects_baseline"] = e["p_detects_baseline"]
                filas.append(fila)
    tabla = pd.DataFrame(filas)
    for col in ("n", "n_nonzero"):
        if col in tabla:
            tabla[col] = tabla[col].astype("Int64")   # enteros aunque los controles no traigan ventaja
    es_conf = (tabla["role"] == "confirmatoria").to_numpy()

    # Holm sobre la familia declarada: los no contrastables cuentan con p = 1.
    p_holm = tabla.loc[es_conf, "p_above_delta"].fillna(1.0).to_numpy()
    holm = holm_correction(p_holm, alpha=alpha)
    tabla["p_above_delta_holm"] = np.nan
    tabla.loc[es_conf, "p_above_delta_holm"] = holm["p_adjusted"].to_numpy()
    relevante = np.zeros(len(tabla), dtype=bool)
    relevante[es_conf] = holm["significant"].to_numpy()

    # Desenlace dentro de la familia: la ventaja relevante es la de tras Holm.
    for i in np.flatnonzero(es_conf):
        if tabla.at[i, "outcome"] == OUTCOME_UNTESTABLE:
            continue
        if tabla.at[i, "p_below_delta"] <= alpha:
            tabla.at[i, "outcome"] = OUTCOME_SUPPORTS
        elif relevante[i]:
            tabla.at[i, "outcome"] = OUTCOME_RELEVANT
        else:
            tabla.at[i, "outcome"] = OUTCOME_INCONCLUSIVE

    suelo = ~(tabla["detects_method"].astype(bool) | tabla["detects_baseline"].astype(bool))
    tabla["floor"] = suelo
    tabla["counts_as_support"] = pd.array([pd.NA] * len(tabla), dtype="boolean")
    tabla.loc[es_conf, "counts_as_support"] = (
        (tabla.loc[es_conf, "outcome"] == OUTCOME_SUPPORTS) & ~suelo[es_conf]
    ).to_numpy()
    tabla["control_ok"] = pd.array([pd.NA] * len(tabla), dtype="boolean")
    tabla.loc[~es_conf, "control_ok"] = (
        tabla.loc[~es_conf, "detects_method"].astype(bool)
        & tabla.loc[~es_conf, "detects_baseline"].astype(bool)
    ).to_numpy()
    # El desenlace de la ventaja de un control (sin Holm, informativo) va aparte:
    # en la columna outcome se leeria como una contradiccion con el veredicto (M4).
    tabla["control_outcome"] = pd.Series([None] * len(tabla), dtype=object)
    tabla.loc[~es_conf, "control_outcome"] = tabla.loc[~es_conf, "outcome"]
    tabla.loc[~es_conf, "outcome"] = None

    def _quien(i):
        return (f"fallo {tabla.at[i, 'fault']}, {tabla.at[i, 'method']} frente a "
                f"{tabla.at[i, 'baseline']}")

    notas = [""] * len(tabla)
    for i in range(len(tabla)):
        if es_conf[i]:
            out = tabla.at[i, "outcome"]
            if out == OUTCOME_UNTESTABLE:
                notas[i] = f"menos de {MIN_PAIRS} simulaciones: no apoya; en Holm con p = 1"
            elif out == OUTCOME_SUPPORTS and suelo[i]:
                notas[i] = "en el suelo: ninguno detecta por encima de su FAR; no cuenta como apoyo"
            elif out == OUTCOME_RELEVANT:
                notas[i] = "ventaja relevante tras Holm sobre la familia: refuta"
        else:
            notas[i] = ("control superado" if tabla.at[i, "control_ok"]
                        else "control fallido: no se puede apoyar la hipotesis")
    tabla["note"] = notas

    # Avisos de potencia (sobre ventaja - delta, que es lo que se contrasta).
    n_min_refute = min_runs_for_significance(alpha, m, "greater")
    n_min_support = min_runs_for_significance(alpha, 1, "less")
    pobres = []
    for i in np.flatnonzero(es_conf):
        nz = tabla.at[i, "n_nonzero"]
        if pd.isna(nz) or nz < n_min_refute:
            pobres.append(f"{_quien(i)} ({'sin contraste' if pd.isna(nz) else int(nz)})")
    _warn_low_power(f"hypothesis_verdict: con alpha = {alpha} y Holm sobre m = {m}, refutar "
                    f"exige al menos {n_min_refute} diferencias no nulas (apoyar, "
                    f"{n_min_support}); no llegan: ", pobres)
    if not ctrls:
        warnings.warn(
            "hypothesis_verdict sin fallos de control: la memoria incluye fallos de "
            "control en los que si se espera deteccion; sin ellos el veredicto no puede "
            "ser 'apoya'.", stacklevel=2,
        )
    if problemas_familia:
        warnings.warn(
            "hypothesis_verdict: la familia no es la declarada en la memoria (parrafo "
            "368): " + "; ".join(problemas_familia) + ". El veredicto no puede ser "
            "'apoya', y un 'refuta' es de esta familia, no de la declarada.", stacklevel=2,
        )

    # Veredicto.
    razones = []
    idx_conf = np.flatnonzero(es_conf)
    refutan = [i for i in idx_conf if relevante[i]]
    controles_fallidos = [i for i in np.flatnonzero(~es_conf) if not tabla.at[i, "control_ok"]]
    if refutan:
        veredicto = VERDICT_REFUTES
        for i in refutan:
            razones.append(
                f"{_quien(i)}: ventaja relevante tras Holm sobre la familia de {m} "
                f"(p ajustado = {tabla.at[i, 'p_above_delta_holm']:.3g} <= {alpha}; "
                f"Hodges-Lehmann = {tabla.at[i, 'hl_estimate']:.4g}, delta = {delta:.4g})."
            )
        if problemas_familia:
            razones.append("La familia evaluada no es la declarada (parrafo 368): "
                           + "; ".join(problemas_familia) + ".")
    else:
        no_apoyan = [i for i in idx_conf if not bool(tabla.at[i, "counts_as_support"])]
        if not no_apoyan and not controles_fallidos and ctrls and not problemas_familia:
            veredicto = VERDICT_SUPPORTS
            razones.append(
                f"Las {m} comparaciones apoyan la no superioridad (p <= {alpha} en cada "
                "una, sin correccion: interseccion-union), ninguna esta en el suelo de "
                "deteccion y ninguna muestra ventaja relevante tras Holm."
            )
            razones.append(f"Los {len(ctrls)} fallos de control se detectan.")
        else:
            veredicto = VERDICT_INCONCLUSIVE
            razones.append(
                f"Ninguna comparacion muestra ventaja relevante tras Holm sobre la "
                f"familia de {m}, pero no se cumple todo lo necesario para apoyar:"
            )
            if problemas_familia:
                razones.append("- la familia no es la declarada en el parrafo 368 (se apoya "
                               "solo si las seis la apoyan): " + "; ".join(problemas_familia)
                               + ".")
            if not ctrls:
                razones.append("- no hay fallos de control (el parrafo 368 los incluye en "
                               "la regla).")
            for i in no_apoyan:
                out = tabla.at[i, "outcome"]
                if out == OUTCOME_UNTESTABLE:
                    razones.append(f"- {_quien(i)}: no contrastable (menos de {MIN_PAIRS} "
                                   "simulaciones con ventaja).")
                elif out == OUTCOME_SUPPORTS:
                    razones.append(f"- {_quien(i)}: apoya, pero esta en el suelo de "
                                   "deteccion (ninguno de los dos detecta por encima de su FAR).")
                else:
                    razones.append(
                        f"- {_quien(i)}: inconclusa (p de apoyo = "
                        f"{tabla.at[i, 'p_below_delta']:.3g} > {alpha}; intervalo "
                        f"[{tabla.at[i, 'ci_low']:.4g}, {tabla.at[i, 'ci_high']:.4g}] "
                        f"frente a delta = {delta:.4g}).")
            for i in controles_fallidos:
                faltan = [q for q, ok in ((tabla.at[i, "method"], tabla.at[i, "detects_method"]),
                                          (tabla.at[i, "baseline"], tabla.at[i, "detects_baseline"]))
                          if not ok]
                razones.append(f"- control {_quien(i)}: no detecta {', '.join(map(str, faltan))}.")
    if not ctrls:
        razones.append("Sin fallos de control evaluados.")

    # La otra condicion de la prediccion (parrafo 349) no entra en el veredicto (I2).
    razones.append(
        "Condicion 'menor que la publicada' (parrafo 349): no evaluada. Este veredicto "
        "solo aplica la regla de no superioridad del parrafo 368 en 3, 9 y 15; como se "
        "combina con la ventaja publicada lo decide el registro de decisiones."
    )
    if publicadas is not None:
        razones.append("Resultados de published_advantage_test que se pasaron (se muestran, "
                       "no cambian el veredicto):")
        for _, p in publicadas.iterrows():
            razones.append(
                f"- {p['article']}, {p['metric']}, {p['method']} frente a {p['baseline']} "
                f"({p['comparison']}): {p['outcome']} (Hodges-Lehmann = "
                f"{p['hl_estimate']:.4g}, publicada = {p['published']:.4g})."
            )

    return {
        "verdict": veredicto,
        "explanation": "\n".join(razones),
        "table": tabla,
        "alpha": alpha,
        "delta": delta,
        "m": m,
        "n_min_refute": n_min_refute,
        "n_min_support": n_min_support,
        "declared_family": not problemas_familia,
        "published_condition": PUBLISHED_NOT_EVALUATED,
        "published": publicadas,
    }


#: Claves que tiene que traer cada resultado de published_advantage_test que se
#: pase a hypothesis_verdict.
_CLAVES_PUBLICADA = ("article", "metric", "method", "baseline", "comparison", "published",
                     "hl_estimate", "ci_low", "ci_high", "p_below_published",
                     "p_above_published", "outcome")


def _published_results(published) -> pd.DataFrame | None:
    """Los resultados de published_advantage_test que se pasan al veredicto, como tabla."""
    if published is None:
        return None
    if isinstance(published, pd.DataFrame):
        filas = published.to_dict("records")
    elif isinstance(published, Mapping):
        filas = [published]
    else:
        filas = list(published)
    for f in filas:
        if not isinstance(f, Mapping) or any(c not in f for c in _CLAVES_PUBLICADA):
            raise ValueError(
                "published: pasa los resultados de published_advantage_test (uno por "
                f"comparacion con la ventaja publicada), con las claves {_CLAVES_PUBLICADA}."
            )
    return pd.DataFrame(filas)


def _published_row(published) -> dict:
    """Una fila de la tabla de ventajas publicadas como dict, con las columnas comprobadas."""
    if isinstance(published, pd.Series):
        published = published.to_dict()
    if not isinstance(published, Mapping):
        raise TypeError("published debe ser una fila de la tabla de ventajas publicadas "
                        "(dict o fila de DataFrame).")
    fila = dict(published)
    faltan = [c for c in PUBLISHED_TABLE_COLUMNS if c not in fila]
    if faltan:
        raise ValueError(f"A la fila de la ventaja publicada le faltan las columnas {faltan} "
                         f"(obligatorias: {PUBLISHED_TABLE_COLUMNS}).")
    if fila["comparison"] not in (COMPARISON_WITHIN, COMPARISON_BETWEEN):
        raise ValueError(
            f"comparison = {fila['comparison']!r}: debe ser {COMPARISON_WITHIN!r} (el "
            f"articulo mide su metodo y el baseline lineal) o {COMPARISON_BETWEEN!r} (el "
            "baseline sale de otro trabajo)."
        )
    try:
        valor = float(fila["value"])
    except (TypeError, ValueError):
        valor = np.nan
    if not np.isfinite(valor):
        raise ValueError(f"value = {fila['value']!r}: la ventaja publicada tiene que ser un "
                         "numero finito.")
    fila["value"] = valor
    return fila


def published_advantage_test(advantage, published, *, metric: str, alpha: float = 0.05) -> dict:
    """Compara la ventaja medida con la ventaja publicada, en la misma metrica.

    La prediccion de la hipotesis tiene dos condiciones, y la memoria la refuta
    si falla cualquiera: "si la ventaja resulta igual o mayor que la publicada,
    o si es relevante en al menos uno de esos tres fallos". La segunda la
    decide hypothesis_verdict; esta funcion da el mecanismo de la primera para
    una comparacion: la ventaja medida bajo el protocolo, simulacion a
    simulacion, frente a la cifra que publica un articulo.

    La regla es la del informe 06 (1c, punto 2) y el borrador del 4.3: "menor
    que la publicada" si la cota superior del intervalo unilateral al
    1 - alpha de la ventaja medida queda por debajo de la publicada; "igual o
    mayor que la publicada" si la cota inferior queda por encima; si no,
    "inconclusa". Es la regla de non_superiority_test con la cifra publicada en
    el lugar de delta (las dos usan el mismo nucleo, _dos_unilaterales, y no
    pueden divergir), con dos diferencias: la cifra puede ser negativa, y los p
    se llaman p_below_published (H1: theta < publicada) y p_above_published
    (H1: theta > publicada). Lo que alli se dice del intervalo y del "Supuesto
    de simetria" vale aqui igual.

    Ojo con "igual": un contraste unilateral no puede probar que dos ventajas
    son iguales. Con esta regla, una ventaja medida igual a la publicada queda
    "inconclusa" (no apoya la prediccion, pero tampoco la refuta). Si el
    registro quiere que "igual" refute, hay que anadir una tolerancia
    alrededor de la cifra publicada: es una decision de la regla, no de este
    codigo.

    Tambien devuelve la diferencia entre la publicada y la medida, con su
    intervalo, que es la parte de la ventaja publicada que no aparece bajo el
    protocolo. Atribuirla al protocolo exige que lo demas sea comparable (el
    conjunto de datos, la configuracion; DIS-16): si la comparacion es "entre
    articulos", la cifra publicada mezcla ademas dos trabajos.

    Tabla de ventajas publicadas (la fija el registro de decisiones antes de
    los experimentos confirmatorios; este codigo no la rellena). Una fila por
    ventaja publicada, con estas columnas obligatorias (PUBLISHED_TABLE_COLUMNS):

    - article: cita corta del articulo que publica la ventaja.
    - metric: metrica en que la publica. La comparacion se hace dentro de la
      misma metrica (memoria): debe ser el nombre de la columna de resultados
      con que se mide la ventaja aqui, y la funcion lo comprueba.
    - method: el metodo no lineal o profundo del articulo.
    - baseline: el baseline lineal con que se compara.
    - value: la ventaja publicada, metodo - baseline orientada como
      paired_advantage (positivo = el metodo es mejor; en FAR o retardo,
      baseline - metodo), en las unidades de la metrica (fraccion: 0,02 son 2
      puntos).
    - page: pagina (y tabla o figura) del PDF de donde sale la cifra.
    - comparison: "dentro del articulo" (COMPARISON_WITHIN) si el articulo mide
      el metodo y el baseline, o "entre articulos" (COMPARISON_BETWEEN) si el
      baseline lineal sale de otro trabajo (memoria: "Si el articulo no tiene
      baseline lineal, la comparacion es entre articulos y se declara asi").
    Columnas recomendadas, no obligatorias: fault (si la ventaja es de un
    fallo), dataset (clasico, Rieth, particiones de Lyu), baseline_article (de
    donde sale el baseline si la comparacion es entre articulos) y verified
    (si la cifra se comprobo en el PDF; DIS-16 pide marcarla [VERIFICAR] hasta
    entonces).

    Uso en una familia. Si se comparan varias celdas (por ejemplo, los fallos
    3, 9 y 15 frente a PCA y DPCA) y basta una "igual o mayor" para refutar,
    la refutacion necesita Holm sobre los p_above_published de la familia (como
    hypothesis_verdict con los p_above_delta), y "menor" en todas, la
    interseccion-union de cada una al nivel alpha. Si entra en el veredicto, y
    con que familia, lo decide el registro: hypothesis_verdict no la evalua,
    pero muestra estos resultados si se le pasan (`published`).

    Parametros
    ----------
    advantage : array
        Ventaja medida, una por simulacion de prueba, con el signo de "mejor"
        (paired_advantage). Los NaN se descartan; un infinito es un error.
    published : dict o fila de DataFrame
        La fila de la tabla de ventajas publicadas (columnas de arriba).
    metric : str
        Metrica en que se midio `advantage`. Tiene que ser la de
        published["metric"].
    alpha : float
        Nivel de cada contraste unilateral (alpha_contrastes).

    Devuelve
    --------
    dict con n, metric, article, method, baseline, page, comparison, published
    (la cifra), alpha, hl_estimate, ci_low, ci_high y ci_level (1 - 2 alpha),
    p_below_published, p_above_published, n_nonzero, p_method,
    published_minus_measured (la publicada menos Hodges-Lehmann), gap_ci_low y
    gap_ci_high (su intervalo), outcome ("menor que la publicada", "igual o
    mayor que la publicada", "inconclusa", o "no contrastable" con menos de
    MIN_PAIRS simulaciones) y note (el tipo de comparacion, en texto).
    """
    alpha = _check_alpha(alpha)
    fila = _published_row(published)
    if fila["metric"] != metric:
        raise ValueError(
            f"La ventaja medida es de {metric!r} y la publicada de {fila['metric']!r}: la "
            "comparacion se hace dentro de la misma metrica (no se compara una ventaja en "
            "FDR con una publicada en exactitud)."
        )
    valor = fila["value"]
    d = np.asarray(advantage, dtype=float).reshape(-1)
    d = _finitos(d[~np.isnan(d)], "advantage")
    n = int(d.size)
    nota = ("comparacion dentro del articulo" if fila["comparison"] == COMPARISON_WITHIN
            else "comparacion entre articulos: el baseline lineal sale de otro trabajo, y "
                 "la cifra mezcla los dos")
    resultado = {"n": n, "metric": metric, "article": fila["article"], "method": fila["method"],
                 "baseline": fila["baseline"], "page": fila["page"],
                 "comparison": fila["comparison"], "published": valor, "alpha": alpha,
                 "hl_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                 "ci_level": 1.0 - 2.0 * alpha, "p_below_published": np.nan,
                 "p_above_published": np.nan, "n_nonzero": np.nan, "p_method": None,
                 "published_minus_measured": np.nan, "gap_ci_low": np.nan,
                 "gap_ci_high": np.nan, "outcome": OUTCOME_UNTESTABLE, "note": nota}
    if n < MIN_PAIRS:
        return resultado

    r = _dos_unilaterales(d, valor, alpha)    # el mismo nucleo que non_superiority_test
    if r["p_below"] <= alpha:
        desenlace = PUBLISHED_LESS
    elif r["p_above"] <= alpha:
        desenlace = PUBLISHED_NOT_LESS
    else:
        desenlace = PUBLISHED_INCONCLUSIVE
    resultado.update({
        "hl_estimate": r["hl_estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
        "p_below_published": r["p_below"], "p_above_published": r["p_above"],
        "n_nonzero": r["n_nonzero"], "p_method": r["p_method"],
        "published_minus_measured": valor - r["hl_estimate"],
        "gap_ci_low": valor - r["ci_high"], "gap_ci_high": valor - r["ci_low"],
        "outcome": desenlace,
    })
    return resultado


# ------------------------------------------------------ mejor compromiso

def _objectives(table: pd.DataFrame, objectives, que: str, con_nan: bool = False) -> list:
    """Lista de (columna, direccion) comprobada: direcciones explicitas y, salvo con_nan, sin NaN."""
    if table.index.has_duplicates:
        raise ValueError("El indice de table tiene que identificar a cada candidato (sin repetidos).")
    if isinstance(objectives, Mapping):
        pares = list(objectives.items())
    else:
        try:
            pares = [tuple(p) for p in objectives]
        except TypeError:
            pares = None
    if not pares or any(len(p) != 2 for p in pares):
        raise ValueError(
            f"{que}: pasa {{columna: 'max' o 'min'}} o una lista de (columna, 'max' o 'min'); "
            "la direccion de cada metrica tiene que ser explicita."
        )
    for col, direccion in pares:
        if direccion not in _DIRECCIONES:
            raise ValueError(f"{que}: la direccion de {col!r} es {direccion!r}; usa 'max' o 'min'.")
        if col not in table:
            raise ValueError(f"{que}: la columna {col!r} no esta en la tabla.")
        if not con_nan and table[col].isna().any():
            raise ValueError(
                f"{que}: la columna {col!r} tiene NaN en {list(table.index[table[col].isna()])}: "
                "un candidato sin la metrica no se puede comparar."
            )
    columnas = [c for c, _ in pares]
    if len(set(columnas)) != len(columnas):
        raise ValueError(f"{que}: columnas repetidas {columnas}.")
    return pares


def pareto_front(table: pd.DataFrame, objectives, *, tolerance=None) -> pd.DataFrame:
    """Frente de Pareto de los candidatos sobre las metricas que se pasen, con su direccion.

    El objetivo general de la memoria pide la familia "de mejor compromiso"
    entre deteccion, falsas alarmas, retardo, diagnostico y coste, con un
    criterio que fija el registro (uno para la deteccion y otro para el
    diagnostico). El frente de Pareto es la parte neutral de esa pregunta: un
    candidato esta en el frente si ningun otro lo domina, es decir, si ningun
    otro es al menos igual de bueno en todas las metricas y mejor en alguna. No
    pondera ni ordena las metricas, asi que no elige: con cuatro o cinco
    metricas casi todo queda en el frente (informe 06, 7.2). Sirve como figura
    descriptiva al lado del criterio (best_compromise), no como criterio.

    Con `tolerance`, la dominancia exige diferencias mayores que el margen de
    cada metrica: a domina a b si en ninguna metrica es peor que b por mas de
    su margen y en alguna es mejor por mas de su margen. Asi una diferencia
    menor que la diferencia minima relevante no saca a nadie del frente.

    Parametros
    ----------
    table : DataFrame
        Una fila por candidato (metodo o familia), indexada por su nombre, con
        las metricas ya agregadas como se haya decidido (por ejemplo, la media
        sobre las simulaciones de prueba, con los fallos 3, 9 y 15 aparte).
    objectives : dict {columna: "max" o "min"} (o lista de pares)
        Las metricas y su direccion, explicitas: "max" si mas es mejor (FDR,
        macro-F1), "min" si menos es mejor (FAR, retardo, coste). Sin valor por
        defecto.
    tolerance : dict {columna: margen >= 0}, opcional
        Margen de cada metrica, en sus unidades. Si se pasa, con todas las
        columnas de objectives. Por defecto, la dominancia de Pareto estricta.

    Devuelve
    --------
    Copia de `table` con dos columnas mas: pareto (True si no lo domina nadie)
    y dominated_by (lista de los candidatos que lo dominan).
    """
    pares = _objectives(table, objectives, "objectives")
    if tolerance is None:
        tol = np.zeros(len(pares))
    else:
        if not isinstance(tolerance, Mapping) or set(tolerance) != {c for c, _ in pares}:
            raise ValueError("tolerance: un dict con un margen para cada columna de objectives.")
        tol = np.array([_check_margin(tolerance[c], f"tolerance[{c!r}]") for c, _ in pares])
    signo = np.array([1.0 if d == "max" else -1.0 for _, d in pares])
    v = table[[c for c, _ in pares]].to_numpy(dtype=float) * signo   # mas es mejor
    no_peor = np.all(v[:, None, :] >= v[None, :, :] - tol, axis=2)   # [a, b]: a no es peor que b
    mejor = np.any(v[:, None, :] > v[None, :, :] + tol, axis=2)      # [a, b]: a es mejor en alguna
    domina = no_peor & mejor
    np.fill_diagonal(domina, False)
    salida = table.copy()
    etiquetas = list(table.index)
    salida["pareto"] = ~domina.any(axis=0)
    salida["dominated_by"] = [[etiquetas[a] for a in np.flatnonzero(domina[:, b])]
                              for b in range(len(etiquetas))]
    return salida


def best_compromise(table: pd.DataFrame, *, constraints, order, ties) -> dict:
    """Aplica un criterio de mejor compromiso que se le pasa: restricciones, orden y empates.

    El criterio de mejor compromiso lo fija el registro de decisiones antes de
    ver resultados, uno para la deteccion y otro para el diagnostico (memoria,
    metodologia). Esta funcion no trae ninguno: aplica el que se le pase, de
    la forma "restriccion y luego orden" (la opcion B del informe 06, 7.2):

    1. Se descartan los candidatos que no cumplen alguna restriccion (por
       ejemplo, un tope de FAR realizada).
    2. Entre los que quedan, se mira la primera metrica de `order`: siguen los
       que empatan con el mejor en ella; los demas quedan fuera en ese paso.
    3. Con los que siguen, la segunda metrica, y asi hasta agotar el orden.
    Lo que queda al final es el compromiso; si son varios, es un compromiso
    compartido (un resultado legitimo, informe 06, 7.3).

    Que es empatar lo dice `ties`, metrica a metrica: un numero delta >= 0
    (empatan los que no se separan del mejor mas de delta; con 0, solo los
    empates exactos) o una funcion tie(columna, mejor, otro) -> bool que
    recibe las etiquetas de los dos candidatos, para decidir el empate con un
    contraste sobre las simulaciones (por ejemplo, que non_superiority_test de
    la ventaja del mejor sobre el otro no salga "ventaja relevante" frente a
    delta), que es lo que recomienda el informe 06.

    Parametros
    ----------
    table : DataFrame
        Una fila por candidato, indexada por su nombre, con las metricas ya
        agregadas como se haya decidido.
    constraints : lista de (columna, operador, valor)
        Restricciones que hay que cumplir, con operador "<=", "<", ">=" o ">".
        Obligatoria; una lista vacia dice explicitamente que no hay ninguna.
    order : lista de (columna, "max" o "min") (o dict ordenado)
        Metricas en orden de prioridad, con su direccion. Obligatoria y no
        vacia.
    ties : dict {columna: delta >= 0 o funcion}
        Regla de empate de cada columna de `order` (todas, sin valores por
        defecto).

    Devuelve
    --------
    dict con:
      'selected': lista de los candidatos elegidos (vacia si ninguno cumple
      las restricciones);
      'table': copia de `table` con meets_constraints, eliminated_at (por que
      y en que paso quedo fuera cada uno; None si sigue) y selected;
      'explanation': el recorrido, en texto.
    """
    try:
        restricciones = [tuple(r) for r in constraints]
    except TypeError:
        raise ValueError("constraints: una lista de (columna, operador, valor); [] si no hay.") from None
    for r in restricciones:
        if len(r) != 3 or r[1] not in _OPERADORES:
            raise ValueError(f"constraints: {r!r} no es (columna, operador, valor) con operador "
                             f"en {list(_OPERADORES)}.")
        if r[0] not in table:
            raise ValueError(f"constraints: la columna {r[0]!r} no esta en la tabla.")
        if table[r[0]].isna().any():
            raise ValueError(f"constraints: la columna {r[0]!r} tiene NaN: no se puede comprobar.")
    pares = _objectives(table, order, "order", con_nan=True)
    if not isinstance(ties, Mapping) or set(ties) != {c for c, _ in pares}:
        raise ValueError("ties: un dict con una regla de empate (delta o funcion) para cada "
                         "columna de order, sin mas ni menos.")
    reglas = {c: (ties[c] if callable(ties[c]) else _check_margin(ties[c], f"ties[{c!r}]"))
              for c, _ in pares}

    salida = table.copy()
    cumple = np.ones(len(table), dtype=bool)
    fuera = pd.Series([None] * len(table), index=table.index, dtype=object)
    texto = []
    for col, op, valor in restricciones:
        ok = _OPERADORES[op](table[col].to_numpy(dtype=float), float(valor))
        for etiqueta in table.index[~ok & cumple]:
            fuera[etiqueta] = f"restriccion {col} {op} {valor}"
        cumple &= ok
    salida["meets_constraints"] = cumple
    vivos = list(table.index[cumple])
    texto.append(f"Cumplen las restricciones: {vivos}" if restricciones
                 else f"Sin restricciones: {vivos}")

    for paso, (col, direccion) in enumerate(pares, start=1):
        if len(vivos) <= 1:
            break
        valores = table.loc[vivos, col].astype(float)
        if valores.isna().any():
            raise ValueError(
                f"order: {col!r} es NaN en {list(valores.index[valores.isna()])}, que llegan al "
                f"paso {paso}: sin la metrica no se puede aplicar el criterio."
            )
        mejor_valor = valores.max() if direccion == "max" else valores.min()
        mejor = valores.index[valores == mejor_valor][0]
        regla = reglas[col]
        if callable(regla):
            sigue = [e for e in vivos if valores[e] == mejor_valor or bool(regla(col, mejor, e))]
            como = "la regla de empate"
        else:
            distancia = _round_noise((valores - mejor_valor).abs().to_numpy() - regla,
                                     max(_max_abs(valores.to_numpy()), regla))
            sigue = [e for e, dist in zip(vivos, distancia) if dist <= 0]
            como = f"un margen de {regla:g}"
        for e in vivos:
            if e not in sigue:
                fuera[e] = f"paso {paso}: {col} ({direccion}), sin empate con {mejor} ({como})"
        texto.append(f"Paso {paso}, {col} ({direccion}): mejor {mejor} ({mejor_valor:g}); "
                     f"empatan con el, con {como}: {sigue}")
        vivos = sigue

    salida["eliminated_at"] = fuera
    salida["selected"] = salida.index.isin(vivos)
    if len(vivos) > 1:
        texto.append(f"Compromiso compartido: {vivos}")
    elif vivos:
        texto.append(f"Compromiso: {vivos[0]}")
    else:
        texto.append("Ningun candidato cumple las restricciones.")
    return {"selected": vivos, "table": salida, "explanation": "\n".join(texto)}
