"""Limites de control para los estadisticos T2 y SPE.

Un limite de control es el umbral por encima del cual se declara fallo. Se
calibra con datos de OPERACION NORMAL y con un nivel de confianza, tipicamente
del 99 %: por debajo del umbral se considera que el proceso esta en control.

Elegir el nivel de confianza es elegir el compromiso entre falsas alarmas y
detecciones perdidas. Al 99 %, por construccion, se espera en torno a un 1 % de
falsas alarmas sobre datos normales; ese numero es la primera comprobacion de
que la implementacion esta bien.

Para el T2 hay dos limites parametricos que conviene no confundir, porque
responden a preguntas distintas:

- Fase II (distribucion F, `t2_limit`): el limite para una observacion NUEVA,
  independiente de las m muestras con que se estimaron el modelo. Es el que
  corresponde a monitorizar datos que el modelo no vio.
- Fase I (distribucion Beta, `t2_limit_beta`): el limite para evaluar las MISMAS
  m muestras con que se estimo el modelo. Con el mismo numero de componentes,
  de muestras y de nivel sale por debajo del de Fase II. Aplicado a datos nuevos
  es la version parametrica de calibrar en muestra, y por eso es un brazo de la
  ablacion del factor 1 (DIS-5); su efecto en la FAR se mide, no se presupone.

Se ofrecen varias formas de calcular el limite del SPE porque la literatura usa
varias y conviene poder compararlas:

- Box: aproxima la distribucion del SPE por una chi-cuadrado ponderada ajustada
  a la media y la varianza observadas en entrenamiento.
- Jackson-Mudholkar: formula clasica basada en los autovalores descartados.
  Ojo: esos autovalores salen de los datos del ajuste, asi que es una
  calibracion en muestra implicita aunque no use datos reservados (DIS-6).
- KDE: estimacion no parametrica, util cuando el SPE no sigue ninguna de las
  distribuciones anteriores.
- Empirico: el percentil directo, como comprobacion de cordura. Tiene dos
  estimadores (ver `limit_empirical`).

KDE y empirico valen para cualquier estadistico (T2, SPE, error de
reconstruccion de un autoencoder). Un NaN en el estadistico significa "muestra
no evaluada" (por ejemplo, las primeras filas de cada trayectoria en DPCA) y no
entra en la calibracion; un infinito es un error. Un limite que saliera NaN o
infinito lanza ValueError: nunca se devuelve en silencio.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special, stats

#: Estimadores del percentil empirico que acepta `limit_empirical`.
EMPIRICAL_METHODS = ("linear", "conformal")


def _validar_nivel(alpha: float) -> float:
    """Comprueba que alpha es un nivel de confianza (0 < alpha < 1)."""
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(
            f"alpha es el nivel de confianza del limite (por ejemplo 0.99) y debe "
            f"estar entre 0 y 1; llego {alpha}"
        )
    return alpha


def _finito(valor: float, que: str) -> float:
    """Devuelve el limite si es finito; si no, lanza un error que dice cual fallo."""
    valor = float(valor)
    if not np.isfinite(valor):
        raise ValueError(
            f"{que}: el limite sale {valor}. Revisa los parametros (alpha, numero "
            "de componentes, tamano de la muestra) y los datos de calibracion."
        )
    return valor


def _valores_calibracion(stat_train: np.ndarray, que: str, minimo: int) -> np.ndarray:
    """Valores del estadistico que entran en la calibracion.

    Quita los NaN (muestras no evaluadas), rechaza los infinitos y exige un
    minimo de valores. Un conjunto de calibracion vacio es un error, no un NaN.
    """
    stat = np.asarray(stat_train, dtype=float).ravel()
    if np.isinf(stat).any():
        raise ValueError(f"{que}: el estadistico de calibracion tiene valores infinitos")
    stat = stat[~np.isnan(stat)]
    if stat.size < minimo:
        raise ValueError(
            f"{que}: hacen falta al menos {minimo} valores evaluados para calibrar "
            f"el limite y llegaron {stat.size}"
        )
    return stat


def t2_limit(n_components: int, n_samples: int, alpha: float = 0.99) -> float:
    """Limite de control del estadistico T2 de Hotelling (Fase II).

    Usa la distribucion F, siguiendo Chiang, Russell y Braatz (2001):

        T2_lim = a (m^2 - 1) / (m (m - a)) * F(a, m - a; alpha)

    donde a es el numero de componentes retenidas y m el numero de muestras de
    entrenamiento. Es el limite para una observacion NUEVA, independiente de las
    m con que se ajusto el modelo. La misma formula aparece en Yin et al. (2012,
    ec. 3), Vanhatalo et al. (2017, ec. 2), Reinartz et al. (2021, ec. 4) y
    Diallo et al. (2025, ec. 11), que la atribuyen a Tracy, Young y Mason (1992).
    No depende de X_cal: con este limite, calibrar en muestra o con reserva da
    lo mismo por construccion.
    """
    alpha = _validar_nivel(alpha)
    a, m = int(n_components), int(n_samples)
    if m <= a:
        raise ValueError(f"Hacen falta mas muestras ({m}) que componentes ({a})")

    f_crit = stats.f.ppf(alpha, a, m - a)
    return _finito(a * (m**2 - 1) / (m * (m - a)) * f_crit, "t2_limit (F, Fase II)")


def t2_limit_beta(n_components: int, n_samples: int, alpha: float = 0.99) -> float:
    """Limite de control del T2 de Fase I (distribucion Beta).

    Cuando el T2 se evalua sobre las MISMAS m muestras con que se estimaron la
    media y la covarianza, no sigue una F sino una Beta:

        m T2 / (m - 1)^2  ~  Beta(a / 2, (m - a - 1) / 2)

        T2_lim = (m - 1)^2 / m * B(alpha; a / 2, (m - a - 1) / 2)

    con a componentes retenidas y m muestras de ajuste. Con el mismo a, m y
    alpha sale por debajo del limite F de Fase II (`t2_limit`). Sirve para dos
    cosas: evaluar en muestra con el limite que toca, y reproducir la practica de
    aplicar a datos nuevos un limite de Fase I, que es la version parametrica de
    calibrar en muestra (DIS-5). Con normales independientes, sobre datos nuevos
    su tasa de falsas alarmas supera 1 - alpha (lo comprueba el Monte Carlo de
    las pruebas); con datos reales el efecto se mide y se reporta con su signo,
    no se presupone.

    Fuente [VERIFICAR]: Vanhatalo et al. (2017, p. 2) y Reinartz et al. (2021,
    p. 8) dicen que el limite de Fase I se basa en la Beta y lo atribuyen a
    Tracy, Young y Mason (1992), pero no escriben la formula, y el PDF de Tracy
    et al. no esta en la bibliografia del proyecto. La formula se comprueba por
    Monte Carlo con normales multivariantes en tests/test_v03_monitor.py: la
    fraccion de muestras del propio ajuste que la superan queda en 1 - alpha.
    """
    alpha = _validar_nivel(alpha)
    a, m = int(n_components), int(n_samples)
    if a < 1:
        raise ValueError(f"Hace falta al menos una componente; llego {a}")
    if m <= a + 1:
        raise ValueError(
            f"El limite de Fase I necesita m > a + 1 muestras (m = {m}, a = {a})"
        )

    b_crit = stats.beta.ppf(alpha, a / 2.0, (m - a - 1) / 2.0)
    return _finito((m - 1) ** 2 / m * b_crit, "t2_limit_beta (Beta, Fase I)")


def spe_limit_box(spe_train: np.ndarray, alpha: float = 0.99) -> float:
    """Limite del SPE por la aproximacion de Box (chi-cuadrado ponderada).

    Ajusta g * chi2(h) a la media y la varianza del SPE de entrenamiento:

        g = v / (2 mu),   h = 2 mu^2 / v
    """
    alpha = _validar_nivel(alpha)
    spe = _valores_calibracion(spe_train, "spe_limit_box", minimo=2)
    mu, v = spe.mean(), spe.var(ddof=1)
    if mu <= 0 or v <= 0:
        raise ValueError("El SPE de entrenamiento no tiene media o varianza positivas")

    g = v / (2.0 * mu)
    h = 2.0 * mu**2 / v
    return _finito(g * stats.chi2.ppf(alpha, h), "spe_limit_box")


def spe_limit_jackson(
    eigenvalues: np.ndarray, n_components: int, alpha: float = 0.99
) -> float:
    """Limite del SPE por Jackson y Mudholkar.

    Se apoya en los autovalores NO retenidos por el modelo:

        theta_i = suma_{j > a} lambda_j^i,   i = 1, 2, 3
        h0      = 1 - 2 theta1 theta3 / (3 theta2^2)
        Q_alpha = theta1 [ c_alpha sqrt(2 theta2 h0^2) / theta1
                           + 1 + theta2 h0 (h0 - 1) / theta1^2 ] ^ (1/h0)

    donde c_alpha es el cuantil de la normal estandar. Los autovalores son los
    de los datos del ajuste: no usa X_cal, pero es una calibracion en muestra
    implicita (DIS-6). Con h0 <= 0 la aproximacion no vale (daria un cuantil
    inferior, MET-7) y se lanza un error en vez de devolver un limite absurdo.

    Los autovalores tienen que llegar en orden DECRECIENTE (los retenidos
    primero), como los deja PCAMonitor: los descartados son los que quedan
    desde la posicion n_components. `np.linalg.eigh` los da en orden creciente;
    con ese orden los "descartados" serian los mayores y el limite saldria
    distinto sin avisar (M6), asi que otro orden es un error.
    """
    alpha = _validar_nivel(alpha)
    lam = np.asarray(eigenvalues, dtype=float)
    if lam.ndim != 1:
        raise ValueError(f"eigenvalues debe ser un vector; llego una matriz {lam.shape}")
    if np.any(np.diff(lam) > 0):
        raise ValueError(
            "spe_limit_jackson: los autovalores tienen que venir en orden decreciente "
            "(los retenidos primero). np.linalg.eigh los da en orden creciente: pasa "
            "autovalores[::-1]."
        )
    descartados = lam[int(n_components) :]
    if descartados.size == 0:
        raise ValueError("No quedan autovalores descartados; el modelo retiene todo")

    th1 = float(descartados.sum())
    th2 = float((descartados**2).sum())
    th3 = float((descartados**3).sum())

    if not (th1 > 0 and th2 > 0):
        raise ValueError("Autovalores descartados degenerados")

    h0 = 1.0 - (2.0 * th1 * th3) / (3.0 * th2**2)
    if h0 <= 0:
        raise ValueError(
            f"h0 = {h0:.4g} <= 0: la aproximacion de Jackson-Mudholkar no vale con "
            "estos autovalores descartados. Usa spe_method='box' o un limite "
            "calibrado con datos (MET-7)."
        )

    c_alpha = stats.norm.ppf(alpha)
    base = (
        c_alpha * np.sqrt(2.0 * th2 * h0**2) / th1
        + 1.0
        + th2 * h0 * (h0 - 1.0) / th1**2
    )
    if base <= 0:
        raise ValueError("Base negativa en Jackson-Mudholkar; revisa los autovalores")

    return _finito(th1 * base ** (1.0 / h0), "spe_limit_jackson")


def limit_kde(stat_train: np.ndarray, alpha: float = 0.99) -> float:
    """Limite no parametrico por estimacion de densidad kernel.

    No supone ninguna forma para la distribucion del estadistico: estima su
    densidad a partir de los datos de entrenamiento y devuelve el cuantil alpha.
    Es la opcion adecuada cuando el proceso es claramente no gaussiano.

    Como se calcula. La KDE gaussiana de scipy (ancho de banda de Scott) es una
    mezcla de normales, una centrada en cada valor y todas con la misma
    desviacion h, asi que su funcion de distribucion es exacta:

        F(t) = media_i Phi((t - x_i) / h)

    y el limite es el t con F(t) = alpha, que se busca con `brentq` entre
    min - 10 h y max + 10 h (ahi F vale 0 y 1 a efectos practicos). Vale para
    cualquier signo y escala del estadistico. La version anterior integraba la
    densidad en una rejilla de min a 1,5 veces el maximo: con un estadistico
    negativo la rejilla acababa antes del maximo y el limite salia en el centro
    de la distribucion, y cerca de cero dejaba fuera la masa por debajo del
    minimo (I5 de la verificacion del 12/10). Los limites KDE de antes cambian
    un poco.
    """
    alpha = _validar_nivel(alpha)
    stat = _valores_calibracion(stat_train, "limit_kde", minimo=20)
    if np.ptp(stat) == 0:
        raise ValueError("limit_kde: todos los valores de calibracion son iguales y no hay densidad que estimar")

    kde = stats.gaussian_kde(stat)
    h = float(np.sqrt(kde.covariance[0, 0]))

    def exceso(t: float) -> float:
        return float(np.mean(special.ndtr((t - stat) / h))) - alpha

    limite = optimize.brentq(exceso, stat.min() - 10.0 * h, stat.max() + 10.0 * h)
    return _finito(limite, "limit_kde")


def limit_empirical(
    stat_train: np.ndarray, alpha: float = 0.99, method: str = "linear"
) -> float:
    """Percentil empirico. El mas simple y el mas robusto de todos.

    Util como comprobacion: si el limite empirico y el parametrico difieren
    mucho, el supuesto distribucional del segundo no se sostiene y conviene
    decirlo en la memoria.

    Parametros
    ----------
    stat_train : array
        Estadistico sobre los datos de calibracion. Los NaN no entran.
    alpha : float
        Nivel de confianza.
    method : {'linear', 'conformal'}
        - 'linear' (por defecto, el de siempre): `np.percentile` con
          interpolacion lineal, en la posicion alpha (n - 1) + 1 (base 1). Con
          datos intercambiables su FAR esperada ronda (n - alpha (n - 1)) / (n + 1),
          algo mayor que 1 - alpha; la diferencia pesa cuando n es pequeno
          (MET-3). Es una propiedad del estimador, no de un fallo de diseno.
        - 'conformal': el estadistico de orden k = ceil(alpha (n + 1)) (base 1),
          el umbral marginal de prediccion conformal (Diallo et al., 2025,
          ec. 25). Con datos intercambiables su FAR esperada es
          (n + 1 - k) / (n + 1) <= 1 - alpha. Si k > n no hay muestras
          suficientes para ese nivel y se lanza un error.

    Que estimador usa el protocolo es una decision del protocolo (informe 06,
    3.3); este parametro solo ofrece el mecanismo.
    """
    alpha = _validar_nivel(alpha)
    if method not in EMPIRICAL_METHODS:
        raise ValueError(f"method debe ser uno de {EMPIRICAL_METHODS}; llego {method!r}")
    stat = _valores_calibracion(stat_train, "limit_empirical", minimo=1)

    if method == "linear":
        return _finito(np.percentile(stat, alpha * 100), "limit_empirical")

    n = stat.size
    # El 1e-9 evita que un redondeo (0.99 * 100 = 99.00000000000001) suba k.
    k = math.ceil(alpha * (n + 1) - 1e-9)
    if k > n:
        raise ValueError(
            f"limit_empirical (conformal): con {n} valores no se puede calibrar al "
            f"nivel {alpha}; hacen falta al menos {math.ceil(alpha / (1 - alpha) - 1e-9)}"
        )
    return _finito(np.sort(stat)[k - 1], "limit_empirical")
