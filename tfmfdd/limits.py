"""Limites de control para los estadisticos T2 y SPE.

Un limite de control es el umbral por encima del cual se declara fallo. Se
calibra con datos de OPERACION NORMAL y con un nivel de confianza, tipicamente
del 99 %: por debajo del umbral se considera que el proceso esta en control.

Elegir el nivel de confianza es elegir el compromiso entre falsas alarmas y
detecciones perdidas. Al 99 %, por construccion, se espera en torno a un 1 % de
falsas alarmas sobre datos normales; ese numero es la primera comprobacion de
que la implementacion esta bien.

Se ofrecen varias formas de calcular el limite del SPE porque la literatura usa
varias y conviene poder compararlas:

- Box: aproxima la distribucion del SPE por una chi-cuadrado ponderada ajustada
  a la media y la varianza observadas en entrenamiento.
- Jackson-Mudholkar: formula clasica basada en los autovalores descartados.
- KDE: estimacion no parametrica, util cuando el SPE no sigue ninguna de las
  distribuciones anteriores.
- Empirico: el percentil directo, como comprobacion de cordura.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def t2_limit(n_components: int, n_samples: int, alpha: float = 0.99) -> float:
    """Limite de control del estadistico T2 de Hotelling.

    Usa la distribucion F, siguiendo Chiang, Russell y Braatz (2001):

        T2_lim = a (m^2 - 1) / (m (m - a)) * F(a, m - a; alpha)

    donde a es el numero de componentes retenidas y m el numero de muestras de
    entrenamiento.
    """
    a, m = int(n_components), int(n_samples)
    if m <= a:
        raise ValueError(f"Hacen falta mas muestras ({m}) que componentes ({a})")

    f_crit = stats.f.ppf(alpha, a, m - a)
    return float(a * (m**2 - 1) / (m * (m - a)) * f_crit)


def spe_limit_box(spe_train: np.ndarray, alpha: float = 0.99) -> float:
    """Limite del SPE por la aproximacion de Box (chi-cuadrado ponderada).

    Ajusta g * chi2(h) a la media y la varianza del SPE de entrenamiento:

        g = v / (2 mu),   h = 2 mu^2 / v
    """
    spe = np.asarray(spe_train, dtype=float)
    mu, v = spe.mean(), spe.var(ddof=1)
    if mu <= 0 or v <= 0:
        raise ValueError("El SPE de entrenamiento no tiene media o varianza positivas")

    g = v / (2.0 * mu)
    h = 2.0 * mu**2 / v
    return float(g * stats.chi2.ppf(alpha, h))


def spe_limit_jackson(
    eigenvalues: np.ndarray, n_components: int, alpha: float = 0.99
) -> float:
    """Limite del SPE por Jackson y Mudholkar.

    Se apoya en los autovalores NO retenidos por el modelo:

        theta_i = suma_{j > a} lambda_j^i,   i = 1, 2, 3
        h0      = 1 - 2 theta1 theta3 / (3 theta2^2)
        Q_alpha = theta1 [ c_alpha sqrt(2 theta2 h0^2) / theta1
                           + 1 + theta2 h0 (h0 - 1) / theta1^2 ] ^ (1/h0)

    donde c_alpha es el cuantil de la normal estandar.
    """
    lam = np.asarray(eigenvalues, dtype=float)
    descartados = lam[int(n_components) :]
    if descartados.size == 0:
        raise ValueError("No quedan autovalores descartados; el modelo retiene todo")

    th1 = float(descartados.sum())
    th2 = float((descartados**2).sum())
    th3 = float((descartados**3).sum())

    if th1 <= 0 or th2 <= 0:
        raise ValueError("Autovalores descartados degenerados")

    h0 = 1.0 - (2.0 * th1 * th3) / (3.0 * th2**2)
    if abs(h0) < 1e-10:
        h0 = 1e-10

    c_alpha = stats.norm.ppf(alpha)
    base = (
        c_alpha * np.sqrt(2.0 * th2 * h0**2) / th1
        + 1.0
        + th2 * h0 * (h0 - 1.0) / th1**2
    )
    if base <= 0:
        raise ValueError("Base negativa en Jackson-Mudholkar; revisa los autovalores")

    return float(th1 * base ** (1.0 / h0))


def limit_kde(stat_train: np.ndarray, alpha: float = 0.99) -> float:
    """Limite no parametrico por estimacion de densidad kernel.

    No supone ninguna forma para la distribucion del estadistico: estima su
    densidad a partir de los datos de entrenamiento y devuelve el cuantil alpha.
    Es la opcion adecuada cuando el proceso es claramente no gaussiano.
    """
    stat = np.asarray(stat_train, dtype=float)
    if stat.size < 20:
        raise ValueError("Hacen falta al menos 20 muestras para estimar la densidad")

    kde = stats.gaussian_kde(stat)
    rejilla = np.linspace(stat.min(), stat.max() * 1.5, 2000)
    acumulada = np.cumsum(kde(rejilla))
    acumulada /= acumulada[-1]
    return float(np.interp(alpha, acumulada, rejilla))


def limit_empirical(stat_train: np.ndarray, alpha: float = 0.99) -> float:
    """Percentil empirico. El mas simple y el mas robusto de todos.

    Util como comprobacion: si el limite empirico y el parametrico difieren
    mucho, el supuesto distribucional del segundo no se sostiene y conviene
    decirlo en la memoria.
    """
    return float(np.percentile(np.asarray(stat_train, dtype=float), alpha * 100))
