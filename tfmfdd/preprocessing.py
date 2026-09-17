"""Preprocesado comun a los tres bloques del TFM.

La regla que no se negocia: el escalado se ajusta UNICAMENTE con datos de
entrenamiento y despues se aplica a validacion y prueba. Ajustarlo con todo el
conjunto es fuga de datos (data leakage) y produce resultados optimistas que no
se sostienen fuera del experimento.
"""

from __future__ import annotations

import numpy as np


class Scaler:
    """Estandarizacion a media cero y desviacion uno.

    Equivale a StandardScaler de scikit-learn, pero se incluye aqui para que el
    preprocesado sea identico en los tres bloques sin depender de la version de
    la libreria, y para dejar explicito que solo se ajusta con entrenamiento.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Scaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=1)
        # Una variable constante en entrenamiento no aporta informacion; se deja
        # su desviacion en 1 para no dividir por cero.
        std[std < 1e-10] = 1.0
        self.std_ = std
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Llama a fit() antes de transform().")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def lag_matrix(X: np.ndarray, lags: int = 2) -> np.ndarray:
    """Matriz de retardos para DPCA.

    Apila cada observacion con las `lags` anteriores, de modo que el modelo pueda
    capturar la dinamica del proceso y no solo la correlacion instantanea entre
    variables.

    Con X de forma (n, m) y lags = l, devuelve (n - l, m * (l + 1)). Las primeras
    l filas se pierden porque no tienen suficiente historia.

    Importante: se aplica DENTRO de cada simulacion, nunca sobre simulaciones
    concatenadas, porque si no se mezclaria el final de una con el principio de
    la siguiente.
    """
    X = np.asarray(X, dtype=float)
    if lags < 0:
        raise ValueError("lags no puede ser negativo")
    if lags == 0:
        return X
    if len(X) <= lags:
        raise ValueError(
            f"La simulacion tiene {len(X)} muestras y no admite {lags} retardos"
        )

    bloques = [X[lags - i : len(X) - i] for i in range(lags + 1)]
    return np.hstack(bloques)


def lag_by_run(X: np.ndarray, runs: np.ndarray, lags: int = 2):
    """Aplica lag_matrix por separado a cada simulacion.

    Devuelve la matriz apilada y el vector de identificadores de simulacion
    correspondiente a las filas que sobreviven.
    """
    X, runs = np.asarray(X, dtype=float), np.asarray(runs)
    salida, ids = [], []
    for r in np.unique(runs):
        m = runs == r
        salida.append(lag_matrix(X[m], lags))
        ids.append(np.full(m.sum() - lags, r))
    return np.vstack(salida), np.concatenate(ids)
