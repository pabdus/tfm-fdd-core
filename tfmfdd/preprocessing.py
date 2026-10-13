"""Preprocesado comun a los tres bloques del TFM.

La regla que no se negocia: el escalado se ajusta UNICAMENTE con datos de
entrenamiento y despues se aplica a validacion y prueba. Ajustarlo con todo el
conjunto es fuga de datos (data leakage): la cifra que sale ya no mide lo que
el metodo haria con datos nuevos, y puede moverse en cualquier sentido (el
signo del efecto no se presupone: se mide). En el protocolo, ademas, el
escalado se ajusta solo con los datos NORMALES del ajuste, tambien en el
diagnostico (ver Scaler).
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#: Ninguna simulacion del TEP pasa de 960 muestras: 480 o 500 en entrenamiento y
#: 960 en prueba, en el clasico y en Rieth. Una trayectoria mas larga casi
#: seguro junta ficheros o clases con el mismo identificador (ver
#: trajectory_starts).
TEP_MAX_RUN_LENGTH = 960

_CARPETA_PAQUETE = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))


def _stacklevel_fuera_del_paquete() -> int:
    """stacklevel para que un aviso senale la primera linea fuera de tfmfdd.

    trajectory_starts se llama desde muchos sitios del paquete (lag_by_run,
    PCAMonitor, las ventanas y bloques de tfmfdd.ablation); el aviso tiene que
    decir donde llamaste tu, no en que linea del paquete se detecto.
    """
    nivel, marco = 2, sys._getframe(2)
    while marco is not None and os.path.normcase(
        os.path.abspath(marco.f_code.co_filename)
    ).startswith(_CARPETA_PAQUETE + os.sep):
        nivel += 1
        marco = marco.f_back
    return nivel


class Scaler(TransformerMixin, BaseEstimator):
    """Estandarizacion a media cero y desviacion uno.

    Hace lo mismo que StandardScaler de scikit-learn salvo en dos detalles:
    divide por la desviacion con ddof = 1 (StandardScaler usa ddof = 0; con
    muchas muestras la diferencia es despreciable) y, cuando recibe etiquetas,
    se ajusta solo con las filas normales (ver normal_label). Se incluye aqui
    para que el preprocesado sea identico en los tres bloques sin depender de la
    version de la libreria, y para dejar explicito que solo se ajusta con
    entrenamiento.

    Funciona dentro de una `Pipeline` de scikit-learn y de `GridSearchCV`: asi el
    escalador se ajusta dentro de cada pliegue, nunca antes (regla 2).

    Parametros
    ----------
    normal_label : escalar o None
        Etiqueta de la clase normal; por defecto 0, la de `faultNumber`.
        Sin etiquetas, `fit(X)` usa todas las filas de X, como siempre (dentro de
        PCAMonitor, X ya es solo operacion normal). Con etiquetas `y` (por
        ejemplo, dentro de una Pipeline de diagnostico), la media y la
        desviacion salen SOLO de las filas con `y == normal_label`, y despues se
        transforma todo. Es la regla del protocolo: la estandarizacion se ajusta
        solo con las simulaciones normales del ajuste, tambien en el diagnostico
        (memoria E2, parr. 367; decision 16 del traspaso, 04/10). Si no hay
        ninguna fila normal, ValueError.
        normal_label=None ajusta con todas las filas de todas las clases e
        ignora `y`. Se aparta del protocolo, y por eso hay que pedirlo de forma
        explicita (por ejemplo, para reproducir un trabajo que escala asi).

        Ojo, es la etiqueta TAL COMO LLEGA en `y`. Si recodificas las clases
        (por ejemplo con LabelEncoder, o para una red que pide clases 0..K-1, o
        en un diagnostico sin la clase normal), la clase 0 puede dejar de ser la
        normal y el escalador se ajustaria con otra sin avisar: pasa aqui la
        etiqueta nueva de la clase normal.

    Con GridSearchCV (o cross_validate) usa `error_score="raise"`. Si el
    entrenamiento de algun pliegue no trae ninguna fila normal, este escalador
    lanza ValueError; con el valor por defecto de scikit-learn ese pliegue
    puntua NaN, la media de todos los candidatos tambien, y la busqueda elige el
    primero con solo un aviso (V-09 de la verificacion del 12/10). Con Rieth y
    GroupKFold por simulationRun no pasa: cada id esta en todas las clases.
    """

    def __init__(self, normal_label=0) -> None:
        self.normal_label = normal_label
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y=None) -> "Scaler":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Scaler espera una matriz 2D (muestras, variables); llego {X.ndim}D")

        filas = X
        if y is not None and self.normal_label is not None:
            y = np.asarray(y)
            if y.shape[0] != X.shape[0]:
                raise ValueError(f"y tiene {y.shape[0]} etiquetas y X {X.shape[0]} filas")
            filas = X[y == self.normal_label]
            if len(filas) == 0:
                raise ValueError(
                    f"No hay filas con etiqueta {self.normal_label!r} (la clase normal) "
                    "para ajustar el escalador. Si la clase normal lleva otra etiqueta, "
                    "pasala en normal_label; normal_label=None ajusta con todas las "
                    "clases y se aparta del protocolo."
                )

        if len(filas) < 2:
            raise ValueError(
                f"Hacen falta al menos 2 filas para estimar la desviacion; llegaron {len(filas)}"
            )
        if not np.isfinite(filas).all():
            raise ValueError("Los datos para ajustar el escalador tienen valores no finitos")

        self.mean_ = filas.mean(axis=0)
        std = filas.std(axis=0, ddof=1)
        # Una variable constante en entrenamiento no aporta informacion; se deja
        # su desviacion en 1 para no dividir por cero.
        std[std < 1e-10] = 1.0
        self.std_ = std
        self.n_features_in_ = X.shape[1]
        #: Filas con que se ajusto (todas, o solo las normales con normal_label).
        self.n_samples_fit_ = len(filas)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Llama a fit() antes de transform().")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"El escalador se ajusto con {self.mean_.shape[0]} variables y llega "
                f"una matriz de forma {X.shape}"
            )
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def __sklearn_is_fitted__(self) -> bool:
        return self.mean_ is not None


def lag_matrix(X: np.ndarray, lags: int = 2) -> np.ndarray:
    """Matriz de retardos para DPCA.

    Apila cada observacion con las `lags` anteriores, de modo que el modelo pueda
    capturar la dinamica del proceso y no solo la correlacion instantanea entre
    variables.

    Con X de forma (n, m) y lags = l, devuelve (n - l, m * (l + 1)). Las primeras
    l filas se pierden porque no tienen suficiente historia: la fila j de la
    salida corresponde a la muestra j + l de la entrada.

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


def trajectory_starts(runs: np.ndarray) -> np.ndarray:
    """Posiciones donde empieza cada trayectoria contigua, mas el final.

    Por que es publica. Es LA definicion de trayectoria del paquete: la usan
    `lag_by_run` (y por tanto el DPCA) y las ventanas, los bloques y la
    estandarizacion por simulacion de `tfmfdd.ablation`. Si un bloque corta sus
    trayectorias con otra regla, sus retardos o ventanas ya no son los mismos
    que los de los demas.

    Una trayectoria nueva empieza cada vez que cambia el identificador respecto a
    la fila anterior. Con `runs` 2D (por ejemplo las columnas faultNumber y
    simulationRun) basta con que cambie una columna.

    Pasa las dos columnas, `df[["faultNumber", "simulationRun"]].to_numpy()`,
    siempre que juntes ficheros o clases. Con solo `simulationRun`, dos
    trayectorias seguidas con el mismo id se funden en una: en el TEP clasico
    `simulationRun` vale 1 en todos los ficheros, y en Rieth la simulacion r de
    cada clase lleva el mismo numero. Como ninguna simulacion del TEP pasa de
    TEP_MAX_RUN_LENGTH (960) muestras, una trayectoria mas larga da un aviso
    (V-02 de la verificacion del 12/10). Si de verdad tienes trayectorias mas
    largas (otro proceso), el aviso se puede filtrar con `warnings`.

    Parametros
    ----------
    runs : array (n,) o (n, k)
        Identificador de trayectoria de cada fila, con las filas de cada
        trayectoria seguidas y en orden temporal.

    Devuelve
    --------
    array con la posicion de la primera fila de cada trayectoria y, al final,
    n: la trayectoria i son las filas de inicios[i] a inicios[i + 1] - 1.
    """
    runs = np.asarray(runs)
    if runs.ndim == 1:
        cambio = runs[1:] != runs[:-1]
    elif runs.ndim == 2:
        cambio = (runs[1:] != runs[:-1]).any(axis=1)
    else:
        raise ValueError(f"runs debe ser 1D o 2D; llego {runs.ndim}D")
    inicios = np.concatenate([[0], np.flatnonzero(cambio) + 1, [len(runs)]])

    largos = np.diff(inicios)
    if largos.size and largos.max() > TEP_MAX_RUN_LENGTH:
        i = int(np.argmax(largos))
        que = "el mismo identificador" if runs.ndim == 1 else "la misma clave"
        remedio = (
            "Pasa como runs las dos columnas, df[['faultNumber', 'simulationRun']].to_numpy(), "
            "o un id distinto por fichero."
            if runs.ndim == 1
            else "Anade una columna que distinga los ficheros (por ejemplo, ajuste y prueba)."
        )
        warnings.warn(
            f"Hay una trayectoria de {int(largos[i])} filas seguidas con {que} "
            f"(filas {int(inicios[i])} a {int(inicios[i + 1]) - 1}), mas que cualquier "
            f"simulacion del TEP ({TEP_MAX_RUN_LENGTH} muestras como mucho). Si juntaste "
            "ficheros o clases con el mismo id (en el clasico, simulationRun vale 1 en "
            f"todos), se tratan como una sola trayectoria. {remedio}",
            UserWarning,
            stacklevel=_stacklevel_fuera_del_paquete(),
        )
    return inicios


# Nombre privado de la v0.3 en desarrollo, para el codigo que ya lo importaba.
_trajectory_starts = trajectory_starts


def lag_by_run(
    X: np.ndarray,
    runs: np.ndarray,
    lags: int = 2,
    return_index: bool = False,
    samples: np.ndarray | None = None,
):
    """Aplica lag_matrix por separado a cada trayectoria contigua.

    Una trayectoria es un bloque de filas SEGUIDAS con el mismo identificador:
    se corta cada vez que el identificador cambia, no por su valor. Asi dos
    clases que comparten ids (en Rieth, la simulacion r de cada fichero de fallo
    se numera igual que la normal r) quedan separadas siempre que vengan una
    detras de otra, y la salida conserva el orden de la entrada (MET-2).

    Si concatenas ficheros o clases, pasa como `runs` las dos columnas a la vez:
    `df[["faultNumber", "simulationRun"]].to_numpy()`. Con solo `simulationRun`,
    dos trayectorias seguidas con el mismo id se funden (en el TEP clasico vale
    1 en todos los ficheros): ver trajectory_starts, que avisa cuando una
    trayectoria es mas larga que cualquier simulacion del TEP. Las filas de cada
    trayectoria tienen que venir seguidas y en orden temporal.

    Parametros
    ----------
    X : array (n, m)
    runs : array (n,) o (n, k)
        Identificador de trayectoria de cada fila.
    lags : int
    return_index : bool
        Si es True, devuelve ademas la posicion en X de la muestra "actual" de
        cada fila de la salida, para alinear los estadisticos con X.
    samples : array (n,) o None
        Opcional: la columna `sample` de cada fila. Si llega y lags > 0, se
        comprueba que dentro de cada trayectoria las muestras sean consecutivas
        (cada una, la anterior mas 1) antes de apilar retardos. Lo que detecta
        es apilar muestras que no son vecinas, por ejemplo tras repartir las
        filas por muestra (V-10 de la verificacion del 12/10): sin `samples`,
        nada en X ni en runs lo delata.

    Devuelve
    --------
    La matriz apilada y los identificadores de las filas que sobreviven (y, con
    return_index=True, sus posiciones en X).
    """
    X, runs = np.asarray(X, dtype=float), np.asarray(runs)
    if len(runs) != len(X):
        raise ValueError(f"runs tiene {len(runs)} filas y X {len(X)}")
    if samples is not None:
        samples = np.asarray(samples)
        if samples.shape != (len(X),):
            raise ValueError(f"samples debe tener una entrada por fila de X ({len(X)}); llego {samples.shape}")

    inicios = trajectory_starts(runs)
    salida, ids, posiciones = [], [], []
    for ini, fin in zip(inicios[:-1], inicios[1:]):
        if fin - ini <= lags:
            raise ValueError(
                f"La trayectoria de las filas {ini} a {fin - 1} tiene {fin - ini} "
                f"muestras y no admite {lags} retardos. Comprueba que las filas de "
                "cada trayectoria vengan seguidas y en orden temporal."
            )
        if samples is not None and lags > 0:
            saltos = np.flatnonzero(np.diff(samples[ini:fin]) != 1)
            if saltos.size:
                j = ini + int(saltos[0])
                raise ValueError(
                    f"En la trayectoria de las filas {ini} a {fin - 1} las muestras no son "
                    f"consecutivas: la fila {j} es la muestra {samples[j]} y la {j + 1}, la "
                    f"{samples[j + 1]} ({saltos.size} saltos). Apilar retardos uniria muestras "
                    "que no son vecinas. Si repartiste las filas por muestra, apila primero y "
                    "reparte despues (ver tfmfdd.ablation.split_by_sample)."
                )
        salida.append(lag_matrix(X[ini:fin], lags))
        ids.append(runs[ini + lags : fin])
        posiciones.append(np.arange(ini + lags, fin))

    matriz, ids = np.vstack(salida), np.concatenate(ids)
    if return_index:
        return matriz, ids, np.concatenate(posiciones)
    return matriz, ids
