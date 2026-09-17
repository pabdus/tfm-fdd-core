"""Carga de datos del Tennessee Eastman Process.

Dos fuentes:

- TEP clasico (Braatz/Chiang): 44 ficheros .dat de texto, una simulacion por
  fallo. Se usa para prototipar y para verificar el protocolo contra las cifras
  publicadas en la literatura.
- TEP-Rieth (2017): cuatro ficheros .RData con 500 simulaciones por condicion.
  Es la fuente de los resultados definitivos.

Ambos cargadores devuelven el MISMO contrato, para que los tres bloques del TFM
sean comparables:

    DataFrame con columnas
        faultNumber    int    0 = operacion normal, 1..21 = tipo de fallo
        simulationRun  int    identificador de la simulacion (1 en el clasico)
        sample         int    indice de muestra, empieza en 1
        xmeas_1 .. xmeas_41   variables medidas
        xmv_1 .. xmv_11       variables manipuladas

El fallo entra despues de la muestra FAULT_ONSET_TEST (160) en los ficheros de
prueba. En los de entrenamiento del TEP clasico el fallo esta activo desde la
primera muestra.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

N_VARS = 52
N_XMEAS = 41
N_XMV = 11

#: Muestra tras la cual entra el fallo en los ficheros de PRUEBA.
FAULT_ONSET_TEST = 160

#: Muestra tras la cual entra el fallo en los ficheros de ENTRENAMIENTO de Rieth.
FAULT_ONSET_TRAIN_RIETH = 20

VAR_NAMES = [f"xmeas_{i}" for i in range(1, N_XMEAS + 1)] + [
    f"xmv_{i}" for i in range(1, N_XMV + 1)
]

COLUMNS = ["faultNumber", "simulationRun", "sample"] + VAR_NAMES


def _as_samples_by_vars(arr: np.ndarray, source: str) -> np.ndarray:
    """Devuelve la matriz como (muestras, variables).

    El fichero d00.dat del TEP clasico viene TRANSPUESTO: llega como (52, 500)
    cuando todos los demas son (muestras, 52). Cargarlo sin transponer no lanza
    ninguna excepcion, simplemente ajusta el modelo sobre 52 observaciones y 500
    variables y produce resultados sin sentido. Esta funcion lo corrige y deja
    constancia.
    """
    if arr.ndim != 2:
        raise ValueError(f"{source}: se esperaba una matriz 2D, llego {arr.ndim}D")

    if arr.shape[1] == N_VARS:
        out = arr
    elif arr.shape[0] == N_VARS:
        warnings.warn(
            f"{source}: matriz transpuesta {arr.shape}, se corrige a "
            f"{arr.T.shape}. Es el caso conocido de d00.dat.",
            stacklevel=2,
        )
        out = arr.T
    else:
        raise ValueError(
            f"{source}: ninguna dimension vale {N_VARS}; la matriz es {arr.shape}. "
            "Revisa el fichero."
        )

    assert out.shape[1] == N_VARS, f"{source}: {out.shape[1]} variables, se esperaban {N_VARS}"
    return out


def load_classic(
    fault: int,
    split: str = "test",
    root: str | Path = "data/tep_classic",
) -> pd.DataFrame:
    """Carga un fichero del TEP clasico.

    Parametros
    ----------
    fault : int
        0 para operacion normal, 1..21 para el tipo de fallo.
    split : {'train', 'test'}
        'train' carga dXX.dat, 'test' carga dXX_te.dat.
    root : ruta
        Carpeta que contiene los ficheros .dat.

    Devuelve
    --------
    DataFrame con el contrato descrito en el modulo.
    """
    if fault not in range(22):
        raise ValueError(f"fault debe estar entre 0 y 21, llego {fault}")
    if split not in ("train", "test"):
        raise ValueError("split debe ser 'train' o 'test'")

    suffix = "_te" if split == "test" else ""
    path = Path(root) / f"d{fault:02d}{suffix}.dat"
    if not path.exists():
        raise FileNotFoundError(
            f"No encuentro {path}. Descomprime ahi el TEP clasico "
            "(44 ficheros .dat)."
        )

    arr = _as_samples_by_vars(np.loadtxt(path), path.name)

    df = pd.DataFrame(arr, columns=VAR_NAMES)
    df.insert(0, "sample", np.arange(1, len(df) + 1))
    df.insert(0, "simulationRun", 1)
    df.insert(0, "faultNumber", fault)
    return df[COLUMNS]


def load_rieth(
    fault: int,
    split: str = "test",
    runs: int | list[int] | None = None,
    root: str | Path = "data/tep_rieth",
) -> pd.DataFrame:
    """Carga datos del conjunto de Rieth et al. (2017) desde Parquet.

    Requiere haber ejecutado antes `python -m tfmfdd.convert_rieth`, que
    convierte los cuatro .RData a Parquet particionado.

    Parametros
    ----------
    fault : int
        0 para operacion normal, 1..20 para el tipo de fallo. El conjunto de
        Rieth NO incluye el fallo 21.
    split : {'train', 'test'}
    runs : int o lista de int, opcional
        Numero de simulaciones a cargar (las primeras n) o lista explicita de
        identificadores. None carga todas. Usa 20 en desarrollo y 100 o mas en
        los resultados definitivos.
    """
    if fault not in range(21):
        raise ValueError(f"fault debe estar entre 0 y 20 en Rieth, llego {fault}")

    path = Path(root) / f"{split}_fault{fault:02d}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No encuentro {path}. Ejecuta 'python -m tfmfdd.convert_rieth' "
            "despues de descargar los cuatro ficheros .RData."
        )

    df = pd.read_parquet(path)

    if runs is not None:
        available = np.sort(df["simulationRun"].unique())
        wanted = available[:runs] if isinstance(runs, int) else np.asarray(runs)
        df = df[df["simulationRun"].isin(wanted)]

    return df.reset_index(drop=True)[COLUMNS]


def split_runs(
    n_runs: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reparte identificadores de simulacion en entrenamiento, validacion y prueba.

    La particion es POR SIMULACION, nunca por muestra: una misma simulacion no
    puede aparecer en dos conjuntos, porque sus muestras estan correlacionadas
    en el tiempo y eso seria fuga de datos.

    Devuelve tres arrays de identificadores (empezando en 1).
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Las fracciones deben sumar 1, suman {sum(fractions)}")

    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, n_runs + 1))

    n_train = int(round(fractions[0] * n_runs))
    n_val = int(round(fractions[1] * n_runs))

    return (
        np.sort(ids[:n_train]),
        np.sort(ids[n_train : n_train + n_val]),
        np.sort(ids[n_train + n_val :]),
    )


def fault_onset(split: str, source: str = "classic") -> int | None:
    """Muestra tras la cual entra el fallo.

    Devuelve None cuando el fallo esta activo desde la primera muestra, que es
    el caso de los ficheros de entrenamiento del TEP clasico.
    """
    if split == "test":
        return FAULT_ONSET_TEST
    if source == "rieth":
        return FAULT_ONSET_TRAIN_RIETH
    return None


def values(df: pd.DataFrame) -> np.ndarray:
    """Extrae solo la matriz de las 52 variables de proceso."""
    return df[VAR_NAMES].to_numpy(dtype=float)
