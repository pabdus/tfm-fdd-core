"""Carga de datos del Tennessee Eastman Process.

Dos fuentes:

- TEP clasico (Braatz/Chiang): 44 ficheros .dat de texto, una simulacion por
  fallo. Se usa para prototipar y para verificar la implementacion (que el
  codigo hace lo que dice, no las reglas del protocolo) contra las cifras
  publicadas en la literatura.
- TEP-Rieth (2017): 500 simulaciones por condicion y split, convertidas de los
  cuatro .RData originales a 42 Parquet ({train,test}_faultNN.parquet). Es la
  fuente de los resultados definitivos.

Ambos cargadores devuelven el MISMO contrato, para que los tres bloques del TFM
sean comparables:

    DataFrame con columnas
        faultNumber    int64  0 = operacion normal, 1..21 = tipo de fallo (1..20 en Rieth)
        simulationRun  int64  identificador de la simulacion (1 en el clasico)
        sample         int64  indice de muestra, empieza en 1
        xmeas_1 .. xmeas_41   variables medidas
        xmv_1 .. xmv_11       variables manipuladas

Los tres identificadores son int64 en las dos fuentes. En los Parquet de Rieth
estan guardados en int16, y en int16 una cuenta como simulationRun * 1000 +
sample se desborda sin avisar (DAT-5): load_rieth los convierte al cargar. Las
variables llegan en float64 en el clasico y en float32 en Rieth, que es la
precision con que se convirtieron (pasarlas a float64 no recupera nada y dobla
la memoria). `values` devuelve siempre float64.

Las columnas de identificacion (ID_COLUMNS) no son variables y ningun modelo
debe recibirlas: `sample` delata el inicio fijo del fallo y `simulationRun`, la
trayectoria. La matriz de entrada se construye siempre con `values`.

Donde entra el fallo (`fault_onset`):

- Ficheros de prueba, en las dos fuentes: tras la muestra FAULT_ONSET_TEST
  (160). Las muestras 1..160 son operacion normal.
- Entrenamiento de Rieth: tras la muestra FAULT_ONSET_TRAIN_RIETH (20).
- Entrenamiento del clasico: los dXX.dat tienen 480 filas con el fallo activo
  desde la primera, y se usan enteros; fault_onset("train", "classic") devuelve
  None. La constante FAULT_ONSET_TRAIN_CLASSIC = 20 de versiones anteriores era
  falsa (revision del 27/09, B2) y se retiro en la v0.3: pedirla da un error que
  dice que usar.

En Rieth, la simulacion r de cada fichero de fallo es gemela de la simulacion
normal r del mismo split: comparten semilla y son la misma trayectoria hasta el
inicio del fallo (y, en algunos fallos, varias muestras despues). Por eso se
parte por simulationRun en todas las clases a la vez (`split_runs`), y
`effective_onset` dice hasta que muestra llega la copia.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

N_VARS = 52
N_XMEAS = 41
N_XMV = 11

#: Muestra tras la cual entra el fallo en los ficheros de PRUEBA.
FAULT_ONSET_TEST = 160

#: Muestra tras la cual entra el fallo en los ficheros de ENTRENAMIENTO de Rieth.
FAULT_ONSET_TRAIN_RIETH = 20

# En los ficheros de ENTRENAMIENTO del TEP clasico (d01.dat ... d21.dat) no hay
# inicio: el fallo esta activo desde la fila 0, y fault_onset("train",
# "classic") devuelve None. La constante FAULT_ONSET_TRAIN_CLASSIC = 20 que
# habia aqui era falsa: en d04, d06 y d07.dat la fila 0 ya esta a varias
# desviaciones tipicas de la operacion normal, como la muestra 161 de la prueba
# (revision del 27/09, B2; informe 11 de la revision del 03/10, A.2). La "hora"
# de Russell, Chiang y Braatz (2000) es tiempo de simulacion antes de empezar a
# registrar, y el 20 es el inicio de Rieth. Se retira con un error que dice que
# usar (ver __getattr__ al final del modulo).

#: Simulaciones por condicion y split en Rieth (ids 1..500) y muestras por
#: simulacion en cada split. Son hechos de los datos (informe 11, B.4).
N_RUNS_RIETH = 500
N_SAMPLES_RIETH = {"train": 500, "test": 960}

VAR_NAMES = [f"xmeas_{i}" for i in range(1, N_XMEAS + 1)] + [
    f"xmv_{i}" for i in range(1, N_XMV + 1)
]

#: Columnas de identificacion: no son variables de proceso (ver `values`).
ID_COLUMNS = ["faultNumber", "simulationRun", "sample"]

COLUMNS = ID_COLUMNS + VAR_NAMES

#: Subconjunto de 33 variables que usan Yin et al. (2012): XMEAS(1..22) y
#: XMV(1..11). Excluye los 19 analizadores de composicion, que se muestrean mas
#: despacio. Es el conjunto que hay que usar para que la verificacion contra sus
#: tablas sea comparable.
YIN_VARS = [f"xmeas_{i}" for i in range(1, 23)] + [f"xmv_{i}" for i in range(1, N_XMV + 1)]

#: Variable de entorno con la carpeta de los Parquet de Rieth. Es la misma que
#: leen los cuadernos B00 y C00 (y la que indica el LEEME de los datos), asi que
#: con definirla una vez basta. Si no existe, load_rieth busca en
#: RIETH_DIR_DEFAULT, relativo al directorio de trabajo.
RIETH_DIR_ENV = "TFM_RIETH_DIR"
RIETH_DIR_DEFAULT = "data/tep_rieth"


def _as_samples_by_vars(arr: np.ndarray, source: str) -> np.ndarray:
    """Devuelve la matriz como (muestras, variables).

    El fichero d00.dat del TEP clasico viene TRANSPUESTO: llega como (52, 500)
    cuando todos los demas son (muestras, 52). Cargarlo sin transponer no falla
    en el ajuste: el PCA se calcula sobre 52 observaciones y 500 "variables", la
    covarianza tiene rango 51 y el modelo "explica" toda la varianza. El error
    aparece lejos de la causa, al puntuar un fichero bien orientado, como un
    fallo de formas del escalador; y si entrenamiento y prueba tuvieran la misma
    longitud ni siquiera apareceria. Esta funcion lo corrige y deja constancia.
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
        'train' carga dXX.dat, 'test' carga dXX_te.dat. Los dXX.dat de fallo
        tienen 480 filas con el fallo activo desde la primera: se usan enteros.
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
    # Los ids en int64 de forma explicita, como en load_rieth (contrato del modulo).
    df.insert(0, "sample", np.arange(1, len(df) + 1, dtype=np.int64))
    df.insert(0, "simulationRun", np.int64(1))
    df.insert(0, "faultNumber", np.int64(fault))
    return df[COLUMNS]


# ------------------------------------------------------------------ Rieth


def _carpeta_rieth(root: str | Path | None) -> tuple[Path, str]:
    """Carpeta de los Parquet de Rieth y de donde sale (para los mensajes)."""
    if root is not None:
        return Path(root).expanduser(), "argumento root"
    env = os.environ.get(RIETH_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser(), f"variable de entorno {RIETH_DIR_ENV}"
    return Path(RIETH_DIR_DEFAULT), (
        f"valor por defecto {RIETH_DIR_DEFAULT!r}, relativo a {Path.cwd()} "
        f"(no esta definida {RIETH_DIR_ENV})"
    )


def _pyarrow_parquet():
    """Importa pyarrow.parquet y, si no se puede, explica por que y que hacer.

    En Windows 11 con el Control inteligente de aplicaciones, una directiva puede
    bloquear la DLL de Parquet de pyarrow ("DLL load failed while importing
    _parquet: Una directiva de Control de aplicaciones bloqueo este archivo";
    informe 11 de la revision del 03/10, B.0). El error original no dice nada
    de los datos ni de este paquete, y lo natural seria buscar el fallo donde no
    esta. Aqui se traduce a un mensaje que lo nombra y remite a la solucion. No
    se toca nada del sistema.
    """
    try:
        import pyarrow.parquet as pq
    except (ImportError, OSError) as e:
        raise ImportError(
            "No puedo leer Parquet: 'import pyarrow.parquet' falla en este Python "
            f"({type(e).__name__}: {e}).\n"
            "- Si el error habla de 'DLL load failed' o de una directiva de Control de "
            "aplicaciones, es Windows (Control inteligente de aplicaciones) bloqueando la "
            "DLL de Parquet de pyarrow. No es un fallo de los datos ni de tfmfdd, y no se "
            "arregla cambiando tu configuracion de seguridad por tu cuenta: sigue la seccion "
            "'pyarrow y Windows' del LEEME.md que acompana a los Parquet de Rieth y avisale "
            "a Pablo.\n"
            "- Si pyarrow no esta instalado, instala las dependencias del paquete "
            "(pyarrow>=12; los Parquet se escribieron con pyarrow 25.0.1).\n"
            '- Para comprobar un entorno: python -c "import pyarrow.parquet"'
        ) from e
    return pq


def _ids_pedidos(runs, disponibles: np.ndarray, nombre: str) -> np.ndarray | None:
    """Valida `runs` contra los ids del fichero y devuelve los ids a cargar.

    None significa todas las simulaciones. Un int de Python n son las n
    primeras. Cualquier otro iterable son ids explicitos. Un entero de numpy es
    un error (ver abajo).
    """
    if runs is None:
        return None
    if isinstance(runs, np.ndarray) and runs.ndim == 0:
        runs = runs[()]          # un escalar envuelto en un array de numpy
    if isinstance(runs, (bool, np.bool_, str, bytes)) or not (
        isinstance(runs, (int, np.integer)) or isinstance(runs, Iterable)
    ):
        raise TypeError(
            "runs debe ser un entero (cuantas simulaciones, las primeras), una lista "
            f"de ids o None; llego {runs!r}"
        )
    if isinstance(runs, np.integer):
        # Un entero de numpy casi siempre es un id sacado de un array (por ejemplo,
        # te[0] o "for r in te" con te de split_runs). Leido como "las n primeras",
        # un id confirmatorio cargaria cientos de simulaciones no miradas antes de
        # que nadie viera un aviso (V-04 de la verificacion del 12/10; regla 3).
        # Por eso es un error que dice que pasar en cada caso.
        n = int(runs)
        raise TypeError(
            f"runs={n} llego como entero de numpy ({type(runs).__name__}) y es ambiguo: "
            f"suele ser un id sacado de un array de ids (por ejemplo, de split_runs). "
            f"Para cargar esa simulacion, pasa una lista: runs=[{n}]. Para las {n} "
            f"primeras simulaciones (ids 1..{n}), pasa un int de Python: runs=int(n)."
        )

    if isinstance(runs, int):
        n = runs
        if not 1 <= n <= disponibles.size:
            raise ValueError(
                f"runs={n}: {nombre} tiene {disponibles.size} simulaciones; "
                f"pide entre 1 y {disponibles.size}."
            )
        return disponibles[:n].astype(np.int64)

    pedidos = np.asarray(list(runs))
    if pedidos.size == 0:
        raise ValueError("runs esta vacio: pide al menos una simulacion.")
    if pedidos.ndim != 1:
        raise ValueError(f"runs debe ser una lista plana de ids; llego con forma {pedidos.shape}")
    if pedidos.dtype.kind not in "iu":
        raise TypeError(
            f"runs debe traer ids enteros; llego un array de tipo {pedidos.dtype} "
            f"({pedidos[:5].tolist()}...)"
        )

    unicos = np.unique(pedidos).astype(np.int64)
    if unicos.size < pedidos.size:
        valores, cuentas = np.unique(pedidos, return_counts=True)
        repetidos = valores[cuentas > 1].tolist()
        warnings.warn(
            f"runs trae ids repetidos ({repetidos[:10]}): cada simulacion se carga una "
            "sola vez. Si salen de juntar dos listas, revisa que no se solapen.",
            UserWarning,
            stacklevel=3,
        )

    faltan = np.setdiff1d(unicos, disponibles)
    if faltan.size:
        raise ValueError(
            f"{nombre} no tiene las simulaciones {faltan[:10].tolist()}"
            f"{' ...' if faltan.size > 10 else ''} ({faltan.size} de {unicos.size} pedidas); "
            f"sus ids van de {int(disponibles.min())} a {int(disponibles.max())}. "
            "No se carga nada a medias: corrige la lista (¿mezclaste posiciones con ids?)."
        )
    return unicos


def load_rieth(
    fault: int,
    split: str = "test",
    runs: int | Iterable[int] | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Carga datos del conjunto de Rieth et al. (2017) desde Parquet.

    Por que esta funcion hace mas que leer un fichero. Cada Parquet trae las 500
    simulaciones de una condicion, y lo habitual es querer unas pocas. Leer el
    fichero entero y filtrar despues ponia en memoria el fichero completo en cada
    llamada, aunque se pidieran dos simulaciones (DAT-4). Ademas, los errores en
    `runs` eran silenciosos: los ids que no existen se ignoraban, un entero de
    numpy fallaba con un error criptico y los ids llegaban en int16, que se
    desborda al operar (DAT-5, DAT-6). Aqui se lee columna a columna y solo se
    guardan las filas de las simulaciones pedidas; los ids salen en int64 y
    cualquier id inexistente o ambiguo es un error.

    Los Parquet se obtienen de Pablo, con su manifiesto SHA-256 (LEEME.md de la
    carpeta de datos). Convertirlos uno mismo con `python -m tfmfdd.convert_rieth`
    pide unos 16 GB de memoria (DAT-2): no es lo recomendado.

    Parametros
    ----------
    fault : int
        0 para operacion normal, 1..20 para el tipo de fallo. El conjunto de
        Rieth NO incluye el fallo 21.
    split : {'train', 'test'}
        En minusculas. 'train': 500 muestras por simulacion, fallo tras la 20;
        'test': 960 muestras, fallo tras la 160.
    runs : int, iterable de int o None
        - None: las 500 simulaciones.
        - Un int de Python n: las n primeras, es decir, los ids 1..n. NO es la
          simulacion n: para una sola, pasa una lista, runs=[n]. Sirve para
          desarrollo; los ids 1..20 ya estan mirados en todos los ficheros
          (regla 3).
        - Una lista, tupla, range o array de ids (de Python o de numpy, por
          ejemplo los de `split_runs`): esas simulaciones. Para los resultados,
          los ids salen de `split_runs` y son los mismos en todas las clases.
        Un entero de numpy suelto (por ejemplo, te[0]) es un TypeError: casi
        siempre es un id, y leido como "las n primeras" cargaria simulaciones
        no miradas; el mensaje dice que pasar (runs=[n] o runs=int(n)). Un id
        que no esta en el fichero es un error (no se devuelve un conjunto a
        medias). Los repetidos se cargan una vez, con un aviso.
    root : ruta, opcional
        Carpeta con los 42 Parquet. Si no se pasa, se usa la variable de entorno
        TFM_RIETH_DIR y, si no esta definida, 'data/tep_rieth' (relativo al
        directorio de trabajo).

    Devuelve
    --------
    DataFrame con el contrato del modulo, en el orden del fichero (por
    simulationRun y sample): ids en int64 y variables en float32.

    Errores
    -------
    FileNotFoundError si falta el fichero; ImportError con la explicacion si
    pyarrow no puede leer Parquet; ValueError si `runs` pide ids que no existen o
    si el fichero no tiene la forma de Rieth (columnas, fallo o numero de
    muestras por simulacion).
    """
    if (
        isinstance(fault, (bool, np.bool_))
        or not isinstance(fault, (int, np.integer))
        or fault not in range(21)
    ):
        raise ValueError(
            f"fault debe ser un entero entre 0 y 20 en Rieth (no incluye el fallo 21), "
            f"llego {fault!r}"
        )
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', en minusculas; llego {split!r}")

    carpeta, origen = _carpeta_rieth(root)
    path = carpeta / f"{split}_fault{int(fault):02d}.parquet"
    if not path.exists():
        if carpeta.is_dir():
            detalle = (
                f"La carpeta existe pero no tiene {path.name}: la copia esta incompleta. "
                "Comprueba los 42 ficheros con verificar_rieth.py y el manifiesto SHA256SUMS.txt."
            )
        else:
            detalle = (
                "Esa carpeta no existe. Pidele a Pablo los 42 Parquet y su manifiesto "
                "SHA256SUMS.txt, guardalos fuera de OneDrive y de Google Drive (por ejemplo, "
                "en ~/datos_tfm/tep_rieth), comprueba que llegaron enteros con "
                f"verificar_rieth.py y define la variable de entorno {RIETH_DIR_ENV} con esa "
                "carpeta (o pasa root=). Convertir los .RData uno mismo pide unos 16 GB de "
                "memoria: no es lo recomendado."
            )
        raise FileNotFoundError(f"No encuentro {path} (carpeta tomada de: {origen}). {detalle}")

    pq = _pyarrow_parquet()
    fichero = pq.ParquetFile(path, pre_buffer=False)
    try:
        esquema = fichero.schema_arrow
        faltan = [c for c in COLUMNS if c not in esquema.names]
        if faltan:
            raise ValueError(
                f"{path.name} no tiene las columnas del contrato: {faltan[:5]}"
                f"{' ...' if len(faltan) > 5 else ''}. ¿Es un Parquet de convert_rieth?"
            )

        def columna(nombre: str) -> np.ndarray:
            # Una columna cada vez: la memoria transitoria es la de una columna,
            # no la del fichero entero (los Parquet tienen un solo row group, asi
            # que un filtro de pyarrow descomprime igual todo el grupo).
            return fichero.read(columns=[nombre], use_threads=False).column(0).to_numpy()

        sim = columna("simulationRun")
        disponibles = np.unique(sim)
        ids = _ids_pedidos(runs, disponibles, path.name)
        mascara = np.ones(sim.size, dtype=bool) if ids is None else np.isin(sim, ids)
        n_filas = int(mascara.sum())

        tipo = np.result_type(*[esquema.field(c).type.to_pandas_dtype() for c in VAR_NAMES])
        valores = np.empty((N_VARS, n_filas), dtype=tipo)
        for j, c in enumerate(VAR_NAMES):
            valores[j] = columna(c)[mascara]
        fallo_col = columna("faultNumber")[mascara].astype(np.int64)
        muestra = columna("sample")[mascara].astype(np.int64)
        sim = sim[mascara].astype(np.int64)
    finally:
        fichero.close()

    if fallo_col.size and not (fallo_col == fault).all():
        raise ValueError(
            f"{path.name} trae faultNumber {np.unique(fallo_col)[:5].tolist()} y se pidio el "
            f"fallo {int(fault)}: el fichero esta mal nombrado o no es de Rieth."
        )
    ids_leidos, por_sim = np.unique(sim, return_counts=True)
    esperadas = N_SAMPLES_RIETH[split]
    raras = ids_leidos[por_sim != esperadas]
    if raras.size:
        raise ValueError(
            f"{path.name}: las simulaciones {raras[:10].tolist()} no tienen {esperadas} "
            f"muestras ({split}). ¿Fichero truncado? Compruebalo con verificar_rieth.py."
        )

    df = pd.DataFrame(valores.T, columns=VAR_NAMES, copy=False)
    df.insert(0, "sample", muestra)
    df.insert(0, "simulationRun", sim)
    df.insert(0, "faultNumber", fallo_col)
    return df


# ------------------------------------------------- particiones y registro


def split_runs(
    n_runs: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
    *,
    exclude: Iterable[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reparte identificadores de simulacion en entrenamiento, validacion y prueba.

    La particion es POR SIMULACION, nunca por muestra: una misma simulacion no
    puede aparecer en dos conjuntos, porque sus muestras estan correlacionadas
    en el tiempo y eso seria fuga de datos.

    Regla de las gemelas (Rieth). La simulacion r de cada fichero de fallo es
    gemela de la normal r del mismo split: comparten semilla, son identicas hasta
    el inicio del fallo y, en los fallos 3, 9 y 15, siguen casi pegadas despues
    (DAT-1, DIS-2; informe 11 de la revision del 03/10, B.5). Por eso:

    - Se llama UNA vez y se usan los mismos ids en todas las clases.
    - En validacion cruzada, los grupos son simulationRun sobre el conjunto de
      todas las clases (GroupKFold con groups = df["simulationRun"]), NUNCA el
      par (fallo, simulacion): con esa clave, la gemela normal de una simulacion
      de prueba se queda en entrenamiento con otra etiqueta y acaba siendo su
      vecina mas cercana (informe 11, C.1).
    - Entre splits no hace falta agrupar: train_* y test_* no comparten
      semillas (informe 11, B.6).

    Simulaciones ya miradas. Las que se han mirado en desarrollo no sirven para
    el resultado confirmatorio (regla 3 de docs/reglas_de_medida.md, DIS-17).
    `exclude` las saca del reparto: split_runs(500, exclude=seen_runs(split,
    faults)), con los ficheros que vaya a usar el diseno (ver RIETH_SEEN_RUNS).
    Que ids son los confirmatorios, y con que procedimiento, se fija en el
    registro de decisiones antes de ejecutar; esta funcion solo da el mecanismo.

    Parametros
    ----------
    n_runs : int
        Los candidatos son los ids 1..n_runs (500 en Rieth).
    fractions : (float, float, float)
        Fracciones de entrenamiento, validacion y prueba; suman 1 y alguna puede
        ser 0. Se aplican sobre los candidatos que quedan tras `exclude`: con n
        candidatos, round(fractions[0] * n) van a entrenamiento,
        round(fractions[1] * n) a validacion y el resto a prueba. Una parte con
        fraccion 0 queda siempre vacia: si la de prueba es 0, lo que el redondeo
        dejaria en ella va a validacion. Con las tres fracciones positivas el
        reparto es el de siempre.
    seed : int
        Semilla del reparto (la del yaml, protocolo.semilla).
    exclude : iterable de int, opcional
        Ids que no entran en el reparto; tienen que estar en 1..n_runs. Sin
        `exclude` (o con una lista vacia), el resultado es exactamente el de las
        versiones anteriores con la misma semilla.

    Devuelve
    --------
    Tres arrays de ids (int64, ordenados y sin solapamiento): entrenamiento,
    validacion y prueba.
    """
    if (
        isinstance(n_runs, (bool, np.bool_))
        or not isinstance(n_runs, (int, np.integer))
        or n_runs < 1
    ):
        raise ValueError(f"n_runs debe ser un entero positivo, llego {n_runs!r}")
    if len(fractions) != 3 or min(fractions) < 0:
        raise ValueError(f"fractions son tres fracciones no negativas, llego {fractions}")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Las fracciones deben sumar 1, suman {sum(fractions)}")

    candidatos = np.arange(1, int(n_runs) + 1, dtype=np.int64)
    if exclude is not None:
        fuera = _como_ids(exclude, "exclude")
        raros = fuera[(fuera < 1) | (fuera > n_runs)]
        if raros.size:
            raise ValueError(
                f"exclude trae ids fuera de 1..{int(n_runs)}: {raros[:10].tolist()}. "
                "¿Mezclaste posiciones con ids?"
            )
        candidatos = np.setdiff1d(candidatos, fuera)
        if candidatos.size == 0:
            raise ValueError("exclude deja sin simulaciones que repartir.")

    return _repartir(candidatos, fractions, seed)


def _repartir(candidatos, fractions, seed) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nucleo del reparto por simulacion, comun a split_runs y ablation.split_by_run.

    Esta en un solo sitio para que los dos repartos no se separen al corregir
    uno (verificacion del 12/10, 2.9). Ordena los ids candidatos sin repetir,
    los permuta con la semilla y da round(f0 * n) a la primera parte,
    round(f1 * n) a la segunda y el resto a la tercera. Si la tercera fraccion
    es 0, el resto va a la segunda: sin eso, el redondeo podia dar simulaciones
    a una parte con fraccion 0 (con n = 5 y (0.5, 0.5, 0.0), una a la tercera;
    V-08). La primera y la segunda no lo necesitan: round(0 * n) es 0. Con las
    tres fracciones positivas, el reparto es exactamente el de siempre.

    `fractions` llega ya validada (tres, no negativas, suman 1).
    """
    candidatos = np.unique(np.asarray(candidatos))
    ids = np.random.default_rng(seed).permutation(candidatos)

    n = candidatos.size
    n_train = int(round(fractions[0] * n))
    if fractions[2] == 0:
        n_val = n - n_train
    else:
        n_val = int(round(fractions[1] * n))

    return (
        np.sort(ids[:n_train]),
        np.sort(ids[n_train : n_train + n_val]),
        np.sort(ids[n_train + n_val :]),
    )


def _como_ids(valores, nombre: str) -> np.ndarray:
    """Convierte un iterable de ids (de Python o de numpy) en un array int64 unico."""
    if isinstance(valores, (str, bytes)) or not isinstance(valores, Iterable) or (
        isinstance(valores, np.ndarray) and valores.ndim == 0
    ):
        raise TypeError(f"{nombre} debe ser una lista de ids, llego {valores!r}")
    arr = np.asarray(list(valores))
    if arr.size == 0:
        return np.array([], dtype=np.int64)
    if arr.ndim != 1 or arr.dtype.kind not in "iu":
        raise TypeError(f"{nombre} debe ser una lista plana de ids enteros; llego {arr.dtype}")
    return np.unique(arr).astype(np.int64)


class SeenRuns(NamedTuple):
    """Una entrada del registro de simulaciones de Rieth ya miradas.

    splits y faults dicen en que ficheros (por ejemplo ("test",) y (0, ..., 20)
    son los 21 test_faultNN); runs, que ids; source, de donde sale el dato.
    """

    splits: tuple[str, ...]
    faults: tuple[int, ...]
    runs: tuple[int, ...]
    source: str


_TODOS = tuple(range(21))          # 0..20: los 21 ficheros de un split
_CON_FALLO = tuple(range(1, 21))   # 1..20

#: Registro de las simulaciones de Rieth ya miradas en desarrollo, que no sirven
#: para el resultado confirmatorio (regla 3 de docs/reglas_de_medida.md,
#: DIS-17). "Mirada" es una simulacion de la que se han leido valores de proceso
#: para calcular algo; leer solo las columnas de ids (conteos) no cuenta, ni
#: decodificar un fichero para validar un lector sin calcular nada por
#: simulacion (informe 11 de la revision del 03/10, B.0). Lo que se mire a partir
#: de ahora se anade aqui, con su fuente. Ultima revision: 11/10/2026.
RIETH_SEEN_RUNS: tuple[SeenRuns, ...] = (
    # Regla 3 (estado del 27/09), punto 1: analisis de datos, verificaciones y
    # cuadernos de arranque. Incluye el piloto de DIS-17 (test_fault00, 01, 02,
    # 03, 05, 06, 09, 10, 15 y 20 y train_fault06, dentro de 1..20), el B00
    # (secciones 7.2 y 8) y el informe 11 (B.5, B.6, B.7).
    SeenRuns(("train", "test"), _TODOS, tuple(range(1, 21)),
             "regla 3, punto 1: los 42 ficheros, ids 1 a 20"),
    # Regla 3, punto 2 (DIS-17: la verificacion del 27/09 cargo las 1-40 de
    # train_fault00; el C00 usa ahi ids de ajuste, parada y calibracion <= 38).
    SeenRuns(("train",), (0,), tuple(range(21, 41)),
             "regla 3, punto 2: train_fault00, ids 21 a 40"),
    # Regla 3, punto 3: diseno de ids distintos del B00 (seccion 7.1), con
    # datos.runs_desarrollo = 20: en train_faultKK, de 20*KK + 1 a 20*(KK + 1),
    # es decir, hasta la 420 en train_fault20.
    *(
        SeenRuns(("train",), (k,), tuple(range(20 * k + 1, 20 * (k + 1) + 1)),
                 f"regla 3, punto 3: B00 seccion 7.1, ids distintos por clase (train_fault{k:02d})")
        for k in _TODOS
    ),
    # Regla 3, punto 4: los ids del C00 guardados en
    # results/raw/C00_configuracion.json (contrastado con ese fichero el 11/10:
    # ids_prueba en los test_*; ids_diag_tr e ids_diag_va en los train_*).
    SeenRuns(("test",), _TODOS, (23, 24, 32, 35, 36, 37, 42, 43, 44, 50, 58, 64, 66),
             "regla 3, punto 4: C00, prueba en los 21 test_*"),
    SeenRuns(("train",), _CON_FALLO, (21, 25, 26),
             "regla 3, punto 4: C00, diagnostico en los train_* con fallo"),
    # Traspaso v12, seccion 9.7, e informe 11 de la revision del 03/10 (B.0 y
    # B.5; script B5_gemelas.py): las simulaciones 250 y 500 de los ficheros 00,
    # 01, 03, 06, 09 y 15, en entrenamiento y en prueba.
    SeenRuns(("train", "test"), (0, 1, 3, 6, 9, 15), (250, 500),
             "traspaso v12, 9.7 (informe 11 del 03/10, B.5)"),
)

#: Entradas del registro PENDIENTES DE CONFIRMAR. No se inventan: hasta que
#: Pablo las confirme o las descarte en el registro de decisiones, seen_runs y
#: unseen_runs avisan de que el registro esta incompleto, y sus ids no se
#: excluyen. Regla 3, punto 5, literal: "Los prototipos del C00 tomaron prefijos
#: de las mismas listas de split_runs(500) con la semilla del yaml; el tamano
#: exacto dependia de sus argumentos, que no quedaron registrados".
RIETH_SEEN_PENDING: tuple[str, ...] = (
    "Prototipos del C00 (regla 3, punto 5): prefijos de longitud desconocida de las "
    "listas de split_runs(500) con la semilla del yaml, en ficheros sin registrar.",
)


def _ficheros(split: str | None, faults) -> tuple[set[str], set[int]]:
    """Valida y normaliza la seleccion de ficheros de seen_runs y unseen_runs."""
    if split is None:
        splits = {"train", "test"}
    elif split in ("train", "test"):
        splits = {split}
    else:
        raise ValueError(f"split debe ser 'train', 'test' o None (los dos); llego {split!r}")

    if faults is None:
        fallos = set(_TODOS)
    elif isinstance(faults, (int, np.integer)):
        fallos = {faults}
    elif isinstance(faults, Iterable) and not isinstance(faults, (str, bytes)):
        fallos = set(list(faults))
    else:
        fallos = set()
    malos = [f for f in fallos
             if isinstance(f, (bool, np.bool_)) or not isinstance(f, (int, np.integer))
             or f not in range(21)]
    if malos or not fallos:
        raise ValueError(
            f"faults son fallos de Rieth entre 0 y 20 (o None, todos); llego {faults!r}"
        )
    return splits, {int(f) for f in fallos}


def _vistos(splits: set[str], fallos: set[int]) -> np.ndarray:
    vistos: set[int] = set()
    for e in RIETH_SEEN_RUNS:
        if splits & set(e.splits) and fallos & set(e.faults):
            vistos.update(e.runs)
    return np.array(sorted(vistos), dtype=np.int64)


def _avisar_pendientes() -> None:
    if RIETH_SEEN_PENDING:
        warnings.warn(
            "El registro de simulaciones miradas de Rieth tiene entradas sin confirmar "
            f"(RIETH_SEEN_PENDING): {' | '.join(RIETH_SEEN_PENDING)} Sus ids NO se tienen "
            "en cuenta aqui. Antes de fijar ids confirmatorios, hay que confirmarlas o "
            "descartarlas en el registro de decisiones.",
            UserWarning,
            stacklevel=3,
        )


def seen_runs(split: str | None = None, faults: int | Iterable[int] | None = None) -> np.ndarray:
    """Ids de Rieth ya mirados en alguno de los ficheros pedidos, segun RIETH_SEEN_RUNS.

    Es la lista que se pasa a split_runs(500, exclude=...) para que el reparto
    confirmatorio no reutilice simulaciones de desarrollo. Mientras haya entradas
    pendientes (RIETH_SEEN_PENDING), avisa.

    Parametros
    ----------
    split : {'train', 'test'} o None
        None son los dos splits.
    faults : int, iterable de int o None
        Fallos 0..20 (0 = test_fault00 o train_fault00). None son los 21.

    Devuelve
    --------
    Array int64 ordenado con la union de los ids mirados en esos ficheros.
    """
    splits, fallos = _ficheros(split, faults)
    _avisar_pendientes()
    return _vistos(splits, fallos)


def unseen_runs(split: str | None = None, faults: int | Iterable[int] | None = None) -> np.ndarray:
    """Ids de Rieth (1..500) que no se han mirado en NINGUNO de los ficheros pedidos.

    Con un solo fichero (por ejemplo, unseen_runs("test", 3)) da sus ids libres.
    Con varios, solo los que estan libres en todos, que es lo que hace falta para
    fijar ids comunes a los tres bloques: por la regla de las gemelas, un id se
    usa en todas las clases a la vez. Que ficheros cuentan (solo los de prueba o
    tambien los de entrenamiento) lo decide el procedimiento confirmatorio del
    registro de decisiones. Mientras haya entradas pendientes
    (RIETH_SEEN_PENDING), avisa.

    Parametros
    ----------
    split, faults : como en `seen_runs`.

    Devuelve
    --------
    Array int64 ordenado.
    """
    splits, fallos = _ficheros(split, faults)
    _avisar_pendientes()
    todos = np.arange(1, N_RUNS_RIETH + 1, dtype=np.int64)
    return np.setdiff1d(todos, _vistos(splits, fallos))


# ------------------------------------------------------- inicio del fallo


def fault_onset(split: str, source: str = "classic") -> int | None:
    """Muestra tras la cual entra el fallo.

    Devuelve None cuando el fallo esta activo desde la primera muestra, que es
    el caso de los ficheros de entrenamiento del TEP clasico: los dXX.dat tienen
    480 filas con el fallo activo desde la fila 0 y se usan enteros, sin
    descartar ni reetiquetar ninguna (revision del 27/09, B2). En Rieth, 20 en
    entrenamiento y 160 en prueba; en la prueba del clasico, 160.

    Parametros
    ----------
    split : {'train', 'test'}
    source : {'classic', 'rieth'}
    """
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', llego {split!r}")
    if source not in ("classic", "rieth"):
        raise ValueError(f"source debe ser 'classic' o 'rieth', llego {source!r}")
    if split == "test":
        return FAULT_ONSET_TEST
    if source == "rieth":
        return FAULT_ONSET_TRAIN_RIETH
    return None


def effective_onset(fault_df: pd.DataFrame, normal_df: pd.DataFrame, onset: int) -> pd.DataFrame:
    """Inicio efectivo del fallo en Rieth: hasta que muestra cada simulacion es copia de su gemela.

    Por que existe. En Rieth, la simulacion r de un fichero de fallo es la misma
    trayectoria que la normal r del mismo split hasta el inicio nominal (la
    muestra 20 en entrenamiento y la 160 en prueba). En varios fallos (8 a 13,
    16 a 18 y 20) la copia exacta sigue bastantes muestras despues del inicio
    (informe 11 de la revision del 03/10, B.5; DAT-1). Esas filas llevan la
    etiqueta del fallo pero son identicas a la clase 0: ningun detector puede
    distinguirlas, y fijan un suelo al retardo y un techo a la FDR. Si ese
    tramo se mide, se declara o se descarta lo decide el registro de decisiones
    (regla 6); esta funcion da el numero, para que los tres bloques lo hagan
    igual.

    Compara las 52 variables con igualdad exacta (los valores son los del
    Parquet; float32 frente a float32, o frente a float64, es exacto).

    Parametros
    ----------
    fault_df : DataFrame
        Filas de uno o varios ficheros de fallo del mismo split (contrato del
        modulo). Puede traer o no las filas previas al inicio; si las trae,
        tienen que ser copia exacta de su gemela, y si no lo son es un error (no
        son gemelas: otro split, otros ids u otro onset). Las posteriores al
        inicio tienen que estar desde la onset + 1 y seguidas, al menos hasta la
        primera distinta de la gemela (hasta la ultima, si no se separa): se
        cuenta por el valor de `sample`, y si faltan es un error.
    normal_df : DataFrame
        La clase 0 del MISMO split, con al menos las simulaciones de fault_df.
    onset : int
        Inicio nominal: fault_onset(split, "rieth").

    Devuelve
    --------
    DataFrame con una fila por (faultNumber, simulationRun) y las columnas:
        onset               el inicio nominal que se paso.
        effective_onset     ultima muestra de la copia exacta: la trayectoria es
                            identica a su gemela hasta esa muestra y distinta en
                            la siguiente. Misma convencion que onset (el fallo
                            se ve a partir de effective_onset + 1), asi que se
                            puede usar como onset en metrics.evaluate_run. Si la
                            simulacion no se separa nunca, es su ultima muestra.
        copied_after_onset  muestras posteriores al inicio nominal que siguen
                            siendo copia exacta (effective_onset - onset).
        separates           False si no se separa de su gemela en ninguna
                            muestra.

    Para descartar el tramo copiado, igual en todos los bloques:

        ef = effective_onset(fallo, normal, fault_onset("test", "rieth"))
        fallo = fallo.merge(ef[["faultNumber", "simulationRun", "effective_onset"]],
                            on=["faultNumber", "simulationRun"])
        fallo = fallo[fallo["sample"] > fallo["effective_onset"]]
    """
    if (
        isinstance(onset, (bool, np.bool_))
        or not isinstance(onset, (int, np.integer))
        or onset < 0
    ):
        raise ValueError(
            "onset debe ser un entero >= 0, por ejemplo fault_onset(split, 'rieth'); "
            f"llego {onset!r}"
        )
    for nombre, d in (("fault_df", fault_df), ("normal_df", normal_df)):
        faltan = [c for c in COLUMNS if c not in d.columns]
        if faltan:
            raise ValueError(f"{nombre} no tiene las columnas del contrato: {faltan[:5]}")
    if fault_df.empty:
        raise ValueError("fault_df esta vacio.")
    if (normal_df["faultNumber"] != 0).any():
        raise ValueError("normal_df tiene que ser la clase 0 (operacion normal) del mismo split.")
    if (fault_df["faultNumber"] == 0).any():
        raise ValueError("fault_df trae filas de la clase 0: pasa solo filas de ficheros de fallo.")

    def clave(d: pd.DataFrame) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [d["simulationRun"].to_numpy(np.int64), d["sample"].to_numpy(np.int64)],
            names=["simulationRun", "sample"],
        )

    idx_normal = clave(normal_df)
    if not idx_normal.is_unique:
        raise ValueError("normal_df repite algun par (simulationRun, sample).")
    if fault_df.duplicated(ID_COLUMNS).any():
        raise ValueError("fault_df repite algun trio (faultNumber, simulationRun, sample).")

    pos = idx_normal.get_indexer(clave(fault_df))
    if (pos < 0).any():
        sin = np.unique(fault_df["simulationRun"].to_numpy(np.int64)[pos < 0])
        raise ValueError(
            f"Las simulaciones {sin[:10].tolist()} de fault_df no tienen todas sus muestras en "
            "normal_df: carga la clase 0 del mismo split con los mismos ids."
        )
    propio = fault_df[VAR_NAMES].to_numpy()
    gemela = normal_df[VAR_NAMES].to_numpy()[pos]
    igual = (propio == gemela).all(axis=1)

    tabla = pd.DataFrame({
        "faultNumber": fault_df["faultNumber"].to_numpy(np.int64),
        "simulationRun": fault_df["simulationRun"].to_numpy(np.int64),
        "sample": fault_df["sample"].to_numpy(np.int64),
        "igual": igual,
    })
    grupo = ["faultNumber", "simulationRun"]

    previas = tabla[tabla["sample"] <= onset]
    malas = previas.loc[~previas["igual"], grupo].drop_duplicates()
    if len(malas):
        pares = list(malas.itertuples(index=False, name=None))
        raise ValueError(
            f"Antes del inicio ({int(onset)}) estas simulaciones no son copia exacta de su "
            f"normal: {pares[:5]}{' ...' if len(pares) > 5 else ''} (fallo, simulacion). No "
            "son gemelas: ¿mezclaste splits o ids, o el onset no es el de este split?"
        )

    ultima = tabla.groupby(grupo)["sample"].max()
    sin_post = ultima[ultima <= onset]
    if len(sin_post):
        raise ValueError(
            f"fault_df no trae muestras posteriores al inicio ({int(onset)}) en "
            f"{list(sin_post.index)[:5]}: no se puede medir donde se separan."
        )

    # Se cuenta por el valor de sample, no por las filas que hayan llegado (V-07
    # de la verificacion del 12/10): si faltan las primeras muestras posteriores
    # al inicio, la primera distinta que llega no es la primera distinta. Por eso,
    # en cada (fallo, simulacion), las muestras posteriores al inicio tienen que
    # estar desde la onset + 1 y seguidas hasta la primera distinta de la gemela
    # (o hasta la ultima, si no se separa); lo que venga despues da igual.
    post = tabla[tabla["sample"] > onset].sort_values(grupo + ["sample"], kind="stable")
    orden = post.groupby(grupo).cumcount().to_numpy()
    seguida = post["sample"].to_numpy() == onset + 1 + orden   # cierto en un prefijo de cada grupo
    aux = post[grupo].assign(orden=orden, seguida=seguida, igual=post["igual"].to_numpy())
    por_grupo = aux.groupby(grupo)
    n_post = por_grupo.size()
    n_seguidas = por_grupo["seguida"].sum()
    primera_distinta = aux[~aux["igual"]].groupby(grupo)["orden"].min().reindex(n_post.index)
    separa = primera_distinta.notna()
    hasta = primera_distinta.fillna(n_post - 1)            # posicion que tiene que estar seguida
    huecos = hasta >= n_seguidas
    if huecos.any():
        pares = list(huecos[huecos].index)
        raise ValueError(
            f"fault_df no trae seguidas las muestras posteriores al inicio en {pares[:5]}"
            f"{' ...' if len(pares) > 5 else ''} (fallo, simulacion): tienen que estar desde la "
            f"{int(onset) + 1} y sin huecos hasta la primera distinta de la gemela (o hasta la "
            "ultima, si no se separa). Sin ellas no se sabe donde se separa: no recortes esas "
            "filas (ni las conviertas en ventanas) antes de llamar a effective_onset."
        )
    efectivo = int(onset) + np.where(separa, primera_distinta.fillna(0), n_post).astype(np.int64)

    out = n_post.index.to_frame(index=False)
    out["onset"] = np.int64(onset)
    out["effective_onset"] = efectivo
    out["copied_after_onset"] = out["effective_onset"] - out["onset"]
    out["separates"] = separa.to_numpy()
    return out


def values(df: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """Extrae la matriz numerica de variables de proceso, en orden fijo.

    Por defecto devuelve las 52 variables (`VAR_NAMES`). Con `columns` se
    selecciona un subconjunto, por ejemplo `YIN_VARS` para reproducir la
    configuracion de Yin et al. (2012). Se selecciona por nombre y no por
    posicion: si falta una columna, falla con KeyError en vez de devolver una
    matriz equivocada en silencio.

    Pedirle una columna de identificacion (faultNumber, simulationRun o sample)
    es un error: no son variables de proceso y un modelo no debe recibirlas. En
    Rieth el fallo entra siempre tras la misma muestra, asi que `sample` sola
    acierta casi siempre en la prueba (informe 11 de la revision del 03/10, C.4);
    simulationRun identifica la trayectoria y faultNumber es la etiqueta. Si
    necesitas los ids para agrupar o emparejar, tomalos aparte, por ejemplo
    df["simulationRun"].to_numpy().
    """
    cols = list(VAR_NAMES) if columns is None else list(columns)
    ids = [c for c in cols if c in ID_COLUMNS]
    if ids:
        raise ValueError(
            f"values() no devuelve columnas de identificacion ({ids}): no son variables de "
            "proceso y ningun modelo debe recibirlas ('sample' delata el inicio fijo del "
            "fallo; simulationRun, la trayectoria; faultNumber es la etiqueta). Si las "
            "necesitas para agrupar o emparejar, tomalas aparte: df['simulationRun'].to_numpy()."
        )
    return df[cols].to_numpy(dtype=float)


# ------------------------------------------------------ nombres retirados

_RETIRADOS = {
    "FAULT_ONSET_TRAIN_CLASSIC": (
        "data.FAULT_ONSET_TRAIN_CLASSIC se retiro en la v0.3 porque era falso (valia 20): "
        "en los dXX.dat del TEP clasico el fallo esta activo desde la fila 0 (revision del "
        "27/09, B2; informe 11 de la revision del 03/10, A.2). Usa "
        "data.fault_onset('train', 'classic'), que devuelve None, y las 480 filas enteras: "
        "no descartes ni reetiquetes las 20 primeras. El 20 es el inicio de Rieth "
        "(data.fault_onset('train', 'rieth'))."
    ),
}


def __getattr__(name: str):
    # PEP 562: solo se llama cuando el nombre no existe en el modulo. El aviso
    # repite el mensaje porque "from tfmfdd.data import X" convierte el
    # AttributeError en un ImportError generico y lo pierde.
    if name in _RETIRADOS:
        warnings.warn(_RETIRADOS[name], FutureWarning, stacklevel=2)
        raise AttributeError(_RETIRADOS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
