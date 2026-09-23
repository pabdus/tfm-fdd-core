"""Conversion de los ficheros .RData de Rieth et al. (2017) a Parquet.

Por que hace falta este paso
----------------------------
Los cuatro ficheros vienen en formato .RData, que es nativo de R. Leerlos desde
Python requiere pyreadr, y pyreadr **carga el objeto entero en memoria**: no hay
lectura por trozos. El fichero de prueba con fallos tiene unos 9,6 millones de
filas por 55 columnas, lo que en float64 son alrededor de 4 GB solo de datos,
mas el gasto temporal de la propia lectura.

Se convierte una sola vez a Parquet particionado por fallo y en float32. A
partir de ahi, cargar un fallo concreto con 20 simulaciones cuesta milisegundos
y unos pocos MB, y los .RData no se vuelven a tocar.

Uso
---
    python -m tfmfdd.convert_rieth

    python -m tfmfdd.convert_rieth --origen data/tep_rieth --destino data/tep_rieth
    python -m tfmfdd.convert_rieth --solo faulty_testing   # uno solo, si falta memoria

Antes de ejecutarlo
-------------------
Cierra el navegador y todo lo que no necesites. El fichero faulty_testing es el
que aprieta; si el equipo tiene 16 GB deberia entrar, pero con Chrome abierto
puede no entrar. Si aun asi falla con MemoryError, ejecuta los ficheros de uno
en uno con --solo, empezando por los pequenos.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLUMNS, VAR_NAMES

#: Nombre del fichero .RData -> (split, con_fallos)
FICHEROS = {
    "TEP_FaultFree_Training.RData": ("train", False),
    "TEP_FaultFree_Testing.RData": ("test", False),
    "TEP_Faulty_Training.RData": ("train", True),
    "TEP_Faulty_Testing.RData": ("test", True),
}

#: Tamano aproximado en disco, para avisar si un fichero llego truncado.
TAMANO_ESPERADO_MB = {
    "TEP_FaultFree_Training.RData": (15, 40),
    "TEP_FaultFree_Testing.RData": (30, 70),
    "TEP_Faulty_Training.RData": (300, 700),
    "TEP_Faulty_Testing.RData": (600, 1300),
}


def _normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Lleva los nombres de columna de R al contrato del paquete.

    En los .RData las columnas vienen como faultNumber, simulationRun, sample,
    xmeas_1..xmeas_41 y xmv_1..xmv_11. Esta funcion tolera variaciones de
    mayusculas y de separador, y falla ruidosamente si falta alguna.
    """
    renombrado = {}
    for c in df.columns:
        limpio = c.strip().lower().replace(".", "_").replace(" ", "_")
        if limpio in ("faultnumber", "fault_number"):
            renombrado[c] = "faultNumber"
        elif limpio in ("simulationrun", "simulation_run"):
            renombrado[c] = "simulationRun"
        elif limpio == "sample":
            renombrado[c] = "sample"
        else:
            renombrado[c] = limpio

    df = df.rename(columns=renombrado)

    faltan = [c for c in COLUMNS if c not in df.columns]
    if faltan:
        raise ValueError(
            f"Al fichero le faltan columnas del contrato: {faltan[:5]}"
            f"{'...' if len(faltan) > 5 else ''}. "
            f"Las que trae son: {list(df.columns)[:8]}..."
        )
    return df[COLUMNS]


def _aligerar(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce la memoria sin perder informacion util.

    Las 52 variables de proceso pasan a float32: siete cifras significativas
    sobran para sensores industriales y la memoria se reduce a la mitad. Los
    tres identificadores pasan a enteros pequenos.
    """
    df["faultNumber"] = df["faultNumber"].astype("int16")
    df["simulationRun"] = df["simulationRun"].astype("int16")
    df["sample"] = df["sample"].astype("int16")
    for c in VAR_NAMES:
        df[c] = df[c].astype("float32")
    return df


def convertir_fichero(ruta: Path, destino: Path, verbose: bool = True) -> int:
    """Convierte un .RData a un Parquet por fallo. Devuelve cuantos escribio."""
    import pyreadr  # se importa aqui para no exigirlo si no se usa

    nombre = ruta.name
    if not ruta.exists():
        raise FileNotFoundError(f"No encuentro {ruta}")

    mb = ruta.stat().st_size / (1024 * 1024)
    rango = TAMANO_ESPERADO_MB.get(nombre)
    if rango and not (rango[0] <= mb <= rango[1]):
        print(
            f"  AVISO: {nombre} pesa {mb:.0f} MB y se esperaban entre "
            f"{rango[0]} y {rango[1]} MB. Puede estar truncado; "
            f"comprueba la descarga antes de fiarte del resultado.",
            file=sys.stderr,
        )

    split, con_fallos = FICHEROS[nombre]
    if verbose:
        print(f"  Leyendo {nombre} ({mb:.0f} MB). Esto tarda y consume memoria...")

    resultado = pyreadr.read_r(str(ruta))
    if not resultado:
        raise ValueError(f"{nombre} no contiene ningun objeto de R")

    # pyreadr devuelve un diccionario {nombre_del_objeto: DataFrame}.
    clave = list(resultado.keys())[0]
    df = resultado[clave]
    del resultado
    gc.collect()

    if verbose:
        print(f"  Leido: {len(df):,} filas. Normalizando...")

    df = _normalizar_columnas(df)
    df = _aligerar(df)
    gc.collect()

    destino.mkdir(parents=True, exist_ok=True)
    fallos = sorted(df["faultNumber"].unique())
    escritos = 0

    for fallo in fallos:
        sub = df[df["faultNumber"] == fallo].reset_index(drop=True)
        salida = destino / f"{split}_fault{int(fallo):02d}.parquet"
        sub.to_parquet(salida, index=False, compression="snappy")
        escritos += 1
        if verbose:
            n_runs = sub["simulationRun"].nunique()
            print(
                f"    fallo {int(fallo):2d}: {len(sub):>9,} filas, "
                f"{n_runs:>3} simulaciones -> {salida.name}"
            )
        del sub

    del df
    gc.collect()
    return escritos


def comprobar(destino: Path) -> None:
    """Revisa que lo convertido tenga la forma esperada."""
    print("\nComprobacion de lo convertido:")
    ficheros = sorted(destino.glob("*.parquet"))
    if not ficheros:
        print("  No hay ningun Parquet. Algo fallo.")
        return

    total = 0
    for f in ficheros:
        df = pd.read_parquet(f, columns=["faultNumber", "simulationRun", "sample"])
        total += len(df)
        n_muestras = df.groupby("simulationRun")["sample"].max().unique()
        esperado = 500 if "train" in f.name else 960
        marca = "ok" if list(n_muestras) == [esperado] else f"REVISAR {n_muestras}"
        print(
            f"  {f.name:<28} {len(df):>9,} filas  "
            f"{df['simulationRun'].nunique():>3} sims  {marca}"
        )
        del df

    print(f"\n  Total: {total:,} filas en {len(ficheros)} ficheros.")
    print(f"  Tamano en disco: {sum(f.stat().st_size for f in ficheros)/(1024**2):.0f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origen", default="data/tep_rieth",
                   help="carpeta con los cuatro .RData")
    p.add_argument("--destino", default="data/tep_rieth",
                   help="carpeta donde dejar los Parquet")
    p.add_argument("--solo", choices=["faultfree_training", "faultfree_testing",
                                      "faulty_training", "faulty_testing"],
                   help="convertir un solo fichero, para no quedarse sin memoria")
    p.add_argument("--callado", action="store_true")
    args = p.parse_args()

    origen, destino = Path(args.origen), Path(args.destino)
    verbose = not args.callado

    objetivo = list(FICHEROS)
    if args.solo:
        clave = args.solo.replace("faultfree", "FaultFree").replace("faulty", "Faulty")
        clave = clave.replace("_training", "_Training").replace("_testing", "_Testing")
        objetivo = [f"TEP_{clave}.RData"]

    if verbose:
        print(f"Convirtiendo de {origen} a {destino}\n")

    total = 0
    for nombre in objetivo:
        ruta = origen / nombre
        if not ruta.exists():
            print(f"  SALTADO: no encuentro {nombre}", file=sys.stderr)
            continue
        try:
            total += convertir_fichero(ruta, destino, verbose)
        except MemoryError:
            print(
                f"\n  SIN MEMORIA con {nombre}.\n"
                f"  Cierra todo lo demas y ejecuta solo este fichero:\n"
                f"    python -m tfmfdd.convert_rieth --solo "
                f"{nombre.replace('TEP_','').replace('.RData','').lower()}\n",
                file=sys.stderr,
            )
            raise
        print()

    if verbose:
        print(f"Listo: {total} ficheros Parquet escritos.")
        comprobar(destino)
        print(
            "\nYa puedes borrar los .RData si necesitas el espacio: el script de\n"
            "descarga los vuelve a traer y los Parquet son la fuente de trabajo."
        )


if __name__ == "__main__":
    main()
