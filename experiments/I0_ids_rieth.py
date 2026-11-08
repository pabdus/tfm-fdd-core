"""I0 - Ids de Rieth del prerregistro: normales de cada replica de ajuste y orden de los ids de prueba.

Por que existe. El acta de decisiones del 18/10 (secciones 8, 9 y 15) y las
filas de esas fechas en docs/decisiones.md fijan REGLAS: cuantas replicas de
ajuste hay (particion.rieth.replicas_ajuste), como se reparten las normales
libres de train_fault00 entre ellas (particion.rieth.reparto) y que ids de
prueba son confirmatorios (datos.ids_confirmatorios: los n mas altos de
data.unseen_runs("test")). La seccion 9 pide calcular y anotar los ids
concretos despues de cerrar RIETH_SEEN_PENDING y antes de la etiqueta. Este
script los calcula con esas reglas, sin ninguna eleccion propia, y los escribe
en configs/referencias/rieth_ids_prerregistro.csv. No lee ningun Parquet: solo
el registro de simulaciones miradas (tfmfdd.data) y el yaml.

Que calcula, con configs/baseline.yaml (`tfmfdd.config`):

1. Normales: candidatos = data.unseen_runs("train", 0); con 34 por replica
   (normales_ajuste + normales_calibracion), R = floor(candidatos / 34), con un
   minimo de 8, y comprueba que es el R del yaml.
   ablation.subsample_runs(candidatos, 34, R, protocolo.semilla, disjoint=True)
   da los 34 ids de cada replica (ordenados); dentro de cada una,
   data.split_runs(34, (10/34, 4/34, 20/34), protocolo.semilla) sobre las
   posiciones 1 a 34 de esos ids da los que entrenan el autoencoder (10), los
   que lo paran (4) y los que calibran los limites (20). PCA y DPCA se ajustan
   con los 14 de entrenar y parar.
2. Prueba: data.unseen_runs("test"), ordenados de mayor a menor. Los ids
   confirmatorios son los n primeros de esa lista (columna orden <= n), con la n
   que salga de datos.regla_n despues del piloto; se escriben los 400 primeros
   (el peldano mas alto). Son los mismos en los 21 ficheros de prueba y en los
   tres bloques.

Uso, desde la raiz del repositorio:
    python experiments/I0_ids_rieth.py              # escribe el CSV
    python experiments/I0_ids_rieth.py --comprobar  # compara con el CSV versionado (codigo 1 si difiere)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

from tfmfdd import ablation, config, data

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "configs" / "referencias" / "rieth_ids_prerregistro.csv"
REPLICAS_MINIMAS = 8          # acta del 18/10, seccion 8 (Holm sobre 4 contrastes)
N_PRUEBA_MAXIMO = 400         # el ultimo peldano de datos.regla_n
COLUMNAS = ["conjunto", "replica", "rol", "orden", "simulationRun"]

CABECERA = (
    "# configs/referencias/rieth_ids_prerregistro.csv: ids de Rieth del prerregistro, calculados por\n"
    "# experiments/I0_ids_rieth.py con las reglas de particion.rieth y datos.ids_confirmatorios de\n"
    "# configs/baseline.yaml y el registro de simulaciones miradas de tfmfdd/data.py (acta de decisiones\n"
    "# del 18/10, secciones 8, 9 y 15; filas de esas fechas en docs/decisiones.md). Normales: conjunto =\n"
    "# normales_train_fault00, una fila por replica, rol (entrenar_ae, parar_ae, calibrar) e id; orden es la\n"
    "# posicion del id entre los 34 ordenados de su replica. Prueba: conjunto = prueba, orden 1 = el id libre\n"
    "# mas alto; los confirmatorios son los de orden <= n (datos.regla_n). No se edita a mano: una prueba lo\n"
    "# compara con lo que da el script. Leer con pandas.read_csv(ruta, comment='#').\n"
)


def calcular(cfg: dict) -> pd.DataFrame:
    """Las filas del CSV, con las reglas del yaml y el registro de simulaciones miradas."""
    if data.RIETH_SEEN_PENDING:
        raise RuntimeError("RIETH_SEEN_PENDING no esta vacio: los ids libres no son definitivos.")
    semilla = config.require(cfg, "protocolo.semilla")
    n_ajuste = config.require(cfg, "particion.rieth.normales_ajuste")
    n_parada = config.require(cfg, "particion.rieth.normales_parada")
    n_cal = config.require(cfg, "particion.rieth.normales_calibracion")
    r_yaml = config.require(cfg, "particion.rieth.replicas_ajuste")
    reparto = config.require(cfg, "particion.rieth.reparto")
    if reparto != "subsample_disjunto_y_split_runs":
        raise ValueError(f"particion.rieth.reparto = {reparto!r}: este script solo conoce subsample_disjunto_y_split_runs")
    por_replica = n_ajuste + n_cal
    n_tr = n_ajuste - n_parada

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        candidatos = data.unseen_runs("train", 0)
        libres_prueba = data.unseen_runs("test")
    r = len(candidatos) // por_replica
    if r < REPLICAS_MINIMAS:
        raise ValueError(f"solo caben {r} replicas de {por_replica}: la decision se reabre (acta, seccion 8)")
    if r != r_yaml:
        raise ValueError(f"con {len(candidatos)} normales libres caben {r} replicas y el yaml dice {r_yaml}")

    replicas = ablation.subsample_runs(candidatos, por_replica, r, semilla, disjoint=True)
    tr, es, cal = data.split_runs(por_replica, fractions=(n_tr / por_replica, n_parada / por_replica,
                                                          n_cal / por_replica), seed=semilla)
    rol_de = {int(p): "entrenar_ae" for p in tr} | {int(p): "parar_ae" for p in es} | {int(p): "calibrar" for p in cal}
    filas = []
    for i, ids in enumerate(replicas, start=1):
        for posicion, sim in enumerate(ids.tolist(), start=1):
            filas.append(("normales_train_fault00", i, rol_de[posicion], posicion, int(sim)))
    for orden, sim in enumerate(sorted(libres_prueba.tolist(), reverse=True)[:N_PRUEBA_MAXIMO], start=1):
        filas.append(("prueba", None, "prueba", orden, int(sim)))
    tabla = pd.DataFrame(filas, columns=COLUMNAS)
    tabla["replica"] = tabla["replica"].astype("Int64")
    return tabla


def leer(ruta: Path = SALIDA) -> pd.DataFrame:
    tabla = pd.read_csv(ruta, comment="#")
    tabla["replica"] = tabla["replica"].astype("Int64")
    return tabla


def escribir(tabla: pd.DataFrame, ruta: Path = SALIDA) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        f.write(CABECERA)
        tabla.to_csv(f, index=False, lineterminator="\n")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="baseline.yaml (por defecto, el orden de tfmfdd.config.load_baseline)")
    p.add_argument("--comprobar", action="store_true", help="no escribe: compara con el CSV versionado")
    args = p.parse_args(argv)
    tabla = calcular(config.load_baseline(args.config))
    normales = tabla[tabla["conjunto"] == "normales_train_fault00"]
    print(f"{normales['replica'].nunique()} replicas de {normales.groupby('replica').size().iloc[0]} normales "
          f"({len(normales)} ids distintos: {normales['simulationRun'].nunique()}); "
          f"{(tabla['conjunto'] == 'prueba').sum()} ids de prueba en orden (del {tabla[tabla['conjunto'] == 'prueba']['simulationRun'].iloc[0]} hacia abajo)")
    if args.comprobar:
        igual = SALIDA.exists() and leer().equals(tabla)
        print("El CSV versionado coincide." if igual else f"El CSV versionado NO coincide con lo que da el script: {SALIDA}")
        return 0 if igual else 1
    escribir(tabla)
    print(f"Guardado en {SALIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
