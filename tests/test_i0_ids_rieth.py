"""Pruebas de experiments/I0_ids_rieth.py y de configs/referencias/rieth_ids_prerregistro.csv.

El CSV es parte del prerregistro (acta de decisiones del 18/10, secciones 8, 9 y
15): no se edita a mano. Estas pruebas comprueban que es exactamente lo que dan
las reglas del yaml con el registro de simulaciones miradas, y que cumple lo que
esas reglas prometen: replicas disjuntas de normales no miradas, el reparto
10 / 4 / 20 dentro de cada una y los ids de prueba libres en los 21 ficheros,
de mayor a menor. No leen ningun Parquet.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from tfmfdd import config, data

RAIZ = Path(__file__).resolve().parents[1]
BASELINE = RAIZ / "configs" / "baseline.yaml"


def _cargar_experimento(nombre: str):
    ruta = RAIZ / "experiments" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


I0 = _cargar_experimento("I0_ids_rieth")


@pytest.fixture(scope="module")
def tabla():
    return I0.leer()


def test_el_csv_versionado_es_lo_que_dan_las_reglas(tabla):
    calculada = I0.calcular(config.load_baseline(BASELINE))
    assert tabla.equals(calculada)
    assert I0.main(["--config", str(BASELINE), "--comprobar"]) == 0


def test_replicas_disjuntas_de_normales_no_miradas(tabla):
    cfg = config.load_baseline(BASELINE)
    normales = tabla[tabla["conjunto"] == "normales_train_fault00"]
    r = config.get(cfg, "particion.rieth.replicas_ajuste")
    assert normales["replica"].nunique() == r == 12
    assert (normales.groupby("replica").size() == 34).all()
    assert normales["simulationRun"].is_unique
    assert set(normales["simulationRun"]) <= set(data.unseen_runs("train", 0).tolist())
    assert not np.isin(normales["simulationRun"], data.seen_runs("train", 0)).any()


def test_reparto_dentro_de_cada_replica(tabla):
    """10 para entrenar el autoencoder, 4 para pararlo y 20 para calibrar, por posiciones (decision 9)."""
    normales = tabla[tabla["conjunto"] == "normales_train_fault00"]
    cuenta = normales.groupby(["replica", "rol"]).size().unstack()
    assert (cuenta["entrenar_ae"] == 10).all() and (cuenta["parar_ae"] == 4).all() and (cuenta["calibrar"] == 20).all()
    tr, es, cal = data.split_runs(34, fractions=(10 / 34, 4 / 34, 20 / 34), seed=42)
    for _, g in normales.groupby("replica"):
        assert g["simulationRun"].is_monotonic_increasing and g["orden"].tolist() == list(range(1, 35))
        por_rol = {rol: sorted(g.loc[g["rol"] == rol, "orden"].tolist()) for rol in ("entrenar_ae", "parar_ae", "calibrar")}
        assert por_rol == {"entrenar_ae": sorted(tr.tolist()), "parar_ae": sorted(es.tolist()),
                           "calibrar": sorted(cal.tolist())}


def test_ids_de_prueba_libres_en_los_21_ficheros_y_de_mayor_a_menor(tabla):
    prueba = tabla[tabla["conjunto"] == "prueba"]
    assert len(prueba) == 400 and prueba["orden"].tolist() == list(range(1, 401))
    ids = prueba["simulationRun"].tolist()
    assert ids == sorted(ids, reverse=True)
    libres = data.unseen_runs("test")
    assert set(ids) <= set(libres.tolist())
    assert ids == sorted(libres.tolist(), reverse=True)[:400]
    assert not np.isin(ids, data.seen_runs("test")).any()
