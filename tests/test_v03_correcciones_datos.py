"""Pruebas de las correcciones del 12/10 en tfmfdd/data.py (verificacion de codigo).

- V-04: un entero de numpy suelto en `runs` de load_rieth es un TypeError que
  dice que pasar. Antes valia "las n primeras" con un aviso, y con un id
  confirmatorio (por ejemplo te[0]) cargaba cientos de simulaciones no miradas.
  Solo Parquet sinteticos: nunca ids nuevos de los datos reales (regla 3).
- V-07: effective_onset cuenta por el valor de `sample` y falla si faltan las
  primeras muestras posteriores al inicio (antes daba un numero falso).
- V-08: en split_runs, una parte con fraccion 0 queda vacia. Con las tres
  fracciones positivas, el reparto es exactamente el de siempre: se compara con
  el algoritmo antiguo, copiado tal cual, en una rejilla de casos. De ese
  reparto dependen las listas del B00 y del C00.

La prueba con datos reales (V-07) se salta si no estan; solo lee los ids 1 y 2
de test_fault00 y test_fault01, ya mirados.
"""

import numpy as np
import pandas as pd
import pytest

from tfmfdd import data


def _hay_rieth():
    try:
        import pyarrow.parquet  # noqa: F401
    except Exception:
        return False
    carpeta = data._carpeta_rieth(None)[0]
    return all((carpeta / f).exists() for f in ("test_fault00.parquet", "test_fault01.parquet"))


HAY_RIETH = _hay_rieth()


def _parquet_sintetico(carpeta, sims, split="test", fallo=0, semilla=0):
    """Parquet con el esquema de convert_rieth (ids int16, variables float32)."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    n = data.N_SAMPLES_RIETH[split]
    sims = np.asarray(list(sims))
    sim = np.repeat(sims, n)
    muestra = np.tile(np.arange(1, n + 1), sims.size)
    X = np.random.default_rng(semilla).normal(size=(sim.size, data.N_VARS)).astype(np.float32)
    cols = {"faultNumber": pa.array(np.full(sim.size, fallo, dtype=np.int16)),
            "simulationRun": pa.array(sim.astype(np.int16)),
            "sample": pa.array(muestra.astype(np.int16))}
    for j, c in enumerate(data.VAR_NAMES):
        cols[c] = pa.array(X[:, j])
    pq.write_table(pa.table(cols), carpeta / f"{split}_fault{fallo:02d}.parquet")


def _tabla(fallo, sims, n, semilla=0):
    sim = np.repeat(np.asarray(sims), n)
    df = pd.DataFrame(np.random.default_rng(semilla).normal(size=(sim.size, data.N_VARS))
                      .astype(np.float32), columns=data.VAR_NAMES)
    df.insert(0, "sample", np.tile(np.arange(1, n + 1), len(sims)).astype(np.int64))
    df.insert(0, "simulationRun", sim.astype(np.int64))
    df.insert(0, "faultNumber", np.int64(fallo))
    return df


# ------------------------------------------------------------------ V-04

def test_v04_iterar_los_ids_de_split_runs_es_error(tmp_path):
    """El caso de la verificacion: `for r in te: load_rieth(..., runs=r)` con te de
    split_runs. Cada r es un np.int64; antes cargaba las r primeras simulaciones."""
    _parquet_sintetico(tmp_path, sims=range(1, 31))
    te = data.split_runs(30, seed=42)[2]
    assert isinstance(te[0], np.integer)
    for r in te:
        with pytest.raises(TypeError, match=rf"runs=\[{int(r)}\]"):
            data.load_rieth(0, "test", runs=r, root=tmp_path)


def test_v04_el_mensaje_da_las_dos_salidas(tmp_path):
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    with pytest.raises(TypeError) as info:
        data.load_rieth(0, "test", runs=np.int64(3), root=tmp_path)
    assert "runs=[3]" in str(info.value) and "runs=int(n)" in str(info.value)


@pytest.mark.parametrize("escalar", [np.int64(2), np.int32(2), np.uint8(2), np.array(2)])
def test_v04_cualquier_entero_de_numpy_suelto_es_error(tmp_path, escalar):
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    with pytest.raises(TypeError, match="entero de numpy"):
        data.load_rieth(0, "test", runs=escalar, root=tmp_path)


def test_v04_int_de_python_sigue_siendo_las_primeras(tmp_path):
    """Compatibilidad: runs=2 (como en el LEEME de los datos) sigue cargando los ids 1 y 2."""
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    df = data.load_rieth(0, "test", runs=2, root=tmp_path)
    assert sorted(df["simulationRun"].unique()) == [1, 2]
    df = data.load_rieth(0, "test", runs=int(np.int64(2)), root=tmp_path)
    assert sorted(df["simulationRun"].unique()) == [1, 2]


def test_v04_listas_y_arrays_de_ids_de_numpy_siguen_valiendo(tmp_path):
    _parquet_sintetico(tmp_path, sims=(1, 2, 3, 4))
    for runs in ([np.int64(3)], np.array([4, 2], dtype=np.int64), np.array([1], dtype=np.int16)):
        df = data.load_rieth(0, "test", runs=runs, root=tmp_path)
        assert set(df["simulationRun"]) == {int(i) for i in runs}


# ------------------------------------------------------------------ V-07

ONSET = 10


@pytest.fixture
def gemelas():
    """Normal y fallo sinteticos (30 muestras, inicio tras la 10): la simulacion 1
    se separa en la 11, la 2 en la 15 y la 3 nunca."""
    normal = _tabla(0, sims=(1, 2, 3), n=30)
    fallo = normal.copy()
    fallo["faultNumber"] = np.int64(5)
    for sim, desde in ((1, 11), (2, 15)):
        m = (fallo["simulationRun"] == sim) & (fallo["sample"] >= desde)
        fallo.loc[m, "xmeas_1"] = fallo.loc[m, "xmeas_1"] + np.float32(1.0)
    return fallo, normal


def _quitar(df, sim, muestras):
    return df[~((df["simulationRun"] == sim) & df["sample"].isin(muestras))]


def test_v07_sin_las_primeras_muestras_posteriores_es_error(gemelas):
    """Antes, sin las muestras 11 a 12, la simulacion 1 daba effective_onset 12 en vez de 10."""
    fallo, normal = gemelas
    with pytest.raises(ValueError, match="seguidas"):
        data.effective_onset(fallo[fallo["sample"] > 12], normal, ONSET)


def test_v07_hueco_antes_de_la_primera_distinta_es_error(gemelas):
    fallo, normal = gemelas
    with pytest.raises(ValueError, match=r"\(5, 2\)"):
        data.effective_onset(_quitar(fallo, 2, [13]), normal, ONSET)   # la 2 se separa en la 15


def test_v07_si_no_se_separa_necesita_todas_sus_muestras(gemelas):
    fallo, normal = gemelas
    with pytest.raises(ValueError, match=r"\(5, 3\)"):
        data.effective_onset(_quitar(fallo, 3, [25]), normal, ONSET)


def test_v07_un_hueco_despues_de_la_primera_distinta_no_cambia_nada(gemelas):
    fallo, normal = gemelas
    completo = data.effective_onset(fallo, normal, ONSET)
    con_huecos = _quitar(_quitar(fallo, 1, [20, 21]), 2, [16, 29])
    pd.testing.assert_frame_equal(data.effective_onset(con_huecos, normal, ONSET), completo)


def test_v07_cuenta_por_el_valor_de_sample_y_no_por_el_orden_de_las_filas(gemelas):
    fallo, normal = gemelas
    completo = data.effective_onset(fallo, normal, ONSET)
    barajado = fallo.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(data.effective_onset(barajado, normal, ONSET), completo)
    ef = completo.set_index("simulationRun")["effective_onset"].to_dict()
    assert ef == {1: 10, 2: 14, 3: 30}


@pytest.mark.skipif(not HAY_RIETH, reason="faltan los Parquet de Rieth o pyarrow no lee Parquet")
def test_v07_el_caso_de_la_verificacion_con_datos_reales():
    """Fallo 1 de prueba sin las muestras 161 a 200: antes daba 200 (y 40 copiadas)."""
    onset = data.fault_onset("test", "rieth")
    normal = data.load_rieth(0, "test", runs=[1, 2])
    fallo = data.load_rieth(1, "test", runs=[1, 2])
    assert (data.effective_onset(fallo, normal, onset)["effective_onset"] == onset).all()
    with pytest.raises(ValueError, match="seguidas"):
        data.effective_onset(fallo[fallo["sample"] > 200], normal, onset)


# ------------------------------------------------------------------ V-08

def _reparto_antiguo(candidatos, fractions, seed):
    """El algoritmo de split_runs antes de la correccion, copiado tal cual."""
    ids = np.random.default_rng(seed).permutation(np.asarray(candidatos))
    n = len(candidatos)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    return (np.sort(ids[:n_train]), np.sort(ids[n_train:n_train + n_val]),
            np.sort(ids[n_train + n_val:]))


FRACCIONES_POSITIVAS = [
    (0.6, 0.2, 0.2), (0.7, 0.15, 0.15), (0.5, 0.25, 0.25), (1 / 3, 1 / 3, 1 / 3),
    (0.8, 0.1, 0.1), (0.5, 0.3, 0.2), (0.34, 0.33, 0.33), (0.9, 0.05, 0.05),
    (0.25, 0.25, 0.5), (0.1, 0.1, 0.8), (0.45, 0.45, 0.1),
]
TAMANOS = list(range(1, 61)) + [79, 100, 465, 500]
SEMILLAS = [0, 1, 7, 42]


def test_v08_con_fracciones_positivas_el_reparto_no_cambia():
    casos = 0
    for n in TAMANOS:
        for fr in FRACCIONES_POSITIVAS:
            for s in SEMILLAS:
                nuevo = data.split_runs(n, fr, s)
                viejo = _reparto_antiguo(np.arange(1, n + 1), fr, s)
                assert all(np.array_equal(a, b) for a, b in zip(nuevo, viejo)), (n, fr, s)
                casos += 1
    assert casos == len(TAMANOS) * len(FRACCIONES_POSITIVAS) * len(SEMILLAS)


def test_v08_con_fracciones_positivas_y_exclude_tampoco_cambia():
    excluidos = {
        "primeras 20": np.arange(1, 21),
        "una de cada tres": np.arange(1, 501, 3),
        "1 a 420": np.arange(1, 421),
    }
    for nombre, fuera in excluidos.items():
        candidatos = np.setdiff1d(np.arange(1, 501), fuera)
        for fr in FRACCIONES_POSITIVAS:
            for s in SEMILLAS:
                nuevo = data.split_runs(500, fr, s, exclude=fuera)
                viejo = _reparto_antiguo(candidatos, fr, s)
                assert all(np.array_equal(a, b) for a, b in zip(nuevo, viejo)), (nombre, fr, s)


def test_v08_las_listas_del_b00_y_del_c00_no_cambian():
    """Ids que guardan las salidas de los cuadernos (configuracion, no resultados):
    B00, celda 35 de la base, split_runs(20) con las fracciones y la semilla del yaml;
    C00, results/raw/C00_configuracion.json, prefijos de split_runs(500)."""
    tr, va, te = data.split_runs(20, (0.6, 0.2, 0.2), 42)
    assert tr.tolist() == [1, 4, 6, 7, 8, 10, 11, 13, 15, 16, 17, 20]
    assert va.tolist() == [3, 5, 12, 19] and te.tolist() == [2, 9, 14, 18]
    tr, va, te = data.split_runs(500, (0.6, 0.2, 0.2), 42)
    assert tr[:10].tolist() == [1, 3, 4, 8, 9, 13, 20, 21, 25, 26]
    assert va[:10].tolist() == [5, 6, 10, 12, 14, 19, 22, 29, 33, 38]
    assert te[:20].tolist() == [2, 7, 11, 15, 16, 17, 18, 23, 24, 32,
                                35, 36, 37, 42, 43, 44, 50, 58, 64, 66]


@pytest.mark.parametrize("fr", [(0.5, 0.5, 0.0), (0.7, 0.3, 0.0), (0.25, 0.75, 0.0),
                                (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.5, 0.5),
                                (0.5, 0.0, 0.5), (0.0, 0.0, 1.0)])
def test_v08_una_fraccion_0_deja_su_parte_vacia(fr):
    for n in TAMANOS:
        for s in SEMILLAS:
            partes = data.split_runs(n, fr, s)
            for f, parte in zip(fr, partes):
                if f == 0:
                    assert parte.size == 0, (n, fr, s)
            juntos = np.concatenate(partes)
            assert np.array_equal(np.sort(juntos), np.arange(1, n + 1))


def test_v08_con_prueba_0_solo_pasa_a_validacion_lo_que_el_redondeo_dejaba_en_prueba():
    for n in TAMANOS:
        for fr in [(0.5, 0.5, 0.0), (0.7, 0.3, 0.0), (0.25, 0.75, 0.0)]:
            for s in SEMILLAS:
                nuevo = data.split_runs(n, fr, s)
                viejo = _reparto_antiguo(np.arange(1, n + 1), fr, s)
                assert np.array_equal(nuevo[0], viejo[0])
                assert np.array_equal(nuevo[1], np.sort(np.concatenate([viejo[1], viejo[2]])))
                assert nuevo[2].size == 0


def test_v08_el_ejemplo_de_la_verificacion():
    """n = 1 y n = 5 con (0.5, 0.5, 0.0): antes, una simulacion iba a la parte vacia."""
    assert [p.size for p in data.split_runs(1, (0.5, 0.5, 0.0), 0)] == [0, 1, 0]
    assert [p.size for p in data.split_runs(5, (0.5, 0.5, 0.0), 0)] == [2, 3, 0]


def test_v08_nucleo_comun_con_ids_cualesquiera_igual_que_split_runs_con_exclude():
    """data._repartir es el reparto de split_runs sobre ids que no empiezan en 1: lo
    que necesita ablation.split_by_run para no duplicar el algoritmo."""
    ids = np.array([41, 5, 9, 77, 12, 13, 40])               # desordenados a proposito
    resto = np.setdiff1d(np.arange(1, 78), ids)
    for fr in FRACCIONES_POSITIVAS + [(0.5, 0.5, 0.0)]:
        for s in SEMILLAS:
            a = data._repartir(ids, fr, s)
            b = data.split_runs(77, fr, s, exclude=resto)
            assert all(np.array_equal(x, y) for x, y in zip(a, b)), (fr, s)
