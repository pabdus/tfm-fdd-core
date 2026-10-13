"""Pruebas de la v0.3 en la carga de datos y las particiones (tfmfdd/data.py).

Cada bloque protege una regla del protocolo (docs/reglas_de_medida.md):
- inicio del fallo en el TEP clasico (regla 6; B2 de la revision del 27/09),
- load_rieth: ids int64, ids inexistentes, enteros de numpy, lectura por
  columnas, carpeta por defecto y mensajes claros (DAT-4 a DAT-6),
- split_runs con exclusion de ids y registro de simulaciones ya miradas
  (regla 3, DIS-17),
- columnas de identificacion fuera de los modelos (regla 1),
- inicio efectivo frente a la gemela normal (regla 6, DAT-1).

Casi todas usan datos sinteticos (Parquet pequenos escritos en una carpeta
temporal). Las que necesitan datos reales se saltan si no estan:
- TEP clasico: carpeta de la variable de entorno TFM_CLASSIC_DIR o, si no esta
  definida, data/tep_classic del repositorio.
- Rieth: la misma carpeta que usaria load_rieth (TFM_RIETH_DIR o
  data/tep_rieth). Solo se leen simulaciones ya miradas (ids 1 y 2, regla 3).

Ninguna cifra de resultado esta escrita a mano: los umbrales se calculan con
los datos y se declaran en la docstring de cada prueba.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfmfdd import data

RAIZ = Path(__file__).resolve().parents[1]
CLASICO = Path(os.environ.get("TFM_CLASSIC_DIR", RAIZ / "data" / "tep_classic"))
HAY_CLASICO = all((CLASICO / f).exists() for f in
                  ("d00.dat", "d00_te.dat", "d04.dat", "d06.dat", "d07.dat", "d06_te.dat"))


def _pyarrow_parquet_carga():
    try:
        import pyarrow.parquet  # noqa: F401
    except Exception:
        return False
    return True


def _hay_rieth(*ficheros):
    if not _pyarrow_parquet_carga():
        return False
    carpeta = data._carpeta_rieth(None)[0]
    return all((carpeta / f).exists() for f in ficheros)


HAY_RIETH = _hay_rieth("test_fault00.parquet", "test_fault06.parquet", "test_fault18.parquet")
SIN_CLASICO = "faltan los .dat del TEP clasico (define TFM_CLASSIC_DIR)"
SIN_RIETH = "faltan los Parquet de Rieth o pyarrow no puede leer Parquet (define TFM_RIETH_DIR)"


# ------------------------------------------------------------- utilidades

def _parquet_sintetico(carpeta, fallo=0, split="test", sims=(1, 2, 3), n=None,
                       semilla=0, fault_number=None, quitar_ultima_de=None):
    """Escribe un Parquet con el esquema de convert_rieth (ids int16, variables float32)."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    n = n or data.N_SAMPLES_RIETH[split]
    sim = np.repeat(np.asarray(sims), n)
    muestra = np.tile(np.arange(1, n + 1), len(sims))
    if quitar_ultima_de is not None:                        # simula una simulacion truncada
        fuera = (sim == quitar_ultima_de) & (muestra == n)
        sim, muestra = sim[~fuera], muestra[~fuera]
    X = np.random.default_rng(semilla).normal(size=(sim.size, data.N_VARS)).astype(np.float32)
    fn = fallo if fault_number is None else fault_number
    cols = {"faultNumber": pa.array(np.full(sim.size, fn, dtype=np.int16)),
            "simulationRun": pa.array(sim.astype(np.int16)),
            "sample": pa.array(muestra.astype(np.int16))}
    for j, c in enumerate(data.VAR_NAMES):
        cols[c] = pa.array(X[:, j])
    ruta = Path(carpeta) / f"{split}_fault{fallo:02d}.parquet"
    pq.write_table(pa.table(cols), ruta)
    return ruta


def _tabla(fallo, sims, n, semilla=0):
    """DataFrame con el contrato del modulo y valores aleatorios en float32."""
    sim = np.repeat(np.asarray(sims), n)
    df = pd.DataFrame(np.random.default_rng(semilla).normal(size=(sim.size, data.N_VARS))
                      .astype(np.float32), columns=data.VAR_NAMES)
    df.insert(0, "sample", np.tile(np.arange(1, n + 1), len(sims)).astype(np.int64))
    df.insert(0, "simulationRun", sim.astype(np.int64))
    df.insert(0, "faultNumber", np.int64(fallo))
    return df


def _split_runs_v02(n_runs, fractions=(0.6, 0.2, 0.2), seed=42):
    """El algoritmo de split_runs de la v0.2, copiado tal cual, como referencia."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, n_runs + 1))
    n_train = int(round(fractions[0] * n_runs))
    n_val = int(round(fractions[1] * n_runs))
    return (np.sort(ids[:n_train]), np.sort(ids[n_train:n_train + n_val]),
            np.sort(ids[n_train + n_val:]))


# ------------------------------------------- inicio del fallo (clasico)

def test_fault_onset_por_fuente_y_split():
    assert data.fault_onset("train", "classic") is None
    assert data.fault_onset("train", "rieth") == data.FAULT_ONSET_TRAIN_RIETH == 20
    assert data.fault_onset("test", "classic") == data.fault_onset("test", "rieth") == 160


@pytest.mark.parametrize("split, source",
                         [("Train", "classic"), ("train", "Rieth"), ("prueba", "rieth")])
def test_fault_onset_rechaza_argumentos_mal_escritos(split, source):
    """Antes, fault_onset("train", "Rieth") devolvia None sin avisar (activo desde la fila 0)."""
    with pytest.raises(ValueError):
        data.fault_onset(split, source)


def test_constante_retirada_falla_y_dice_que_usar():
    with pytest.warns(FutureWarning, match="fault_onset"):
        with pytest.raises(AttributeError, match=r"fault_onset\('train', 'classic'\)"):
            data.FAULT_ONSET_TRAIN_CLASSIC


def test_constante_retirada_con_from_import_tambien_avisa():
    """'from ... import' convierte el AttributeError en un ImportError generico:
    el aviso lleva el mensaje."""
    with pytest.warns(FutureWarning, match="fault_onset"):
        with pytest.raises(ImportError):
            from tfmfdd.data import FAULT_ONSET_TRAIN_CLASSIC  # noqa: F401


@pytest.fixture(scope="module")
def clasico_normal():
    """Media y desviacion de d00.dat y envolvente de normalidad de d00_te.dat.

    Envolvente: el maximo, en las 960 filas de d00_te.dat, de max|z| sobre las 52
    variables, con z calculado con la media y la desviacion (ddof = 1) de d00.dat.
    Es la fila normal de prueba mas alejada de la normal de entrenamiento. Mismo
    criterio que el informe 11 de la revision del 03/10 (A.2).
    """
    with pytest.warns(UserWarning, match="transpuesta"):
        normal = data.values(data.load_classic(0, "train", root=CLASICO))
    mu, sd = normal.mean(axis=0), normal.std(axis=0, ddof=1)

    def zmax(X):
        return np.abs((X - mu) / sd).max(axis=1)

    envolvente = zmax(data.values(data.load_classic(0, "test", root=CLASICO))).max()
    return zmax, envolvente


@pytest.mark.skipif(not HAY_CLASICO, reason=SIN_CLASICO)
@pytest.mark.parametrize("fallo", [4, 6, 7])
def test_clasico_fallo_activo_desde_la_fila_0(clasico_normal, fallo):
    """B2 con datos: las 20 primeras filas de dXX.dat, que la constante retirada daba por
    normales, estan todas mas lejos de la normal que cualquier fila de d00_te.dat.

    Fallos 4, 6 y 7: los que cita el informe 11 (A.2) porque su efecto es inmediato.
    Umbral declarado: la envolvente de la fixture clasico_normal.
    """
    zmax, envolvente = clasico_normal
    X = data.values(data.load_classic(fallo, "train", root=CLASICO))
    assert (zmax(X[:20]) > envolvente).all()


@pytest.mark.skipif(not HAY_CLASICO, reason=SIN_CLASICO)
def test_clasico_prueba_fallo_justo_tras_la_160(clasico_normal):
    """En d06_te.dat las filas 0-159 estan dentro de la envolvente y la 160 (muestra 161)
    fuera: el inicio de prueba es FAULT_ONSET_TEST y el de entrenamiento no existe."""
    zmax, envolvente = clasico_normal
    z = zmax(data.values(data.load_classic(6, "test", root=CLASICO)))
    assert (z[:data.FAULT_ONSET_TEST] <= envolvente).all()
    assert z[data.FAULT_ONSET_TEST] > envolvente


# ---------------------------------------------------------- load_rieth

def test_load_rieth_ids_int64_y_variables_float32(tmp_path):
    """DAT-5: en int16, simulationRun * 1000 + sample se desborda sin avisar."""
    _parquet_sintetico(tmp_path, sims=(1, 66))
    df = data.load_rieth(0, "test", runs=[1, 66], root=tmp_path)
    assert list(df.columns) == data.COLUMNS
    assert all(df[c].dtype == np.int64 for c in data.ID_COLUMNS)
    assert all(df[c].dtype == np.float32 for c in data.VAR_NAMES)
    clave = df["simulationRun"] * 1000 + df["sample"]
    assert clave.nunique() == len(df)


def test_load_rieth_entero_son_las_primeras(tmp_path):
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    df = data.load_rieth(0, "test", runs=2, root=tmp_path)
    assert sorted(df["simulationRun"].unique()) == [1, 2]
    assert len(df) == 2 * data.N_SAMPLES_RIETH["test"]


def test_load_rieth_entero_de_numpy_es_error_que_dice_que_pasar(tmp_path):
    """DAT-6: un np.int64 suelto fallaba con un error criptico. Desde la correccion V-04
    (12/10) es un TypeError que dice que pasar: casi siempre es un id, y leido como
    "las n primeras" cargaria simulaciones no miradas. Mas casos en
    tests/test_v03_correcciones_datos.py."""
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    with pytest.raises(TypeError, match=r"runs=\[2\]"):
        data.load_rieth(0, "test", runs=np.int64(2), root=tmp_path)


def test_load_rieth_acepta_ids_de_split_runs(tmp_path):
    _parquet_sintetico(tmp_path, sims=(1, 2, 3, 4, 5))
    ids = np.array([5, 2], dtype=np.int64)
    for runs in (ids, list(ids), range(2, 4)):
        df = data.load_rieth(0, "test", runs=runs, root=tmp_path)
        assert set(df["simulationRun"]) == {int(i) for i in runs}


def test_load_rieth_ids_inexistentes_es_error(tmp_path):
    """DAT-6: runs=[1, 2, 999] devolvia 2 simulaciones sin avisar."""
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    with pytest.raises(ValueError, match="999"):
        data.load_rieth(0, "test", runs=[1, 2, 999], root=tmp_path)


def test_load_rieth_ids_repetidos_avisa_y_carga_una_vez(tmp_path):
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    with pytest.warns(UserWarning, match="repetidos"):
        df = data.load_rieth(0, "test", runs=[2, 2, 3], root=tmp_path)
    assert len(df) == 2 * data.N_SAMPLES_RIETH["test"]


@pytest.mark.parametrize("runs, error", [(0, ValueError), (4, ValueError), ([], ValueError),
                                         ([1.0, 2.0], TypeError), (True, TypeError), ("12", TypeError)])
def test_load_rieth_runs_invalido_es_error(tmp_path, runs, error):
    """Antes, runs mayor que las disponibles devolvia todas sin avisar."""
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    with pytest.raises(error):
        data.load_rieth(0, "test", runs=runs, root=tmp_path)


@pytest.mark.parametrize("split", ["Train", "TEST", "prueba"])
def test_load_rieth_split_mal_escrito_es_error(tmp_path, split):
    """DAT-6: en Windows 'Train' cargaba el fichero; en Linux daba un error enganoso."""
    with pytest.raises(ValueError, match="split"):
        data.load_rieth(0, split, runs=1, root=tmp_path)


def test_load_rieth_carpeta_desde_variable_de_entorno(tmp_path, monkeypatch):
    """Sin root, se usa TFM_RIETH_DIR, la variable que ya leen los cuadernos B00 y C00."""
    _parquet_sintetico(tmp_path, sims=(1, 2))
    monkeypatch.setenv(data.RIETH_DIR_ENV, str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)               # lejos de cualquier data/tep_rieth
    assert len(data.load_rieth(0, "test", runs=[1])) == data.N_SAMPLES_RIETH["test"]


def test_load_rieth_sin_variable_usa_data_tep_rieth(tmp_path, monkeypatch):
    carpeta = tmp_path / "data" / "tep_rieth"
    carpeta.mkdir(parents=True)
    _parquet_sintetico(carpeta, sims=(1, 2))
    monkeypatch.delenv(data.RIETH_DIR_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert len(data.load_rieth(0, "test", runs=[2])) == data.N_SAMPLES_RIETH["test"]


def test_load_rieth_root_explicito_manda_sobre_la_variable(tmp_path, monkeypatch):
    buena, otra = tmp_path / "buena", tmp_path / "otra"
    buena.mkdir()
    otra.mkdir()
    _parquet_sintetico(buena, sims=(1, 2))
    monkeypatch.setenv(data.RIETH_DIR_ENV, str(otra))   # aqui no hay ficheros
    assert len(data.load_rieth(0, "test", runs=[1], root=buena)) == data.N_SAMPLES_RIETH["test"]


def test_load_rieth_sin_fichero_explica_que_hacer(tmp_path, monkeypatch):
    monkeypatch.setenv(data.RIETH_DIR_ENV, str(tmp_path / "no_existe"))
    with pytest.raises(FileNotFoundError, match=data.RIETH_DIR_ENV):
        data.load_rieth(0, "test", runs=1)


def test_load_rieth_pyarrow_bloqueado_lo_explica(tmp_path, monkeypatch):
    """El Control de aplicaciones de Windows puede bloquear la DLL de Parquet: el error
    tiene que decirlo y remitir al LEEME, no dejar un 'DLL load failed' suelto."""
    (tmp_path / "test_fault00.parquet").write_bytes(b"")      # existe; no hace falta leerlo
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)  # import pyarrow.parquet falla
    with pytest.raises(ImportError, match="Control de aplicaciones") as info:
        data.load_rieth(0, "test", runs=1, root=tmp_path)
    assert "LEEME" in str(info.value)


def test_load_rieth_fichero_truncado_es_error(tmp_path):
    _parquet_sintetico(tmp_path, sims=(1, 2), quitar_ultima_de=2)
    with pytest.raises(ValueError, match="truncado"):
        data.load_rieth(0, "test", runs=[1, 2], root=tmp_path)


def test_load_rieth_fallo_distinto_en_el_fichero_es_error(tmp_path):
    _parquet_sintetico(tmp_path, fallo=3, sims=(1, 2), fault_number=4)
    with pytest.raises(ValueError, match="faultNumber"):
        data.load_rieth(3, "test", runs=[1], root=tmp_path)


def test_load_rieth_lee_una_columna_cada_vez(tmp_path, monkeypatch):
    """DAT-4: leer el fichero entero y filtrar despues pedia cientos de MB por llamada.
    La lectura va columna a columna; si alguien vuelve a leerlo de golpe, esta prueba falla."""
    pq = pytest.importorskip("pyarrow.parquet")
    _parquet_sintetico(tmp_path, sims=(1, 2, 3))
    leidas = []
    original = pq.ParquetFile.read

    def espia(self, columns=None, *args, **kwargs):
        leidas.append(columns)
        return original(self, columns, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read", espia)
    monkeypatch.setattr(pd, "read_parquet",
                        lambda *a, **k: pytest.fail("no debe leer el fichero entero"))
    df = data.load_rieth(0, "test", runs=[2], root=tmp_path)
    assert set(df["simulationRun"]) == {2}
    assert leidas and all(c is not None and len(c) == 1 for c in leidas)


@pytest.mark.skipif(not HAY_RIETH, reason=SIN_RIETH)
def test_load_rieth_real_como_dice_el_leeme():
    """El ejemplo del LEEME de los datos: dos simulaciones de test_fault00 (ids 1 y 2, ya miradas)."""
    df = data.load_rieth(0, "test", runs=2)
    assert df.shape == (2 * data.N_SAMPLES_RIETH["test"], len(data.COLUMNS))
    assert sorted(df["simulationRun"].unique()) == [1, 2]
    assert df["simulationRun"].dtype == np.int64


# ---------------------------------------------------------- split_runs

@pytest.mark.parametrize("n, seed", [(500, 42), (20, 42), (100, 1), (7, 3)])
def test_split_runs_sin_exclude_igual_que_la_v02(n, seed):
    """Los ids de B00 y C00 salen de aqui: sin exclude, el reparto no puede cambiar."""
    for nuevo, viejo in zip(data.split_runs(n, seed=seed), _split_runs_v02(n, seed=seed)):
        assert np.array_equal(nuevo, viejo)


def test_split_runs_exclude_vacio_no_cambia_nada():
    for a, b in zip(data.split_runs(500, seed=42, exclude=[]), data.split_runs(500, seed=42)):
        assert np.array_equal(a, b)


def test_split_runs_exclude_saca_esos_ids():
    fuera = np.arange(1, 101)
    tr, va, te = data.split_runs(500, seed=42, exclude=fuera)
    juntos = np.concatenate([tr, va, te])
    assert not np.isin(juntos, fuera).any()
    assert np.array_equal(np.sort(juntos), np.arange(101, 501))
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))


def test_split_runs_exclude_fuera_de_rango_es_error():
    """Un id fuera de 1..n_runs suele ser una posicion confundida con un id."""
    with pytest.raises(ValueError, match="fuera de 1..20"):
        data.split_runs(20, seed=42, exclude=[250])


def test_split_runs_acepta_enteros_de_numpy():
    tr, va, te = data.split_runs(np.int64(50), seed=7, exclude=np.array([1, 2], dtype=np.int64))
    assert len(tr) + len(va) + len(te) == 48


# ------------------------------------------- registro de simulaciones miradas

@pytest.fixture
def sin_aviso_de_pendientes(monkeypatch):
    """Las pruebas del contenido del registro no comprueban el aviso de pendientes."""
    monkeypatch.setattr(data, "RIETH_SEEN_PENDING", ())


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("fallo", [0, 3, 20])
def test_registro_ids_1_a_20_mirados_en_todos(sin_aviso_de_pendientes, split, fallo):
    assert np.isin(np.arange(1, 21), data.seen_runs(split, fallo)).all()


def test_registro_ids_distintos_del_b00_solo_en_entrenamiento(sin_aviso_de_pendientes):
    """Regla 3: en train_faultKK, de 20*KK + 1 a 20*(KK + 1); no en la prueba."""
    bloque = np.arange(101, 121)                      # KK = 5
    assert np.isin(bloque, data.seen_runs("train", 5)).all()
    assert not np.isin(bloque, data.seen_runs("test", 5)).any()


def test_registro_250_y_500_solo_en_los_seis_ficheros(sin_aviso_de_pendientes):
    """Traspaso v12, 9.7: ficheros 00, 01, 03, 06, 09 y 15, en entrenamiento y en prueba."""
    for split in ("train", "test"):
        for fallo in (0, 1, 3, 6, 9, 15):
            assert np.isin([250, 500], data.seen_runs(split, fallo)).all()
        assert not np.isin([250, 500], data.seen_runs(split, 2)).any()


def test_registro_c00_en_la_prueba(sin_aviso_de_pendientes):
    ids_c00 = [23, 24, 32, 35, 36, 37, 42, 43, 44, 50, 58, 64, 66]
    assert np.isin(ids_c00, data.seen_runs("test", 12)).all()


def test_seen_y_unseen_se_complementan(sin_aviso_de_pendientes):
    vistos, libres = data.seen_runs("test", 9), data.unseen_runs("test", 9)
    assert not np.isin(vistos, libres).any()
    todos = np.sort(np.concatenate([vistos, libres]))
    assert np.array_equal(todos, np.arange(1, data.N_RUNS_RIETH + 1))


def test_unseen_de_varios_ficheros_es_la_interseccion(sin_aviso_de_pendientes):
    """Un id comun a los tres bloques tiene que estar libre en todos los ficheros que se usen."""
    juntos = data.unseen_runs("train", [0, 5])
    por_separado = np.intersect1d(data.unseen_runs("train", 0), data.unseen_runs("train", 5))
    assert np.array_equal(juntos, por_separado)
    todos = np.intersect1d(data.unseen_runs("train"), data.unseen_runs("test"))
    assert np.array_equal(data.unseen_runs(), todos)


def test_registro_pendiente_avisa():
    """Regla 3, punto 5: los prefijos de los prototipos del C00 no quedaron registrados.
    Hasta que se confirmen, quien pida ids libres tiene que saberlo."""
    assert data.RIETH_SEEN_PENDING
    with pytest.warns(UserWarning, match="sin confirmar"):
        data.unseen_runs("test")
    with pytest.warns(UserWarning, match="sin confirmar"):
        data.seen_runs("test")


@pytest.mark.parametrize("split, faults", [("Test", None), (None, 21), (None, [0, 21]), (None, "3")])
def test_registro_rechaza_ficheros_que_no_existen(split, faults):
    with pytest.raises(ValueError):
        data.seen_runs(split, faults)


def test_split_runs_con_el_registro_no_reparte_mirados(sin_aviso_de_pendientes):
    vistos = data.seen_runs("test")
    tr, va, te = data.split_runs(data.N_RUNS_RIETH, seed=42, exclude=vistos)
    assert not np.isin(np.concatenate([tr, va, te]), vistos).any()


# ------------------------------------------- columnas de identificacion

@pytest.mark.parametrize("col", data.ID_COLUMNS)
def test_values_rechaza_columnas_de_identificacion(col):
    """Regla 1: 'sample' delata el inicio fijo del fallo; ningun modelo debe recibirla."""
    df = _tabla(0, sims=(1,), n=10)
    with pytest.raises(ValueError, match=col):
        data.values(df, columns=data.VAR_NAMES + [col])


def test_values_sigue_aceptando_subconjuntos_de_variables():
    df = _tabla(0, sims=(1,), n=10)
    assert data.values(df, columns=data.YIN_VARS).shape == (10, 33)


# ------------------------------------------------------ inicio efectivo

ONSET_SINT = 10


@pytest.fixture
def gemelas():
    """Normal y fallo sinteticos (30 muestras, inicio tras la 10) con tres casos:
    la simulacion 1 se separa en la 11 (enseguida), la 2 en la 15 y la 3 nunca."""
    normal = _tabla(0, sims=(1, 2, 3), n=30)
    fallo = normal.copy()
    fallo["faultNumber"] = np.int64(5)
    for sim, desde in ((1, 11), (2, 15)):
        m = (fallo["simulationRun"] == sim) & (fallo["sample"] >= desde)
        fallo.loc[m, "xmeas_1"] = fallo.loc[m, "xmeas_1"] + np.float32(1.0)
    return fallo, normal


def test_effective_onset_sintetico(gemelas):
    fallo, normal = gemelas
    ef = data.effective_onset(fallo, normal, ONSET_SINT).set_index("simulationRun")
    assert ef.loc[1, "effective_onset"] == 10 and ef.loc[1, "copied_after_onset"] == 0
    assert ef.loc[2, "effective_onset"] == 14 and ef.loc[2, "copied_after_onset"] == 4
    assert ef.loc[3, "effective_onset"] == 30 and not ef.loc[3, "separates"]
    assert ef["separates"].tolist() == [True, True, False]
    assert (ef["faultNumber"] == 5).all() and (ef["onset"] == ONSET_SINT).all()


def test_effective_onset_igual_sin_las_filas_previas(gemelas):
    """B00 y C00 descartan sample <= onset antes de medir: el resultado no cambia."""
    fallo, normal = gemelas
    completo = data.effective_onset(fallo, normal, ONSET_SINT)
    recortado = data.effective_onset(fallo[fallo["sample"] > ONSET_SINT], normal, ONSET_SINT)
    pd.testing.assert_frame_equal(completo, recortado)


def test_effective_onset_permite_descartar_el_tramo_copiado(gemelas):
    """La receta de la docstring deja en cada simulacion solo filas distintas de su gemela."""
    fallo, normal = gemelas
    ef = data.effective_onset(fallo, normal, ONSET_SINT)
    fallo = fallo.merge(ef[["faultNumber", "simulationRun", "effective_onset"]],
                        on=["faultNumber", "simulationRun"])
    fallo = fallo[fallo["sample"] > fallo["effective_onset"]]
    primeras = fallo.groupby("simulationRun")["sample"].min()
    assert primeras.to_dict() == {1: 11, 2: 15}            # la 3 no se separa: se queda sin filas


def test_effective_onset_si_no_son_gemelas_es_error(gemelas):
    fallo, normal = gemelas
    otra = _tabla(0, sims=(1, 2, 3), n=30, semilla=99)       # otra semilla: no es su gemela
    with pytest.raises(ValueError, match="no son"):
        data.effective_onset(fallo, otra, ONSET_SINT)


def test_effective_onset_sin_la_gemela_es_error(gemelas):
    fallo, normal = gemelas
    with pytest.raises(ValueError, match="clase 0 del mismo split"):
        data.effective_onset(fallo, normal[normal["simulationRun"] != 2], ONSET_SINT)


def test_effective_onset_pide_la_normal_como_referencia(gemelas):
    fallo, _ = gemelas
    with pytest.raises(ValueError, match="clase 0"):
        data.effective_onset(fallo, fallo, ONSET_SINT)


@pytest.mark.skipif(not HAY_RIETH, reason=SIN_RIETH)
def test_effective_onset_real():
    """Con datos de Rieth (prueba, ids 1 y 2, ya mirados). Informe 11 (B.5, hecho 2): en el
    fallo 6 la trayectoria se separa de su gemela justo tras el inicio en todas las
    simulaciones; en el 18, ninguna se separa en la primera muestra con fallo."""
    onset = data.fault_onset("test", "rieth")
    normal = data.load_rieth(0, "test", runs=[1, 2])
    claro = data.effective_onset(data.load_rieth(6, "test", runs=[1, 2]), normal, onset)
    lento = data.effective_onset(data.load_rieth(18, "test", runs=[1, 2]), normal, onset)
    assert (claro["effective_onset"] == onset).all()
    assert (lento["effective_onset"] > onset).all()
