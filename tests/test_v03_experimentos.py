"""Pruebas de la configuracion, la procedencia y los experimentos (fase B de la v0.3).

Cubren:
- tfmfdd.config: orden de busqueda de baseline.yaml y copia empaquetada identica
  a configs/baseline.yaml (REP-4); claves nuevas y decisiones pendientes en null.
- tfmfdd.provenance: el JSON junto a cada CSV (regla 11, H15), leyendo git sin
  escribir en el repositorio.
- experiments/V1_verificacion_yin.py: la regla de orden (se niega sin tolerancia,
  sin fila fechada con la huella o con cambios sin confirmar), la FDR de la
  muestra 161 a la 960, la FAR en la ventana del yaml, los recuentos, las cifras
  de Yin leidas del CSV, el veredicto sin 3, 9 y 15 y --solo-limites sin ficheros
  de prueba. Con ficheros .dat sinteticos en una carpeta temporal: ninguna de
  estas pruebas lee un fichero de prueba real.
- La regresion de los limites con d00.dat frente a los umbrales que dibujan Yin
  et al. (2012), leidos del CSV (se salta si no esta d00.dat).
- A0 y A1 con datos sinteticos: brazos, columnas de configuracion y procedencia.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tfmfdd import config, data, provenance
from tfmfdd.pca import PCAMonitor

RAIZ = Path(__file__).resolve().parents[1]
BASELINE = RAIZ / "configs" / "baseline.yaml"
YIN_YAML = RAIZ / "configs" / "yin2012.yaml"
REFERENCIAS = RAIZ / "configs" / "referencias"
CLASSIC_DIR = Path(os.environ.get("TFM_CLASSIC_DIR", RAIZ / "data" / "tep_classic"))

#: Tolerancia de lectura de la prueba de regresion de los limites: el umbral
#: calculado tiene que caer a menos de UNA vez el residuo maximo del ajuste de
#: las marcas del eje (columna residuo_max_marcas del CSV de umbrales), que es la
#: precision con que se leyo la linea dibujada. Declarada aqui, antes de comparar.
TOLERANCIA_LECTURA = 1.0


def _cargar_experimento(nombre: str):
    """Importa un script de experiments/ como modulo (no son un paquete)."""
    ruta = RAIZ / "experiments" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


V1 = _cargar_experimento("V1_verificacion_yin")


# --------------------------------------------------------- datos sinteticos

def _tep_sintetico(carpeta: Path, semilla: int = 0, solo_d00: bool = False) -> Path:
    """Escribe un TEP clasico falso: d00.dat (500 x 52) y d00_te.dat a d21_te.dat (960 x 52).

    Datos normales correlacionados (5 factores latentes y ruido: con 5 factores, ni el
    PCA con 9 componentes ni el DPCA con 17 y h = 2 dejan un factor en el residuo, y
    Jackson es aplicable, h0 > 0). En los ficheros
    con fallo, salvo 3, 9 y 15, a partir del indice 160 (la muestra 161) se suma
    un escalon a unas variables de las 33 de Yin.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    cargas = rng.normal(size=(52, 5))
    media = rng.normal(size=52) * 5

    def normales(n):
        return rng.normal(size=(n, 5)) @ cargas.T + 0.5 * rng.normal(size=(n, 52)) + media

    np.savetxt(carpeta / "d00.dat", normales(500), fmt="%.6f")
    if solo_d00:
        return carpeta
    columnas_yin = [data.VAR_NAMES.index(v) for v in data.YIN_VARS]
    for f in range(22):
        X = normales(960)
        if f not in (0, 3, 9, 15):
            X[160:, columnas_yin[f % 33]] += 8.0
            X[160:, columnas_yin[(5 * f) % 33]] += 8.0
        np.savetxt(carpeta / f"d{f:02d}_te.dat", X, fmt="%.6f")
    return carpeta


@pytest.fixture(scope="module")
def tep_sintetico(tmp_path_factory) -> Path:
    return _tep_sintetico(tmp_path_factory.mktemp("tep_sintetico"))


def _yaml_fijado(**cambios) -> dict:
    """El yaml real de la verificacion con las decisiones pendientes rellenadas (solo para probar)."""
    cfg = yaml.safe_load(YIN_YAML.read_text(encoding="utf-8"))
    cfg["referencia"]["transcripcion_verificada"] = "2026-01-01"
    cfg["supuestos"]["S9_far_ventana"] = [160, None]
    cfg["tolerancia"].update(comparacion="conteo", fdr_umbral_por_fallo_pp=2, fdr_min_fallos_dentro=16,
                             fdr_max_pp=5, fdr_media_max_pp="no_aplica", far_max_pp=2, dpca="descriptivo")
    cfg["secundaria"]["incluir"] = False
    cfg["variantes"] = [v for v in cfg["variantes"] if v["nombre"] in ("S5_retardar_y_escalar", "S9_far_1_960")]
    for clave, valor in cambios.items():
        seccion, nombre = clave.split("__")
        cfg[seccion][nombre] = valor
    return cfg


def _repo_falso(carpeta: Path, cfg: dict, fila: str | None = "huella") -> dict:
    """Arbol configs/ + docs/ en una carpeta temporal, con la fila del registro."""
    (carpeta / "configs" / "referencias").mkdir(parents=True, exist_ok=True)
    (carpeta / "docs").mkdir(parents=True, exist_ok=True)
    for f in REFERENCIAS.glob("*.csv"):
        shutil.copy(f, carpeta / "configs" / "referencias" / f.name)
    ruta_yaml = carpeta / "configs" / "yin2012.yaml"
    ruta_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    actual = V1.huella(cfg, V1.rutas_referencia(cfg, ruta_yaml.parent))
    lineas = ["# Registro de decisiones", "", "| Fecha | Decision | Quien | Motivo |", "|---|---|---|---|"]
    if fila == "huella":
        lineas.append(f"| 2026-01-01 | Verificacion frente a Yin [V1-yin huella={actual}] | Pablo | prueba |")
    elif fila is not None:
        lineas.append(fila)
    decisiones = carpeta / "docs" / "decisiones.md"
    decisiones.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return {"yaml": ruta_yaml, "decisiones": decisiones, "huella": actual}


def _git_limpio(monkeypatch):
    monkeypatch.setattr(V1, "estado_git", lambda rutas: {"repository": True, "commit": "abc123", "status": []})


def _espiar_cargas(monkeypatch) -> list:
    """Registra cada (fallo, split) que se pide a data.load_classic."""
    llamadas = []
    original = data.load_classic

    def espia(fault, split="test", root="data/tep_classic"):
        llamadas.append((fault, split))
        return original(fault, split, root=root)

    monkeypatch.setattr(data, "load_classic", espia)
    return llamadas


# ================================================================ config

def test_config_copia_empaquetada_identica_a_configs_baseline():
    """REP-4: la copia que viaja con el paquete es la misma que configs/baseline.yaml."""
    empaquetada = config.packaged_baseline()
    assert empaquetada.is_file(), "falta tfmfdd/configs/baseline.yaml"
    assert empaquetada.read_text(encoding="utf-8") == BASELINE.read_text(encoding="utf-8")


def test_config_ruta_explicita_manda(tmp_path, monkeypatch):
    otra = tmp_path / "otra.yaml"
    otra.write_text("pca:\n  variance: 0.5\n", encoding="utf-8")
    monkeypatch.setenv(config.CONFIG_ENV, str(BASELINE))
    cfg, info = config.load_baseline(otra, with_info=True)
    assert cfg["pca"]["variance"] == 0.5 and info["source"] == "explicit"


def test_config_ruta_explicita_inexistente_falla_sin_buscar_otra(tmp_path):
    with pytest.raises(FileNotFoundError, match="explicita"):
        config.load_baseline(tmp_path / "no_existe.yaml")


def test_config_variable_de_entorno_antes_que_el_directorio(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "baseline.yaml").write_text("origen: cwd\n", encoding="utf-8")
    entorno = tmp_path / "entorno.yaml"
    entorno.write_text("origen: env\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(config.CONFIG_ENV, str(entorno))
    assert config.load_baseline()["origen"] == "env"
    assert config.find_baseline().source == "env"


def test_config_variable_de_entorno_inexistente_falla(tmp_path, monkeypatch):
    monkeypatch.setenv(config.CONFIG_ENV, str(tmp_path / "no_existe.yaml"))
    with pytest.raises(FileNotFoundError, match=config.CONFIG_ENV):
        config.load_baseline()


def test_config_directorio_de_trabajo_antes_que_el_paquete(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "baseline.yaml").write_text("origen: cwd\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    # Desde la correccion V-15 avisa: esta copia no es la del paquete instalado.
    with pytest.warns(UserWarning, match="no es el de la version instalada"):
        assert config.load_baseline()["origen"] == "cwd"
    assert config.find_baseline().source == "cwd"


def test_config_sin_nada_usa_la_copia_del_paquete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    cfg, info = config.load_baseline(with_info=True)
    assert info["source"] == "package"
    assert cfg == yaml.safe_load(BASELINE.read_text(encoding="utf-8"))


def test_config_huella_no_depende_del_fin_de_linea(tmp_path):
    texto = BASELINE.read_text(encoding="utf-8")
    lf, crlf = tmp_path / "lf.yaml", tmp_path / "crlf.yaml"
    lf.write_bytes(texto.encode("utf-8"))
    crlf.write_bytes(texto.replace("\n", "\r\n").encode("utf-8"))
    assert config.load_baseline(lf, with_info=True)[1]["sha256"] == config.load_baseline(crlf, with_info=True)[1]["sha256"]


def test_config_get_no_tiene_valor_por_defecto():
    cfg = config.load_baseline(BASELINE)
    with pytest.raises(KeyError, match="pca.no_existe"):
        config.get(cfg, "pca.no_existe")


def test_config_require_dice_que_la_clave_esta_pendiente():
    cfg = config.load_baseline(BASELINE)
    with pytest.raises(ValueError, match="pendiente"):
        config.require(cfg, "contrastes.delta.fdr")


def test_baseline_alpha_contrastes_separado_del_nivel_de_confianza():
    """Parrafo 358: alfa = 0,05 para los contrastes, distinto del 0,99 de los limites."""
    cfg = config.load_baseline(BASELINE)
    assert config.get(cfg, "contrastes.alpha_contrastes") == 0.05
    assert config.get(cfg, "protocolo.alpha") == 0.99


def test_baseline_valores_por_defecto_nuevos_coinciden_con_pcamonitor():
    """Las claves nuevas de pca son los valores por defecto de PCAMonitor (M9)."""
    cfg = config.load_baseline(BASELINE)
    m = PCAMonitor()
    assert m.t2_method == cfg["pca"]["t2_method"]
    assert m.quantile_method == cfg["pca"]["quantile_method"]
    assert cfg["pca"]["calibration"] == "holdout"


def test_baseline_decisiones_pendientes_estan_en_null():
    """Las decisiones de Pablo que piden los informes de la fase A existen y estan en null."""
    cfg = config.load_baseline(BASELINE)
    pendientes = set(config.pending(cfg))
    for clave in ("contrastes.delta.fdr", "contrastes.delta.far", "contrastes.delta.retardo_muestras",
                  "contrastes.delta.macro_f1", "deteccion.ventana_far_normal",
                  "deteccion.falsas_alarmas_por_hora", "diagnostico.incluir_normal",
                  "deteccion.suelo.margen", "hipotesis.fallos_control", "robustez.niveles_deteccion",
                  "robustez.niveles_diagnostico", "coste.maquina_referencia", "datos.ids_confirmatorios",
                  "compromiso.deteccion", "particion.rieth.replicas_ajuste"):
        assert clave in pendientes, clave


def test_baseline_no_cambia_ningun_valor_anterior():
    """Las 17 claves de la v0.2 conservan su valor (cambiar el yaml es un evento de coordinacion)."""
    cfg = config.load_baseline(BASELINE)
    anteriores = {"protocolo.alpha": 0.99, "protocolo.alpha_sensibilidad": 0.95, "protocolo.k_alarmas": 3,
                  "protocolo.sample_minutes": 3.0, "protocolo.semilla": 42, "particion.fraccion_ajuste": 0.6,
                  "particion.fraccion_validacion": 0.2, "particion.fraccion_prueba": 0.2,
                  "particion.calibracion_limites": 0.3, "datos.fuente": "classic", "datos.runs_desarrollo": 20,
                  "datos.runs_finales": 100, "datos.fault_onset_test": 160, "pca.variance": 0.85,
                  "pca.spe_method": "box", "pca.lags": 0, "fallos_dificiles": [3, 9, 15]}
    for clave, valor in anteriores.items():
        assert config.get(cfg, clave) == valor, clave


# ============================================================ procedencia

def test_procedencia_json_junto_al_csv(tmp_path):
    csv = tmp_path / "resultado.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(csv, index=False)
    ruta = provenance.write_provenance(csv, config={"n_components": 9, "alpha": 0.99},
                                       ids={"prueba": ["d01_te.dat"]}, machine=False)
    assert ruta == tmp_path / "resultado.provenance.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert datos["csv"] == "resultado.csv"
    assert datos["config"] == {"n_components": 9, "alpha": 0.99}
    assert datos["ids"] == {"prueba": ["d01_te.dat"]}
    for clave in ("created_utc", "created_local", "git", "versions", "schema", "argv"):
        assert clave in datos


def test_procedencia_exige_el_csv_ya_escrito(tmp_path):
    with pytest.raises(FileNotFoundError, match="primero el CSV"):
        provenance.write_provenance(tmp_path / "no.csv", config={}, machine=False)


def test_procedencia_huellas_del_csv_y_de_las_entradas(tmp_path):
    entrada = tmp_path / "d00.dat"
    entrada.write_bytes(b"1 2 3\n4 5 6\n")
    csv = tmp_path / "r.csv"
    csv.write_bytes(b"a,b\r\n1,2\r\n")
    datos = json.loads(provenance.write_provenance(csv, config={}, inputs=[entrada], machine=False)
                       .read_text(encoding="utf-8"))
    assert datos["inputs"] == [{"name": "d00.dat", "bytes": 12,
                                "sha256": hashlib.sha256(b"1 2 3\n4 5 6\n").hexdigest()}]
    assert datos["csv_sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()   # texto con LF


def test_procedencia_trae_maquina_y_versiones(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    datos = provenance.read_provenance(provenance.write_provenance(csv, config={}))
    assert datos["machine"]["numpy_version"] == np.__version__
    assert "pyyaml_version" in datos["versions"]


def test_procedencia_nan_se_escribe_como_null(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    texto = provenance.write_provenance(csv, config={"x": float("nan"), "y": np.float64(1.5),
                                                     "z": np.int64(3)}, machine=False).read_text(encoding="utf-8")
    datos = json.loads(texto, parse_constant=lambda c: pytest.fail(f"JSON no estandar: {c}"))
    assert datos["config"] == {"x": None, "y": 1.5, "z": 3}


def test_procedencia_no_guarda_la_carpeta_personal(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    casa = str(Path.home())
    datos = provenance.build_provenance(config={"root": str(Path.home() / "datos")}, argv=["--root", casa],
                                        machine=False)

    def textos(x):
        if isinstance(x, dict):
            for v in x.values():
                yield from textos(v)
        elif isinstance(x, list):
            for v in x:
                yield from textos(v)
        elif isinstance(x, str):
            yield x

    assert not any(casa in t for t in textos(datos))
    assert datos["config"]["root"].startswith("~") and datos["argv"][1] == "~"


def test_procedencia_rutas_de_configuracion_relativas_a_la_raiz(monkeypatch):
    monkeypatch.chdir(RAIZ)
    datos = provenance.build_provenance(
        config={}, config_files=[{"path": str(BASELINE), "source": "cwd", "sha256": "x"}],
        argv=["--root", str(RAIZ / "configs")], machine=False)
    assert datos["config_files"][0]["path"] == "configs/baseline.yaml"
    assert datos["argv"] == ["--root", "configs"]


def test_procedencia_git_solo_lee_y_sin_bloqueos(monkeypatch, tmp_path):
    """git_info usa --no-optional-locks, --literal-pathspecs y GIT_OPTIONAL_LOCKS=0, sin
    variables GIT_* heredadas (correccion V-05), y solo ordenes de lectura."""
    llamadas = []

    def falso(args, **kw):
        llamadas.append((args, kw.get("env", {})))
        salida = {"rev-parse": "x\n", "tag": "", "status": " M a.py\n?? b.py\n"}[args[3]]
        return subprocess.CompletedProcess(args, 0, stdout=salida, stderr="")

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "otro"))
    monkeypatch.setattr(provenance.subprocess, "run", falso)
    info = provenance.git_info(tmp_path)
    assert info["repository"] and info["dirty"] and info["untracked"] == 1
    for args, entorno in llamadas:
        assert args[:3] == ["git", "--no-optional-locks", "--literal-pathspecs"]
        assert args[3] in ("rev-parse", "status", "tag")
        assert entorno.get("GIT_OPTIONAL_LOCKS") == "0"
        assert [k for k in entorno if k.upper().startswith("GIT_")] == ["GIT_OPTIONAL_LOCKS"]


def test_procedencia_sin_git_no_rompe(monkeypatch, tmp_path):
    def sin_git(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", sin_git)
    info = provenance.git_info(tmp_path)
    assert info["available"] is False and info["commit"] is None


@pytest.mark.skipif(shutil.which("git") is None or not (RAIZ / ".git").exists(), reason="sin git o sin repositorio")
def test_procedencia_git_real_da_el_commit_sin_tocar_el_indice():
    indice = RAIZ / ".git" / "index"
    antes = indice.stat().st_mtime_ns if indice.exists() else None
    info = provenance.git_info(RAIZ)
    assert info["repository"] and info["commit"] and len(info["commit"]) == 40
    despues = indice.stat().st_mtime_ns if indice.exists() else None
    assert antes == despues


# ===================================================== V1: referencia y huella

def test_v1_lee_las_cifras_de_yin_del_csv():
    ref = V1.leer_referencia(REFERENCIAS / "yin2012_tablas_transcripcion.csv", 4, 5)
    assert len(ref) == 44                                  # 21 FDR y 1 FAR por metodo
    assert set(ref.loc[ref["excluido"], "fallo"]) == {3, 9, 15}
    crudo = pd.read_csv(REFERENCIAS / "yin2012_tablas_transcripcion.csv", comment="#")
    celda = crudo[(crudo.tabla == 4) & (crudo.metodo == "PCA") & (crudo.fallo == 5)]["valor_num"].iloc[0]
    assert ref[(ref.metodo == "PCA") & (ref.fallo == 5)]["valor"].iloc[0] == celda


def test_v1_si_cambia_el_csv_cambia_la_cifra_de_yin(tmp_path):
    crudo = pd.read_csv(REFERENCIAS / "yin2012_tablas_transcripcion.csv", comment="#")
    mascara = (crudo.tabla == 4) & (crudo.metodo == "DPCA") & (crudo.fallo == 7)
    crudo.loc[mascara, "valor_num"] = 12.5
    ruta = tmp_path / "ref.csv"
    crudo.to_csv(ruta, index=False)
    ref = V1.leer_referencia(ruta, 4, 5)
    assert ref[(ref.metodo == "DPCA") & (ref.fallo == 7)]["valor"].iloc[0] == 12.5


def test_v1_no_teclea_cifras_de_yin():
    """Ninguna cifra de Yin de PCA o DPCA (Tablas 4 y 5) ni ningun umbral dibujado esta en el script."""
    fuente = (RAIZ / "experiments" / "V1_verificacion_yin.py").read_text(encoding="utf-8")
    crudo = pd.read_csv(REFERENCIAS / "yin2012_tablas_transcripcion.csv", comment="#", dtype={"valor_impreso": str})
    celdas = crudo[crudo.tabla.isin([4, 5]) & crudo.metodo.isin(["PCA", "DPCA"])]["valor_impreso"]
    umbrales = pd.read_csv(REFERENCIAS / "yin2012_umbrales_dibujados.csv", comment="#", dtype={"umbral_leido": str})
    for valor in set(celdas) | set(umbrales["umbral_leido"]):
        if "." in valor or "," in valor:
            assert valor not in fuente, valor


def test_v1_huella_cambia_con_los_valores_y_las_referencias_no_con_los_comentarios(tmp_path):
    cfg = _yaml_fijado()
    rutas = _repo_falso(tmp_path, cfg)
    referencias = V1.rutas_referencia(cfg, rutas["yaml"].parent)
    base = V1.huella(cfg, referencias)
    con_comentario = yaml.safe_load(("# otro comentario\n" + rutas["yaml"].read_text(encoding="utf-8")))
    assert V1.huella(con_comentario, referencias) == base
    otro = _yaml_fijado(tolerancia__far_max_pp=1)
    assert V1.huella(otro, referencias) != base
    with open(referencias["umbrales"], "a", encoding="utf-8") as f:
        f.write("# cambio\n")
    assert V1.huella(cfg, referencias) != base


def test_v1_yaml_real_valido_y_pendientes_en_una_copia():
    # Desde el 2026-10-14 el yaml real tiene todas las decisiones fijadas (fila de
    # esa fecha en docs/decisiones.md): es valido y no le queda nada pendiente.
    cfg = config.load_yaml(YIN_YAML)
    V1.validar_config(cfg)
    assert V1.pendientes(cfg) == []
    # Si esas claves vuelven a null en una copia, V1 las da como pendientes.
    copia = yaml.safe_load(YIN_YAML.read_text(encoding="utf-8"))
    copia["supuestos"]["S9_far_ventana"] = None
    copia["tolerancia"].update(comparacion=None, far_max_pp=None, dpca=None)
    copia["referencia"]["transcripcion_verificada"] = None
    copia["secundaria"]["incluir"] = None
    faltan = V1.pendientes(copia)
    for clave in ("supuestos.S9_far_ventana", "tolerancia.comparacion", "tolerancia.far_max_pp",
                  "tolerancia.dpca", "referencia.transcripcion_verificada", "secundaria.incluir"):
        assert clave in faltan, clave


def test_v1_variante_con_valor_invalido_falla():
    cfg = _yaml_fijado()
    cfg["variantes"].append({"nombre": "mala", "cambios": {"S5_orden_dpca": "al_reves"}})
    with pytest.raises(ValueError, match="S5_orden_dpca"):
        V1.validar_config(cfg)


def test_v1_tolerancia_sin_far_cuenta_como_pendiente():
    cfg = _yaml_fijado(tolerancia__far_max_pp="no_aplica")
    assert any("far_max_pp" in c for c in V1.pendientes(cfg))


# ========================================================= V1: medida

def test_v1_fdr_usa_las_muestras_161_a_960():
    indice = np.zeros(960)
    indice[:160] = 5.0          # alarmas antes del fallo: no cuentan en la FDR
    indice[160:360] = 5.0       # 200 alarmas en las muestras 161 a 360
    r = V1.medir_fdr(indice, 160, k=3)
    assert (r["n_alarms"], r["n_evaluated"]) == (200, 800)
    assert r["rate"] == 200 / 800


def test_v1_far_usa_la_ventana_del_yaml():
    indice = np.zeros(960)
    indice[:100] = 5.0          # en las muestras 1 a 100
    indice[900:] = 5.0          # en las muestras 901 a 960
    assert V1.medir_far(indice, (160, None), k=3)["n_alarms"] == 60
    assert V1.medir_far(indice, (0, None), k=3)["n_alarms"] == 160
    r = V1.medir_far(indice, (0, 800), k=3)
    assert (r["n_alarms"], r["n_evaluated"]) == (100, 800)


def test_v1_igual_al_limite_no_es_alarma():
    """Desigualdad estricta (p. 1568): T2 o SPE igual a su limite no es fallo."""
    indice = np.ones(960)
    assert V1.medir_fdr(indice, 160, k=3)["n_alarms"] == 0


def test_v1_far_con_las_primeras_muestras_del_dpca_sin_evaluar():
    indice = np.zeros(960)
    indice[:2] = np.nan         # lags = 2: no evaluadas (S12)
    assert V1.medir_far(indice, (0, None), k=3)["n_evaluated"] == 958
    assert V1.medir_far(indice, (160, None), k=3)["n_evaluated"] == 800


def test_v1_comparar_recupera_el_recuento_exacto_de_yin():
    ref = {"valor": 3.38, "compatible_k800": True}       # 27 de 800 = 3,375 %, impreso 3,38
    c = V1.comparar({"n_alarms": 31, "n_evaluated": 800}, ref)
    assert (c["yin_count"], c["diff_samples"]) == (27, 4)
    assert c["diff_pp_count"] == 0.5                     # exacto; con el impreso saldria 0,495
    assert np.isclose(c["diff_pp"], 31 / 8 - 3.38)
    sin_rejilla = V1.comparar({"n_alarms": 31, "n_evaluated": 958}, ref)
    assert np.isnan(sin_rejilla["yin_count"])


def _tabla_veredicto(diferencias: dict, far: float) -> pd.DataFrame:
    filas = [{"metric": "fdr", "faultNumber": f, "abs_diff_verdict": d, "excluded_369": f in (3, 9, 15)}
             for f, d in diferencias.items()]
    filas.append({"metric": "far", "faultNumber": 0, "abs_diff_verdict": far, "excluded_369": False})
    return pd.DataFrame(filas)


def _tolerancia(**kw) -> dict:
    tol = {"comparacion": "conteo", "fdr_umbral_por_fallo_pp": 0.5, "fdr_min_fallos_dentro": 18,
           "fdr_max_pp": "no_aplica", "fdr_media_max_pp": "no_aplica", "far_max_pp": 0.5, "dpca": "criterio"}
    tol.update(kw)
    return tol


def test_v1_veredicto_no_cuenta_los_fallos_3_9_15():
    difs = {f: 0.25 for f in range(1, 22)}
    difs.update({3: 40.0, 9: 40.0, 15: 40.0})
    res = V1.evaluar_tolerancia(_tabla_veredicto(difs, 0.25), _tolerancia())
    assert res["n_faults_verdict"] == 18 and res["fdr_n_within"] == 18
    assert res["verdict"] == "verificado"


def test_v1_veredicto_falla_si_un_fallo_contado_se_sale():
    difs = {f: 0.25 for f in range(1, 22)}
    difs[10] = 0.625
    assert V1.evaluar_tolerancia(_tabla_veredicto(difs, 0.25), _tolerancia())["verdict"] == "no verificado"


def test_v1_veredicto_en_el_borde_del_umbral_cuenta_dentro():
    difs = {f: 0.5 for f in range(1, 22)}
    assert V1.evaluar_tolerancia(_tabla_veredicto(difs, 0.5), _tolerancia())["verdict"] == "verificado"


def test_v1_veredicto_con_criterio_global():
    """Opcion C: media y maximo, sin umbral por fallo."""
    difs = {f: 0.0 for f in range(1, 22)}
    difs[10] = 4.0
    tol = _tolerancia(fdr_umbral_por_fallo_pp="no_aplica", fdr_min_fallos_dentro="no_aplica",
                      fdr_max_pp=5, fdr_media_max_pp=1, far_max_pp=2)
    res = V1.evaluar_tolerancia(_tabla_veredicto(difs, 1.0), tol)
    assert res["verdict"] == "verificado" and np.isnan(res["fdr_n_within"])
    difs[11] = 20.0
    assert V1.evaluar_tolerancia(_tabla_veredicto(difs, 1.0), tol)["verdict"] == "no verificado"


def test_v1_veredicto_la_far_cuenta():
    difs = {f: 0.0 for f in range(1, 22)}
    assert V1.evaluar_tolerancia(_tabla_veredicto(difs, 0.75), _tolerancia())["verdict"] == "no verificado"


# ===================================================== V1: regla de orden

def test_v1_se_niega_con_la_tolerancia_en_null(tmp_path, tep_sintetico, monkeypatch, capsys):
    cfg = yaml.safe_load(YIN_YAML.read_text(encoding="utf-8"))
    cfg["tolerancia"].update(comparacion=None, far_max_pp=None, dpca=None)  # el real ya la fija (2026-10-14)
    rutas = _repo_falso(tmp_path, cfg)
    _git_limpio(monkeypatch)
    llamadas = _espiar_cargas(monkeypatch)
    codigo = V1.ejecutar(["--config", str(rutas["yaml"]), "--root", str(tep_sintetico),
                          "--out", str(tmp_path / "out")])
    assert codigo == 2
    assert "tolerancia" in capsys.readouterr().err
    assert llamadas == []                                    # no se leyo ningun fichero
    assert not (tmp_path / "out" / V1.SALIDA).exists()


def test_v1_se_niega_sin_fila_fechada_en_decisiones(tmp_path, tep_sintetico, monkeypatch, capsys):
    rutas = _repo_falso(tmp_path, _yaml_fijado(), fila=None)
    _git_limpio(monkeypatch)
    llamadas = _espiar_cargas(monkeypatch)
    codigo = V1.ejecutar(["--config", str(rutas["yaml"]), "--root", str(tep_sintetico),
                          "--out", str(tmp_path / "out")])
    assert codigo == 2 and "Ninguna fila" in capsys.readouterr().err
    assert all(split != "test" for _, split in llamadas)


def test_v1_se_niega_si_la_huella_registrada_es_otra(tmp_path, tep_sintetico, monkeypatch, capsys):
    fila = "| 2026-01-01 | Verificacion frente a Yin [V1-yin huella=000000000000] | Pablo | prueba |"
    rutas = _repo_falso(tmp_path, _yaml_fijado(), fila=fila)
    _git_limpio(monkeypatch)
    codigo = V1.ejecutar(["--config", str(rutas["yaml"]), "--root", str(tep_sintetico),
                          "--out", str(tmp_path / "out")])
    assert codigo == 2 and "otra huella" in capsys.readouterr().err


def test_v1_se_niega_con_fecha_futura(tmp_path, monkeypatch):
    cfg = _yaml_fijado()
    rutas = _repo_falso(tmp_path, cfg, fila=None)
    actual = V1.huella(cfg, V1.rutas_referencia(cfg, rutas["yaml"].parent))
    with open(rutas["decisiones"], "a", encoding="utf-8") as f:
        f.write(f"| 2999-01-01 | Verificacion [V1-yin huella={actual}] | Pablo | prueba |\n")
    _git_limpio(monkeypatch)
    with pytest.raises(V1.RegistroNoFijado, match="fecha futura"):
        V1.comprobar_registro(cfg, rutas["yaml"].parent, rutas["yaml"], rutas["decisiones"])


def test_v1_se_niega_con_cambios_sin_confirmar(tmp_path, monkeypatch):
    cfg = _yaml_fijado()
    rutas = _repo_falso(tmp_path, cfg)
    monkeypatch.setattr(V1, "estado_git", lambda r: {"repository": True, "commit": "x",
                                                     "status": [" M configs/yin2012.yaml"]})
    with pytest.raises(V1.RegistroNoFijado, match="sin confirmar"):
        V1.comprobar_registro(cfg, rutas["yaml"].parent, rutas["yaml"], rutas["decisiones"])


@pytest.mark.skipif(shutil.which("git") is None, reason="sin git")
def test_v1_fuera_de_un_repositorio_git_se_niega(tmp_path):
    fichero = tmp_path / "a.txt"
    fichero.write_text("x", encoding="utf-8")
    with pytest.raises(V1.RegistroNoFijado, match="repositorio"):
        V1.estado_git([fichero])


@pytest.mark.skipif(shutil.which("git") is None, reason="sin git")
def test_v1_fichero_sin_versionar_cuenta_como_cambio(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    fichero = tmp_path / "decisiones.md"
    fichero.write_text("x", encoding="utf-8")
    assert V1.estado_git([fichero])["status"]


def test_v1_cargar_prueba_exige_el_permiso(tep_sintetico):
    with pytest.raises(V1.RegistroNoFijado):
        V1.cargar_prueba(1, tep_sintetico, permiso=None)


def test_v1_conteo_imposible_con_una_ventana_sin_800_muestras(tmp_path, monkeypatch):
    cfg = _yaml_fijado(supuestos__S9_far_ventana=[0, 800])          # el DPCA evalua 798
    rutas = _repo_falso(tmp_path, cfg)
    _git_limpio(monkeypatch)
    with pytest.raises(V1.RegistroNoFijado, match="conteo"):
        V1.comprobar_registro(cfg, rutas["yaml"].parent, rutas["yaml"], rutas["decisiones"])


# ===================================================== V1: ejecucion completa

def test_v1_con_el_registro_fijado_escribe_la_tabla_y_el_veredicto(tmp_path, tep_sintetico, monkeypatch):
    cfg = _yaml_fijado()
    rutas = _repo_falso(tmp_path, cfg)
    _git_limpio(monkeypatch)
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # ni avisos de calibracion ni de runs
        codigo = V1.ejecutar(["--config", str(rutas["yaml"]), "--root", str(tep_sintetico), "--out", str(out)])
    assert codigo == 0
    detalle = pd.read_csv(out / V1.SALIDA)
    resumen = pd.read_csv(out / V1.SALIDA_RESUMEN)
    principal = detalle[(detalle.configuracion == "principal") & (detalle.variant == "principal")]
    assert len(principal) == 44                                      # 22 filas por metodo
    fdr = principal[principal.metric == "fdr"]
    assert (fdr.n_evaluated == 800).all() and (fdr.fdr_from == 160).all()
    far = principal[principal.metric == "far"]
    assert (far.n_evaluated == 800).all() and (far.far_window == "[160, null]").all()
    assert set(fdr.loc[fdr.excluded_369, "faultNumber"]) == {3, 9, 15}
    assert fdr.loc[fdr.excluded_369, "within_tolerance"].isna().all()       # fuera del veredicto
    assert fdr.loc[~fdr.excluded_369, "within_tolerance"].notna().all()
    ref = V1.leer_referencia(REFERENCIAS / "yin2012_tablas_transcripcion.csv", 4, 5)
    unido = principal.merge(ref.assign(method=ref.metodo.str.lower()),
                            left_on=["method", "faultNumber"], right_on=["method", "fallo"])
    assert np.allclose(unido.yin_value, unido.valor)
    assert np.allclose(principal.ours, 100 * principal.n_alarms / principal.n_evaluated)
    pca = resumen[(resumen.method == "pca") & resumen.counts_for_verdict]
    assert pca.verdict.iloc[0] in ("verificado", "no verificado")
    assert resumen.loc[(resumen.method == "dpca") & resumen.counts_for_verdict, "verdict"].iloc[0] == "descriptivo"
    assert set(resumen.loc[~resumen.counts_for_verdict, "verdict"]) == {"informativo"}
    variantes = set(detalle.loc[detalle.method == "pca", "variant"])
    assert variantes == {"principal", "S9_far_1_960"}                # S5 no cambia nada en el PCA
    assert (detalle.fingerprint == rutas["huella"]).all()
    for nombre in (V1.SALIDA, V1.SALIDA_RESUMEN):
        datos = provenance.read_provenance(out / nombre)
        assert datos["extra"]["huella"] == rutas["huella"]
        assert datos["extra"]["registro"]["fecha"] == "2026-01-01"
        assert len(datos["inputs"]) == 23


def test_v1_fdr_del_fichero_sintetico_cuenta_solo_tras_el_inicio(tmp_path, tep_sintetico, monkeypatch):
    """Con el escalon desde la muestra 161, la FDR del PCA es alta y la FAR previa no entra en ella."""
    cfg = _yaml_fijado()
    rutas = _repo_falso(tmp_path, cfg)
    _git_limpio(monkeypatch)
    out = tmp_path / "out"
    assert V1.ejecutar(["--config", str(rutas["yaml"]), "--root", str(tep_sintetico), "--out", str(out)]) == 0
    d = pd.read_csv(out / V1.SALIDA)
    pca = d[(d.method == "pca") & (d.variant == "principal") & (d.metric == "fdr")].set_index("faultNumber")
    assert pca.loc[1, "ours"] > 90 and pca.loc[3, "ours"] < 20


def test_v1_solo_limites_no_abre_ficheros_de_prueba(tmp_path, monkeypatch):
    solo_d00 = _tep_sintetico(tmp_path / "datos", solo_d00=True)
    llamadas = _espiar_cargas(monkeypatch)
    out = tmp_path / "out"
    assert V1.ejecutar(["--config", str(YIN_YAML), "--root", str(solo_d00), "--out", str(out),
                        "--solo-limites"]) == 0
    assert llamadas and all((f, s) == (0, "train") for f, s in llamadas)
    tabla = pd.read_csv(out / V1.SALIDA_LIMITES)
    assert {"pca", "dpca"} <= set(tabla.method) and {"T2", "SPE"} <= set(tabla.statistic)
    assert provenance.read_provenance(out / V1.SALIDA_LIMITES)["extra"]["modo"] == "solo_limites"


def test_v1_huella_avisa_si_quedan_pendientes(tmp_path, capsys):
    cfg = yaml.safe_load(YIN_YAML.read_text(encoding="utf-8"))
    cfg["referencia"]["transcripcion_verificada"] = None     # el real ya no tiene pendientes (2026-10-14)
    rutas = _repo_falso(tmp_path, cfg, fila=None)
    assert V1.ejecutar(["--config", str(rutas["yaml"]), "--huella"]) == 1
    salida = capsys.readouterr().out
    assert "[V1-yin huella=" in salida and "pendientes" in salida


# ============================================ regresion de los limites (datos reales)

@pytest.mark.skipif(not (CLASSIC_DIR / "d00.dat").exists(), reason="sin d00.dat del TEP clasico")
def test_limites_d00_reproducen_los_umbrales_dibujados_por_yin():
    """Con d00.dat y la configuracion principal del yaml, los limites del PCA (T2 y SPE) y
    el SPE del DPCA caen dentro de la precision de lectura de las lineas que dibujan Yin
    et al. (2012) en las Fig. 2 (p. 1573) y 5 (p. 1576). Los valores de referencia se leen
    de configs/referencias/yin2012_umbrales_dibujados.csv. El T2 del DPCA no entra: el
    supuesto S11 sigue abierto (la ec. 3 con el N del ajuste no reproduce el dibujado)."""
    cfg = config.load_yaml(YIN_YAML)
    umbrales = V1.leer_umbrales(REFERENCIAS / "yin2012_umbrales_dibujados.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # d00.dat viene transpuesto: lo corrige load_classic
        X = V1.cargar_d00(CLASSIC_DIR, cfg["explicito"]["variables"])
    comprobados = 0
    for metodo, estadisticos in (("PCA", ("T2", "SPE")), ("DPCA", ("SPE",))):
        modelo = V1.ajustar(X, V1.configuracion_efectiva(cfg, metodo), umbrales)
        for estadistico in estadisticos:
            calculado = modelo.t2_limit_ if estadistico == "T2" else modelo.spe_limit_
            filas = umbrales[(umbrales.metodo == metodo) & (umbrales.estadistico == estadistico)]
            assert len(filas) == 2                           # Fig. 2 y Fig. 5
            for fila in filas.itertuples():
                tolerancia = TOLERANCIA_LECTURA * fila.residuo_max_marcas
                assert abs(calculado - fila.umbral_leido) <= tolerancia, (metodo, estadistico, fila.figura)
                comprobados += 1
    assert comprobados == 6


# ===================================================== A0 y A1 (sinteticos)

A1 = _cargar_experimento("A1_pca_baseline")
A0 = _cargar_experimento("A0_calibracion_limites")


def test_a1_escribe_method_configuracion_y_procedencia(tmp_path, tep_sintetico):
    out = tmp_path / "a1"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A1.ejecutar(["--config", str(BASELINE), "--root", str(tep_sintetico), "--out", str(out)])
    runs = pd.read_csv(out / "A1_pca_baseline_runs.csv")
    assert (runs["method"] == "pca").all() and len(runs) == 44
    for columna in ("n_components", "alpha", "variance", "t2_method", "spe_method", "calibration",
                    "t2_limit_source", "spe_limit_source", "n_fit", "n_cal", "lags", "k"):
        assert columna in runs, columna
    assert (runs["spe_limit_source"] == "holdout").all() and (runs["n_cal"] == 150).all()
    for nombre in ("runs", "T2", "SPE"):
        csv = out / f"A1_pca_baseline_{nombre}.csv"
        datos = provenance.read_provenance(csv)
        assert datos["config"]["n_components"] == runs["n_components"].iloc[0]
        assert datos["config_files"][0]["source"] == "explicit"


def test_a1_dpca_pasa_runs_y_alinea_el_inicio(tmp_path, tep_sintetico):
    out = tmp_path / "a1d"
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # sin runs, PCAMonitor avisaria
        A1.ejecutar(["--config", str(BASELINE), "--root", str(tep_sintetico), "--out", str(out), "--lags", "2"])
    runs = pd.read_csv(out / "A1_pca_baseline_runs.csv")
    assert (runs["method"] == "dpca").all() and (runs["lags"] == 2).all()
    normal = runs[runs.faultNumber == 0]
    assert (normal["n_far"] == 958).all()                    # las 2 primeras no se evaluan
    con_fallo = runs[runs.faultNumber > 0]
    assert (con_fallo["n_far"] == 158).all() and (con_fallo["n_fdr"] == 800).all()


def test_a1_calibracion_en_muestra_explicita_sin_aviso(tmp_path, tep_sintetico):
    out = tmp_path / "a1m"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        A1.ejecutar(["--config", str(BASELINE), "--root", str(tep_sintetico), "--out", str(out),
                     "--calibration", "in_sample"])
    runs = pd.read_csv(out / "A1_pca_baseline_runs.csv")
    assert (runs["spe_limit_source"] == "in_sample").all() and (runs["n_cal_rows"] == 0).all()


def test_a1_jackson_y_t2_beta_desde_la_linea_de_ordenes(tmp_path, tep_sintetico):
    out = tmp_path / "a1j"
    A1.ejecutar(["--config", str(BASELINE), "--root", str(tep_sintetico), "--out", str(out),
                 "--spe-method", "jackson", "--t2-method", "beta", "--calibration", "in_sample"])
    runs = pd.read_csv(out / "A1_pca_baseline_runs.csv")
    assert set(runs["spe_limit_source"]) == {"parametric_fit"} and set(runs["t2_method"]) == {"beta"}


def test_a0_brazos_del_t2_y_del_spe_con_el_mismo_modelo(tmp_path, tep_sintetico):
    out = tmp_path / "a0"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = A0.ejecutar(["--config", str(BASELINE), "--root", str(tep_sintetico), "--out", str(out)])
    spe = pd.read_csv(out / "A0_calibracion_limites.csv").set_index("calibracion")
    t2 = pd.read_csv(out / "A0_calibracion_limites_t2.csv").set_index("calibracion")
    assert list(spe.index) == ["en_muestra", "reservada", "en_muestra_mismo_ncal", "jackson"]
    assert list(t2.index) == ["f", "beta", "empirico_reservada", "empirico_en_muestra"]
    assert spe.loc["en_muestra", "n_cal_used"] == 350 and spe.loc["en_muestra_mismo_ncal", "n_cal_used"] == 150
    assert t2.loc["empirico_en_muestra", "n_cal_used"] == t2.loc["empirico_reservada", "n_cal_used"] == 150
    assert spe.loc["jackson", "limit_source"] == "parametric_fit" and np.isnan(spe.loc["jackson", "n_cal_used"])
    assert np.isclose(t2.loc["f", "far_teorica_faseII"], 1 - 0.99)
    assert t2.loc["beta", "t2_limit"] < t2.loc["f", "t2_limit"]
    assert len(spe["n_components"].unique()) == 1
    detalle = pd.read_csv(out / "A0_calibracion_limites_por_fichero.csv")
    assert list(detalle.columns[:3]) == ["faultNumber", "far_spe_en_muestra", "far_spe_reservada"]
    assert set(res) == {"spe", "t2", "detalle"}
    for nombre in ("A0_calibracion_limites.csv", "A0_calibracion_limites_t2.csv",
                   "A0_calibracion_limites_por_fichero.csv"):
        assert provenance.provenance_path(out / nombre).is_file()


def test_a0_y_a1_no_se_presentan_como_ablacion_ni_como_verificacion():
    """M6 y M7 de rev_trazabilidad: A0 no es una fila de la tabla de ablaciones (parrafo 367) y
    A1 no es la verificacion frente a Yin (parrafo 369)."""
    assert "primera fila de la tabla" not in (A0.__doc__ or "")
    assert "NO es una fila de la tabla de ablaciones" in A0.__doc__
    assert "VERIFICAR EL PROTOCOLO" not in (A1.__doc__ or "")
    assert "V1_verificacion_yin.py" in A1.__doc__
