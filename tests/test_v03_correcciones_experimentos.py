"""Pruebas de las correcciones del 12/10 en configuracion, procedencia y V1.

Cubren los hallazgos de la revision del codigo del 12/10 (verif_codigo):
- V-05: la comprobacion de git de la regla de orden, ruta por ruta: dentro de
  .git, repositorio anidado, skip-worktree, assume-unchanged, ignorado, corchetes.
- V-11: la regla de orden cubre tambien el codigo (el script y el paquete).
- V-12: variante con un supuesto en null, tolerancia NaN, fechas imposibles, filas
  del registro escondidas en comentarios o bloques de codigo.
- V-13: la carpeta personal se oculta con cualquier capitalizacion y en la forma
  de Git Bash.
- V-14: git y la maquina se calculan una vez por ejecucion.
- V-15: aviso si el baseline.yaml del directorio de trabajo no es el del paquete.

Las pruebas de git crean repositorios TEMPORALES de prueba (en la carpeta temporal
de pytest), con commits de un autor ficticio pasado con -c, sin tocar ninguna
configuracion de git: es la unica forma de probar la comprobacion con ficheros
confirmados de verdad. Ningun commit en el repositorio del proyecto. Se saltan si
git no esta instalado. Ninguna prueba lee un fichero de prueba del TEP.
"""

from __future__ import annotations

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

RAIZ = Path(__file__).resolve().parents[1]
YIN_YAML = RAIZ / "configs" / "yin2012.yaml"
REFERENCIAS = RAIZ / "configs" / "referencias"
BASELINE = RAIZ / "configs" / "baseline.yaml"

con_git = pytest.mark.skipif(shutil.which("git") is None, reason="sin git")


def _cargar_experimento(nombre: str):
    ruta = RAIZ / "experiments" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(f"{nombre}_correcciones", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


V1 = _cargar_experimento("V1_verificacion_yin")
A1 = _cargar_experimento("A1_pca_baseline")


# ------------------------------------------------------------- ayudas de git

GIT = ["git", "-c", "user.name=prueba", "-c", "user.email=prueba@example.invalid",
       "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"]


def _git(repo: Path, *args):
    entorno = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    return subprocess.run(GIT + list(args), cwd=repo, capture_output=True, text=True, check=True, env=entorno)


def _repo(carpeta: Path, ficheros: dict, *, commit: bool = True) -> Path:
    """Un repositorio temporal con esos ficheros ({ruta relativa: texto}), confirmados."""
    for rel, texto in ficheros.items():
        p = carpeta / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto, encoding="utf-8")
    _git(carpeta, "init", "-q")
    if commit:
        _git(carpeta, "add", "-A")
        _git(carpeta, "commit", "-q", "-m", "prueba")
    return carpeta


def _yaml_fijado() -> dict:
    """El yaml real con las decisiones pendientes rellenadas (valores de prueba, no una propuesta)."""
    cfg = yaml.safe_load(YIN_YAML.read_text(encoding="utf-8"))
    cfg["referencia"]["transcripcion_verificada"] = "2026-01-01"
    cfg["supuestos"]["S9_far_ventana"] = [160, None]
    cfg["tolerancia"].update(comparacion="conteo", fdr_umbral_por_fallo_pp=2, fdr_min_fallos_dentro=16,
                             fdr_max_pp=5, fdr_media_max_pp="no_aplica", far_max_pp=2, dpca="descriptivo")
    cfg["secundaria"]["incluir"] = False
    return cfg


def _repo_verificacion(carpeta: Path, cfg: dict | None = None) -> dict:
    """Repositorio temporal con todo lo que exige la regla de orden de V1, confirmado:
    el yaml, las referencias, el registro con la fila de la huella y el codigo
    (una copia del script y del paquete)."""
    cfg = cfg or _yaml_fijado()
    (carpeta / "configs" / "referencias").mkdir(parents=True)
    for f in REFERENCIAS.glob("*.csv"):
        shutil.copy(f, carpeta / "configs" / "referencias" / f.name)
    ruta_yaml = carpeta / "configs" / "yin2012.yaml"
    ruta_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    actual = V1.huella(cfg, V1.rutas_referencia(cfg, ruta_yaml.parent))
    (carpeta / "docs").mkdir()
    decisiones = carpeta / "docs" / "decisiones.md"
    decisiones.write_text("| Fecha | Decision | Quien | Motivo |\n|---|---|---|---|\n"
                          f"| 2026-01-01 | Yin [V1-yin huella={actual}] | Pablo | prueba |\n", encoding="utf-8")
    (carpeta / "experiments").mkdir()
    shutil.copy(RAIZ / "experiments" / "V1_verificacion_yin.py", carpeta / "experiments")
    shutil.copytree(RAIZ / "tfmfdd", carpeta / "tfmfdd", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(RAIZ / ".gitignore", carpeta / ".gitignore")
    _repo(carpeta, {})
    return {"cfg": cfg, "yaml": ruta_yaml, "decisiones": decisiones, "repo": carpeta,
            "codigo": [carpeta / "experiments" / "V1_verificacion_yin.py", carpeta / "tfmfdd"]}


def _comprobar(r: dict, **kw):
    return V1.comprobar_registro(r["cfg"], r["yaml"].parent, r["yaml"], r["decisiones"],
                                 codigo=kw.pop("codigo", r["codigo"]), **kw)


# ================================================== V-05: git, ruta por ruta

@con_git
def test_fichero_confirmado_no_tiene_problemas(tmp_path):
    repo = _repo(tmp_path / "r", {"docs/decisiones.md": "| x |\n"})
    estado = provenance.committed_status([repo / "docs" / "decisiones.md"])
    assert estado["problems"] == [] and len(estado["roots"]) == 1


@con_git
def test_fichero_dentro_de_git_es_un_error(tmp_path):
    repo = _repo(tmp_path / "r", {"a.txt": "x"})
    (repo / ".git" / "decisiones.md").write_text("| x |\n", encoding="utf-8")
    with pytest.raises(provenance.GitCheckError, match=".git"):
        provenance.committed_status([repo / ".git" / "decisiones.md"])


@con_git
def test_repositorio_anidado_es_otra_raiz(tmp_path):
    repo = _repo(tmp_path / "r", {"configs/yin2012.yaml": "a: 1\n"})
    anidado = _repo(repo / "sub", {"docs/decisiones.md": "| x |\n"})
    estado = provenance.committed_status([repo / "configs" / "yin2012.yaml", anidado / "docs" / "decisiones.md"])
    assert len(estado["roots"]) == 2
    with pytest.raises(V1.RegistroNoFijado, match="mismo repositorio"):
        V1.estado_git([repo / "configs" / "yin2012.yaml", anidado / "docs" / "decisiones.md"])


@con_git
@pytest.mark.parametrize("marca", ["--skip-worktree", "--assume-unchanged"])
def test_fichero_marcado_para_que_git_no_vea_sus_cambios_cuenta(tmp_path, marca):
    repo = _repo(tmp_path / "r", {"docs/decisiones.md": "| x |\n"})
    _git(repo, "update-index", marca, "docs/decisiones.md")
    with open(repo / "docs" / "decisiones.md", "a", encoding="utf-8") as f:
        f.write("| fila sin confirmar |\n")
    problemas = provenance.committed_status([repo / "docs" / "decisiones.md"])["problems"]
    assert any("skip-worktree o assume-unchanged" in p for p in problemas)


@con_git
def test_fichero_ignorado_y_sin_versionar_cuenta(tmp_path):
    repo = _repo(tmp_path / "r", {".gitignore": "docs/decisiones.md\n"})
    (repo / "docs").mkdir()
    (repo / "docs" / "decisiones.md").write_text("| x |\n", encoding="utf-8")
    problemas = provenance.committed_status([repo / "docs" / "decisiones.md"])["problems"]
    assert "docs/decisiones.md: sin versionar" in problemas and any(p.startswith("!!") for p in problemas)


@con_git
def test_corchetes_en_la_ruta_se_toman_tal_cual(tmp_path):
    repo = _repo(tmp_path / "r", {"conf[1]/yin2012.yaml": "a: 1\n", "conf1/yin2012.yaml": "a: 1\n"})
    assert provenance.committed_status([repo / "conf[1]" / "yin2012.yaml"])["problems"] == []
    with open(repo / "conf[1]" / "yin2012.yaml", "a", encoding="utf-8") as f:
        f.write("# comentario\n")
    assert provenance.committed_status([repo / "conf[1]" / "yin2012.yaml"])["problems"]


def test_git_no_hereda_variables_git(monkeypatch):
    """Una GIT_DIR o GIT_INDEX_FILE heredada haria mirar otro repositorio u otro indice."""
    monkeypatch.setenv("GIT_DIR", "otro/.git")
    monkeypatch.setenv("GIT_INDEX_FILE", "otro_indice")
    entorno = provenance._entorno_git()
    assert [k for k in entorno if k.upper().startswith("GIT_")] == ["GIT_OPTIONAL_LOCKS"]
    assert entorno["GIT_OPTIONAL_LOCKS"] == "0"


@con_git
def test_fuera_de_un_repositorio_es_un_error(tmp_path):
    fichero = tmp_path / "decisiones.md"
    fichero.write_text("| x |\n", encoding="utf-8")
    with pytest.raises(provenance.GitCheckError, match="repositorio"):
        provenance.committed_status([fichero])


# ================================================== V-11: el codigo tambien

@con_git
def test_carpeta_de_codigo_limpia_y_pycache_ignorado_no_cuentan(tmp_path):
    repo = _repo(tmp_path / "r", {".gitignore": "__pycache__/\n*.py[cod]\n", "pkg/__init__.py": "",
                                  "pkg/m.py": "X = 1\n"})
    (repo / "pkg" / "__pycache__").mkdir()
    (repo / "pkg" / "__pycache__" / "m.cpython-312.pyc").write_bytes(b"\x00")
    assert provenance.committed_status([repo / "pkg"])["problems"] == []


@con_git
@pytest.mark.parametrize("caso", ["modificado", "sin_versionar", "ignorado"])
def test_carpeta_de_codigo_con_cambios_cuenta(tmp_path, caso):
    repo = _repo(tmp_path / "r", {".gitignore": "pkg/oculto.py\n", "pkg/__init__.py": "", "pkg/m.py": "X = 1\n"})
    if caso == "modificado":
        (repo / "pkg" / "m.py").write_text("X = 2\n", encoding="utf-8")
    elif caso == "sin_versionar":
        (repo / "pkg" / "nuevo.py").write_text("Y = 1\n", encoding="utf-8")
    else:
        (repo / "pkg" / "oculto.py").write_text("Z = 1\n", encoding="utf-8")
    assert provenance.committed_status([repo / "pkg"])["problems"]


def test_el_codigo_por_defecto_es_el_que_se_ejecuta():
    import tfmfdd

    script, paquete = V1.codigo_en_ejecucion()
    assert script == (RAIZ / "experiments" / "V1_verificacion_yin.py").resolve()
    assert paquete == Path(tfmfdd.__file__).resolve().parent


@con_git
def test_regla_de_orden_completa_con_un_commit_de_verdad(tmp_path):
    """Control positivo: todo fijado y confirmado (codigo incluido) da el permiso."""
    r = _repo_verificacion(tmp_path / "r")
    permiso = _comprobar(r)
    cabeza = _git(r["repo"], "rev-parse", "HEAD").stdout.strip()
    assert isinstance(permiso, V1._PermisoPrueba) and permiso.git["commit"] == cabeza


@con_git
@pytest.mark.parametrize("fichero", ["experiments/V1_verificacion_yin.py", "tfmfdd/metrics.py"])
def test_regla_de_orden_se_niega_con_el_codigo_sin_confirmar(tmp_path, fichero):
    r = _repo_verificacion(tmp_path / "r")
    with open(r["repo"] / fichero, "a", encoding="utf-8") as f:
        f.write("# cambiado despues de fijar el registro\n")
    with pytest.raises(V1.RegistroNoFijado, match="sin confirmar"):
        _comprobar(r)


@con_git
def test_regla_de_orden_se_niega_si_el_codigo_esta_en_otro_repositorio(tmp_path):
    """Con el codigo por defecto (el de este clon), el registro del repositorio temporal
    no fija nada: no es el mismo repositorio."""
    r = _repo_verificacion(tmp_path / "r")
    with pytest.raises(V1.RegistroNoFijado):
        _comprobar(r, codigo=None)


# ================================================== V-12: validacion

def test_variante_con_un_supuesto_en_null_es_un_error():
    cfg = _yaml_fijado()
    cfg["variantes"].append({"nombre": "S4_nulo", "cambios": {"S4_ddof": None}})
    with pytest.raises(ValueError, match="en null"):
        V1.validar_config(cfg)


@pytest.mark.parametrize("valor", [float("nan"), float("inf")])
def test_tolerancia_no_finita_es_un_error(valor):
    cfg = _yaml_fijado()
    cfg["tolerancia"]["far_max_pp"] = valor
    with pytest.raises(ValueError, match="finito"):
        V1.validar_config(cfg)


def test_v1_con_el_yaml_no_valido_se_niega_sin_leer_nada(tmp_path, monkeypatch, capsys):
    cfg = _yaml_fijado()
    cfg["tolerancia"]["far_max_pp"] = float("nan")
    ruta = tmp_path / "configs" / "yin2012.yaml"
    ruta.parent.mkdir()
    ruta.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    leidos = []
    monkeypatch.setattr(data, "load_classic", lambda *a, **k: leidos.append(a))
    assert V1.ejecutar(["--config", str(ruta), "--root", str(tmp_path), "--out", str(tmp_path / "o")]) == 2
    assert "no es valido" in capsys.readouterr().err and leidos == []


def test_fecha_imposible_en_la_fila_se_niega_sin_reventar(tmp_path):
    registro = tmp_path / "decisiones.md"
    registro.write_text("| 2026-02-30 | Yin [V1-yin huella=0123456789ab] | Pablo | prueba |\n", encoding="utf-8")
    with pytest.raises(V1.RegistroNoFijado, match="que exista"):
        V1.buscar_fila_registro(registro, "0123456789ab")


@pytest.mark.parametrize("texto", [
    "<!--\n| 2026-01-01 | Yin [V1-yin huella=0123456789ab] | Pablo | x |\n-->\n",
    "<!-- | 2026-01-01 | Yin [V1-yin huella=0123456789ab] | Pablo | x | -->\n",
    "```\n| 2026-01-01 | Yin [V1-yin huella=0123456789ab] | Pablo | x |\n```\n",
    "~~~\n| 2026-01-01 | Yin [V1-yin huella=0123456789ab] | Pablo | x |\n~~~\n",
    "    | 2026-01-01 | Yin [V1-yin huella=0123456789ab] | Pablo | x |\n",
])
def test_fila_escondida_o_en_un_ejemplo_no_cuenta(tmp_path, texto):
    registro = tmp_path / "decisiones.md"
    registro.write_text(texto, encoding="utf-8")
    with pytest.raises(V1.RegistroNoFijado, match="Ninguna fila"):
        V1.buscar_fila_registro(registro, "0123456789ab")


def test_fila_visible_tras_un_comentario_si_cuenta(tmp_path):
    registro = tmp_path / "decisiones.md"
    registro.write_text("<!-- nota -->\n| Fecha | Decision |\n|---|---|\n"
                        "| 2026-01-01 | Yin [V1-yin huella=0123456789ab] | Pablo | x |\n", encoding="utf-8")
    assert V1.buscar_fila_registro(registro, "0123456789ab")[0] == "2026-01-01"


@pytest.mark.parametrize("fecha", ["2026-02-30", "2999-01-01"])
def test_transcripcion_verificada_con_fecha_imposible_o_futura_queda_pendiente(fecha):
    cfg = _yaml_fijado()
    cfg["referencia"]["transcripcion_verificada"] = fecha
    assert any(p.startswith("referencia.transcripcion_verificada") for p in V1.pendientes(cfg))


def test_solo_limites_se_niega_con_un_supuesto_del_modelo_en_null(tmp_path, capsys):
    cfg = yaml.safe_load(YIN_YAML.read_text(encoding="utf-8"))
    cfg["supuestos"]["S4_ddof"] = None
    carpeta = tmp_path / "configs"
    (carpeta / "referencias").mkdir(parents=True)
    for f in REFERENCIAS.glob("*.csv"):
        shutil.copy(f, carpeta / "referencias" / f.name)
    (carpeta / "yin2012.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    datos = tmp_path / "datos"
    datos.mkdir()
    np.savetxt(datos / "d00.dat", np.random.default_rng(0).normal(size=(500, 52)), fmt="%.6f")
    codigo = V1.ejecutar(["--config", str(carpeta / "yin2012.yaml"), "--root", str(datos),
                          "--out", str(tmp_path / "o"), "--solo-limites"])
    assert codigo == 2 and "S4_ddof" in capsys.readouterr().err


# ================================================== V-13: carpeta personal

def test_carpeta_personal_oculta_con_cualquier_forma():
    casa = Path.home()
    variantes = [str(casa) + os.sep + "datos", casa.as_posix() + "/datos"]
    if os.name == "nt" and casa.as_posix()[1:2] == ":":
        unidad, cuerpo = casa.as_posix()[0], casa.as_posix()[2:]
        variantes += [str(casa).lower() + "\\datos", unidad.lower() + ":" + cuerpo + "/datos",
                      "/" + unidad.lower() + cuerpo + "/datos", str(casa).upper() + "\\datos"]
    for texto in variantes:
        oculto = provenance._sin_casa(texto)
        assert oculto.startswith("~") and casa.name.lower() not in oculto.lower(), texto


def test_otra_carpeta_que_empieza_igual_no_se_toca():
    casa = Path.home().as_posix()
    assert provenance._sin_casa(casa + "2/datos") == casa + "2/datos"


# ================================================== V-14: una vez por ejecucion

def test_write_provenance_usa_el_contexto_dado_sin_recalcular(tmp_path, monkeypatch):
    def prohibido(*a, **k):
        raise AssertionError("no se debe recalcular")

    monkeypatch.setattr(provenance, "git_info", prohibido)
    monkeypatch.setattr(provenance, "_maquina", prohibido)
    csv = tmp_path / "r.csv"
    csv.write_text("a\n1\n", encoding="utf-8")
    contexto = {"git": {"commit": "abc", "dirty": False}, "machine": {"cpu_model": "x"}}
    datos = provenance.read_provenance(provenance.write_provenance(csv, config={}, **contexto))
    assert datos["git"]["commit"] == "abc" and datos["machine"] == {"cpu_model": "x"}


def test_run_context_sin_maquina(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "_maquina", lambda: pytest.fail("no se pidio la maquina"))
    contexto = provenance.run_context(git_path=tmp_path, machine=False)
    assert contexto["machine"] is False and "commit" in contexto["git"]


def _tep_sintetico(carpeta: Path) -> Path:
    rng = np.random.default_rng(0)
    cargas = rng.normal(size=(52, 5))
    carpeta.mkdir()
    np.savetxt(carpeta / "d00.dat", rng.normal(size=(500, 5)) @ cargas.T + 0.5 * rng.normal(size=(500, 52)),
               fmt="%.6f")
    for f in range(22):
        X = rng.normal(size=(960, 5)) @ cargas.T + 0.5 * rng.normal(size=(960, 52))
        np.savetxt(carpeta / f"d{f:02d}_te.dat", X, fmt="%.6f")
    return carpeta


def test_a1_calcula_git_y_maquina_una_vez(tmp_path, monkeypatch):
    llamadas = {"git": 0, "maquina": 0}
    original = provenance.git_info

    def git_contado(*a, **k):
        llamadas["git"] += 1
        return original(*a, **k)

    def maquina_contada():
        llamadas["maquina"] += 1
        return {"cpu_model": "prueba"}

    monkeypatch.setattr(provenance, "git_info", git_contado)
    monkeypatch.setattr(provenance, "_maquina", maquina_contada)
    out = tmp_path / "a1"
    A1.ejecutar(["--config", str(BASELINE), "--root", str(_tep_sintetico(tmp_path / "datos")), "--out", str(out)])
    assert llamadas == {"git": 1, "maquina": 1}
    jsons = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.provenance.json"))]
    assert len(jsons) == 3 and all(j["machine"] == {"cpu_model": "prueba"} for j in jsons)


# ================================================== V-15: copia vieja del yaml

def test_copia_vieja_del_yaml_en_el_directorio_de_trabajo_avisa(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    viejo = BASELINE.read_text(encoding="utf-8").replace("variance: 0.85", "variance: 0.90")
    (tmp_path / "configs" / "baseline.yaml").write_text(viejo, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    with pytest.warns(UserWarning, match="no es el de la version instalada"):
        assert config.load_baseline()["pca"]["variance"] == 0.90


def test_copia_igual_del_yaml_no_avisa(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    texto = config.packaged_baseline().read_text(encoding="utf-8")
    (tmp_path / "configs" / "baseline.yaml").write_bytes(texto.replace("\n", "\r\n").encode("utf-8"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.CONFIG_ENV, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert config.load_baseline(with_info=True)[1]["source"] == "cwd"


def test_ruta_explicita_no_avisa(tmp_path):
    otra = tmp_path / "otra.yaml"
    otra.write_text("pca:\n  variance: 0.5\n", encoding="utf-8")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert config.load_baseline(otra)["pca"]["variance"] == 0.5


# ================================================== remisiones y README

def test_yaml_y_docstring_de_v1_remiten_a_docs_verificacion_yin():
    assert (RAIZ / "docs" / "verificacion_yin.md").is_file()
    assert "docs/verificacion_yin.md" in YIN_YAML.read_text(encoding="utf-8")
    assert "docs/verificacion_yin.md" in V1.__doc__
    assert "especificacion de\nref_yin" not in V1.__doc__


def test_readme_de_resultados_no_cita_el_csv_retirado():
    texto = (RAIZ / "results" / "summary" / "README.md").read_text(encoding="utf-8")
    assert "B1_clasificadores_clasico" not in texto
    assert not (RAIZ / "results" / "summary" / "B1_clasificadores_clasico.csv").exists()


def test_v1_limites_versionado_sigue_igual_con_el_codigo_corregido(tmp_path):
    """--solo-limites con el d00.dat real da el mismo CSV que el versionado (si estan los datos)."""
    clasico = Path(os.environ.get("TFM_CLASSIC_DIR", RAIZ / "data" / "tep_classic"))
    if not (clasico / "d00.dat").exists():
        pytest.skip("sin d00.dat del TEP clasico")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert V1.ejecutar(["--config", str(YIN_YAML), "--root", str(clasico), "--out", str(tmp_path),
                            "--solo-limites"]) == 0
    nuevo = pd.read_csv(tmp_path / V1.SALIDA_LIMITES)
    versionado = pd.read_csv(RAIZ / "results" / "summary" / V1.SALIDA_LIMITES)
    pd.testing.assert_frame_equal(nuevo, versionado)
