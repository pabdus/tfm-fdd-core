"""Pruebas de los ultimos arreglos de la regla de orden de V1 (segunda revision, W-01 a W-03).

- W-01: los bloques de codigo de docs/decisiones.md siguen las reglas de las vallas de
  CommonMark (la de cierre, del mismo caracter, al menos tan larga como la de apertura
  y sin texto detras), asi que una fila dentro de un ejemplo de codigo no cuenta.
- W-02: la fila tiene que estar en docs/decisiones.md, el registro del repositorio
  (otro fichero solo con --permitir-otro-registro), y la procedencia dice cual fue.
- W-03: un .pyc "unchecked-hash" (o un .pyc sin su .py) en el codigo hace que la
  regla de orden se niegue: Python lo ejecutaria en lugar del fuente confirmado.

Las pruebas de git crean repositorios TEMPORALES de prueba (en la carpeta temporal de
pytest), con commits de un autor ficticio pasado con -c, sin tocar ninguna
configuracion de git; ningun commit en el repositorio del proyecto. Se saltan si git
no esta instalado. Los ficheros del TEP son sinteticos: ninguna prueba lee un fichero
de prueba real.
"""

from __future__ import annotations

import importlib.util
import json
import os
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tfmfdd import data, provenance

RAIZ = Path(__file__).resolve().parents[1]
YIN_YAML = RAIZ / "configs" / "yin2012.yaml"
REFERENCIAS = RAIZ / "configs" / "referencias"

con_git = pytest.mark.skipif(shutil.which("git") is None, reason="sin git")


def _cargar_v1():
    ruta = RAIZ / "experiments" / "V1_verificacion_yin.py"
    spec = importlib.util.spec_from_file_location("V1_verificacion_yin_final", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


V1 = _cargar_v1()

GIT = ["git", "-c", "user.name=prueba", "-c", "user.email=prueba@example.invalid",
       "-c", "commit.gpgsign=false", "-c", "core.autocrlf=false"]
IGNORAR = "__pycache__/\n*.py[cod]\n"


def _git(repo: Path, *args):
    entorno = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    return subprocess.run(GIT + list(args), cwd=repo, capture_output=True, text=True, check=True, env=entorno)


def _repo(carpeta: Path, ficheros: dict) -> Path:
    for rel, texto in ficheros.items():
        p = carpeta / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(texto, encoding="utf-8")
    _git(carpeta, "init", "-q")
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
    cfg["variantes"] = []                     # menos modelos: la prueba es de la regla de orden
    return cfg


def _repo_verificacion(carpeta: Path, *, fila_en: str = "docs/decisiones.md") -> dict:
    """Repositorio temporal con el yaml, las referencias, el registro (con la fila de la
    huella en `fila_en`) y una copia del codigo (script y paquete), todo confirmado."""
    cfg = _yaml_fijado()
    (carpeta / "configs" / "referencias").mkdir(parents=True)
    for f in REFERENCIAS.glob("*.csv"):
        shutil.copy(f, carpeta / "configs" / "referencias" / f.name)
    ruta_yaml = carpeta / "configs" / "yin2012.yaml"
    ruta_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    actual = V1.huella(cfg, V1.rutas_referencia(cfg, ruta_yaml.parent))
    cabecera = "| Fecha | Decision | Quien | Motivo |\n|---|---|---|---|\n"
    fila = f"| 2026-01-01 | Yin [V1-yin huella={actual}] | Pablo | prueba |\n"
    ficheros = {"docs/decisiones.md": cabecera + (fila if fila_en == "docs/decisiones.md" else "")}
    if fila_en != "docs/decisiones.md":
        ficheros[fila_en] = cabecera + fila
    for rel, texto in ficheros.items():
        (carpeta / rel).parent.mkdir(parents=True, exist_ok=True)
        (carpeta / rel).write_text(texto, encoding="utf-8")
    (carpeta / "experiments").mkdir()
    shutil.copy(RAIZ / "experiments" / "V1_verificacion_yin.py", carpeta / "experiments")
    shutil.copytree(RAIZ / "tfmfdd", carpeta / "tfmfdd", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(RAIZ / ".gitignore", carpeta / ".gitignore")
    _repo(carpeta, {})
    return {"cfg": cfg, "yaml": ruta_yaml, "repo": carpeta, "registro": carpeta / fila_en,
            "codigo": [carpeta / "experiments" / "V1_verificacion_yin.py", carpeta / "tfmfdd"]}


def _comprobar(r: dict, decisiones: Path | None = None, **kw):
    return V1.comprobar_registro(r["cfg"], r["yaml"].parent, r["yaml"], decisiones or r["registro"],
                                 codigo=r["codigo"], **kw)


# ================================================ W-01: vallas de CommonMark

FILA = "| 2026-01-01 | Yin [V1-yin huella=abcdefabcdef] | Pablo | prueba |"

CASOS_VALLAS = {
    # los diez casos de la propuesta del revisor (v_propuestas_w01_w03.py)
    "valla de 4 con lineas de 3 dentro (N1)": ("````\n```\n" + FILA + "\n```\n````", False),
    "linea ```python dentro de un bloque (N2)": ("```\n```python\n" + FILA + "\n```", False),
    "bloque normal de 3": ("```\n" + FILA + "\n```", False),
    "bloque de tildes": ("~~~\n" + FILA + "\n~~~", False),
    "comentario de varias lineas": ("<!--\n" + FILA + "\n-->", False),
    "visible tras un bloque cerrado": ("```\nx\n```\n" + FILA, True),
    "visible tras un comentario": ("<!-- x -->\n" + FILA, True),
    "visible sin nada": (FILA, True),
    "con un comentario dentro (visible)": (FILA.replace(" | Pablo", " <!-- nota --> | Pablo"), True),
    "valla de 4 cerrada con 4 y fila despues": ("````\nx\n````\n" + FILA, True),
    # casos de las reglas de CommonMark que no estaban en la propuesta
    "cierre con texto detras no cierra": ("```\n" + FILA + "\n``` fin", False),
    "cierre mas largo si cierra": ("```\nx\n`````\n" + FILA, True),
    "tildes no se cierran con comillas": ("~~~\nx\n```\n" + FILA + "\n~~~", False),
    "bloque sin cerrar llega al final": ("```\nx\n" + FILA, False),
    "valla con comilla en el texto no abre": ("``` a`b\n" + FILA, True),
    "valla sangrada con 4 espacios no abre": ("    ```\n" + FILA, True),
    "cierre sangrado con 4 espacios no cierra": ("```\nx\n    ```\n" + FILA + "\n```", False),
}


@pytest.mark.parametrize("nombre", list(CASOS_VALLAS))
def test_vallas_de_codigo_segun_commonmark(nombre):
    texto, debe_contar = CASOS_VALLAS[nombre]
    cuenta = any(l.startswith("|") and "V1-yin" in l for l in V1.lineas_visibles(texto))
    assert cuenta is debe_contar


def test_fila_en_una_valla_larga_no_fija_la_verificacion(tmp_path):
    registro = tmp_path / "decisiones.md"
    registro.write_text("````\n```\n" + FILA + "\n```\n````\n", encoding="utf-8")
    with pytest.raises(V1.RegistroNoFijado, match="Ninguna fila"):
        V1.buscar_fila_registro(registro, "abcdefabcdef")


# ================================================ W-02: el fichero del registro

def test_otro_fichero_de_registro_se_niega_sin_la_opcion():
    with pytest.raises(V1.RegistroNoFijado, match="docs/decisiones.md"):
        V1.comprobar_fichero_de_registro("notas/borrador.md", permitir_otro_registro=False)
    V1.comprobar_fichero_de_registro("notas/borrador.md", permitir_otro_registro=True)
    V1.comprobar_fichero_de_registro("docs/decisiones.md", permitir_otro_registro=False)


def test_fichero_de_registro_relativo_a_la_raiz(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "decisiones.md").write_text("x", encoding="utf-8")
    assert V1.fichero_de_registro(tmp_path / "docs" / "decisiones.md", tmp_path) == "docs/decisiones.md"


@con_git
def test_registro_estandar_da_el_permiso_y_dice_el_fichero(tmp_path):
    r = _repo_verificacion(tmp_path / "r")
    assert _comprobar(r).registro == "docs/decisiones.md"


@con_git
def test_fila_en_otro_fichero_confirmado_se_niega_salvo_que_se_pida(tmp_path):
    r = _repo_verificacion(tmp_path / "r", fila_en="notas/borrador.md")
    with pytest.raises(V1.RegistroNoFijado, match="permitir-otro-registro"):
        _comprobar(r)
    assert _comprobar(r, permitir_otro_registro=True).registro == "notas/borrador.md"


def _tep_sintetico(carpeta: Path) -> Path:
    rng = np.random.default_rng(0)
    cargas = rng.normal(size=(52, 5))
    carpeta.mkdir()

    def normales(n):
        return rng.normal(size=(n, 5)) @ cargas.T + 0.5 * rng.normal(size=(n, 52))

    np.savetxt(carpeta / "d00.dat", normales(500), fmt="%.6f")
    columnas = [data.VAR_NAMES.index(v) for v in data.YIN_VARS]
    for f in range(22):
        X = normales(960)
        if f not in (0, 3, 9, 15):
            X[160:, columnas[f % 33]] += 8.0
        np.savetxt(carpeta / f"d{f:02d}_te.dat", X, fmt="%.6f")
    return carpeta


@con_git
def test_la_procedencia_dice_que_fichero_de_registro_se_uso(tmp_path, monkeypatch):
    """De punta a punta con un commit de verdad y datos sinteticos: con
    --permitir-otro-registro, la tabla y el .provenance.json dicen el fichero."""
    r = _repo_verificacion(tmp_path / "r", fila_en="notas/borrador.md")
    monkeypatch.setattr(V1, "codigo_en_ejecucion", lambda: r["codigo"])
    out = tmp_path / "out"
    argumentos = ["--config", str(r["yaml"]), "--root", str(_tep_sintetico(tmp_path / "datos")),
                  "--out", str(out), "--decisiones", str(r["registro"])]
    assert V1.ejecutar(argumentos) == 2                      # sin la opcion, se niega
    assert not (out / V1.SALIDA).exists()
    assert V1.ejecutar(argumentos + ["--permitir-otro-registro"]) == 0
    assert set(pd.read_csv(out / V1.SALIDA)["decision_file"]) == {"notas/borrador.md"}
    procedencia = json.loads(provenance.provenance_path(out / V1.SALIDA).read_text(encoding="utf-8"))
    assert procedencia["extra"]["registro"]["fichero"] == "notas/borrador.md"


# ================================================ W-03: bytecode que no es el fuente

@pytest.mark.parametrize("modo, sin_comprobar", [
    (py_compile.PycInvalidationMode.TIMESTAMP, False),
    (py_compile.PycInvalidationMode.CHECKED_HASH, False),
    (py_compile.PycInvalidationMode.UNCHECKED_HASH, True),
])
def test_cabecera_del_pyc(tmp_path, modo, sin_comprobar):
    fuente = tmp_path / "m.py"
    fuente.write_text("X = 1\n", encoding="utf-8")
    pyc = tmp_path / "m.pyc"
    py_compile.compile(str(fuente), cfile=str(pyc), invalidation_mode=modo)
    assert provenance._pyc_sin_comprobar(pyc) is sin_comprobar


def _pyc(fuente: Path, modo, texto: str | None = None) -> Path:
    """Compila `fuente` (o un fuente con `texto`, que despues se devuelve a su contenido) a
    su .pyc de __pycache__, el que usaria esta version de Python."""
    original = fuente.read_bytes()
    if texto is not None:
        fuente.write_text(texto, encoding="utf-8")
    pyc = fuente.parent / "__pycache__" / f"{fuente.stem}.{sys.implementation.cache_tag}.pyc"
    py_compile.compile(str(fuente), cfile=str(pyc), invalidation_mode=modo)
    fuente.write_bytes(original)
    return pyc


@con_git
@pytest.mark.parametrize("modo, cuenta", [
    (py_compile.PycInvalidationMode.UNCHECKED_HASH, True),
    (py_compile.PycInvalidationMode.CHECKED_HASH, False),
    (py_compile.PycInvalidationMode.TIMESTAMP, False),
])
def test_pyc_en_el_paquete(tmp_path, modo, cuenta):
    repo = _repo(tmp_path / "r", {".gitignore": IGNORAR, "pkg/__init__.py": "", "pkg/m.py": "X = 1\n"})
    _pyc(repo / "pkg" / "m.py", modo, texto="X = 2\n")
    problemas = provenance.committed_status([repo / "pkg"])["problems"]
    assert any("unchecked-hash" in p for p in problemas) is cuenta
    assert _git(repo, "status", "--porcelain").stdout == ""          # git no ve nada


@con_git
def test_pyc_sin_su_fuente_en_el_paquete_cuenta(tmp_path):
    repo = _repo(tmp_path / "r", {".gitignore": IGNORAR, "pkg/__init__.py": "", "pkg/m.py": "X = 1\n"})
    suelto = repo / "pkg" / "extra.py"
    suelto.write_text("Y = 1\n", encoding="utf-8")
    py_compile.compile(str(suelto), cfile=str(repo / "pkg" / "extra.pyc"))
    suelto.unlink()                                    # queda extra.pyc sin extra.py: importable
    problemas = provenance.committed_status([repo / "pkg"])["problems"]
    assert any("sin su .py" in p for p in problemas)


@con_git
def test_pyc_unchecked_del_script_cuenta(tmp_path):
    repo = _repo(tmp_path / "r", {".gitignore": IGNORAR, "experiments/s.py": "print(1)\n"})
    _pyc(repo / "experiments" / "s.py", py_compile.PycInvalidationMode.UNCHECKED_HASH, texto="print(2)\n")
    problemas = provenance.committed_status([repo / "experiments" / "s.py"])["problems"]
    assert any("unchecked-hash" in p for p in problemas)


@con_git
def test_regla_de_orden_se_niega_con_un_pyc_unchecked_en_el_paquete(tmp_path):
    """El caso N6 del revisor: el fuente esta confirmado y git no ve nada, pero el .pyc
    que Python usaria es de otro codigo."""
    r = _repo_verificacion(tmp_path / "r")
    _pyc(r["repo"] / "tfmfdd" / "plots.py", py_compile.PycInvalidationMode.UNCHECKED_HASH,
         texto="print('codigo no confirmado')\n")
    with pytest.raises(V1.RegistroNoFijado, match="unchecked-hash"):
        _comprobar(r)
