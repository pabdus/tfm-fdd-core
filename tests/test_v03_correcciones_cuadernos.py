"""Correcciones del 12/10 en los cuadernos (revision independiente; ronda fix_cuadernos).

Estas pruebas leen las fuentes de los cuadernos como JSON (no ejecutan ninguno) y comprueban:

- que leen el yaml con `tfmfdd.config.load_baseline()` y no lo abren a mano, y que el alfa de los
  contrastes no se declara a mano (sale de `contrastes.alpha_contrastes`);
- que las llamadas a `metrics.evaluate_run` y `metrics.summary` pasan los argumentos del protocolo que
  salen del yaml (`DETECCION`, con `min_fdr` y `delay_convention`; `min_fdr=MIN_FDR`);
- que toda celda que escribe `GridSearchCV(` lleva `error_score="raise"` (hallazgo V-09), y que con
  `Scaler(normal_label=0)` un pliegue sin la clase normal hace fallar la busqueda;
- que las listas de reglas del B00 y del C00 tienen la numeracion y los titulos de
  `docs/reglas_de_medida.md`, y que ninguna cita "regla N" de los cuadernos se sale de esa numeracion;
- que `notebooks/README.md` y los cuadernos ya no dicen que la version es la 0.1.0 ni que el alfa de
  los contrastes falta en el yaml.

Si cambia un titulo de `docs/reglas_de_medida.md`, hay que cambiar el mismo titulo en las listas de
reglas del B00 (celda `ce4b61c1`) y del C00 (celda `c00-01`).
"""

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
CUADERNOS = RAIZ / "notebooks"
REGLAS = RAIZ / "docs" / "reglas_de_medida.md"
NOMBRES = [
    "01_primeros_pasos",
    "02_arranque_diagnostico",
    "03_arranque_deeplearning",
    "B00_punto_de_partida_diagnostico",
    "C00_punto_de_partida_deeplearning",
]
LISTAS_DE_REGLAS = [
    ("B00_punto_de_partida_diagnostico", "ce4b61c1"),
    ("C00_punto_de_partida_deeplearning", "c00-01"),
]


def _celdas(nombre):
    """(id, tipo, fuente) de cada celda del cuaderno."""
    nb = json.loads((CUADERNOS / f"{nombre}.ipynb").read_text(encoding="utf-8"))
    celdas = []
    for c in nb["cells"]:
        fuente = c["source"]
        celdas.append((c.get("id"), c["cell_type"], "".join(fuente) if isinstance(fuente, list) else fuente))
    return celdas


def _codigo(nombre):
    return [(cid, fuente) for cid, tipo, fuente in _celdas(nombre) if tipo == "code"]


def _titulos_del_documento():
    """{numero: titulo} de las secciones '## N. Titulo' de docs/reglas_de_medida.md."""
    titulos = {}
    for linea in REGLAS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^## (\d+)\. (.+?)\s*$", linea)
        if m:
            titulos[int(m.group(1))] = m.group(2)
    return titulos


@pytest.mark.parametrize("nombre", NOMBRES)
def test_leen_el_yaml_con_tfmfdd_config(nombre):
    codigo = "\n".join(fuente for _, fuente in _codigo(nombre))
    assert "config.load_baseline(" in codigo, f"{nombre}: no lee el yaml con tfmfdd.config.load_baseline()"
    for prohibido in ("yaml.safe_load", "import yaml", 'open("configs/baseline.yaml"'):
        assert prohibido not in codigo, f"{nombre}: lee el yaml a mano ({prohibido})"
    assert not re.search(r"ALPHA_CONTRASTES?\s*=\s*[0-9]", codigo), \
        f"{nombre}: el alfa de los contrastes se declara a mano; sale de contrastes.alpha_contrastes"


@pytest.mark.parametrize("nombre", NOMBRES)
def test_el_alfa_de_los_contrastes_sale_del_yaml(nombre):
    codigo = "\n".join(fuente for _, fuente in _codigo(nombre))
    if "ALPHA_CONTRASTES" in codigo:
        assert re.search(r'ALPHA_CONTRASTES\s*=\s*config\.require\(\w+, "contrastes\.alpha_contrastes"\)', codigo), \
            f"{nombre}: ALPHA_CONTRASTES no se lee de contrastes.alpha_contrastes"


@pytest.mark.parametrize("nombre", NOMBRES)
def test_evaluate_run_y_summary_con_los_argumentos_del_yaml(nombre):
    for cid, fuente in _codigo(nombre):
        for m in re.finditer(r"evaluate_run\(", fuente):
            llamada = fuente[m.start(): fuente.index(")", m.end()) + 1]
            assert "**DETECCION" in llamada, f"{nombre}, celda {cid}: {llamada}"
        for m in re.finditer(r"metrics\.summary\(", fuente):
            fin = fuente.find("\n", m.end())
            linea = fuente[m.start(): fin if fin >= 0 else len(fuente)]
            assert "min_fdr=MIN_FDR" in linea, f"{nombre}, celda {cid}: {linea}"


@pytest.mark.parametrize("nombre", NOMBRES)
def test_deteccion_lee_min_fdr_y_convencion_del_yaml(nombre):
    codigo = "\n".join(fuente for _, fuente in _codigo(nombre))
    if "**DETECCION" in codigo:
        assert '"protocolo.min_fdr"' in codigo and '"protocolo.retardo_convencion"' in codigo, \
            f"{nombre}: DETECCION no lee protocolo.min_fdr y protocolo.retardo_convencion"


@pytest.mark.parametrize("nombre", NOMBRES)
def test_gridsearchcv_con_error_score_raise(nombre):
    for cid, _, fuente in _celdas(nombre):
        if "GridSearchCV(" in fuente:
            assert 'error_score="raise"' in fuente, f"{nombre}, celda {cid}: GridSearchCV sin error_score=\"raise\""


def test_las_busquedas_de_b00_y_c00_lo_dicen():
    for nombre in ("B00_punto_de_partida_diagnostico", "C00_punto_de_partida_deeplearning"):
        assert any("GridSearchCV(" in fuente for _, _, fuente in _celdas(nombre)), nombre


def test_v09_un_pliegue_sin_la_clase_normal_hace_fallar_la_busqueda():
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline

    from tfmfdd.preprocessing import Scaler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 4))
    y = np.repeat([0, 1, 2], 30)
    grupos = np.repeat([1, 2, 3], 30)          # un grupo por clase: un pliegue entrena sin la clase 0
    tuberia = Pipeline([("escalado", Scaler(normal_label=0)), ("knn", KNeighborsClassifier())])
    busqueda = GridSearchCV(tuberia, {"knn__n_neighbors": [1, 3]}, cv=GroupKFold(n_splits=3),
                            error_score="raise")
    with pytest.raises(ValueError):
        busqueda.fit(X, y, groups=grupos)


def test_el_documento_tiene_las_reglas_seguidas():
    titulos = _titulos_del_documento()
    assert len(titulos) >= 14
    assert sorted(titulos) == list(range(1, len(titulos) + 1))


@pytest.mark.parametrize("nombre, celda", LISTAS_DE_REGLAS)
def test_lista_de_reglas_con_la_numeracion_y_los_titulos_del_documento(nombre, celda):
    fuente = next(f for cid, _, f in _celdas(nombre) if cid == celda)
    lista = {}
    for linea in fuente.splitlines():
        m = re.match(r"^(\d+)\. \*\*(.+?)\*\*", linea)
        if m:
            lista[int(m.group(1))] = m.group(2).rstrip(".:")
    assert lista == _titulos_del_documento()


@pytest.mark.parametrize("nombre", NOMBRES)
def test_las_reglas_citadas_existen(nombre):
    n = len(_titulos_del_documento())
    for cid, _, fuente in _celdas(nombre):
        for m in re.finditer(r"[Rr]eglas? \d+(?:(?:, | y | a )\d+)*", fuente):
            numeros = [int(x) for x in re.findall(r"\d+", m.group(0))]
            assert all(1 <= x <= n for x in numeros), f"{nombre}, celda {cid}: {m.group(0)}"


@pytest.mark.parametrize("nombre", NOMBRES)
def test_sin_version_ni_alfa_desfasados(nombre):
    for cid, _, fuente in _celdas(nombre):
        assert not re.search(r"(todavía dice|sigue diciendo) 0\.1\.0", fuente), f"{nombre}, celda {cid}"
        for frase in ("aún no está en el yaml", "sigue sin estar en el yaml", "no viaja con el paquete"):
            assert frase not in fuente, f"{nombre}, celda {cid}: {frase}"


def test_readme_de_los_cuadernos():
    texto = (CUADERNOS / "README.md").read_text(encoding="utf-8")
    assert "0.3.0" in texto and "0.1.0" not in texto
    assert "contrastes.alpha_contrastes" in texto
    assert "hasta que esté en el yaml" not in texto
    assert 'error_score="raise"' in texto
