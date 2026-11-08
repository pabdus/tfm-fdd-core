"""Pruebas de la tabla de ventajas publicadas (hipotesis.tabla_ventajas_publicadas).

La tabla se fijo el 2026-10-26 (filas del 2026-10-15 y del 2026-10-26 en
docs/decisiones.md) y la genera construir_tabla.py, fuera del repositorio. Estas
pruebas comprueban lo que el codigo del paquete necesita de ella:

- la ruta del yaml existe y la tabla tiene las columnas de
  stats.PUBLISHED_TABLE_COLUMNS, 72 filas, todas dentro del articulo y en los
  fallos 3, 9 y 15 (fallos_dificiles);
- la regla de la columna metric: nombres del paquete (fdr, far,
  exceso_far_gemela) solo en las filas de la misma metrica, y nombres propios en
  las de Golyadkin et al. (2023), para que la funcion no las contraste;
- la fila comparadora del parrafo 349 (papel_349): tres filas de Diallo et al.
  (2025), el autoencoder frente al PCA con SPE y umbral tradicional, en exceso,
  coherentes con hipotesis.baselines, con sus cifras y en VERIFICAR; y nueve de
  sensibilidad, sin filas repetidas;
- la nota de las dos FAR, solo en las filas de exceso de Diallo;
- stats.published_advantage_test contrasta la comparadora con la metrica
  exceso_far_gemela, sin tolerancia (una ventaja medida igual a la publicada
  queda "inconclusa"), y devuelve la diferencia con su intervalo; y se niega
  con las filas de Golyadkin y con las de Lomov et al. (2021);
- la tabla de citadas no contrastables y su LEEME estan en el repositorio.

No leen datos del TEP: las ventajas medidas son sinteticas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfmfdd import config, stats

RAIZ = Path(__file__).resolve().parents[1]
BASELINE = RAIZ / "configs" / "baseline.yaml"
REFERENCIAS = RAIZ / "configs" / "referencias"
CITADAS = REFERENCIAS / "ventajas_citadas_no_contrastables.csv"
LEEME = REFERENCIAS / "LEEME_ventajas_publicadas.md"

DIALLO = "Diallo et al. (2025)"
GOLYADKIN = "Golyadkin et al. (2023)"
LOMOV = "Lomov et al. (2021)"
METRIC_PAQUETE = {"fdr_cruda": "fdr", "far": "far", "exceso_fdr_far": "exceso_far_gemela"}
METRIC_GOLYADKIN = {"fdr_cruda": "tpr_diagnostico", "far": "fpr_diagnostico",
                    "exceso_fdr_far": "exceso_tpr_fpr_diagnostico"}
COMPARADORA = ("AE (PyOD) [KDE umbral tradicional]", "PCA (12 CP) [SPE umbral tradicional]")
SENSIBILIDAD = {("AE (PyOD) [KDE umbral tradicional]", "PCA (12 CP) [T2 umbral tradicional]"),
                ("AE (PyOD) [conformal marginal]", "PCA (12 CP) [SPE conformal marginal]"),
                ("AE (PyOD) [conformal marginal]", "PCA (12 CP) [T2 conformal marginal]")}
# Cifras de la comparadora en exceso (fraccion), por fallo: acta del 26/10, seccion 3
# (0,08, 0,12 y 0,24 puntos). Si construir_tabla.py las cambia, esta prueba avisa.
VALORES_COMPARADORA = {3: 0.0008, 9: 0.0012, 15: 0.0024}
NOTA_DOS_FAR = "El exceso de este trabajo resta la FAR de la gemela normal"
CLAVE_FILA = ["article", "method", "baseline", "dataset", "fault", "metric"]


def _ruta_tabla() -> Path:
    cfg = config.load_baseline(BASELINE)
    return RAIZ / config.get(cfg, "hipotesis.tabla_ventajas_publicadas")


def _tabla() -> pd.DataFrame:
    return pd.read_csv(_ruta_tabla(), comment="#")


def test_tabla_fijada_en_el_yaml_y_en_su_copia():
    """La clave ya no esta en null, la copia empaquetada dice lo mismo y el fichero existe."""
    cfg = config.load_baseline(BASELINE)
    ruta = config.get(cfg, "hipotesis.tabla_ventajas_publicadas")
    assert ruta == "configs/referencias/ventajas_publicadas.csv"
    copia = config.load_yaml(RAIZ / "tfmfdd" / "configs" / "baseline.yaml")
    assert config.get(copia, "hipotesis.tabla_ventajas_publicadas") == ruta
    assert "hipotesis.tabla_ventajas_publicadas" not in config.pending(cfg)
    assert (RAIZ / ruta).is_file()


def test_tabla_columnas_filas_y_fallos():
    """Columnas obligatorias de la funcion, 72 filas, dentro del articulo, fallos 3, 9 y 15."""
    tabla = _tabla()
    faltan = [c for c in stats.PUBLISHED_TABLE_COLUMNS if c not in tabla.columns]
    assert not faltan, faltan
    assert "papel_349" in tabla.columns and "marca_metrica" in tabla.columns
    assert len(tabla) == 72
    assert set(tabla["comparison"]) == {stats.COMPARISON_WITHIN}
    cfg = config.load_baseline(BASELINE)
    assert sorted(set(tabla["fault"])) == sorted(config.get(cfg, "fallos_dificiles")) == [3, 9, 15]
    assert set(tabla["article"]) == {DIALLO, GOLYADKIN}
    assert np.isfinite(tabla["value"].astype(float)).all()
    assert not tabla.duplicated(CLAVE_FILA).any()


def test_regla_de_la_columna_metric():
    """Nombres del paquete solo en la misma metrica; Golyadkin, con nombres propios."""
    tabla = _tabla()
    assert set(tabla["marca_metrica"]) == {"misma_metrica", "otra_metrica"}
    misma = tabla[tabla["marca_metrica"] == "misma_metrica"]
    otra = tabla[tabla["marca_metrica"] == "otra_metrica"]
    assert list(misma["metric"]) == [METRIC_PAQUETE[m] for m in misma["metrica"]]
    assert set(otra["article"]) == {GOLYADKIN}
    assert set(tabla.loc[tabla["article"] == GOLYADKIN, "marca_metrica"]) == {"otra_metrica"}
    assert list(otra["metric"]) == [METRIC_GOLYADKIN[m] for m in otra["metrica"]]
    assert not set(otra["metric"]) & set(METRIC_PAQUETE.values())


def test_fila_comparadora_del_349():
    """Tres filas comparadoras (una por fallo) y nueve de sensibilidad, todas de Diallo en exceso."""
    tabla = _tabla()
    papel = tabla["papel_349"].fillna("")
    assert set(papel) == {"comparadora", "sensibilidad", ""}
    comp = tabla[papel == "comparadora"]
    assert len(comp) == 3
    assert sorted(comp["fault"]) == [3, 9, 15]
    assert set(comp["article"]) == {DIALLO}
    assert set(comp["metric"]) == {"exceso_far_gemela"}
    assert set(zip(comp["method"], comp["baseline"])) == {COMPARADORA}
    assert dict(zip(comp["fault"], comp["value"])) == pytest.approx(VALORES_COMPARADORA)
    sens = tabla[papel == "sensibilidad"]
    assert len(sens) == 9
    assert set(sens["article"]) == {DIALLO} and set(sens["metric"]) == {"exceso_far_gemela"}
    assert set(zip(sens["method"], sens["baseline"])) == SENSIBILIDAD
    for par in SENSIBILIDAD:
        filas = sens[(sens["method"] == par[0]) & (sens["baseline"] == par[1])]
        assert sorted(filas["fault"]) == [3, 9, 15], par
    # Las demas filas de exceso de Diallo en la misma metrica son todas comparadora o sensibilidad
    exceso_diallo = tabla[(tabla["article"] == DIALLO) & (tabla["metric"] == "exceso_far_gemela")]
    assert len(exceso_diallo) == 12 and set(exceso_diallo["papel_349"]) == {"comparadora", "sensibilidad"}
    # El exceso de Diallo es derivado (alpha 1 % inferido): sigue en VERIFICAR (acta, seccion 1)
    assert set(exceso_diallo["verified"]) == {"VERIFICAR"}
    assert set(exceso_diallo["origen_metrica"]) == {"derivada"}


def test_nota_de_las_dos_far_y_notas_sin_repetir():
    """La nota de las dos FAR va en las 12 filas de exceso de Diallo y en ninguna otra."""
    tabla = _tabla()
    nota = tabla["nota"].fillna("")
    con_nota = nota.str.contains(NOTA_DOS_FAR, regex=False)
    exceso_diallo = (tabla["article"] == DIALLO) & (tabla["metric"] == "exceso_far_gemela")
    assert (con_nota == exceso_diallo).all()
    for texto in nota:
        trozos = [x.strip() for x in texto.split("; ") if x.strip()]
        assert len(trozos) == len(set(trozos)), texto


def test_comparadora_coherente_con_los_baselines_del_yaml():
    """El baseline de la comparadora lleva el estadistico de "PCA" en la familia (DIS-10).

    Diallo no tiene DPCA: la condicion del parrafo 349 solo existe para el PCA sin
    retardos. Si hipotesis.baselines cambiara de estadistico, la comparadora
    tendria que cambiar con el (regla del 15/10; fila del 18/10).
    """
    cfg = config.load_baseline(BASELINE)
    pca = [b for b in config.get(cfg, "hipotesis.baselines") if b["nombre"] == "pca"]
    assert len(pca) == 1 and pca[0]["lags"] == 0
    etiqueta = f"[{pca[0]['estadistico'].upper()} umbral tradicional]"
    comp = _tabla().query("papel_349 == 'comparadora'")
    assert all(b.endswith(etiqueta) for b in comp["baseline"])


def test_published_advantage_test_con_la_comparadora():
    """Con cada fila comparadora, en exceso_far_gemela: los tres desenlaces y la diferencia.

    Muy por debajo de la publicada sale "menor" con una diferencia positiva y su
    intervalo por encima de cero; muy por encima, "igual o mayor" con el
    intervalo por debajo de cero; y, sin tolerancia, una ventaja medida igual a
    la publicada queda "inconclusa" (acta del 26/10, seccion 3).
    """
    rng = np.random.default_rng(20261026)
    comp = _tabla().query("papel_349 == 'comparadora'")
    for _, fila in comp.iterrows():
        publicada = float(fila["value"])
        debajo = stats.published_advantage_test(publicada - 0.02 + rng.normal(0, 0.002, 20), fila,
                                                metric="exceso_far_gemela")
        assert debajo["metric"] == "exceso_far_gemela" and debajo["n"] == 20
        assert debajo["published"] == pytest.approx(publicada)
        assert debajo["outcome"] == "menor que la publicada"
        assert 0 < debajo["gap_ci_low"] <= debajo["published_minus_measured"] <= debajo["gap_ci_high"]
        assert debajo["published_minus_measured"] == pytest.approx(0.02, abs=0.003)
        encima = stats.published_advantage_test(publicada + 0.02 + rng.normal(0, 0.002, 20), fila,
                                                metric="exceso_far_gemela")
        assert encima["outcome"] == "igual o mayor que la publicada"
        assert encima["gap_ci_low"] <= encima["published_minus_measured"] <= encima["gap_ci_high"] < 0
        # simetrica alrededor de la publicada: Hodges-Lehmann = publicada
        igual = stats.published_advantage_test(publicada + np.linspace(-0.004, 0.004, 21), fila,
                                               metric="exceso_far_gemela")
        assert igual["outcome"] == "inconclusa"
        assert igual["published_minus_measured"] == pytest.approx(0.0, abs=1e-12)
        assert igual["gap_ci_low"] < 0 < igual["gap_ci_high"]


@pytest.mark.parametrize("metric", sorted(METRIC_PAQUETE.values()))
def test_published_advantage_test_se_niega_con_golyadkin(metric):
    """Con cualquier metrica del paquete, las filas de Golyadkin dan ValueError (otra metrica)."""
    gol = _tabla().query("article == @GOLYADKIN")
    assert len(gol) == 36
    for _, fila in gol.iterrows():
        with pytest.raises(ValueError, match="misma metrica"):
            stats.published_advantage_test(np.zeros(10) + 0.01, fila, metric=metric)


def test_citadas_no_contrastables_y_leeme():
    """Las tres filas de Lomov (entre articulos, otra metrica) y el LEEME estan en el repositorio."""
    citadas = pd.read_csv(CITADAS, comment="#")
    assert len(citadas) == 3
    assert set(citadas["article"]) == {LOMOV}
    assert set(citadas["metric"]) == {"tpr_clase_clasificador"}
    assert set(citadas["comparison"]) == {stats.COMPARISON_BETWEEN}
    for _, fila in citadas.iterrows():
        for metric in METRIC_PAQUETE.values():
            with pytest.raises(ValueError, match="misma metrica"):
                stats.published_advantage_test(np.zeros(10) + 0.01, fila, metric=metric)
    texto = LEEME.read_text(encoding="utf-8")
    assert "papel_349" in texto and "comparadora" in texto


def test_registro_fija_la_tabla_el_2026_10_26():
    """docs/decisiones.md tiene las filas del 2026-10-26 que fijan la tabla y su uso."""
    texto = (RAIZ / "docs" / "decisiones.md").read_text(encoding="utf-8")
    filas = [l for l in texto.splitlines() if l.startswith("| 2026-10-26 |")]
    assert any(l.startswith("| 2026-10-26 | Nombres de la columna metric") and "papel_349" in l for l in filas)
    assert any(l.startswith("| 2026-10-26 | Baseline lineal clasico de referencia") for l in filas)
    assert any(l.startswith("| 2026-10-26 | Tolerancia alrededor de la cifra publicada: ninguna") for l in filas)
