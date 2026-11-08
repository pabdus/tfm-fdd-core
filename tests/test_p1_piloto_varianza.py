"""Pruebas de experiments/P1_piloto_varianza.py (piloto de la varianza de la ventaja).

Cubren, sin tocar ningun Parquet de Rieth:
- la negativa con las seis claves en null de configs/baseline.yaml, con un
  --provisional incompleto, con uno que quiere pisar una clave con valor, con
  una metrica que el piloto no calcula, con mas normales de las que caben en
  1..datos.runs_desarrollo, con --epocas-max 0 y con una salida provisional en
  results/summary: codigo 2, sin ninguna carga;
- la ejecucion que cuenta (sin --provisional): se niega con limites_protocolo
  en null o sin particion.rieth.normales_parada (fila 29);
- la comprobacion de ids (mayores que el maximo o no mirados) antes de leer;
- el reparto de las normales de train_fault00: uno solo, el del C00;
- los limites (limites_protocolo o, en provisional, pca.*), el nivel de Sidak
  y la carpeta de salida;
- estimar_sigma: desviacion muestral, intervalo y cota superior por chi
  cuadrado, con su cobertura en una simulacion normal;
- el manifiesto: unos bytes que no coinciden paran el piloto antes de leer;
- la ejecucion completa sobre Parquet SINTETICOS (ids 1 a 8) con un yaml
  provisional: CSV por celda y por simulacion, columnas, procedencia y el
  manifiesto, con los limites de desarrollo y con los de limites_protocolo.
  Se salta si PyTorch no esta instalado.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats as sps

from tfmfdd import data, provenance

RAIZ = Path(__file__).resolve().parents[1]
BASELINE = RAIZ / "configs" / "baseline.yaml"


def _cargar_experimento(nombre: str):
    """Importa un script de experiments/ como modulo (no son un paquete)."""
    ruta = RAIZ / "experiments" / f"{nombre}.py"
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


P1 = _cargar_experimento("P1_piloto_varianza")

#: Valores de prueba completos para las seis claves (los de la hoja del 15/10).
PROVISIONAL = {
    "hipotesis": {
        "metrica_ventaja": "exceso_far_gemela",
        "mayor_es_mejor": True,
        "baselines": [{"nombre": "pca", "estadistico": "spe", "lags": 0},
                      {"nombre": "dpca", "estadistico": "spe", "lags": 2}],
    },
    "deteccion": {"ventana_far_normal": [160, None]},
    "particion": {"rieth": {"normales_ajuste": 14, "normales_calibracion": 6}},
}
#: Valores de prueba para los cinco limites_protocolo.* (todos calibrados con las reservadas).
LIMITES = {"t2_method": "empirical", "spe_method": "empirical", "quantile_method": "linear",
           "nivel_por_estadistico": "sidak", "orden_escalado_dpca": "escalar_y_retardar"}


def _yaml(tmp_path: Path, nombre: str, contenido: dict) -> Path:
    ruta = tmp_path / nombre
    ruta.write_text(yaml.safe_dump(contenido, sort_keys=False), encoding="utf-8")
    return ruta


def _cfg_con_las_seis_en_null() -> dict:
    """Copia del yaml real con las seis claves de P1 en null.

    Desde el 2026-10-15 (filas de esa fecha en docs/decisiones.md) el yaml real fija
    tres de ellas (metrica_ventaja, mayor_es_mejor y ventana_far_normal): las
    negativas y el provisional completo se prueban sobre esta copia. Tambien deja en
    null los cinco limites_protocolo.* y quita particion.rieth.normales_parada, para
    que las pruebas no dependan de cuando se fijen.
    """
    cfg = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    cfg["hipotesis"].update(metrica_ventaja=None, mayor_es_mejor=None, baselines=None)
    cfg["deteccion"]["ventana_far_normal"] = None
    cfg["particion"]["rieth"].update(normales_ajuste=None, normales_calibracion=None)
    cfg["particion"]["rieth"].pop("normales_parada", None)
    cfg["limites_protocolo"].update({k: None for k in LIMITES})
    return cfg


def _cfg_decidida(con_limites: bool, con_parada: bool) -> dict:
    """Copia del yaml con las seis claves fijadas en el propio yaml, como en la ejecucion que cuenta."""
    cfg = _cfg_con_las_seis_en_null()
    cfg["hipotesis"].update(PROVISIONAL["hipotesis"])
    cfg["deteccion"].update(PROVISIONAL["deteccion"])
    cfg["particion"]["rieth"].update(normales_ajuste=5, normales_calibracion=3)
    if con_limites:
        cfg["limites_protocolo"].update(LIMITES)
    if con_parada:
        cfg["particion"]["rieth"]["normales_parada"] = 2
    return cfg


def _baseline_en_null(tmp_path: Path) -> Path:
    return _yaml(tmp_path, "baseline_seis_en_null.yaml", _cfg_con_las_seis_en_null())


def _espiar_cargas(monkeypatch) -> list:
    """Sustituye data.load_rieth por un espia que falla: ninguna negativa puede llegar a cargar."""
    llamadas = []

    def falso(fault, split="test", runs=None, root=None):
        llamadas.append((fault, split))
        raise AssertionError("P1 intento cargar un Parquet cuando debia negarse")

    monkeypatch.setattr(data, "load_rieth", falso)
    return llamadas


def _ejecutar(tmp_path, *extra) -> int:
    return P1.ejecutar(["--config", str(_baseline_en_null(tmp_path)), "--root", str(tmp_path / "rieth"),
                        "--out", str(tmp_path / "out"), *extra])


# =============================================================== negativas

def test_p1_se_niega_con_las_seis_claves_en_null(tmp_path, monkeypatch, capsys):
    """Con las seis en null (copia del yaml real): codigo 2, las lista y no carga nada."""
    llamadas = _espiar_cargas(monkeypatch)
    codigo = _ejecutar(tmp_path)
    err = capsys.readouterr().err
    assert codigo == 2 and "no se ejecuta" in err
    for clave in P1.CLAVES_NECESARIAS:
        assert clave in err, clave
    assert llamadas == []
    assert not (tmp_path / "out").exists()


def test_p1_yaml_real_trae_todo_lo_que_pide_la_ejecucion_que_cuenta_desde_el_2026_10_18():
    """Filas del 2026-10-15 y del 2026-10-18: las seis claves, los cinco limites y la parada tienen valor.

    Con el yaml real (sin --provisional), la validacion pasa, los limites salen de
    limites_protocolo y el reparto de las normales cabe en 1..MAXIMO_NORMALES (commit 2b):
    14 de ajuste (10 para entrenar y 4 para parar) y 20 de calibracion, en los ids 1 a 34.
    """
    cfg = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    decisiones = P1.validar_decisiones(cfg)
    assert [b["nombre"] for b in decisiones["baselines"]] == ["pca", "dpca"]
    assert [(b["estadistico"], b["lags"]) for b in decisiones["baselines"]] == [("spe", 0), ("spe", 2)]
    lim = P1.parametros_limites(cfg, provisional=False)
    assert lim["fuente"] != "pca_desarrollo"
    assert {k: lim[k] for k in LIMITES} == LIMITES
    n_parada, fuente = P1.resolver_normales_parada(cfg, decisiones["normales_ajuste"],
                                                   decisiones["normales_calibracion"], provisional=False)
    assert (decisiones["normales_ajuste"], n_parada, decisiones["normales_calibracion"]) == (14, 4, 20)
    assert fuente == P1.CLAVE_PARADA
    ids = P1.repartir_normales(14, 20, n_parada=4, semilla=42, maximo=P1.MAXIMO_NORMALES)
    assert (len(ids["tr"]), len(ids["es"]), len(ids["cal"])) == (10, 4, 20)
    assert set(ids["ajuste"] + ids["cal"]) == set(range(1, 35))


def test_p1_provisional_incompleto_sigue_negandose(tmp_path, monkeypatch, capsys):
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "parcial.yaml", {"hipotesis": PROVISIONAL["hipotesis"]})
    codigo = _ejecutar(tmp_path, "--provisional", str(prov))
    err = capsys.readouterr().err
    assert codigo == 2
    assert "deteccion.ventana_far_normal" in err and "particion.rieth.normales_ajuste" in err
    assert "hipotesis.metrica_ventaja" not in err          # esa ya la trae el provisional
    assert llamadas == []


def test_p1_provisional_no_pisa_una_clave_con_valor(tmp_path, monkeypatch, capsys):
    """pca.variance tiene valor en el yaml del protocolo: el yaml manda y el provisional se rechaza."""
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "pisa.yaml", {**PROVISIONAL, "pca": {"variance": 0.9}})
    codigo = _ejecutar(tmp_path, "--provisional", str(prov))
    err = capsys.readouterr().err
    assert codigo == 2 and "pca.variance" in err and "manda" in err
    assert llamadas == []


def test_p1_provisional_con_clave_inexistente_se_niega(tmp_path, monkeypatch, capsys):
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "rara.yaml", {**PROVISIONAL, "hipotesis": {**PROVISIONAL["hipotesis"], "delta": 0.02}})
    codigo = _ejecutar(tmp_path, "--provisional", str(prov))
    assert codigo == 2 and "hipotesis.delta" in capsys.readouterr().err
    assert llamadas == []


def test_p1_metrica_no_implementada_se_niega(tmp_path, monkeypatch, capsys):
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "igualada.yaml",
                 {**PROVISIONAL, "hipotesis": {**PROVISIONAL["hipotesis"], "metrica_ventaja": "fdr_a_far_igualada"}})
    codigo = _ejecutar(tmp_path, "--provisional", str(prov))
    assert codigo == 2 and "fdr_a_far_igualada" in capsys.readouterr().err
    assert llamadas == []


def test_p1_mas_normales_que_el_maximo_se_niega(tmp_path, monkeypatch, capsys):
    """21 + 20 = 41 normales de train_fault00 pasarian del id 40 (MAXIMO_NORMALES): se niega antes de leer."""
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "muchas.yaml",
                 {**PROVISIONAL, "particion": {"rieth": {"normales_ajuste": 21, "normales_calibracion": 20}}})
    codigo = _ejecutar(tmp_path, "--provisional", str(prov))
    assert codigo == 2 and "MAXIMO_NORMALES" in capsys.readouterr().err
    assert llamadas == []


def test_p1_ejecutar_reparte_y_carga_las_normales_con_maximo_normales(tmp_path, monkeypatch):
    """Commit 2b en ejecutar: con el yaml real, el reparto y la carga de train_fault00 usan MAXIMO_NORMALES (40).

    La primera carga es la de las normales (ids 1 a 34); la prueba sigue en 1..datos.runs_desarrollo
    (--n-ids por defecto, 20). La carga se sustituye por un espia que para la ejecucion: no se lee nada.
    """
    pytest.importorskip("torch")

    class Alto(Exception):
        pass

    vistos = {"reparto": [], "cargas": []}
    original = P1.repartir_normales

    def reparto(n_ajuste, n_cal, n_parada, semilla, maximo):
        vistos["reparto"].append(maximo)
        return original(n_ajuste, n_cal, n_parada, semilla, maximo)

    def carga(fallo, split, ids, root, maximo):
        vistos["cargas"].append((fallo, split, min(ids), max(ids), maximo))
        raise Alto

    monkeypatch.setattr(P1, "repartir_normales", reparto)
    monkeypatch.setattr(P1, "cargar_mirados", carga)
    with pytest.raises(Alto):
        P1.ejecutar(["--config", str(BASELINE), "--root", str(tmp_path / "rieth"), "--out", str(tmp_path / "out")])
    assert vistos["reparto"] == [P1.MAXIMO_NORMALES] == [40]
    assert vistos["cargas"] == [(0, "train", 1, 34, 40)]
    # la prueba no pasa de datos.runs_desarrollo: --n-ids 21 se niega antes de cargar nada
    assert P1.ejecutar(["--config", str(BASELINE), "--n-ids", "21", "--root", str(tmp_path / "rieth")]) == 2
    assert len(vistos["cargas"]) == 1


def test_maximo_normales_son_las_40_que_el_registro_da_por_miradas():
    """Commit 2b: las normales 1 a 40 de train_fault00 estan en el registro por la regla 3, puntos 1 y 2."""
    assert P1.MAXIMO_NORMALES == 40
    fuentes = {e.source for e in data.RIETH_SEEN_RUNS
               if "train" in e.splits and 0 in e.faults and set(e.runs) & set(range(1, 41))}
    assert any("punto 1" in f for f in fuentes) and any("punto 2" in f for f in fuentes)
    assert np.isin(np.arange(1, 41), data.seen_runs("train", 0)).all()


def test_p1_n_ids_fuera_de_rango_se_niega(tmp_path, monkeypatch, capsys):
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "ok.yaml", PROVISIONAL)
    codigo = _ejecutar(tmp_path, "--provisional", str(prov), "--n-ids", "21")
    assert codigo == 2 and "--n-ids" in capsys.readouterr().err
    assert llamadas == []


def test_p1_epocas_max_menor_que_uno_se_niega(tmp_path, monkeypatch, capsys):
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "ok.yaml", PROVISIONAL)
    codigo = _ejecutar(tmp_path, "--provisional", str(prov), "--epocas-max", "0")
    assert codigo == 2 and "--epocas-max" in capsys.readouterr().err
    assert llamadas == []


def test_p1_provisional_no_escribe_en_results_summary(tmp_path, monkeypatch, capsys):
    llamadas = _espiar_cargas(monkeypatch)
    monkeypatch.chdir(tmp_path)
    prov = _yaml(tmp_path, "ok.yaml", PROVISIONAL)
    codigo = P1.ejecutar(["--config", str(_baseline_en_null(tmp_path)), "--provisional", str(prov),
                          "--root", str(tmp_path / "rieth"), "--out", "results/summary"])
    assert codigo == 2 and "provisional no escribe" in capsys.readouterr().err
    assert llamadas == [] and not (tmp_path / "results").exists()


def test_p1_la_que_cuenta_exige_limites_protocolo_y_la_parada(tmp_path, monkeypatch, capsys):
    """Sin --provisional (fila 29): con limites_protocolo en null o sin la clave de la parada, no corre."""
    llamadas = _espiar_cargas(monkeypatch)
    for nombre, cfg, clave in (
        ("sin_limites.yaml", _cfg_decidida(con_limites=False, con_parada=True), "limites_protocolo"),
        ("sin_parada.yaml", _cfg_decidida(con_limites=True, con_parada=False), P1.CLAVE_PARADA),
    ):
        codigo = P1.ejecutar(["--config", str(_yaml(tmp_path, nombre, cfg)), "--root", str(tmp_path / "rieth"),
                              "--out", str(tmp_path / "out")])
        assert codigo == 2 and clave in capsys.readouterr().err, nombre
    assert llamadas == [] and not (tmp_path / "out").exists()


def test_aplicar_provisional_solo_rellena_las_null():
    cfg = _cfg_con_las_seis_en_null()
    nuevo, aplicadas = P1.aplicar_provisional(cfg, PROVISIONAL)
    assert set(aplicadas) == set(P1.CLAVES_NECESARIAS)
    assert cfg["hipotesis"]["metrica_ventaja"] is None            # el original no cambia
    assert nuevo["hipotesis"]["metrica_ventaja"] == "exceso_far_gemela"
    assert P1.pendientes(nuevo) == []
    decisiones = P1.validar_decisiones(nuevo)
    assert decisiones["ventana_far_normal"] == (160, None)
    assert [b["nombre"] for b in decisiones["baselines"]] == ["pca", "dpca"]


# ===================================================================== ids

def test_comprobar_ids_mirados_rechaza_mayores_que_el_maximo_y_no_mirados():
    with pytest.raises(ValueError, match="pasan de 20"):
        P1.comprobar_ids_mirados("test", 3, [1, 21], maximo=20)
    with pytest.raises(ValueError, match="no estan en el registro"):
        P1.comprobar_ids_mirados("test", 3, [401], maximo=500)     # el 401 no esta en RIETH_SEEN_RUNS
    with pytest.raises(ValueError, match="enteros >= 1"):
        P1.comprobar_ids_mirados("test", 3, [0, 1], maximo=20)
    assert P1.comprobar_ids_mirados("test", 3, range(20, 0, -1), maximo=20) == list(range(1, 21))
    # train_fault00 (commit 2b): las 1 a 40, si; la 41, no
    assert P1.comprobar_ids_mirados("train", 0, range(1, 41), maximo=P1.MAXIMO_NORMALES) == list(range(1, 41))
    with pytest.raises(ValueError, match="pasan de 40"):
        P1.comprobar_ids_mirados("train", 0, [40, 41], maximo=P1.MAXIMO_NORMALES)


def test_repartir_normales_un_solo_reparto_como_el_c00():
    """14 de ajuste (4 de parada) y 6 de calibracion: el split_runs(20, (0.5, 0.2, 0.3)) del C00, celda 35."""
    ids = P1.repartir_normales(14, 6, n_parada=4, semilla=42, maximo=20)
    assert (len(ids["tr"]), len(ids["es"]), len(ids["cal"])) == (10, 4, 6)
    assert sorted(ids["tr"] + ids["es"]) == ids["ajuste"]
    assert not set(ids["tr"]) & set(ids["es"]) and not set(ids["ajuste"]) & set(ids["cal"])
    assert set(ids["ajuste"] + ids["cal"]) == set(range(1, 21))
    tr, es, cal = data.split_runs(20, fractions=(0.5, 0.2, 0.3), seed=42)
    assert (ids["tr"], ids["es"], ids["cal"]) == (tr.tolist(), es.tolist(), cal.tolist())
    assert ids == P1.repartir_normales(14, 6, n_parada=4, semilla=42, maximo=20)   # reproducible
    with pytest.raises(P1.DecisionPendiente, match="pasa de 20"):
        P1.repartir_normales(15, 6, n_parada=4, semilla=42, maximo=20)


def test_repartir_normales_con_las_decisiones_del_2026_10_18():
    """14 (10 + 4) y 20: split_runs(34, (10/34, 4/34, 20/34)) sobre los ids 1 a 34, el reparto de cada replica."""
    ids = P1.repartir_normales(14, 20, n_parada=4, semilla=42, maximo=P1.MAXIMO_NORMALES)
    tr, es, cal = data.split_runs(34, fractions=(10 / 34, 4 / 34, 20 / 34), seed=42)
    assert (ids["tr"], ids["es"], ids["cal"]) == (tr.tolist(), es.tolist(), cal.tolist())
    assert set(ids["ajuste"] + ids["cal"]) == set(range(1, 35))
    with pytest.raises(P1.DecisionPendiente, match="MAXIMO_NORMALES"):
        P1.repartir_normales(21, 20, n_parada=4, semilla=42, maximo=P1.MAXIMO_NORMALES)
    for n_parada in (0, 4):
        with pytest.raises(P1.DecisionPendiente, match="para entrenar y otra para parar"):
            P1.repartir_normales(4, 2, n_parada=n_parada, semilla=42, maximo=20)


def test_normales_parada_fijada_o_provisional():
    cfg = _cfg_con_las_seis_en_null()                  # sin la clave, como el yaml del 16/10
    assert P1.resolver_normales_parada(cfg, 14, 6, provisional=True) == (
        4, "particion.fraccion_validacion (provisional)")                  # round(0,2 x 20), la cuenta del C00
    with pytest.raises(P1.DecisionPendiente, match="normales_parada no existe en el yaml"):
        P1.resolver_normales_parada(cfg, 14, 6, provisional=False)
    cfg["particion"]["rieth"]["normales_parada"] = None
    with pytest.raises(P1.DecisionPendiente, match="normales_parada sigue en null"):
        P1.resolver_normales_parada(cfg, 14, 6, provisional=False)
    cfg["particion"]["rieth"]["normales_parada"] = 3
    assert P1.resolver_normales_parada(cfg, 14, 6, provisional=False) == (3, P1.CLAVE_PARADA)


def test_parametros_limites_de_desarrollo_solo_en_provisional():
    cfg = _cfg_con_las_seis_en_null()
    assert P1.parametros_limites(cfg, provisional=True) == {
        "fuente": "pca_desarrollo", "t2_method": cfg["pca"]["t2_method"], "spe_method": cfg["pca"]["spe_method"],
        "quantile_method": cfg["pca"]["quantile_method"], "nivel_por_estadistico": "mismo_alpha",
        "orden_escalado_dpca": "retardar_y_escalar"}
    with pytest.raises(P1.DecisionPendiente, match="limites_protocolo"):
        P1.parametros_limites(cfg, provisional=False)
    cfg["limites_protocolo"].update(LIMITES)
    assert P1.parametros_limites(cfg, provisional=False) == {"fuente": "limites_protocolo", **LIMITES}
    cfg["limites_protocolo"]["orden_escalado_dpca"] = None
    with pytest.raises(P1.DecisionPendiente, match="a medias"):
        P1.parametros_limites(cfg, provisional=True)
    cfg["limites_protocolo"]["orden_escalado_dpca"] = "en_medio"
    with pytest.raises(P1.DecisionPendiente, match="orden_escalado_dpca"):
        P1.parametros_limites(cfg, provisional=False)


def test_alpha_por_estadistico_sidak_solo_en_la_regla_t2_o_spe():
    assert P1.alpha_por_estadistico(0.99, "t2_o_spe", "sidak") == pytest.approx(0.99 ** 0.5)
    assert P1.alpha_por_estadistico(0.99, "t2_o_spe", "mismo_alpha") == 0.99
    assert P1.alpha_por_estadistico(0.99, "spe", "sidak") == 0.99


def test_carpeta_salida_provisional_fuera_de_results_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert P1.carpeta_salida(None, provisional=True) == Path("results/raw/P1_provisional")
    assert P1.carpeta_salida(None, provisional=False) == Path("results/summary")
    for prohibida in ("results/summary", "results/summary/sub", str(P1.RAIZ / "results" / "summary")):
        with pytest.raises(P1.DecisionPendiente, match="provisional no escribe"):
            P1.carpeta_salida(prohibida, provisional=True)
    assert P1.carpeta_salida("results/summary", provisional=False) == Path("results/summary")
    assert P1.carpeta_salida(str(tmp_path / "otra"), provisional=True) == tmp_path / "otra"


# =================================================================== sigma

def test_estimar_sigma_desviacion_muestral_e_intervalo_chi2():
    d = np.array([0.01, -0.02, 0.03, 0.0, 0.015, -0.005, 0.02, np.nan])
    r = P1.estimar_sigma(d, nivel=0.95)
    limpio = d[~np.isnan(d)]
    n = limpio.size
    s = np.std(limpio, ddof=1)
    assert r["n"] == n and r["n_nonzero"] == 6
    assert np.isclose(r["sigma_hat"], s) and np.isclose(r["mean_advantage"], limpio.mean())
    assert np.isclose(r["sigma_ci_low"], s * np.sqrt((n - 1) / sps.chi2.ppf(0.975, n - 1)))
    assert np.isclose(r["sigma_ci_high"], s * np.sqrt((n - 1) / sps.chi2.ppf(0.025, n - 1)))
    assert np.isclose(r["sigma_upper"], s * np.sqrt((n - 1) / sps.chi2.ppf(0.05, n - 1)))
    assert r["sigma_ci_low"] < s < r["sigma_upper"] < r["sigma_ci_high"]
    assert r["min_advantage"] == -0.02 and r["max_advantage"] == 0.03


def test_estimar_sigma_con_menos_de_dos_valores_da_nan():
    assert np.isnan(P1.estimar_sigma([]).get("sigma_hat")) and P1.estimar_sigma([])["n"] == 0
    r = P1.estimar_sigma([0.02, np.nan])
    assert r["n"] == 1 and np.isnan(r["sigma_hat"]) and r["mean_advantage"] == 0.02
    with pytest.raises(ValueError):
        P1.estimar_sigma([0.1, 0.2], nivel=1.0)


def test_estimar_sigma_cobertura_con_ventaja_normal():
    """Con ventaja normal y n = 20, el intervalo al 95 % y la cota superior cubren sigma a su nivel."""
    rng = np.random.default_rng(20261015)
    sigma, n, reps = 0.05, 20, 2000
    dentro = cota = 0
    for _ in range(reps):
        r = P1.estimar_sigma(rng.normal(0.0, sigma, size=n), nivel=0.95)
        dentro += r["sigma_ci_low"] <= sigma <= r["sigma_ci_high"]
        cota += sigma <= r["sigma_upper"]
    assert 0.93 <= dentro / reps <= 0.97
    assert 0.93 <= cota / reps <= 0.97


# ================================================== ejecucion con Parquet sinteticos

def _rieth_sintetico(carpeta: Path, n_sims: int = 8, semilla: int = 0) -> Path:
    """Escribe train_fault00, test_fault00, test_fault03, 09 y 15 con el contrato de Rieth.

    Normales correlacionadas (5 factores y ruido); en los ficheros con fallo, un
    escalon pequeno desde el indice 160 en unas variables, para que haya alguna
    alarma. Ids en int16 y variables en float32, como los Parquet reales.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    carpeta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semilla)
    cargas = rng.normal(size=(52, 5))
    media = rng.normal(size=52) * 5

    def fichero(fault: int, split: str) -> None:
        n = data.N_SAMPLES_RIETH[split]
        bloques = []
        for sim in range(1, n_sims + 1):
            X = rng.normal(size=(n, 5)) @ cargas.T + 0.5 * rng.normal(size=(n, 52)) + media
            if fault:
                X[160:, :4] += 1.5 + 0.1 * fault
            df = pd.DataFrame(X.astype(np.float32), columns=data.VAR_NAMES)
            df.insert(0, "sample", np.arange(1, n + 1, dtype=np.int16))
            df.insert(0, "simulationRun", np.full(n, sim, dtype=np.int16))
            df.insert(0, "faultNumber", np.full(n, fault, dtype=np.int16))
            bloques.append(df)
        tabla = pa.Table.from_pandas(pd.concat(bloques, ignore_index=True), preserve_index=False)
        pq.write_table(tabla, carpeta / f"{split}_fault{fault:02d}.parquet")

    for fault, split in ((0, "train"), (0, "test"), (3, "test"), (9, "test"), (15, "test")):
        fichero(fault, split)
    lineas = ["manifiesto sintetico"]
    for p in sorted(carpeta.glob("*.parquet")):
        lineas.append(f"{p.name} {p.stat().st_size} {provenance.file_sha256(p)}")
    (carpeta / P1.MANIFIESTO).write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return carpeta


@pytest.fixture(scope="module")
def rieth_sintetico(tmp_path_factory) -> Path:
    return _rieth_sintetico(tmp_path_factory.mktemp("rieth"))


def test_p1_manifiesto_con_bytes_distintos_se_niega(tmp_path, monkeypatch, capsys, rieth_sintetico):
    """Si los bytes de un Parquet no son los del manifiesto, no se abre ninguno."""
    import shutil

    carpeta = tmp_path / "rieth"
    shutil.copytree(rieth_sintetico, carpeta)
    lineas = []
    for linea in (carpeta / P1.MANIFIESTO).read_text(encoding="utf-8").splitlines():
        partes = linea.split()
        if partes[0] == "test_fault09.parquet":
            linea = f"{partes[0]} {int(partes[1]) + 1} {partes[2]}"
        lineas.append(linea)
    (carpeta / P1.MANIFIESTO).write_text("\n".join(lineas) + "\n", encoding="utf-8")
    llamadas = _espiar_cargas(monkeypatch)
    prov = _yaml(tmp_path, "prov.yaml",
                 {**PROVISIONAL, "particion": {"rieth": {"normales_ajuste": 5, "normales_calibracion": 3}}})
    codigo = P1.ejecutar(["--config", str(_baseline_en_null(tmp_path)), "--provisional", str(prov),
                          "--root", str(carpeta), "--out", str(tmp_path / "out"), "--n-ids", "8"])
    err = capsys.readouterr().err
    assert codigo == 2 and "test_fault09.parquet" in err and "no coinciden" in err
    assert llamadas == [] and not (tmp_path / "out").exists()


def test_p1_corre_con_rieth_sintetico_y_provisional(tmp_path, rieth_sintetico):
    pytest.importorskip("torch")
    prov = _yaml(tmp_path, "prov.yaml",
                 {**PROVISIONAL, "particion": {"rieth": {"normales_ajuste": 5, "normales_calibracion": 3}}})
    out = tmp_path / "out"
    codigo = P1.ejecutar(["--config", str(_baseline_en_null(tmp_path)), "--provisional", str(prov),
                          "--root", str(rieth_sintetico),
                          "--out", str(out), "--n-ids", "8", "--epocas-max", "2"])
    assert codigo == 0

    celdas = pd.read_csv(out / P1.SALIDA)
    assert len(celdas) == 6                                    # 3 fallos x 2 baselines
    assert set(celdas["faultNumber"]) == {3, 9, 15} and set(celdas["baseline"]) == {"pca", "dpca"}
    assert (celdas["method"] == "ae").all() and (celdas["n"] == 8).all()
    assert (celdas["metric"] == "exceso_far_gemela").all() and celdas["higher_is_better"].all()
    assert np.isfinite(celdas["sigma_hat"]).all() and (celdas["sigma_hat"] >= 0).all()
    assert (celdas["sigma_ci_low"] <= celdas["sigma_hat"]).all()
    assert (celdas["sigma_hat"] <= celdas["sigma_upper"]).all()
    assert (celdas["sigma_upper"] <= celdas["sigma_ci_high"]).all()
    assert np.isfinite(celdas["hl_estimate"]).all()
    assert celdas["provisional"].all() and "hipotesis.metrica_ventaja" in celdas["provisional_keys"].iloc[0]
    for columna in ("far_window_normal", "normales_ajuste", "normales_calibracion", "n_components_pca",
                    "n_components_dpca", "spe_limit_pca", "ae_limit", "hp_ae_capas", "alpha", "k", "semilla",
                    "baseline_statistic", "baseline_lags", "torch_threads"):
        assert columna in celdas, columna
    assert (celdas["far_window_normal"] == "[160, null]").all()
    assert (celdas["normales_ajuste"] == 5).all() and (celdas["normales_calibracion"] == 3).all()
    assert celdas.set_index("baseline")["baseline_lags"].to_dict() == {"pca": 0, "dpca": 2}
    assert (celdas["hl_ci_level"] == 0.90).all()                      # el nivel del contraste (fila 27)
    assert (celdas["limites_fuente"] == "pca_desarrollo").all()       # provisional: los de desarrollo
    assert (celdas["orden_escalado_dpca"] == "retardar_y_escalar").all()
    assert (celdas["nivel_por_estadistico"] == "mismo_alpha").all()
    assert celdas["limite_reservado_pca"].all() and celdas["limite_reservado_dpca"].all()   # SPE por caja
    # Un solo reparto: 5 + 3 normales, round(0,2 x 8) = 2 para parar y 3 para entrenar.
    assert (celdas["normales_parar_ae"] == 2).all() and (celdas["normales_entrenar_ae"] == 3).all()
    assert (celdas["normales_parada_fuente"] == "particion.fraccion_validacion (provisional)").all()

    runs = pd.read_csv(out / P1.SALIDA_RUNS)
    assert len(runs) == 4 * 8 * 3                              # fallos 0, 3, 9, 15 x 8 sims x 3 metodos
    normales = runs[runs["faultNumber"] == 0]
    assert (normales["n_far"] == 800).all() and normales["fdr"].isna().all()   # ventana [160, null]
    con_fallo = runs[runs["faultNumber"] > 0]
    assert (con_fallo["n_fdr"] == 800).all()
    assert (con_fallo.loc[con_fallo["method"] == "dpca", "n_far"] == 158).all()   # 2 retardos sin evaluar
    assert (con_fallo.loc[con_fallo["method"] != "dpca", "n_far"] == 160).all()
    assert np.allclose(con_fallo["exceso_far_gemela"], con_fallo["fdr"] - con_fallo["far_gemela"])
    fila = con_fallo[(con_fallo["method"] == "ae") & (con_fallo["faultNumber"] == 3) & (con_fallo["simulationRun"] == 2)]
    gemela = normales[(normales["method"] == "ae") & (normales["simulationRun"] == 2)]
    assert np.isclose(fila["far_gemela"].iloc[0], gemela["far"].iloc[0])
    assert runs["simulationRun"].max() == 8

    # La ventaja de la celda sale de las filas: media de (fdr - far_gemela) del ae menos la del baseline.
    sub = con_fallo[con_fallo["faultNumber"] == 9].set_index(["method", "simulationRun"])["exceso_far_gemela"]
    ventaja = (sub.loc["ae"] - sub.loc["pca"]).sort_index()
    celda = celdas[(celdas["faultNumber"] == 9) & (celdas["baseline"] == "pca")].iloc[0]
    assert np.isclose(celda["mean_advantage"], ventaja.mean())
    assert np.isclose(celda["sigma_hat"], ventaja.std(ddof=1))

    for nombre in (P1.SALIDA, P1.SALIDA_RUNS):
        datos = provenance.read_provenance(out / nombre)
        assert datos["config"]["provisional"] is True
        assert [f["source"] for f in datos["config_files"]] == ["explicit", "provisional"]
        assert datos["ids"]["prueba"]["simulationRun"] == list(range(1, 9))
        assert datos["ids"]["max_id"] == 20
        ajuste = datos["ids"]["ajuste"]
        assert len(ajuste["ajuste_pca_dpca"]) == 5 and len(ajuste["calibracion"]) == 3
        assert sorted(ajuste["entrenar_ae"] + ajuste["parar_ae"]) == ajuste["ajuste_pca_dpca"]
        assert max(ajuste["ajuste_pca_dpca"] + ajuste["calibracion"]) <= 8
        assert datos["extra"]["provisional"]["claves"] == list(P1.CLAVES_NECESARIAS)
        rieth = datos["extra"]["rieth"]
        assert rieth["manifiesto"] == P1.MANIFIESTO and rieth["discrepancias"] == []
        f03 = rieth["ficheros"]["test_fault03.parquet"]
        assert f03["sha256_segun_manifiesto"] == provenance.file_sha256(rieth_sintetico / "test_fault03.parquet")
        assert f03["bytes_coinciden"]
        assert f03["bytes_en_disco"] == (rieth_sintetico / "test_fault03.parquet").stat().st_size
        assert datos["extra"]["limites"]["fuente"] == "pca_desarrollo"
        assert datos["extra"]["normales_parada"] == {"n": 2, "fuente": "particion.fraccion_validacion (provisional)"}
        assert datos["inputs"] == []                           # no se vuelven a leer los Parquet


def test_p1_fdr_cruda_con_la_regla_t2_o_spe(tmp_path, rieth_sintetico, capsys):
    pytest.importorskip("torch")
    prov = _yaml(tmp_path, "prov.yaml", {
        "hipotesis": {"metrica_ventaja": "fdr", "mayor_es_mejor": True,
                      "baselines": [{"nombre": "pca_t2_o_spe", "estadistico": "t2_o_spe", "lags": 0}]},
        "deteccion": {"ventana_far_normal": [0, None]},
        "particion": {"rieth": {"normales_ajuste": 4, "normales_calibracion": 2}},
    })
    out = tmp_path / "out"
    codigo = P1.ejecutar(["--config", str(_baseline_en_null(tmp_path)), "--provisional", str(prov),
                          "--root", str(rieth_sintetico),
                          "--out", str(out), "--n-ids", "6", "--fallos", "3", "15", "--epocas-max", "1"])
    assert codigo == 0
    celdas = pd.read_csv(out / P1.SALIDA)
    assert len(celdas) == 2 and (celdas["metric"] == "fdr").all() and (celdas["n"] == 6).all()
    runs = pd.read_csv(out / P1.SALIDA_RUNS)
    assert (runs.loc[runs["method"] == "pca_t2_o_spe", "limit"] == 1.0).all()
    assert (runs.loc[runs["faultNumber"] == 0, "n_far"] == 960).all()        # ventana [0, null]
    sub = runs[runs["faultNumber"] == 15].set_index(["method", "simulationRun"])["fdr"]
    ventaja = sub.loc["ae"] - sub.loc["pca_t2_o_spe"]
    celda = celdas[celdas["faultNumber"] == 15].iloc[0]
    assert np.isclose(celda["mean_advantage"], ventaja.mean())
    # Con los limites de desarrollo, el T2 por F es parametrico: no usa las reservadas, y se avisa.
    assert not celdas["limite_reservado_pca_t2_o_spe"].any()
    salida = capsys.readouterr().out
    assert "AVISO: en pca_t2_o_spe" in salida and "parametric_fit" in salida


def test_p1_con_limites_protocolo_sidak_y_escalar_y_retardar(tmp_path, rieth_sintetico, monkeypatch):
    """Con los cinco limites fijados (aqui, de prueba) y sin --out: va a results/raw/P1_provisional."""
    pytest.importorskip("torch")
    monkeypatch.chdir(tmp_path)
    prov = _yaml(tmp_path, "prov.yaml", {
        "hipotesis": {"metrica_ventaja": "exceso_far_gemela", "mayor_es_mejor": True,
                      "baselines": [{"nombre": "dpca", "estadistico": "t2_o_spe", "lags": 2}]},
        "deteccion": {"ventana_far_normal": [160, None]},
        "particion": {"rieth": {"normales_ajuste": 5, "normales_calibracion": 3}},
        "limites_protocolo": LIMITES,
    })
    codigo = P1.ejecutar(["--config", str(_baseline_en_null(tmp_path)), "--provisional", str(prov),
                          "--root", str(rieth_sintetico), "--n-ids", "4", "--fallos", "3", "--epocas-max", "1"])
    assert codigo == 0
    out = tmp_path / "results" / "raw" / "P1_provisional"
    fila = pd.read_csv(out / P1.SALIDA).iloc[0]
    alpha = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))["protocolo"]["alpha"]
    assert fila["limites_fuente"] == "limites_protocolo" and fila["orden_escalado_dpca"] == "escalar_y_retardar"
    assert fila["nivel_por_estadistico"] == "sidak" and fila["t2_method"] == "empirical"
    assert np.isclose(fila["alpha_por_estadistico_dpca"], alpha ** 0.5)
    assert bool(fila["limite_reservado_dpca"]) and fila["scaler_source_dpca"] == "external"
    datos = provenance.read_provenance(out / P1.SALIDA)
    assert datos["extra"]["limites"] == {"fuente": "limites_protocolo", **LIMITES}
    assert datos["config"]["provisional"] is True
