"""Pruebas de las decisiones del 26/10 al 04/11/2026 (acta del 26/10, secciones 4 a 16).

- datos.tramo_copiado y diagnostico.incluir_normal, con su valor en los dos yaml.
- El factor 1 de las ablaciones: unidad (diseno anidado), metrica principal, brazos (clave
  nueva), familia de Holm y regla de equivalencia.
- El reparto anidado de las simulaciones de prueba entre las replicas, con la llamada que
  declara el yaml (ablation.subsample_runs, disjunto), sin leer ningun dato.
- stats.compare_arms con ese diseno: una familia de 4 por factor (PCA y DPCA con T2 y SPE),
  la replica como unidad y el margen de contrastes.delta.far.
- El orden de los ids (fila del 2026-11-03): los grupos del reparto, con los ids en orden
  creciente, fijados para n = 100, 200 y 400, y las 14 reservadas de cada replica.
- La unidad de la familia de la hipotesis (hipotesis.unidad_replica, fila del 2026-11-03): el
  diseno anidado del factor 1 con las funciones de hoy del paquete (los grupos, la ventaja por
  replica, el suelo con su unidad, los controles y el veredicto) y lo que cambia con una cola.
- El criterio de mejor compromiso (compromiso.deteccion y compromiso.diagnostico, filas del
  2026-11-03 y del 2026-11-04): la lista cerrada de candidatos con su familia, que es la familia
  declarada aunque falte alguno, la restriccion y el orden del yaml
  y la eliminacion frente a cualquiera con Holm en cada paso, escrita aqui (_criterio) porque
  stats.best_compromise compara solo con el mejor, en escenarios sinteticos y en sus fronteras.
- Las claves del OE3 (filas del 2026-11-04): la mediana del retardo, las falsas alarmas por hora
  y el conjunto de la PR-AUC que se reportan.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import operator
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfmfdd import ablation, config, data, metrics, stats

RAIZ = Path(__file__).resolve().parents[1]
BASELINE = RAIZ / "configs" / "baseline.yaml"
DECIDIDAS = ("datos.tramo_copiado", "diagnostico.incluir_normal", "ablaciones.unidad_replica",
             "ablaciones.metrica_principal.factor1", "ablaciones.familia_holm",
             "ablaciones.regla_equivalencia", "compromiso.deteccion", "compromiso.diagnostico",
             "deteccion.mediana_retardo", "deteccion.falsas_alarmas_por_hora",
             "deteccion.pr_auc_conjunto")


def _cfg():
    return config.load_baseline(BASELINE)


def test_valores_del_yaml_y_de_su_copia():
    """Los valores fijados del 26 al 29/10, iguales en la copia empaquetada."""
    cfg = _cfg()
    copia = config.load_yaml(RAIZ / "tfmfdd" / "configs" / "baseline.yaml")
    esperado = {
        "datos.tramo_copiado": "inicio_nominal_medido",
        "diagnostico.incluir_normal": False,
        "ablaciones.unidad_replica": {"diseno": "anidado", "unidad": "replicate",
                                      "grupos_prueba": "subsample_runs_disjunto"},
        "ablaciones.metrica_principal.factor1": "far",
        "ablaciones.familia_holm": "por_factor",
        "ablaciones.regla_equivalencia": "effect_test",
        "hipotesis.unidad_replica": {"diseno": "anidado", "unidad": "replicate",
                                     "grupos_prueba": "ablaciones.unidad_replica.grupos_prueba",
                                     "agregado": "media"},
        "compromiso.deteccion": {
            "candidatos": [{"nombre": "pca", "familia": 1}, {"nombre": "dpca", "familia": 1},
                           {"nombre": "kpca", "familia": 1}, {"nombre": "ae", "familia": 3}],
            "eliminacion": "frente_a_cualquiera_con_holm",
            "restricciones": [{"metrica": "far", "operador": "<=", "valor": 0.015}],
            "orden": [{"metrica": "exceso_far_gemela", "direccion": "max",
                       "empate": "contrastes.delta.fdr"},
                      {"metrica": "retardo", "direccion": "min",
                       "empate": "contrastes.delta.retardo_muestras"},
                      {"metrica": "log10_inferencia_s", "direccion": "min", "empate": 1.0}]},
        "compromiso.diagnostico": {
            "candidatos": None,
            "eliminacion": "frente_a_cualquiera_con_holm",
            "restricciones": [],
            "orden": [{"metrica": "macro_f1_rest", "direccion": "max",
                       "empate": "contrastes.delta.macro_f1"},
                      {"metrica": "log10_inferencia_s", "direccion": "min", "empate": 1.0}]},
        "ablaciones.brazos_factor1": {"limpio": "limites_protocolo",
                                      "sucio_principal": "en_muestra_mismo_estimador",
                                      "secundarios": ["t2_f", "t2_beta", "spe_jackson", "reservada_14"],
                                      "estadisticos": ["t2", "spe"]},
    }
    for clave, valor in esperado.items():
        assert config.get(cfg, clave) == valor, clave
        assert config.get(copia, clave) == valor, clave
    assert config.get(cfg, "diagnostico.incluir_normal") is False


def test_quedan_30_pendientes_y_ninguna_de_estas():
    pendientes = set(config.pending(_cfg()))
    assert len(pendientes) == 30
    # ninguna de las decididas, ni ninguna clave por debajo de ellas, salvo la lista de candidatos
    # del diagnostico, que sigue pendiente
    debajo = {p for p in pendientes for d in DECIDIDAS if p == d or p.startswith(d + ".")}
    assert debajo == {"compromiso.diagnostico.candidatos"}
    # siguen pendientes las de los factores 2 a 4 (adendas), la etiqueta del prefallo y la lista
    # de candidatos del diagnostico (nueva el 2026-11-03)
    for clave in ("ablaciones.metrica_principal.factor2", "ablaciones.estimanda_factor3",
                  "ablaciones.brazo_sucio_factor2", "diagnostico.etiqueta_prefallo",
                  "compromiso.diagnostico.candidatos"):
        assert clave in pendientes, clave


def test_reparto_anidado_de_las_simulaciones_de_prueba():
    """R grupos disjuntos de n // R ids, con la semilla del protocolo; sobran n mod R.

    Usa los 100 ids mas altos de los no mirados en los ficheros de prueba (el procedimiento de
    datos.ids_confirmatorios con el primer peldano de datos.regla_n). Solo consulta el registro:
    no lee ningun dato."""
    cfg = _cfg()
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    semilla = config.get(cfg, "protocolo.semilla")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ids = np.sort(data.unseen_runs("test"))[-100:]
    grupos = ablation.subsample_runs(ids, ids.size // R, R, semilla, disjoint=True)
    assert R == 12 and grupos.shape == (12, 8)
    usados = np.unique(grupos)
    assert usados.size == 96 and set(usados) <= set(ids.tolist())       # disjuntos y de la lista
    assert ids.size - usados.size == ids.size % R == 4                    # sobran n mod R
    otra = ablation.subsample_runs(ids, ids.size // R, R, semilla, disjoint=True)
    assert np.array_equal(grupos, otra)                                   # reproducible


def test_compare_arms_con_el_diseno_anidado_y_la_familia_de_cuatro():
    """Una fila por (metodo, estadistico), 12 replicas como unidad, familia de 4 y el margen del yaml."""
    cfg = _cfg()
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    margen = config.get(cfg, "contrastes.delta.far")
    grupos = np.arange(1, R * 8 + 1).reshape(R, 8)                         # grupos disjuntos
    rng = np.random.default_rng(20261029)
    filas = []
    for metodo in ("pca", "dpca"):
        for estadistico, efecto in (("t2", 0.0005), ("spe", 0.02)):
            for r in range(R):
                for sim in grupos[r]:
                    base = 0.01 + rng.normal(0, 0.001)
                    for brazo, d in (("protocolo", 0.0), ("en_muestra", efecto)):
                        filas.append({"metodo": metodo, "estadistico": estadistico, "replica": r + 1,
                                      "simulationRun": int(sim), "brazo": brazo,
                                      "far": base + d + rng.normal(0, 0.0005)})
    salida = stats.compare_arms(pd.DataFrame(filas), "far", "brazo", "protocolo", margin=margen,
                                family=(), by=("metodo", "estadistico"), replicate_col="replica",
                                unit="replicate", alpha=config.get(cfg, "contrastes.alpha_contrastes"))
    assert len(salida) == 4
    assert set(salida["family_m"]) == {4}
    assert set(salida["n"]) == {R} and set(salida["n_cells"]) == {R * 8}
    por = salida.set_index("estadistico")["outcome"]
    assert set(por.loc["spe"]) == {"efecto relevante"}                    # 2 puntos, fuera de 0,5
    assert set(por.loc["t2"]) == {"equivalente a cero"}                   # 0,05 puntos, dentro

# ------------------------------------------------- el orden de los ids (fila del 2026-11-03)
# Calculados el 2026-11-03 con el codigo de este paquete: el registro de simulaciones miradas
# cerrado el 2026-10-21 (408 ids libres en los ficheros de prueba), la semilla del protocolo y
# los ids en orden creciente. Si cambia el orden, la semilla, subsample_runs o el registro, estas
# pruebas fallan: los grupos son parte del prerregistro. Dependen tambien del generador de NumPy
# (default_rng), cuya secuencia no se promete igual entre versiones: dieron lo mismo con la 2.4.6
# de requirements-probado.txt y con la 2.5.3. HUELLA_GRUPOS fija los 12 grupos de cada peldano.
GRUPO_1 = {
    100: [418, 421, 427, 433, 442, 450, 456, 459],
    200: [307, 320, 324, 326, 338, 346, 356, 379, 406, 415, 444, 448, 452, 471, 478, 497],
    400: [125, 132, 143, 149, 152, 160, 171, 200, 230, 232, 246, 254, 300, 303, 332, 338, 344,
          353, 360, 364, 370, 371, 402, 407, 444, 446, 458, 462, 471, 472, 478, 483, 489],
}
SOBRAN = {
    100: [408, 436, 465, 477],
    200: [314, 336, 397, 436, 441, 481, 493, 494],
    400: [135, 196, 235, 294],
}
GRUPOS_N100 = [
    [418, 421, 427, 433, 442, 450, 456, 459],
    [402, 424, 425, 452, 471, 481, 492, 499],
    [404, 428, 437, 451, 461, 480, 494, 497],
    [400, 407, 426, 438, 457, 476, 478, 479],
    [410, 417, 429, 439, 445, 446, 470, 486],
    [401, 405, 440, 468, 484, 489, 491, 495],
    [403, 420, 443, 463, 473, 483, 487, 493],
    [415, 423, 432, 448, 455, 469, 475, 496],
    [409, 416, 431, 444, 460, 462, 472, 482],
    [430, 434, 454, 458, 467, 474, 488, 498],
    [406, 411, 419, 422, 441, 449, 485, 490],
    [412, 413, 414, 435, 447, 453, 464, 466],
]
HUELLA_GRUPOS = {                     # sha256 de json.dumps(grupos.tolist())
    100: "0829dbb3550aab15f447e4ae5375610098305fb4cca3c12501be98ac3aed968f",
    200: "4a6dddfc2a97319d1f2c58cbf738b91113a45ec291e1a477ed06f44ef2f53ff7",
    400: "62fdf9156ce6da8070e5177aa60f0dff48b29edd2efd9599d6463f529eea7bda",
}
RESERVADAS_14 = {
    1: [114, 123, 141, 149, 152, 174, 177, 224, 241, 260, 268, 276, 296, 307],
    12: [74, 87, 92, 102, 125, 140, 188, 201, 204, 227, 230, 232, 255, 267],
}
CSV_IDS = RAIZ / "configs" / "referencias" / "rieth_ids_prerregistro.csv"


def _ids_confirmatorios(n):
    """Los n ids mas altos de los no mirados en los ficheros de prueba, en orden creciente."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        libres = data.unseen_runs("test")
    assert np.all(np.diff(libres) > 0)        # data.unseen_runs ya los da en orden creciente
    return libres[-n:]


@pytest.mark.parametrize("n", sorted(GRUPO_1))
def test_grupos_del_reparto_fijados_con_orden_creciente(n):
    """Los tres peldanos de datos.regla_n: forma, grupo 1, los n mod R que sobran y la huella de
    los 12 grupos.

    Con los mismos ids al reves sale otro reparto: por eso el orden es parte de la regla."""
    cfg = _cfg()
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    semilla = config.get(cfg, "protocolo.semilla")
    assert n in [p["n"] for p in config.get(cfg, "datos.regla_n.peldanos")]
    ids = _ids_confirmatorios(n)
    grupos = ablation.subsample_runs(ids, n // R, R, semilla, disjoint=True)
    assert grupos.shape == (R, n // R)
    assert grupos[0].tolist() == GRUPO_1[n]
    assert np.setdiff1d(ids, grupos.ravel()).tolist() == SOBRAN[n]
    assert len(SOBRAN[n]) == n % R
    assert hashlib.sha256(json.dumps(grupos.tolist()).encode()).hexdigest() == HUELLA_GRUPOS[n]
    al_reves = ablation.subsample_runs(ids[::-1], n // R, R, semilla, disjoint=True)
    assert not np.array_equal(grupos, al_reves)


def test_grupos_de_n100_completos_y_el_csv_da_los_mismos_ids():
    """Los 12 grupos del primer peldano, y los ids del CSV del prerregistro (orden 1 = el mas alto)
    son, ordenados de menor a mayor, los que entran en subsample_runs en cada peldano."""
    cfg = _cfg()
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    semilla = config.get(cfg, "protocolo.semilla")
    grupos = ablation.subsample_runs(_ids_confirmatorios(100), 100 // R, R, semilla, disjoint=True)
    assert grupos.tolist() == GRUPOS_N100
    csv = pd.read_csv(CSV_IDS, comment="#")
    prueba = csv[csv["conjunto"] == "prueba"]
    for n in GRUPO_1:
        del_csv = np.sort(prueba.loc[prueba["orden"] <= n, "simulationRun"].to_numpy())
        assert np.array_equal(del_csv, _ids_confirmatorios(n)), n


def test_reservadas_14_son_las_de_menor_id_de_la_calibracion():
    """El brazo reservada_14: en cada replica, las 14 de menor id de sus 20 de calibracion."""
    cfg = _cfg()
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    n_cal = config.get(cfg, "particion.rieth.normales_calibracion")
    n_res = config.get(cfg, "particion.rieth.normales_ajuste")   # el tamano del brazo en muestra
    assert (n_cal, n_res) == (20, 14)
    assert "reservada_14" in config.get(cfg, "ablaciones.brazos_factor1.secundarios")
    csv = pd.read_csv(CSV_IDS, comment="#")
    cal = csv[(csv["conjunto"] == "normales_train_fault00") & (csv["rol"] == "calibrar")]
    assert sorted(cal["replica"].astype(int).unique()) == list(range(1, R + 1))
    reservadas = {}
    for r, g in cal.groupby("replica"):
        ids = np.sort(g["simulationRun"].to_numpy())
        assert ids.size == n_cal
        reservadas[int(r)] = ids[:n_res].tolist()
        # orden es la posicion entre los 34 ids ordenados de la replica: menor orden, menor id
        assert sorted(g.nsmallest(n_res, "orden")["simulationRun"].tolist()) == reservadas[int(r)]
    for r, esperado in RESERVADAS_14.items():
        assert reservadas[r] == esperado, r


# ------------------------------- la unidad de la familia de la hipotesis (fila del 2026-11-03)
AGREGADOS = {"media": "mean"}         # hipotesis.unidad_replica.agregado, como lo aplica pandas


def _familia_anidada(cfg, ventaja_ae, n=100, semilla_datos=20261103, exceso_base=None,
                     cola_ae=None):
    """Resultados sinteticos con el diseno de hipotesis.unidad_replica y la familia que sale.

    La replica r evalua solo su grupo r (los grupos de ablaciones.unidad_replica, con los ids en
    orden creciente). Por (replica, simulacion, metodo, fallo): far de la gemela y fdr = far mas el
    exceso base del fallo (0,05 si no se pide otro), mas la ventaja del autoencoder que se pida por
    fallo y, si se pide cola_ae, un exceso extra del autoencoder en la primera simulacion del grupo
    de cada replica. El agregado dentro de la replica sale del yaml. No lee ningun dato."""
    agregar = AGREGADOS[config.get(cfg, "hipotesis.unidad_replica.agregado")]
    exceso_base = exceso_base or {}
    cola_ae = cola_ae or {}
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    semilla = config.get(cfg, "protocolo.semilla")
    grupos = ablation.subsample_runs(_ids_confirmatorios(n), n // R, R, semilla, disjoint=True)
    rng = np.random.default_rng(semilla_datos)
    baselines = [b["nombre"] for b in config.get(cfg, "hipotesis.baselines")]
    controles = config.get(cfg, "hipotesis.fallos_control.fallos")
    filas = []
    for r in range(R):
        u = rng.normal(0.0, 0.002)                          # la parte que depende del ajuste
        for sim in grupos[r]:
            for fallo in tuple(stats.CONFIRMATORY_FAULTS) + tuple(controles):
                far = 0.01 + rng.normal(0.0, 0.002)
                for metodo in ["ae"] + baselines:
                    extra = ventaja_ae.get(fallo, 0.0) + u if metodo == "ae" else 0.0
                    if metodo == "ae" and sim == grupos[r][0]:
                        extra += cola_ae.get(fallo, 0.0)
                    fdr = far + exceso_base.get(fallo, 0.05) + extra + rng.normal(0.0, 0.003)
                    filas.append({"replica": r + 1, "simulationRun": int(sim),
                                  "faultNumber": fallo, "method": metodo, "fdr": fdr, "far": far,
                                  "exceso_far_gemela": fdr - far})
    res = pd.DataFrame(filas)
    margen = config.get(cfg, "deteccion.suelo.margen")
    alpha = config.get(cfg, "contrastes.alpha_contrastes")

    def detecta(metodo, fallo):
        """detects_above_far sobre las R medias de replica de fdr y de far."""
        celda = res[(res["method"] == metodo) & (res["faultNumber"] == fallo)]
        celda = celda.dropna(subset=["fdr", "far"])      # las simulaciones con fdr y far validas
        medias = celda.groupby("replica")[["fdr", "far"]].agg(agregar)
        return stats.detects_above_far(medias["fdr"].to_numpy(), medias["far"].to_numpy(),
                                       margin=margen, alpha=alpha)

    def ventaja(baseline, fallo):
        """paired_advantage dentro del grupo de cada replica y la media por replica: R valores."""
        mejor = config.get(cfg, "hipotesis.mayor_es_mejor")
        por_replica = [stats.paired_advantage(g, "exceso_far_gemela", "ae", baseline,
                                              higher_is_better=mejor, fault=fallo).agg(agregar)
                       for _, g in res.groupby("replica")]
        return np.array(por_replica)

    familia = [{"fault": f, "method": "ae", "baseline": b, "advantage": ventaja(b, f),
                "detects_method": detecta("ae", f), "detects_baseline": detecta(b, f)}
               for f in stats.CONFIRMATORY_FAULTS for b in baselines]
    ctrl = [{"fault": f, "method": "ae", "baseline": b,
             "detects_method": detecta("ae", f), "detects_baseline": detecta(b, f)}
            for f in controles for b in baselines]
    return res, familia, ctrl, grupos


def test_hipotesis_con_el_diseno_anidado_apoya_sin_ventaja():
    """Sin ventaja del autoencoder: seis comparaciones de R = 12 unidades, y "apoya"."""
    cfg = _cfg()
    unidad = config.get(cfg, "hipotesis.unidad_replica")
    assert config.get(cfg, unidad["grupos_prueba"]) == "subsample_runs_disjunto"
    assert unidad["unidad"] == config.get(cfg, "ablaciones.unidad_replica.unidad") == "replicate"
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    res, familia, ctrl, grupos = _familia_anidada(cfg, {})
    assert grupos.tolist() == GRUPOS_N100                                  # los del factor 1
    assert np.setdiff1d(_ids_confirmatorios(100), grupos.ravel()).tolist() == SOBRAN[100]
    assert res.groupby("simulationRun")["replica"].nunique().max() == 1   # una por simulacion
    assert res["simulationRun"].nunique() == R * (100 // R)                # sobran n mod R
    suelo = familia[0]["detects_method"]
    assert suelo["n"] == R and suelo["p_value"] == 2.0 ** -R             # el suelo, con R unidades
    out = stats.hypothesis_verdict(familia, delta=config.get(cfg, "contrastes.delta.fdr"),
                                   alpha=config.get(cfg, "contrastes.alpha_contrastes"),
                                   controls=ctrl)
    tabla = out["table"]
    confirmatoria = tabla[tabla["role"] == "confirmatoria"]
    assert out["m"] == 6 and len(confirmatoria) == 6
    assert set(confirmatoria["n"]) == {R}                                  # la unidad es la replica
    assert out["verdict"] == stats.VERDICT_SUPPORTS


def test_hipotesis_con_el_diseno_anidado_refuta_con_ventaja_relevante():
    """Con 5 puntos de ventaja del autoencoder en el fallo 9 (delta = 2), "refuta" tras Holm."""
    cfg = _cfg()
    _, familia, ctrl, _ = _familia_anidada(cfg, {9: 0.05})
    out = stats.hypothesis_verdict(familia, delta=config.get(cfg, "contrastes.delta.fdr"),
                                   alpha=config.get(cfg, "contrastes.alpha_contrastes"),
                                   controls=ctrl)
    assert out["verdict"] == stats.VERDICT_REFUTES


def test_hipotesis_anidada_con_cola_hacia_arriba():
    """Lo que acepta la fila del 2026-11-03, por el mismo procedimiento que las demas pruebas.

    En el fallo 9 ningun metodo detecta (exceso base de 0,2 puntos), salvo el autoencoder en una
    de las 8 simulaciones de cada replica, con 24 puntos mas. Con medias de replica, su suelo pasa
    (3,2 puntos) y el del baseline no; por simulacion no pasaria (la pseudomediana es de unos 0,2).
    Su ventaja media por replica es de 3 puntos, mas que delta = 2: la comparacion sale "ventaja
    relevante" y el veredicto "refuta". Con la mediana dentro de cada replica, el suelo no pasaria
    y la ventaja seria de 0: fija el agregado y la unidad del suelo donde se aplica la regla."""
    cfg = _cfg()
    R = config.get(cfg, "particion.rieth.replicas_ajuste")
    margen = config.get(cfg, "deteccion.suelo.margen")
    alpha = config.get(cfg, "contrastes.alpha_contrastes")
    res, familia, ctrl, _ = _familia_anidada(cfg, {}, exceso_base={9: 0.002}, cola_ae={9: 0.24})
    nueve = [c for c in familia if c["fault"] == 9]
    assert len(nueve) == 2
    for c in nueve:
        assert c["detects_method"]["detects_above_far"] and c["detects_method"]["n"] == R
        assert not c["detects_baseline"]["detects_above_far"]
    ae9 = res[(res["method"] == "ae") & (res["faultNumber"] == 9)]
    por_simulacion = stats.detects_above_far(ae9["fdr"].to_numpy(), ae9["far"].to_numpy(),
                                             margin=margen, alpha=alpha)
    assert not por_simulacion["detects_above_far"]
    out = stats.hypothesis_verdict(familia, delta=config.get(cfg, "contrastes.delta.fdr"),
                                   alpha=alpha, controls=ctrl)
    tabla = out["table"]
    fila9 = tabla[(tabla["role"] == "confirmatoria") & (tabla["fault"] == 9)]
    assert set(fila9["outcome"]) == {stats.OUTCOME_RELEVANT}
    assert out["verdict"] == stats.VERDICT_REFUTES


# ---------------------- el criterio de mejor compromiso (filas del 2026-11-03 y del 2026-11-04)
OPERADORES = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt}


def _contraste_par(x, y, vx, vy, delta, mayor_es_mejor, alpha):
    """Un par de candidatos en un paso con delta (compromiso.*, "empate: <clave de un delta>").

    Las diferencias por unidad, orientadas para que positivo sea "y es mejor que x"; las unidades
    en que falta uno de los dos no entran, y se redondean a 10 decimales para que la coma
    flotante no cree diferencias que no existen. El contraste va en el sentido de su
    pseudomediana (el estimador de Hodges-Lehmann): non_superiority_test sobre la ventaja del
    mejor, y su p_above_delta. Con pseudomediana 0, o con menos de stats.MIN_PAIRS unidades en
    comun (no contrastable, contrastes.no_contrastables), p = 1 y ninguno es mejor."""
    d = np.asarray(vy, dtype=float) - np.asarray(vx, dtype=float)
    if not mayor_es_mejor:
        d = -d
    d = np.round(d[~np.isnan(d)], 10)
    par = {"par": (x, y), "n": int(d.size), "p": 1.0, "mejor": None, "peor": None,
           "contrastable": d.size >= stats.MIN_PAIRS}
    if not par["contrastable"]:
        return par
    hl = stats.hodges_lehmann(d)["hl_estimate"]
    if hl == 0:
        return par
    mejor, peor, ventaja = (y, x, d) if hl > 0 else (x, y, -d)
    t = stats.non_superiority_test(ventaja, delta, alpha)
    return {**par, "p": float(t["p_above_delta"]), "mejor": mejor, "peor": peor}


def _paso_con_delta(siguen, valores, metrica, delta, mayor_es_mejor, alpha):
    """Un paso con delta: todos los pares de los que siguen, Holm sobre ellos (los no
    contrastables cuentan con p = 1) y salen los peores de los pares significativos."""
    pares = [_contraste_par(x, y, valores[x][metrica], valores[y][metrica], delta,
                            mayor_es_mejor, alpha)
             for x, y in itertools.combinations(siguen, 2)]
    if pares:
        holm = stats.holm_correction([p["p"] for p in pares], alpha=alpha)
        for p, significativo in zip(pares, holm["significant"]):
            p["elimina"] = bool(significativo)
    return pares, {p["peor"] for p in pares if p["elimina"]}


def _criterio(cfg, tarea, por_unidad, de_prueba=None):
    """El criterio de compromiso.<tarea> del yaml, paso a paso (filas del 2026-11-03 y 04).

    stats.best_compromise compara cada paso solo con el mejor; esta regla, frente a cualquiera
    con Holm, vive aqui hasta que llegue la funcion del paquete con el commit de codigo.

    por_unidad: {candidato: {metrica: valores por unidad, NaN donde la unidad no es valida}}. En
    deteccion, la unidad es la replica, y la r de un candidato se empareja con la r de otro (las
    mismas normales y el mismo grupo de prueba). El coste es un valor por metodo. Una lista o un
    delta que sigan en null los pasa la prueba en de_prueba, y no son una decision.

    - Candidatos: los de la lista cerrada; uno de fuera es un error. La familia es la declarada
      (fila del 2026-11-04, como contrastes.no_contrastables): uno de la lista con menos de
      stats.MIN_PAIRS unidades validas (con todas sus metricas por unidad), o sin coste, sigue en
      todos los pasos con p = 1 en sus pares, no entra en la restriccion ni en los pasos con un
      numero, no elimina a nadie y no puede quedar elegido. Se reporta como desviacion.
    - Restricciones: sobre la media de las unidades validas, redondeada a 6 decimales.
    - Paso con delta: _paso_con_delta sobre los que siguen y las desviaciones, con Holm al
      contrastes.alpha_contrastes.
    - Paso con un numero: sale quien queda a mas de ese margen del mejor, con las diferencias
      redondeadas a 10 decimales (10 veces justas, en log10, son un empate).
    - Si en un paso salieran todos los que siguen, siguen todos.

    Devuelve seleccion, resultado (sin_compromiso, metodo, familia o compartido), familias,
    desviaciones, tabla (las medias) y pasos."""
    spec = config.get(cfg, f"compromiso.{tarea}")
    assert spec["eliminacion"] == "frente_a_cualquiera_con_holm"
    de_prueba = de_prueba or {}
    alpha = config.get(cfg, "contrastes.alpha_contrastes")
    lista = spec["candidatos"] if spec["candidatos"] is not None else de_prueba["candidatos"]
    familia = {c["nombre"]: c["familia"] for c in lista}
    fuera = sorted(set(por_unidad) - set(familia))
    if fuera:
        raise ValueError(f"fuera de la lista cerrada de compromiso.{tarea}: {fuera}")
    metricas = [r["metrica"] for r in spec["restricciones"]] + [o["metrica"] for o in spec["orden"]]
    valores = {c: {m: np.atleast_1d(np.asarray(v[m], dtype=float)) for m in metricas}
               for c, v in por_unidad.items()}

    def evaluable(c):
        if c not in valores:
            return False
        por_unidad_c = [valores[c][m] for m in metricas if valores[c][m].size > 1]
        un_valor = [valores[c][m] for m in metricas if valores[c][m].size == 1]
        validas = (np.all([~np.isnan(a) for a in por_unidad_c], axis=0) if por_unidad_c
                   else np.zeros(0, dtype=bool))
        return (int(validas.sum()) >= stats.MIN_PAIRS
                and not any(np.isnan(a).any() for a in un_valor))

    desviaciones = [c for c in familia if not evaluable(c)]
    siguen = [c for c in familia if c not in desviaciones]           # en el orden de la lista
    sin_datos = {m: np.array([np.nan]) for m in metricas}           # sus pares, con p = 1
    en_pasos = {**valores, **{c: sin_datos for c in desviaciones}}
    tabla = pd.DataFrame({c: {m: float(np.nanmean(valores[c][m])) for m in metricas}
                          for c in siguen}, index=metricas).T
    pasos = []
    for r in spec["restricciones"]:
        cumple, tope = OPERADORES[r["operador"]], round(float(r["valor"]), 6)
        salen = {c for c in siguen if not cumple(round(tabla.loc[c, r["metrica"]], 6), tope)}
        pasos.append({"paso": r["metrica"], "siguen": list(siguen), "pares": [],
                      "salen": [c for c in siguen if c in salen], "todos_salian": False})
        siguen = [c for c in siguen if c not in salen]
    for o in spec["orden"]:
        m, mayor = o["metrica"], o["direccion"] == "max"
        if isinstance(o["empate"], str):
            delta = config.get(cfg, o["empate"])
            if delta is None:
                delta = de_prueba[o["empate"]]
            carrera = [c for c in familia if c in siguen or c in desviaciones]
            pares, salen = _paso_con_delta(carrera, en_pasos, m, delta, mayor, alpha)
            salen &= set(siguen)
        else:
            pares, salen = [], set()
            if siguen:
                mejor = (max if mayor else min)(tabla.loc[c, m] for c in siguen)
                salen = {c for c in siguen
                         if round(abs(tabla.loc[c, m] - mejor), 10) > float(o["empate"])}
        # Que salgan todos exigiria un ciclo de ventajas relevantes (con un margen fijo no puede
        # pasar: el mejor no sale). Si pasara, siguen todos (fila del 2026-11-04).
        todos_salian = bool(siguen) and salen == set(siguen)
        if todos_salian:
            salen = set()
        pasos.append({"paso": m, "siguen": list(siguen), "pares": pares,
                      "salen": [c for c in siguen if c in salen], "todos_salian": todos_salian})
        siguen = [c for c in siguen if c not in salen]
    familias = sorted({familia[c] for c in siguen})
    if not siguen:
        resultado = "sin_compromiso"
    elif len(siguen) == 1:
        resultado = "metodo"
    else:
        resultado = "familia" if len(familias) == 1 else "compartido"
    return {"seleccion": siguen, "resultado": resultado, "familias": familias,
            "desviaciones": desviaciones, "tabla": tabla, "pasos": pasos}


CUATRO = ("pca", "dpca", "kpca", "ae")             # la lista cerrada de deteccion, en su orden
RETARDO_DE_PRUEBA = {"contrastes.delta.retardo_muestras": 10.0}   # de prueba: sigue en null
SEMILLA_RUIDOSO = 20261104                         # una en que el ruidoso tiene la mejor media
COSTE = {"pca": 2e-5, "dpca": 4e-5, "kpca": 5e-3, "ae": 1e-3}      # s por muestra, de prueba


def _cuatro(pca, dpca, kpca, ae):
    return dict(zip(CUATRO, (pca, dpca, kpca, ae)))


def _deteccion_sintetica(exceso, retardo, far, coste_s, semilla):
    """Por replica: exceso medio en los 17 fallos, retardo y FAR realizada; un coste por metodo."""
    rng = np.random.default_rng(semilla)
    R = 12
    out = {}
    for c in exceso:
        u = rng.normal(0.0, 0.003)                                     # la parte del ajuste
        out[c] = {"exceso_far_gemela": exceso[c] + u + rng.normal(0.0, 0.004, R),
                  "retardo": retardo[c] + rng.normal(0.0, 2.0, R),
                  "far": far[c] + rng.normal(0.0, 0.0008, R),
                  "log10_inferencia_s": np.log10(coste_s[c])}
    return out


def _fijas(exceso, retardo=30.0, far=0.010, coste_s=1e-4):
    """Unidades fijadas a mano: el exceso que se pase y lo demas igual en las 12 replicas."""
    return {"exceso_far_gemela": np.asarray(exceso, dtype=float),
            "retardo": np.broadcast_to(np.asarray(retardo, dtype=float), (12,)),
            "far": np.broadcast_to(np.asarray(far, dtype=float), (12,)),
            "log10_inferencia_s": np.log10(coste_s)}


def _sin(unidades, replicas):
    """Las mismas unidades con NaN en las replicas que se pidan (sin simulaciones validas)."""
    out = dict(unidades)
    for m in ("exceso_far_gemela", "retardo", "far"):
        out[m] = np.array(unidades[m], dtype=float)
        out[m][list(replicas)] = np.nan
    return out


def _e(negativos):
    """12 diferencias de 0,001 a 0,012, negativas en los rangos que se pidan: fija el W+ exacto."""
    e = 0.001 * np.arange(1, 13)
    e[[r - 1 for r in negativos]] *= -1
    return e


def test_compromiso_lista_cerrada_restriccion_y_referencias():
    """La lista cerrada con su familia, la regla de eliminacion, el tope de la FAR y los delta."""
    cfg = _cfg()
    spec = config.get(cfg, "compromiso.deteccion")
    assert [(c["nombre"], c["familia"]) for c in spec["candidatos"]] == [
        ("pca", 1), ("dpca", 1), ("kpca", 1), ("ae", 3)]
    # PCA y DPCA son los de la hipotesis: la misma configuracion en los dos analisis
    assert [b["nombre"] for b in config.get(cfg, "hipotesis.baselines")] == ["pca", "dpca"]
    for tarea in ("deteccion", "diagnostico"):
        assert config.get(cfg, f"compromiso.{tarea}.eliminacion") == "frente_a_cualquiera_con_holm"
    nominal = 1.0 - config.get(cfg, "protocolo.alpha")
    tope = nominal + config.get(cfg, "contrastes.delta.far")
    assert spec["restricciones"] == [{"metrica": "far", "operador": "<=", "valor": 0.015}]
    assert tope != 0.015 and round(tope, 6) == 0.015        # por eso se redondea a 6 decimales
    for tarea in ("deteccion", "diagnostico"):
        for o in config.get(cfg, f"compromiso.{tarea}.orden"):
            assert o["direccion"] in ("max", "min")
            if isinstance(o["empate"], str):
                config.get(cfg, o["empate"])                               # la clave existe
    # lo que falta sigue pendiente y se fija antes de sus confirmatorios
    pendientes = set(config.pending(cfg))
    assert {"contrastes.delta.retardo_muestras", "contrastes.delta.macro_f1",
            "diagnostico.etiqueta_prefallo", "compromiso.diagnostico.candidatos"} <= pendientes
    with pytest.raises(KeyError):                     # sin su lista, el diagnostico no se aplica
        _criterio(cfg, "diagnostico", {}, {"contrastes.delta.macro_f1": 0.02})


@pytest.mark.parametrize("escenario, esperado, resultado", [
    ("el autoencoder saca 5 puntos de exceso", ["ae"], "metodo"),
    ("empate en exceso, decide el retardo", ["ae"], "metodo"),
    ("empate en exceso y retardo, decide el coste", ["dpca"], "metodo"),
    ("el autoencoder no cumple la FAR", ["pca", "dpca"], "familia"),
    ("KPCA y autoencoder empatan en todo", ["kpca", "ae"], "compartido"),
])
def test_compromiso_deteccion_en_cinco_escenarios(escenario, esperado, resultado):
    """El criterio del yaml con los cuatro candidatos de la lista (el delta del retardo, de
    prueba: 10 muestras)."""
    cfg = _cfg()
    casos = {                       # exceso, retardo (muestras), FAR realizada, coste, semilla
        "el autoencoder saca 5 puntos de exceso": (
            _cuatro(0.40, 0.42, 0.42, 0.47), _cuatro(30, 28, 29, 25),
            _cuatro(0.010, 0.011, 0.011, 0.012), COSTE, 1),
        "empate en exceso, decide el retardo": (
            _cuatro(0.40, 0.455, 0.45, 0.46), _cuatro(30, 40, 38, 22),
            _cuatro(0.010, 0.011, 0.011, 0.012), COSTE, 2),
        "empate en exceso y retardo, decide el coste": (
            _cuatro(0.40, 0.455, 0.45, 0.46), _cuatro(30, 26, 27, 25),
            _cuatro(0.010, 0.011, 0.011, 0.012), COSTE, 3),
        "el autoencoder no cumple la FAR": (
            _cuatro(0.40, 0.41, 0.405, 0.50), _cuatro(30, 28, 29, 20),
            _cuatro(0.010, 0.011, 0.011, 0.019), COSTE, 4),
        "KPCA y autoencoder empatan en todo": (
            _cuatro(0.40, 0.41, 0.46, 0.465), _cuatro(30, 28, 24, 25),
            _cuatro(0.010, 0.011, 0.012, 0.012), _cuatro(2e-5, 4e-5, 2e-3, 1e-3), 5),
    }
    exceso, retardo, far, coste, semilla = casos[escenario]
    datos = _deteccion_sintetica(exceso, retardo, far, coste, semilla)
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["seleccion"] == esperado, out["pasos"]
    assert out["resultado"] == resultado and out["desviaciones"] == []
    k = len(out["pasos"][1]["siguen"])
    assert len(out["pasos"][1]["pares"]) == k * (k - 1) // 2       # todos los pares del paso
    if escenario == "el autoencoder saca 5 puntos de exceso":
        spec = config.get(cfg, "compromiso.deteccion")
        tolerancia = {"exceso_far_gemela": config.get(cfg, "contrastes.delta.fdr"),
                      "retardo": 10.0, "log10_inferencia_s": 1.0}
        objetivos = {o["metrica"]: o["direccion"] for o in spec["orden"]}
        frente = stats.pareto_front(out["tabla"].drop(columns="far"), objetivos,
                                    tolerance=tolerancia)
        assert frente.index[frente["pareto"]].tolist() == ["pca", "dpca", "ae"]  # no elige


def test_compromiso_holm_dentro_del_paso_y_lista_cerrada():
    """Holm sobre los 6 pares del paso: una ventaja del autoencoder de p = 0,026 sobre los otros
    tres no elimina a nadie; sin Holm, o en un paso con solo dos, eliminaria. Por eso la familia es
    la declarada: si solo se ejecutan dos, los que faltan cuentan con p = 1 y m sigue en 6. Con
    p = 0,065, ni en un paso de dos."""
    cfg = _cfg()
    alpha = config.get(cfg, "contrastes.alpha_contrastes")
    delta = config.get(cfg, "contrastes.delta.fdr")
    base = 0.45 + 0.001 * np.arange(12)
    datos = {c: _fijas(base) for c in ("pca", "dpca", "kpca")}
    datos["ae"] = _fijas(base + delta + _e([5, 9]))               # W+ = 64 sobre ventaja - delta
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    paso = out["pasos"][1]
    assert paso["paso"] == "exceso_far_gemela" and len(paso["pares"]) == 6
    del_ae = [p for p in paso["pares"] if "ae" in p["par"]]
    assert len(del_ae) == 3
    assert all(p["mejor"] == "ae" and alpha / 6 < p["p"] < alpha for p in del_ae)
    assert all(p["p"] == 1.0 and p["mejor"] is None for p in paso["pares"] if "ae" not in p["par"])
    assert paso["salen"] == [] and out["seleccion"] == list(CUATRO)
    assert out["resultado"] == "compartido"
    dos = _criterio(cfg, "deteccion", {"kpca": datos["kpca"], "ae": datos["ae"]},
                    RETARDO_DE_PRUEBA)
    assert dos["desviaciones"] == ["pca", "dpca"] and len(dos["pasos"][1]["pares"]) == 6
    assert dos["seleccion"] == ["kpca", "ae"] and dos["resultado"] == "compartido"
    m = "exceso_far_gemela"
    pares, salen = _paso_con_delta(["kpca", "ae"], datos, m, delta, True, alpha)
    assert salen == {"kpca"}                                       # en un paso de dos, si
    flojo = {"kpca": datos["kpca"], "ae": _fijas(base + delta + _e([8, 11]))}   # W+ = 59
    pares, salen = _paso_con_delta(["kpca", "ae"], flojo, m, delta, True, alpha)
    assert alpha < pares[0]["p"] < 2 * alpha and salen == set()
    with pytest.raises(ValueError, match="fuera de la lista"):
        _criterio(cfg, "deteccion", {**datos, "kfcm": datos["ae"]}, RETARDO_DE_PRUEBA)


def test_compromiso_frente_a_cualquiera_con_un_ruidoso():
    """Un candidato ruidoso con la mejor media no salva a otro 4 puntos peor que un tercero (con
    la comparacion frente al mejor sobrevivia el 42 % de las veces, en las simulaciones)."""
    cfg = _cfg()
    rng = np.random.default_rng(SEMILLA_RUIDOSO)
    medias = _cuatro((0.46, 0.004), (0.50, 0.004), (0.51, 0.10), (0.50, 0.004))
    datos = {c: _fijas(m + rng.normal(0.0, sd, 12)) for c, (m, sd) in medias.items()}
    assert max(datos, key=lambda c: np.mean(datos[c]["exceso_far_gemela"])) == "kpca"
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["pasos"][1]["salen"] == ["pca"]
    assert out["seleccion"] == ["dpca", "kpca", "ae"] and out["resultado"] == "compartido"


def test_compromiso_sentido_por_la_pseudomediana():
    """El sentido de cada par lo da la pseudomediana, no la media ni la mediana. Con una replica en
    800 (la mediana no se alcanza), el PCA es 15 muestras mas rapido en 11 de 12 replicas y peor en
    media: en un paso de dos sale el autoencoder (p = 0,017). Con 40 unidades (la del OE2 puede ser
    mayor que 12), 21 diferencias de -0,001 y 19 de +0,10: la mediana es negativa, la pseudomediana
    positiva, y sale el que pierde por la pseudomediana."""
    cfg = _cfg()
    alpha = config.get(cfg, "contrastes.alpha_contrastes")
    pca = _fijas(np.full(12, 0.45), retardo=np.array([20.0] * 11 + [800.0]))
    ae = _fijas(np.full(12, 0.45), retardo=np.full(12, 35.0))
    assert np.mean(pca["retardo"]) > np.mean(ae["retardo"])
    pares, salen = _paso_con_delta(["pca", "ae"], {"pca": pca, "ae": ae}, "retardo", 10.0, False,
                                   alpha)
    assert pares[0]["mejor"] == "pca" and pares[0]["p"] < alpha and salen == {"ae"}
    d = np.r_[np.full(21, -0.001), np.full(19, 0.10)]
    assert np.median(d) < 0 < stats.hodges_lehmann(d)["hl_estimate"]
    cuarenta = {"x": {"macro_f1_rest": np.full(40, 0.70)},
                "y": {"macro_f1_rest": 0.70 + d}}
    pares, salen = _paso_con_delta(["x", "y"], cuarenta, "macro_f1_rest", 0.02, True, alpha)
    assert pares[0]["mejor"] == "y" and salen == {"x"}


def test_compromiso_datos_que_faltan_y_no_contrastables():
    """Una replica sin simulaciones validas sale de los pares de ese candidato. Un candidato con
    menos de 5 replicas validas es una desviacion: sus pares cuentan en Holm con p = 1, no elimina,
    no entra en el coste y no puede quedar elegido (fila del 2026-11-04). Con ellos, una ventaja
    del autoencoder de p = 0,013 no elimina; sin ellos (m = 3), eliminaria. Igual con 4 replicas,
    con 0 o sin ejecutar; con 5 replicas ya cuenta."""
    cfg = _cfg()
    alpha = config.get(cfg, "contrastes.alpha_contrastes")
    datos = _deteccion_sintetica(_cuatro(0.40, 0.42, 0.42, 0.47), _cuatro(30, 28, 29, 25),
                                 _cuatro(0.010, 0.011, 0.011, 0.012), COSTE, 1)
    datos["ae"] = _sin(datos["ae"], [3])               # la replica 4 del autoencoder
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["seleccion"] == ["ae"]
    assert {p["n"] for p in out["pasos"][1]["pares"] if "ae" in p["par"]} == {11}
    assert {p["n"] for p in out["pasos"][1]["pares"] if "ae" not in p["par"]} == {12}
    base = 0.45 + 0.001 * np.arange(12)
    pocos = {"pca": _fijas(base), "dpca": _fijas(base),
             "kpca": _sin(_fijas(base - 0.15, coste_s=5e-3), range(4, 12)),  # 4 replicas
             "ae": _fijas(base + config.get(cfg, "contrastes.delta.fdr") + _e([11]))}  # W+ = 67
    out = _criterio(cfg, "deteccion", pocos, RETARDO_DE_PRUEBA)
    exceso = out["pasos"][1]
    del_kpca = [p for p in exceso["pares"] if "kpca" in p["par"]]
    assert len(exceso["pares"]) == 6 and len(del_kpca) == 3
    assert all(not p["contrastable"] and p["p"] == 1.0 and p["n"] == 0 for p in del_kpca)
    assert all(alpha / 6 < p["p"] < alpha / 3 for p in exceso["pares"]
               if "ae" in p["par"] and "kpca" not in p["par"])
    assert exceso["salen"] == [] and out["desviaciones"] == ["kpca"]
    assert "kpca" not in out["pasos"][0]["siguen"] + out["pasos"][3]["siguen"]
    assert out["seleccion"] == ["pca", "dpca", "ae"] and out["resultado"] == "compartido"
    sin_datos = {**pocos, "kpca": _sin(pocos["kpca"], range(12))}
    sin_ejecutar = {c: v for c, v in pocos.items() if c != "kpca"}
    for otro in (sin_datos, sin_ejecutar):
        o = _criterio(cfg, "deteccion", otro, RETARDO_DE_PRUEBA)
        assert o["desviaciones"] == ["kpca"] and o["seleccion"] == out["seleccion"]
        assert [len(p["pares"]) for p in o["pasos"]] == [len(p["pares"]) for p in out["pasos"]]
    cinco = {**pocos, "kpca": _sin(_fijas(base - 0.15, coste_s=5e-3), range(5, 12))}
    o = _criterio(cfg, "deteccion", cinco, RETARDO_DE_PRUEBA)
    assert o["desviaciones"] == [] and "kpca" in o["pasos"][1]["siguen"]


def test_compromiso_fronteras_de_la_far_y_del_coste():
    """La FAR se compara redondeada a 6 decimales: una media que en coma flotante pasa de 0,015
    cumple. En el coste, 10 veces justas son un empate aunque en coma flotante el log10 pase de 1;
    algo mas, no."""
    cfg = _cfg()
    far = np.array([0.015297, 0.015446, 0.01516, 0.014591, 0.015366, 0.014749, 0.01544, 0.014464,
                    0.015457, 0.01499, 0.014376, 0.014664])
    assert np.mean(far) > 0.015 and round(float(np.mean(far)), 6) == 0.015
    exceso = np.full(12, 0.45)
    assert np.log10(6e-3) - np.log10(6e-4) > 1.0
    datos = {"pca": _fijas(exceso, coste_s=6e-4), "dpca": _fijas(exceso, coste_s=6e-4),
             "kpca": _fijas(exceso, coste_s=6e-3), "ae": _fijas(exceso, far=far, coste_s=6e-3)}
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["pasos"][0]["salen"] == []                      # la FAR del autoencoder cumple
    assert out["seleccion"] == list(CUATRO)                    # 10 veces justas: empate
    datos["kpca"] = _fijas(exceso, coste_s=6.006e-3)
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["pasos"][3]["salen"] == ["kpca"]
    datos["ae"] = _fijas(exceso, far=0.015001, coste_s=6e-3)       # una millonesima por encima
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["pasos"][0]["salen"] == ["ae"]


def test_compromiso_sin_candidatos_que_cumplan():
    """Si ninguno cumple el tope de la FAR, esa tarea no tiene compromiso y se reporta asi."""
    cfg = _cfg()
    datos = _deteccion_sintetica(_cuatro(0.40, 0.42, 0.42, 0.47), _cuatro(30, 28, 29, 25),
                                 _cuatro(0.02, 0.021, 0.019, 0.03), COSTE, 7)
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["seleccion"] == [] and out["resultado"] == "sin_compromiso"
    assert out["pasos"][0]["salen"] == list(CUATRO)
    assert all(p["pares"] == [] and p["salen"] == [] for p in out["pasos"][1:])


def test_compromiso_diagnostico_macro_f1_y_coste():
    """Diagnostico: sin restriccion; la macro-F1 (delta de prueba: 0,02) y despues el coste, con
    una lista de prueba (la de verdad sigue en null)."""
    cfg = _cfg()
    rng = np.random.default_rng(20261103)
    U = 12
    de_prueba = {"contrastes.delta.macro_f1": 0.02,
                 "candidatos": [{"nombre": "rf", "familia": 2}, {"nombre": "knn", "familia": 2},
                                {"nombre": "rnn", "familia": 3}]}

    def unidades(f1, coste_s):
        return {"macro_f1_rest": f1 + rng.normal(0.0, 0.004, U),
                "log10_inferencia_s": np.log10(coste_s)}

    claro = {"rf": unidades(0.80, 1e-5), "knn": unidades(0.72, 1e-4), "rnn": unidades(0.74, 1e-3)}
    out = _criterio(cfg, "diagnostico", claro, de_prueba)
    assert out["pasos"][0]["paso"] == "macro_f1_rest"          # sin restriccion
    assert out["seleccion"] == ["rf"] and out["resultado"] == "metodo"
    empate = {"rf": unidades(0.79, 1e-3), "knn": unidades(0.70, 1e-4), "rnn": unidades(0.80, 5e-3)}
    out = _criterio(cfg, "diagnostico", empate, de_prueba)
    assert out["seleccion"] == ["rf", "rnn"] and out["resultado"] == "compartido"
    assert out["familias"] == [2, 3]                           # 5 veces de coste: empate
    caro = {"rf": unidades(0.79, 1e-4), "knn": unidades(0.70, 1e-4), "rnn": unidades(0.80, 2e-3)}
    out = _criterio(cfg, "diagnostico", caro, de_prueba)
    assert out["pasos"][1]["salen"] == ["rnn"] and out["seleccion"] == ["rf"]   # 20 veces
    # 2,8 puntos de diferencia: con el delta de prueba de 3 puntos empatan; con el de la FDR (2),
    # el rf dejaria fuera a la rnn. Fija que el paso usa contrastes.delta.macro_f1
    cerca = {"rf": unidades(0.80, 1e-3), "knn": unidades(0.70, 1e-4), "rnn": unidades(0.772, 2e-3)}
    out = _criterio(cfg, "diagnostico", cerca, {**de_prueba, "contrastes.delta.macro_f1": 0.03})
    assert out["seleccion"] == ["rf", "rnn"]
    out = _criterio(cfg, "diagnostico", cerca,
                    {**de_prueba, "contrastes.delta.macro_f1": config.get(cfg, "contrastes.delta.fdr")})
    assert out["seleccion"] == ["rf"]


def test_compromiso_holm_por_pasos_y_no_bonferroni():
    """Holm rechaza por pasos: con cuatro p muy pequenos delante, el quinto (p = 0,0105, mayor que
    alpha / 6) tambien elimina, y gana el autoencoder; con Bonferroni, el KPCA seguiria."""
    cfg = _cfg()
    alpha = config.get(cfg, "contrastes.alpha_contrastes")
    delta = config.get(cfg, "contrastes.delta.fdr")
    base = 0.45 + 0.001 * np.arange(12)
    datos = {"pca": _fijas(base - 0.05), "dpca": _fijas(base - 0.05), "kpca": _fijas(base),
             "ae": _fijas(base + delta + _e([10]))}                    # W+ = 68 frente al KPCA
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    kpca_ae = [p for p in out["pasos"][1]["pares"] if p["par"] == ("kpca", "ae")][0]
    assert alpha / 6 < kpca_ae["p"] < alpha / 2 and kpca_ae["elimina"]
    assert out["seleccion"] == ["ae"] and out["resultado"] == "metodo"


def test_compromiso_cinco_unidades_justas():
    """Con 5 unidades en comun (stats.MIN_PAIRS) el par ya es contrastable: si las 5 van a favor
    por mas de delta, p = 1/32 y, en un paso de dos, elimina; con 4, no."""
    cfg = _cfg()
    alpha = config.get(cfg, "contrastes.alpha_contrastes")
    x = np.r_[np.full(5, 0.40), np.full(7, np.nan)]
    valores = {"x": {"exceso_far_gemela": x}, "y": {"exceso_far_gemela": x + 0.05}}
    pares, salen = _paso_con_delta(["x", "y"], valores, "exceso_far_gemela", 0.02, True, alpha)
    assert pares[0]["contrastable"] and pares[0]["p"] == 1 / 32 and salen == {"x"}
    valores["y"]["exceso_far_gemela"] = np.r_[x[:4] + 0.05, np.full(8, np.nan)]
    pares, salen = _paso_con_delta(["x", "y"], valores, "exceso_far_gemela", 0.02, True, alpha)
    assert not pares[0]["contrastable"] and salen == set()


def test_compromiso_si_salieran_todos_siguen_todos(monkeypatch):
    """Que salgan todos en un paso exigiria un ciclo de ventajas relevantes, que no se ha
    encontrado con 12 replicas; la regla dice que entonces siguen todos (fila del 2026-11-04)."""
    cfg = _cfg()
    datos = {c: _fijas(np.full(12, 0.45)) for c in CUATRO}

    def todos_fuera(siguen, valores, metrica, delta, mayor_es_mejor, alpha):
        return [], set(siguen) if metrica == "exceso_far_gemela" else set()

    monkeypatch.setattr(sys.modules[__name__], "_paso_con_delta", todos_fuera)
    out = _criterio(cfg, "deteccion", datos, RETARDO_DE_PRUEBA)
    assert out["pasos"][1]["todos_salian"] and out["pasos"][1]["salen"] == []
    assert out["seleccion"] == list(CUATRO)

def test_compromiso_que_es_una_replica_valida_y_sin_coste():
    """Una replica es valida si tiene todas las metricas del criterio, y un candidato sin coste
    medido tambien es una desviacion (fila del 2026-11-04). Con el exceso valido en 6 replicas y la
    FAR y el retardo en las otras 6, no hay ninguna valida."""
    cfg = _cfg()
    base = 0.45 + 0.001 * np.arange(12)
    datos = {c: _fijas(base) for c in ("pca", "dpca", "ae")}
    cruzado = _fijas(base)
    cruzado["exceso_far_gemela"] = np.r_[base[:6], np.full(6, np.nan)]
    cruzado["far"] = np.r_[np.full(6, np.nan), np.full(6, 0.010)]
    cruzado["retardo"] = np.r_[np.full(6, np.nan), np.full(6, 30.0)]
    out = _criterio(cfg, "deteccion", {**datos, "kpca": cruzado}, RETARDO_DE_PRUEBA)
    assert out["desviaciones"] == ["kpca"]
    sin_coste = _fijas(base, coste_s=np.nan)
    out = _criterio(cfg, "deteccion", {**datos, "kpca": sin_coste}, RETARDO_DE_PRUEBA)
    assert out["desviaciones"] == ["kpca"] and "kpca" not in out["seleccion"]


# ------------------------------------------------- las claves del OE3 (filas del 2026-11-04)
MEDIANAS = {"censurada": "delay_samples_median_censored",     # deteccion.mediana_retardo
            "condicionada": "delay_samples_median"}


def test_mediana_del_retardo_censurada():
    """La mediana que se reporta es la censurada: con 8 simulaciones (las de una replica con
    n = 100), si solo 2 detectan, a las 10 muestras, no se alcanza; la condicionada daria 10.
    Con 4 de 8 sin detectar tampoco se alcanza (la regla del compromiso la cuenta como 800
    muestras)."""
    cfg = _cfg()
    assert config.get(cfg, "deteccion.mediana_retardo") == "censurada"
    columna = MEDIANAS[config.get(cfg, "deteccion.mediana_retardo")]

    def mediana(detectan, retardos):
        n = len(detectan)
        df = pd.DataFrame({"faultNumber": 1, "fdr": [0.5] * n, "detected": detectan,
                           "detected_at_chance": [False] * n,
                           "delay_samples": [r if d else np.nan
                                             for r, d in zip(retardos, detectan)]})
        return metrics.aggregate(df).iloc[0]

    pocas = mediana([True, True] + [False] * 6, [10, 10] + [0] * 6)
    assert np.isinf(pocas[columna]) and pocas["delay_samples_median"] == 10
    assert pocas["valid_detection_ratio"] == 0.25
    todas = mediana([True] * 8, [20, 25, 30, 35, 40, 45, 50, 55])
    assert todas[columna] == todas["delay_samples_median"] == 37.5
    mitad = mediana([True] * 4 + [False] * 4, [20, 25, 30, 35, 0, 0, 0, 0])
    assert np.isinf(mitad[columna])


FALSAS_ALARMAS = {"episodios": "false_alarm_episodes_per_hour",  # deteccion.falsas_alarmas_por_hora
                  "muestras": "false_alarm_samples_per_hour"}


def test_falsas_alarmas_por_hora_en_episodios():
    """Las falsas alarmas por hora son episodios: en las 800 muestras de la ventana de la FAR de
    test_fault00 (40 horas), una racha de 3 alarmas, una de 2 y dos sueltas son 1 episodio, 0,025
    por hora; una racha de 3 fuera de la ventana no cuenta. Las muestras en alarma por hora son la
    FAR por 20."""
    cfg = _cfg()
    assert config.get(cfg, "deteccion.falsas_alarmas_por_hora") == "episodios"
    columna = FALSAS_ALARMAS[config.get(cfg, "deteccion.falsas_alarmas_por_hora")]
    k = config.get(cfg, "protocolo.k_alarmas")
    minutos = config.get(cfg, "protocolo.sample_minutes")
    ventana = config.get(cfg, "deteccion.ventana_far_normal")
    assert minutos == 3.0 and ventana == [160, None]                # 800 muestras: 40 horas
    stat = np.zeros(960)
    stat[[300, 301, 302]] = 2.0                        # una racha de k = 3
    stat[[400, 401]] = 2.0                             # una racha de k - 1
    stat[[500, 700]] = 2.0                             # dos sueltas
    stat[100:103] = 2.0                                # una racha de k fuera de la ventana
    fila = metrics.evaluate_run(stat, 1.0, onset=None, k=k, sample_minutes=minutos,
                                far_window=ventana)
    assert fila["n_far"] == 800 and fila["far"] == 7 / 800
    assert fila[columna] == 1 / 40
    assert fila["false_alarm_samples_per_hour"] == 7 / 40
    assert fila["false_alarm_samples_per_hour"] == pytest.approx(fila["far"] * 60 / minutos)


def _conjunto_pr_auc(cfg, fallo, gemela):
    """Las puntuaciones y etiquetas de deteccion.pr_auc_conjunto para una simulacion."""
    assert config.get(cfg, "deteccion.pr_auc_conjunto") == "por_simulacion_con_gemela"
    inicio, fin = config.get(cfg, "deteccion.ventana_far_normal")
    positivas, negativas = fallo[inicio:fin], gemela[inicio:fin]
    return (np.r_[positivas, negativas],
            np.r_[np.ones(positivas.size), np.zeros(negativas.size)])


def test_pr_auc_por_simulacion_con_su_gemela():
    """La PR-AUC de cada simulacion enfrenta las muestras 161 a 960 del fichero con fallo a las de
    su gemela de test_fault00: prevalencia del 50 %, asi que una puntuacion que no separa da 0,5
    (con el fichero entero daria 800/960) y una que separa del todo da 1."""
    cfg = _cfg()
    gemela = np.zeros(960)
    igual = gemela.copy()                                  # no separa: todo empata
    s, y = _conjunto_pr_auc(cfg, igual, gemela)
    assert y.mean() == 0.5 and metrics.pr_auc(s, y) == 0.5
    assert metrics.pr_auc(igual, np.r_[np.zeros(160), np.ones(800)]) == 800 / 960
    separa = np.r_[np.zeros(160), np.ones(800)]
    s, y = _conjunto_pr_auc(cfg, separa, gemela)
    assert metrics.pr_auc(s, y) == 1.0

