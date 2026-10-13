"""A0 - Calibrar los limites de control con los datos del ajuste o con datos reservados (desarrollo).

Experimento controlado sobre el TEP clasico: MISMO modelo (mismo escalador,
mismas cargas, mismas componentes) y solo cambia de donde sale el limite.

Para que sirve. Es el desarrollo del factor 1 de las ablaciones (calibracion de
limites en muestra frente a reservada) y la prueba de que el codigo mide lo que
dice. NO es una fila de la tabla de ablaciones: las cuatro ablaciones se hacen
sobre Rieth, con muchas simulaciones independientes y un contraste pareado
(memoria, parrafo 367), y el TEP clasico, con una sola simulacion normal, solo
se usa en el desarrollo y en la verificacion. Aqui no hay contraste ni
intervalo: cada cifra es una realizacion. Calibrar con los datos del ajuste no
es una fuga sino un sesgo, y su signo no se presupone (parrafos 279, 283 y 284):
se mide y se reporta con el que salga.

Brazos del SPE (con --spe-method, box por defecto; mismas cargas en todos):

- en_muestra: el limite se calibra con el SPE de todas las filas del ajuste
  (calibration="in_sample", a proposito). Es el brazo de siempre.
- reservada: con el SPE de X_cal, las ultimas filas de d00.dat, que el modelo no
  vio. Es el brazo de siempre.
- en_muestra_mismo_ncal: en muestra, pero con tantas filas del ajuste como tiene
  X_cal (las ultimas de X_fit). Separa el efecto de la procedencia del efecto del
  tamano de calibracion, que en los limites empiricos tambien mueve la FAR
  (DIS-4).
- jackson: el limite de Jackson y Mudholkar, que no usa X_cal pero sale de los
  autovalores del ajuste: una calibracion en muestra implicita (DIS-6). Se omite
  si --spe-method ya es jackson (entonces en_muestra y reservada dan el mismo
  limite, y PCAMonitor avisa de que X_cal no interviene).

Brazos del T2 (DIS-5), en un CSV aparte:

- f: el limite F de Fase II, el de siempre, parametrico con el n del ajuste.
- beta: el limite Beta de Fase I, la version parametrica de calibrar en muestra.
- empirico_reservada: percentil empirico del T2 de X_cal.
- empirico_en_muestra: percentil empirico del T2 de las ultimas filas del ajuste,
  con el mismo n_cal que el reservado (DIS-4).
La columna far_teorica_faseII da la FAR que cada limite parametrico tendria en
datos nuevos independientes, segun la distribucion F de Fase II (1 - alpha para
el F); en los empiricos no hay formula y va vacia.

FAR: la de las muestras sanas, las 160 previas al fallo de d01_te a d21_te
(media en far_*_media_prefallo) y d00_te entero (far_*_d00_te), con el limite de
cada brazo. Las ventanas van declaradas en las columnas far_window_*. La ventana
del protocolo para la FAR normal es una decision pendiente
(deteccion.ventana_far_normal del yaml); aqui se mantiene la de siempre.

El CSV guarda la configuracion junto a cada cifra, y cada CSV lleva al lado su
.provenance.json (configuracion, ficheros con su huella, commit y versiones):
sin eso, dos corridas con distinto numero de componentes acaban citadas como si
fueran la misma (paso con 17 y 27 componentes). La columna n_cal es el tamano
del bloque reservado X_cal (el mismo en todas las filas, como siempre); cuantos
valores calibraron de verdad cada limite lo dice n_cal_used (vacia en los
limites parametricos).

Uso:
    python experiments/A0_calibracion_limites.py
    python experiments/A0_calibracion_limites.py --yin-vars
    python experiments/A0_calibracion_limites.py --variance 0.90 --spe-method kde --out results/raw/A0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from tfmfdd import config, data, metrics, provenance
from tfmfdd.limits import EMPIRICAL_METHODS
from tfmfdd.pca import SPE_METHODS, PCAMonitor

#: Ventanas de la FAR, en el formato de far_window de metrics.evaluate_run.
VENTANA_D00_TE = "[0, null]"   # d00_te entero
VENTANA_PREFALLO = "[0, 160]"  # las 160 muestras previas al fallo de d01_te a d21_te


def _argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="baseline.yaml (por defecto, el orden de tfmfdd.config.load_baseline)")
    p.add_argument("--alpha", type=float, default=None, help="nivel de confianza (protocolo.alpha)")
    p.add_argument("--variance", type=float, default=None, help="varianza acumulada (pca.variance)")
    p.add_argument("--spe-method", default=None, choices=SPE_METHODS,
                   help="limite del SPE de los brazos en_muestra y reservada (pca.spe_method)")
    p.add_argument("--quantile-method", default=None, choices=EMPIRICAL_METHODS,
                   help="percentil de los limites empirical (pca.quantile_method)")
    p.add_argument("--cal-fraction", type=float, default=None,
                   help="fraccion de d00.dat reservada para calibrar (particion.calibracion_limites)")
    p.add_argument("--yin-vars", action="store_true",
                   help="usar las 33 variables de Yin et al. (2012) en vez de las 52")
    p.add_argument("--root", default="data/tep_classic")
    p.add_argument("--out", default="results/summary")
    return p.parse_args(argv)


def _valor(args, nombre: str, cfg: dict, clave: str):
    v = getattr(args, nombre)
    return config.get(cfg, clave) if v is None else v


def cargar_prueba(root: Path, columns) -> dict:
    """Los 22 ficheros de prueba, una vez (se puntuan con todos los brazos)."""
    return {f: data.values(data.load_classic(f, "test", root=root), columns=columns) for f in range(22)}


def far_por_fichero(modelo: PCAMonitor, prueba: dict, onset: int) -> pd.DataFrame:
    """FAR de T2 y SPE sobre las muestras sanas: las 160 previas de d01..d21 y d00_te entero."""
    filas = []
    for f, X in prueba.items():
        t2, spe = modelo.score(X)
        corte = slice(None) if f == 0 else slice(None, onset)
        filas.append({"faultNumber": f,
                      "far_t2": metrics.far(t2[corte], modelo.t2_limit_),
                      "far_spe": metrics.far(spe[corte], modelo.spe_limit_)})
    return pd.DataFrame(filas)


def _valores_calibracion(modelo: PCAMonitor, estadistico: str):
    """Cuantos valores calibraron el limite de ese estadistico (None si es parametrico)."""
    return None if modelo.limit_source_[estadistico] == "parametric_fit" else modelo.n_cal_


def far_teorica_fase2(t2_limit: float, n_components: int, n_fit: int) -> float:
    """P(T2 > limite) de una observacion nueva e independiente, con la F de Fase II.

    T2 ~ a (m + 1)(m - 1) / [m (m - a)] F(a, m - a), con a componentes y m filas
    del ajuste (normal multivariante; Tracy, Young y Mason, 1992, segun Yin et
    al., 2012, ec. 3). Para el limite F sale 1 - alpha por construccion; para el
    Beta de Fase I, la FAR que la teoria le predice en datos nuevos (DIS-5).
    """
    a, m = n_components, n_fit
    c = a * (m + 1) * (m - 1) / (m * (m - a))
    return float(sps.f.sf(t2_limit / c, a, m - a))


def ejecutar(argv=None) -> dict:
    args = _argumentos(argv)
    cfg, info_cfg = config.load_baseline(args.config, with_info=True)
    alpha = _valor(args, "alpha", cfg, "protocolo.alpha")
    variance = _valor(args, "variance", cfg, "pca.variance")
    spe_method = _valor(args, "spe_method", cfg, "pca.spe_method")
    quantile_method = _valor(args, "quantile_method", cfg, "pca.quantile_method")
    cal_fraction = _valor(args, "cal_fraction", cfg, "particion.calibracion_limites")
    t2_method = config.get(cfg, "pca.t2_method")   # el de los brazos del SPE (no interviene)
    root = Path(args.root)

    columns = data.YIN_VARS if args.yin_vars else None
    normal = data.values(data.load_classic(0, "train", root=root), columns=columns)
    corte = int(len(normal) * (1 - cal_fraction))
    X_fit, X_cal = normal[:corte], normal[corte:]
    n_cal = len(X_cal)
    X_fit_ultimas = X_fit[-n_cal:]       # en muestra, con el mismo n_cal que el reservado
    onset = data.FAULT_ONSET_TEST
    prueba = cargar_prueba(root, columns)

    def monitor(**kw) -> PCAMonitor:
        base = dict(variance=variance, alpha=alpha, spe_method=spe_method, t2_method=t2_method,
                    quantile_method=quantile_method)
        return PCAMonitor(**{**base, **kw})

    # Brazos del SPE. "en_muestra" y "reservada" son los de siempre.
    brazos_spe = {
        "en_muestra": monitor().fit(X_fit, calibration="in_sample"),
        "reservada": monitor().fit(X_fit, X_cal=X_cal, calibration="holdout"),
        "en_muestra_mismo_ncal": monitor().fit(X_fit, X_cal=X_fit_ultimas, calibration="in_sample"),
    }
    if spe_method != "jackson":
        brazos_spe["jackson"] = monitor(spe_method="jackson").fit(X_fit, calibration="in_sample")
    # Brazos del T2 (DIS-5): mismo modelo, solo cambia el limite del T2.
    brazos_t2 = {
        "f": monitor(t2_method="f").fit(X_fit, calibration="in_sample"),
        "beta": monitor(t2_method="beta").fit(X_fit, calibration="in_sample"),
        "empirico_reservada": monitor(t2_method="empirical").fit(X_fit, X_cal=X_cal, calibration="holdout"),
        "empirico_en_muestra": monitor(t2_method="empirical").fit(
            X_fit, X_cal=X_fit_ultimas, calibration="in_sample"),
    }

    referencia = brazos_spe["en_muestra"]
    for nombre, m in {**brazos_spe, **brazos_t2}.items():
        assert m.n_components_ == referencia.n_components_, f"{nombre}: otro numero de componentes"
        assert np.allclose(m.loadings_, referencia.loadings_), f"{nombre}: los modelos deben ser identicos"

    fars = {("spe", n): far_por_fichero(m, prueba, onset) for n, m in brazos_spe.items()}
    fars.update({("t2", n): far_por_fichero(m, prueba, onset) for n, m in brazos_t2.items()})

    comun = {"n_components": referencia.n_components_, "variance": variance, "alpha": alpha,
             "n_vars": len(columns) if columns else data.N_VARS, "n_fit": len(X_fit), "n_cal": n_cal}

    filas_spe = []
    for nombre, m in brazos_spe.items():
        tabla = fars[("spe", nombre)]
        pre = tabla[tabla["faultNumber"] > 0]["far_spe"]
        filas_spe.append({
            "calibracion": nombre,
            "spe_limit": m.spe_limit_,
            "far_spe_media_prefallo": pre.mean(),
            "far_spe_d00_te": float(tabla.loc[tabla["faultNumber"] == 0, "far_spe"].iloc[0]),
            "n_components": comun["n_components"], "variance": variance, "alpha": alpha,
            "spe_method": "jackson" if nombre == "jackson" else spe_method,
            "n_vars": comun["n_vars"], "n_fit": len(X_fit), "n_cal": n_cal,
            # Columnas nuevas (v0.3): de donde sale el limite y con cuantos valores.
            "statistic": "SPE", "limit_source": m.limit_source_["spe"],
            "n_cal_used": _valores_calibracion(m, "spe"),
            "quantile_method": quantile_method,
            "far_window_d00_te": VENTANA_D00_TE, "far_window_prefallo": VENTANA_PREFALLO,
        })
    resumen_spe = pd.DataFrame(filas_spe).astype({"n_cal_used": "Int64"})

    filas_t2 = []
    for nombre, m in brazos_t2.items():
        tabla = fars[("t2", nombre)]
        pre = tabla[tabla["faultNumber"] > 0]["far_t2"]
        parametrico = m.limit_source_["t2"] == "parametric_fit"
        filas_t2.append({
            "calibracion": nombre, "t2_method": m.t2_method, "t2_limit": m.t2_limit_,
            "far_t2_media_prefallo": pre.mean(),
            "far_t2_d00_te": float(tabla.loc[tabla["faultNumber"] == 0, "far_t2"].iloc[0]),
            "far_teorica_faseII": (far_teorica_fase2(m.t2_limit_, m.n_components_, m.n_fit_)
                                   if parametrico else np.nan),
            **comun, "statistic": "T2", "limit_source": m.limit_source_["t2"],
            "n_cal_used": _valores_calibracion(m, "t2"),
            "quantile_method": quantile_method,
            "far_window_d00_te": VENTANA_D00_TE, "far_window_prefallo": VENTANA_PREFALLO,
        })
    resumen_t2 = pd.DataFrame(filas_t2).astype({"n_cal_used": "Int64"})

    # Por fichero: las dos columnas de siempre y una por brazo nuevo.
    detalle = fars[("spe", "en_muestra")][["faultNumber", "far_spe"]].rename(
        columns={"far_spe": "far_spe_en_muestra"})
    detalle["far_spe_reservada"] = fars[("spe", "reservada")]["far_spe"].to_numpy()
    for nombre in brazos_spe:
        if nombre not in ("en_muestra", "reservada"):
            detalle[f"far_spe_{nombre}"] = fars[("spe", nombre)]["far_spe"].to_numpy()
    for nombre in brazos_t2:
        detalle[f"far_t2_{nombre}"] = fars[("t2", nombre)]["far_t2"].to_numpy()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sufijo = "_yin33" if args.yin_vars else ""
    configuracion = {**comun, "spe_method": spe_method, "t2_method_spe_arms": t2_method,
                     "quantile_method": quantile_method, "cal_fraction": cal_fraction,
                     "variables": "yin33" if args.yin_vars else "todas (52)",
                     "brazos_spe": list(brazos_spe), "brazos_t2": list(brazos_t2),
                     "far_window_d00_te": VENTANA_D00_TE, "far_window_prefallo": VENTANA_PREFALLO}
    # Git y maquina, una vez y antes de escribir ningun CSV (los tres JSON comparten ese estado).
    procedencia = dict(
        config=configuracion, config_files=[info_cfg], script=__file__, **provenance.run_context(__file__),
        ids={"fuente": "classic", "ajuste": f"d00.dat, filas 1 a {len(X_fit)}",
             "calibracion_reservada": f"d00.dat, filas {len(X_fit) + 1} a {len(normal)}",
             "calibracion_en_muestra_mismo_ncal": f"d00.dat, filas {len(X_fit) - n_cal + 1} a {len(X_fit)}",
             "prueba": [f"d{f:02d}_te.dat" for f in range(22)], "simulationRun": [1]},
        inputs=[root / "d00.dat"] + [root / f"d{f:02d}_te.dat" for f in range(22)],
        extra={"aviso": "desarrollo en el TEP clasico, no una fila de la tabla de ablaciones (parrafo 367)"},
    )
    rutas = {
        "spe": out / f"A0_calibracion_limites{sufijo}.csv",
        "t2": out / f"A0_calibracion_limites_t2{sufijo}.csv",
        "detalle": out / f"A0_calibracion_limites_por_fichero{sufijo}.csv",
    }
    for clave, tabla in (("spe", resumen_spe), ("t2", resumen_t2), ("detalle", detalle)):
        tabla.to_csv(rutas[clave], index=False)
        provenance.write_provenance(rutas[clave], **procedencia)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(resumen_spe[["calibracion", "spe_limit", "limit_source", "n_cal_used",
                           "far_spe_media_prefallo", "far_spe_d00_te"]].to_string(index=False))
        print()
        print(resumen_t2[["calibracion", "t2_limit", "limit_source", "n_cal_used", "far_t2_media_prefallo",
                          "far_t2_d00_te", "far_teorica_faseII"]].to_string(index=False))
    por = resumen_spe.set_index("calibracion")["far_spe_media_prefallo"]
    ratio = por["en_muestra"] / max(por["reservada"], 1e-12)
    print(f"\nFAR del SPE con calibracion en muestra / con calibracion reservada = {ratio:.2f} "
          f"(una realizacion, sin contraste: desarrollo)")
    print(f"Guardado en {out}/ ({', '.join(r.name for r in rutas.values())}, con su .provenance.json)")
    return {"spe": resumen_spe, "t2": resumen_t2, "detalle": detalle}


def main(argv=None) -> None:
    ejecutar(argv)


if __name__ == "__main__":
    main()
