"""A0 - Efecto de calibrar los limites de control con datos del ajuste.

Experimento controlado: MISMO modelo (mismo escalador, mismas cargas, mismas
componentes) y solo cambia de donde sale el limite del SPE.

  A) calibrado con los propios datos del ajuste (lo que hace buena parte de
     la literatura),
  B) calibrado con datos normales reservados que el modelo no vio.

La version anterior de este experimento (celda del notebook 01) ajustaba A con
las 500 muestras y B con 350, asi que cambiaban a la vez el tamano del ajuste y
la procedencia de la calibracion: un confundido. Aqui los dos modelos se ajustan
con X_fit y solo difiere el limite. El efecto existe unicamente para los
limites calibrados con datos (box, kde, empirical); el limite de T2 sale de la
distribucion F y no depende de X_cal, por eso no se reporta.

El CSV guarda la configuracion completa junto a cada cifra: sin eso, dos
corridas con distinto numero de componentes acaban citadas como si fueran la
misma (paso con 17 y 27 componentes). Es la primera fila de la tabla de
ablaciones del protocolo (capitulo 5).

Uso:
    python experiments/A0_calibracion_limites.py
    python experiments/A0_calibracion_limites.py --variance 0.90 --spe-method kde
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tfmfdd import data, metrics
from tfmfdd.pca import PCAMonitor

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def cargar_protocolo(path: str | Path = "configs/baseline.yaml") -> dict:
    path = Path(path)
    if yaml is None or not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def far_por_fichero(modelo: PCAMonitor, root: str, onset: int, columns) -> pd.DataFrame:
    """FAR del SPE sobre las muestras sanas: las 160 previas de d01..d21 y d00_te entero."""
    filas = []
    for f in range(0, 22):
        X = data.values(data.load_classic(f, "test", root=root), columns=columns)
        _, spe = modelo.score(X)
        tramo = spe if f == 0 else spe[:onset]
        filas.append({"faultNumber": f, "far_spe": metrics.far(tramo, modelo.spe_limit_)})
    return pd.DataFrame(filas)


def main() -> None:
    cfg = cargar_protocolo()
    prot, pca_cfg, part = cfg.get("protocolo", {}), cfg.get("pca", {}), cfg.get("particion", {})

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=prot.get("alpha", 0.99))
    p.add_argument("--variance", type=float, default=pca_cfg.get("variance", 0.85))
    p.add_argument("--spe-method", default=pca_cfg.get("spe_method", "box"),
                   choices=["box", "kde", "empirical"])
    p.add_argument("--cal-fraction", type=float, default=part.get("calibracion_limites", 0.3))
    p.add_argument("--yin-vars", action="store_true",
                   help="usar las 33 variables de Yin et al. (2012) en vez de las 52")
    p.add_argument("--root", default="data/tep_classic")
    p.add_argument("--out", default="results/summary")
    args = p.parse_args()

    columns = data.YIN_VARS if args.yin_vars else None
    normal = data.values(data.load_classic(0, "train", root=args.root), columns=columns)
    corte = int(len(normal) * (1 - args.cal_fraction))
    X_fit, X_cal = normal[:corte], normal[corte:]
    onset = data.FAULT_ONSET_TEST

    comun = dict(variance=args.variance, alpha=args.alpha, spe_method=args.spe_method)
    # A) mismo ajuste, limite calibrado en muestra (sobre X_fit)
    en_muestra = PCAMonitor(**comun).fit(X_fit)
    # B) mismo ajuste, limite calibrado con datos reservados (X_cal)
    reservada = PCAMonitor(**comun).fit(X_fit, X_cal=X_cal)

    assert en_muestra.n_components_ == reservada.n_components_
    assert np.allclose(en_muestra.loadings_, reservada.loadings_), "los modelos deben ser identicos"

    far_a = far_por_fichero(en_muestra, args.root, onset, columns)
    far_b = far_por_fichero(reservada, args.root, onset, columns)

    filas = []
    for nombre, m, tabla in (("en_muestra", en_muestra, far_a), ("reservada", reservada, far_b)):
        pre = tabla[tabla["faultNumber"] > 0]["far_spe"]
        filas.append({
            "calibracion": nombre,
            "spe_limit": m.spe_limit_,
            "far_spe_media_prefallo": pre.mean(),
            "far_spe_d00_te": float(tabla.loc[tabla["faultNumber"] == 0, "far_spe"].iloc[0]),
            "n_components": m.n_components_,
            "variance": args.variance,
            "alpha": args.alpha,
            "spe_method": args.spe_method,
            "n_vars": len(columns) if columns else data.N_VARS,
            "n_fit": len(X_fit),
            "n_cal": len(X_cal),
        })
    resumen = pd.DataFrame(filas)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sufijo = "_yin33" if args.yin_vars else ""
    resumen.to_csv(out / f"A0_calibracion_limites{sufijo}.csv", index=False)
    detalle = far_a.merge(far_b, on="faultNumber", suffixes=("_en_muestra", "_reservada"))
    detalle.to_csv(out / f"A0_calibracion_limites_por_fichero{sufijo}.csv", index=False)

    print(resumen.to_string(index=False))
    ratio = resumen.loc[0, "far_spe_media_prefallo"] / max(resumen.loc[1, "far_spe_media_prefallo"], 1e-12)
    print(f"\nFAR con calibracion en muestra / FAR con calibracion reservada = {ratio:.2f}")
    print(f"Guardado en {out}/A0_calibracion_limites{sufijo}.csv")


if __name__ == "__main__":
    main()
