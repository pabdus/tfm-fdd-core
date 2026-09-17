"""A1 - Baseline PCA sobre el TEP clasico, con verificacion contra la literatura.

Ajusta un modelo PCA con datos de operacion normal (d00.dat), calcula los
limites de control de T2 y SPE al nivel de confianza indicado, y evalua los 21
fallos del conjunto de prueba midiendo FAR, FDR y retardo de deteccion.

Este script cumple dos funciones. La primera es dar la linea base contra la que
se comparan todos los demas metodos del TFM. La segunda, y mas importante al
principio, es VERIFICAR EL PROTOCOLO: como Yin et al. (2012) y Russell et al.
(2000) publicaron sus cifras sobre exactamente estos ficheros, si nuestras
tasas se parecen a las suyas sabemos que la implementacion es correcta; si no,
el error es nuestro y hay que encontrarlo antes de construir nada encima.

Uso:
    python experiments/A1_pca_baseline.py --alpha 0.99 --variance 0.90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tfmfdd import data, metrics
from tfmfdd.pca import PCAMonitor


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.99, help="nivel de confianza")
    p.add_argument("--variance", type=float, default=0.90, help="varianza acumulada")
    p.add_argument("--lags", type=int, default=0, help="0 = PCA, 1 o 2 = DPCA")
    p.add_argument("--spe-method", default="box",
                   choices=["box", "jackson", "kde", "empirical"])
    p.add_argument("--k", type=int, default=3, help="alarmas consecutivas")
    p.add_argument("--cal-fraction", type=float, default=0.3,
                   help="fraccion de datos normales reservada para calibrar los limites")
    p.add_argument("--root", default="data/tep_classic")
    p.add_argument("--out", default="results/summary")
    args = p.parse_args()

    normal = data.values(data.load_classic(0, "train", root=args.root))

    # Se reserva una parte de los datos normales para calibrar los limites. Ver
    # la docstring de PCAMonitor.fit: calibrar con los mismos datos del ajuste
    # estrecha el umbral y dispara las falsas alarmas.
    corte = int(len(normal) * (1 - args.cal_fraction))
    X_fit, X_cal = normal[:corte], normal[corte:]

    modelo = PCAMonitor(
        variance=args.variance, alpha=args.alpha,
        lags=args.lags, spe_method=args.spe_method,
    ).fit(X_fit, X_cal=X_cal)

    print(f"Ajuste con {len(X_fit)} muestras, calibracion con {len(X_cal)}")
    print(f"Componentes retenidas: {modelo.n_components_} "
          f"({args.variance:.0%} de varianza)")
    print(f"Limite T2  = {modelo.t2_limit_:.2f}")
    print(f"Limite SPE = {modelo.spe_limit_:.2f}  ({args.spe_method})\n")

    filas = []
    for fault in range(22):
        df = data.load_classic(fault, "test", root=args.root)
        t2, spe = modelo.score(data.values(df))
        onset = None if fault == 0 else data.FAULT_ONSET_TEST

        for nombre, stat, limite in (("T2", t2, modelo.t2_limit_),
                                     ("SPE", spe, modelo.spe_limit_)):
            r = metrics.evaluate_run(stat, limite, onset, k=args.k)
            filas.append({"faultNumber": fault, "statistic": nombre,
                          "simulationRun": 1, **r})

    res = pd.DataFrame(filas)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "A1_pca_baseline_runs.csv", index=False)

    for nombre in ("T2", "SPE"):
        sub = res[res["statistic"] == nombre]
        agg = metrics.aggregate(sub)
        agg.insert(1, "statistic", nombre)
        agg.to_csv(out / f"A1_pca_baseline_{nombre}.csv", index=False)

        s = metrics.summary(agg)
        print(f"--- {nombre} ---")
        print(f"  FAR sobre operacion normal : "
              f"{agg.loc[agg.faultNumber == 0, 'far_mean'].iloc[0]:.3f}")
        print(f"  FDR medio (todos los fallos): {s['fdr_medio_todos']:.3f}")
        print(f"  FDR medio (sin 3, 9 y 15)  : {s['fdr_medio_sin_dificiles']:.3f}")
        print(f"  FDR medio (3, 9 y 15)      : {s['fdr_medio_dificiles']:.3f}")
        print(f"  Retardo medio (muestras)   : {s['retardo_medio_muestras']:.1f}")
        print(f"  Fallos con deteccion < 50 %: {s['fallos_nunca_detectados']}\n")

    print(f"Resultados guardados en {out}/")


if __name__ == "__main__":
    main()
