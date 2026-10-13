"""A1 - Linea base PCA (o DPCA) sobre el TEP clasico, con la configuracion del protocolo.

Ajusta un monitor PCA con datos de operacion normal (d00.dat), calcula los
limites de control de T2 y SPE al nivel de confianza indicado y evalua los 22
ficheros de prueba (d00_te.dat y d01_te.dat a d21_te.dat) con FAR, FDR y retardo
de deteccion.

Para que sirve. Es la linea base de desarrollo del bloque A con la configuracion
del protocolo comun (52 variables, componentes por varianza acumulada, limites
calibrados con datos normales reservados). Sus cifras son de desarrollo: el TEP
clasico solo se usa en el desarrollo y en la verificacion (memoria, parrafo 367),
y los resultados del TFM salen de Rieth.

Lo que NO es: la verificacion frente a Yin et al. (2012). Esa compara la
implementacion con sus tablas usando la configuracion de esos autores (33
variables, 9 o 17 componentes, limites en muestra, regla T2 o SPE) y una
tolerancia fijada antes de ejecutar (parrafos 355 y 369), y vive en
experiments/V1_verificacion_yin.py. Parecerse a sus cifras con OTRA
configuracion no verifica nada. Por eso este script no ofrece las 33 variables
ni un numero de componentes fijo: la configuracion de Yin sobre los ficheros de
prueba pasa solo por V1, con su regla de orden.

Configuracion. Los valores por defecto se leen de baseline.yaml con
`tfmfdd.config.load_baseline` (ruta de --config, variable TFMFDD_CONFIG,
configs/baseline.yaml del directorio de trabajo o la copia del paquete; nunca se
sigue con valores escritos en el codigo). Un argumento de la linea de ordenes
cambia un valor para un experimento puntual, y el valor efectivo queda en cada
fila del CSV y en el JSON de procedencia que acompana a cada CSV
(`tfmfdd.provenance`).

- Ajuste y calibracion. d00.dat se corta en el tiempo: las primeras filas
  (1 - particion.calibracion_limites) ajustan el modelo y las ultimas, X_cal,
  calibran los limites que se calibran con datos (--calibration holdout, el
  protocolo). Con --calibration in_sample el MISMO modelo calibra con las filas
  del ajuste: es el brazo en muestra del factor 1, a proposito.
- --t2-method, --spe-method (tambien jackson) y --quantile-method eligen los
  limites de PCAMonitor. Con F y Jackson los dos limites son parametricos con
  cantidades del ajuste y X_cal no interviene: lo dicen las columnas
  t2_limit_source y spe_limit_source.
- --lags 1 o 2: DPCA. Cada fichero es una simulacion y se pasa como tal (runs) al
  ajuste, a la calibracion y a la puntuacion; las `lags` primeras muestras de
  cada fichero no se evaluan (NaN) y el inicio del fallo es el mismo que en el
  PCA (indice 160).

Salidas (en --out, por defecto results/summary): A1_pca_baseline_runs.csv (una
fila por fallo y estadistico, en el formato comun, con `method`), y
A1_pca_baseline_T2.csv y A1_pca_baseline_SPE.csv (metrics.aggregate por fallo),
cada uno con su .provenance.json. Para pruebas de humo, usa --out results/raw/...
(git lo ignora).

Uso:
    python experiments/A1_pca_baseline.py                      # protocolo comun
    python experiments/A1_pca_baseline.py --variance 0.90      # sensibilidad
    python experiments/A1_pca_baseline.py --lags 2 --out results/raw/A1_dpca
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tfmfdd import config, data, metrics, provenance
from tfmfdd.pca import CALIBRATIONS, SPE_METHODS, T2_METHODS, PCAMonitor
from tfmfdd.limits import EMPIRICAL_METHODS


def _argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None,
                   help="baseline.yaml (por defecto, el orden de tfmfdd.config.load_baseline)")
    p.add_argument("--alpha", type=float, default=None, help="nivel de confianza (protocolo.alpha)")
    p.add_argument("--variance", type=float, default=None, help="varianza acumulada (pca.variance)")
    p.add_argument("--lags", type=int, default=None, help="0 = PCA, 1 o 2 = DPCA (pca.lags)")
    p.add_argument("--t2-method", default=None, choices=T2_METHODS, help="limite del T2 (pca.t2_method)")
    p.add_argument("--spe-method", default=None, choices=SPE_METHODS, help="limite del SPE (pca.spe_method)")
    p.add_argument("--quantile-method", default=None, choices=EMPIRICAL_METHODS,
                   help="percentil de los limites empirical (pca.quantile_method)")
    p.add_argument("--calibration", default=None, choices=CALIBRATIONS,
                   help="holdout = con X_cal reservado; in_sample = con el ajuste (pca.calibration)")
    p.add_argument("--k", type=int, default=None, help="alarmas consecutivas (protocolo.k_alarmas)")
    p.add_argument("--cal-fraction", type=float, default=None,
                   help="fraccion de d00.dat reservada para calibrar (particion.calibracion_limites)")
    p.add_argument("--root", default="data/tep_classic")
    p.add_argument("--out", default="results/summary")
    return p.parse_args(argv)


def _valor(args, nombre: str, cfg: dict, clave: str):
    """El argumento si se paso; si no, la clave del yaml (KeyError si falta)."""
    v = getattr(args, nombre)
    return config.get(cfg, clave) if v is None else v


def ejecutar(argv=None) -> dict:
    args = _argumentos(argv)
    cfg, info_cfg = config.load_baseline(args.config, with_info=True)
    alpha = _valor(args, "alpha", cfg, "protocolo.alpha")
    variance = _valor(args, "variance", cfg, "pca.variance")
    lags = _valor(args, "lags", cfg, "pca.lags")
    t2_method = _valor(args, "t2_method", cfg, "pca.t2_method")
    spe_method = _valor(args, "spe_method", cfg, "pca.spe_method")
    quantile_method = _valor(args, "quantile_method", cfg, "pca.quantile_method")
    calibration = _valor(args, "calibration", cfg, "pca.calibration")
    k = _valor(args, "k", cfg, "protocolo.k_alarmas")
    cal_fraction = _valor(args, "cal_fraction", cfg, "particion.calibracion_limites")
    sample_minutes = config.get(cfg, "protocolo.sample_minutes")
    min_fdr = config.get(cfg, "protocolo.min_fdr")
    delay_convention = config.get(cfg, "protocolo.retardo_convencion")
    root = Path(args.root)

    normal = data.values(data.load_classic(0, "train", root=root))

    # Corte temporal de d00.dat: el modelo se ajusta con las primeras filas y los
    # limites que se calibran con datos usan las ultimas, que el modelo no ve
    # (regla 3). Ver la docstring de PCAMonitor.fit.
    corte = int(len(normal) * (1 - cal_fraction))
    X_fit, X_cal = normal[:corte], normal[corte:]

    # Con DPCA, cada bloque es una trayectoria contigua de la misma simulacion:
    # se pasa su identificador para que los retardos no crucen de una a otra.
    def ids(n: int):
        return np.ones(n, dtype=int) if lags else None

    modelo = PCAMonitor(variance=variance, alpha=alpha, lags=lags, spe_method=spe_method,
                        t2_method=t2_method, quantile_method=quantile_method)
    if calibration == "holdout":
        modelo.fit(X_fit, runs=ids(len(X_fit)), X_cal=X_cal, runs_cal=ids(len(X_cal)), calibration="holdout")
    else:
        modelo.fit(X_fit, runs=ids(len(X_fit)), calibration="in_sample")

    metodo = "pca" if lags == 0 else "dpca"
    configuracion = {
        "n_components": modelo.n_components_, "variance": variance, "alpha": alpha, "lags": lags,
        "t2_method": t2_method, "spe_method": spe_method, "quantile_method": quantile_method,
        "calibration": calibration, "t2_limit": modelo.t2_limit_, "spe_limit": modelo.spe_limit_,
        "t2_limit_source": modelo.limit_source_["t2"], "spe_limit_source": modelo.limit_source_["spe"],
        "n_vars": modelo.n_features_in_, "n_fit": modelo.n_fit_, "n_cal": modelo.n_cal_,
        "n_cal_rows": len(X_cal) if calibration == "holdout" else 0, "k": k, "min_fdr": min_fdr,
        "delay_convention": delay_convention, "sample_minutes": sample_minutes,
        "far_window_normal": "serie entera",
    }

    print(f"Ajuste con {len(X_fit)} muestras de d00.dat; calibracion: {calibration}"
          + (f" con las {len(X_cal)} ultimas" if calibration == "holdout" else " (con las del ajuste)"))
    print(f"Metodo: {metodo} (lags = {lags}), componentes retenidas: {modelo.n_components_} "
          f"({variance:.0%} de varianza)")
    print(f"Limite T2  = {modelo.t2_limit_:.2f}  ({t2_method}, {modelo.limit_source_['t2']})")
    print(f"Limite SPE = {modelo.spe_limit_:.2f}  ({spe_method}, {modelo.limit_source_['spe']})\n")

    filas = []
    for fault in range(22):
        X = data.values(data.load_classic(fault, "test", root=root))
        t2, spe = modelo.score(X, runs=ids(len(X)))
        onset = None if fault == 0 else data.FAULT_ONSET_TEST

        for nombre, stat, limite in (("T2", t2, modelo.t2_limit_),
                                     ("SPE", spe, modelo.spe_limit_)):
            r = metrics.evaluate_run(stat, limite, onset, k=k, sample_minutes=sample_minutes,
                                     min_fdr=min_fdr, delay_convention=delay_convention)
            filas.append({"method": metodo, "faultNumber": fault, "statistic": nombre,
                          "simulationRun": 1, **r, **configuracion})

    res = pd.DataFrame(filas)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Git y maquina, una vez y antes de escribir ningun CSV (los tres JSON comparten ese estado).
    procedencia = dict(
        config=configuracion, config_files=[info_cfg], script=__file__, **provenance.run_context(__file__),
        ids={"fuente": "classic", "ajuste": f"d00.dat, filas 1 a {len(X_fit)}",
             "calibracion": (f"d00.dat, filas {len(X_fit) + 1} a {len(normal)}"
                             if calibration == "holdout" else "las filas del ajuste"),
             "prueba": [f"d{f:02d}_te.dat" for f in range(22)], "simulationRun": [1]},
        inputs=[root / "d00.dat"] + [root / f"d{f:02d}_te.dat" for f in range(22)],
    )
    ruta_runs = out / "A1_pca_baseline_runs.csv"
    res.to_csv(ruta_runs, index=False)
    provenance.write_provenance(ruta_runs, **procedencia)

    resumenes = {}
    for nombre in ("T2", "SPE"):
        sub = res[res["statistic"] == nombre]
        agg = metrics.aggregate(sub)
        agg.insert(0, "method", metodo)
        agg.insert(2, "statistic", nombre)
        for clave, valor in configuracion.items():
            agg[clave] = valor
        ruta = out / f"A1_pca_baseline_{nombre}.csv"
        agg.to_csv(ruta, index=False)
        provenance.write_provenance(ruta, **procedencia)

        s = metrics.summary(agg, min_fdr=min_fdr)
        resumenes[nombre] = s
        print(f"--- {nombre} ---")
        print(f"  FAR sobre operacion normal (d00_te, serie entera): {s['far_normal']:.3f}")
        print(f"  FDR medio (todos los fallos): {s['fdr_medio_todos']:.3f}  (n = {s['fdr_medio_todos_n']})")
        print(f"  FDR medio (sin 3, 9 y 15)  : {s['fdr_medio_sin_dificiles']:.3f}  "
              f"(n = {s['fdr_medio_sin_dificiles_n']})")
        print(f"  FDR medio (3, 9 y 15)      : {s['fdr_medio_dificiles']:.3f}  (n = {s['fdr_medio_dificiles_n']})")
        print(f"  Retardo medio de las detecciones validas (muestras): "
              f"{s['retardo_medio_muestras']:.1f}  (n = {s['retardo_medio_muestras_n']} fallos)")
        print(f"  Retardo mediano (muestras): {s['retardo_mediano_muestras']:.1f}  "
              f"(n = {s['retardo_mediano_muestras_n']} fallos)")
        print(f"  Fallos con FDR media < {min_fdr:.2f}: {s['fallos_bajo_umbral']}")
        print(f"  Fallos con racha de {k} en menos de la mitad de las simulaciones: "
              f"{s['fallos_sin_racha']}\n")

    print(f"Resultados guardados en {out}/ (con su .provenance.json)")
    return {"runs": res, "summary": resumenes, "config": configuracion}


def main(argv=None) -> None:
    ejecutar(argv)


if __name__ == "__main__":
    main()
