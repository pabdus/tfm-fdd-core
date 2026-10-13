"""tfmfdd: infraestructura comun del TFM de deteccion de fallos industriales.

Paquete de soporte para la comparativa de metodos data-driven sobre el
Tennessee Eastman Process. Lo usan los tres bloques del trabajo, de modo que la
carga de datos, las particiones, el escalado y las metricas sean identicas y los
resultados comparables entre si.

Uso tipico (TEP clasico en data/tep_classic; la configuracion sale de
configs/baseline.yaml, ver tfmfdd.config):

    from tfmfdd import config, data, metrics
    from tfmfdd.pca import PCAMonitor

    cfg = config.load_baseline()
    normal = data.values(data.load_classic(0, "train"))     # d00.dat
    corte = int(len(normal) * (1 - cfg["particion"]["calibracion_limites"]))
    X_fit, X_cal = normal[:corte], normal[corte:]           # X_cal: normales reservadas

    modelo = PCAMonitor(variance=cfg["pca"]["variance"], alpha=cfg["protocolo"]["alpha"],
                        spe_method=cfg["pca"]["spe_method"]).fit(X_fit, X_cal=X_cal)
    print(modelo.limit_source_)     # de donde sale cada limite

    prueba = data.load_classic(1, "test")
    t2, spe = modelo.score(data.values(prueba))
    print(metrics.evaluate_run(spe, modelo.spe_limit_, onset=data.FAULT_ONSET_TEST,
                               k=cfg["protocolo"]["k_alarmas"]))

Los limites se calibran con datos normales que el modelo no vio (X_cal), como
pide el protocolo. Calibrar con los datos del ajuste es un sesgo, no una
fuga (memoria, 2.2.4), y solo se hace a proposito y a la vista:
`fit(X, calibration="in_sample")`, como en la verificacion frente a Yin et al.
(2012). Un `fit(X)` sin X_cal tambien calibra en muestra los limites que
dependen de datos, pero avisa.

Modulos: data (carga y particiones), preprocessing (escalado y retardos),
limits (limites de control), pca (monitor PCA y DPCA), metrics (deteccion y
diagnostico), stats (contrastes), ablation (mecanismos de las cuatro
ablaciones), cost (coste computacional), robustness (indice de robustez),
config (configs/baseline.yaml), provenance (procedencia de cada resultado) y
plots (estilo grafico).

Autor: Pablo Alberto Duque Marin
"""

__version__ = "0.3.0"

from . import (  # noqa: F401
    ablation,
    config,
    cost,
    data,
    limits,
    metrics,
    plots,
    preprocessing,
    provenance,
    robustness,
    stats,
)
from .pca import PCAMonitor  # noqa: F401

__all__ = [
    "ablation",
    "config",
    "cost",
    "data",
    "limits",
    "metrics",
    "plots",
    "preprocessing",
    "provenance",
    "robustness",
    "stats",
    "PCAMonitor",
]
