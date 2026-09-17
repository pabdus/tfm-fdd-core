"""tfmfdd: infraestructura comun del TFM de deteccion de fallos industriales.

Paquete de soporte para la comparativa de metodos data-driven sobre el
Tennessee Eastman Process. Lo usan los tres bloques del trabajo, de modo que la
carga de datos, las particiones, el escalado y las metricas sean identicas y los
resultados comparables entre si.

Uso tipico:

    from tfmfdd import data, metrics
    from tfmfdd.pca import PCAMonitor

    normal = data.load_classic(0, "train")
    modelo = PCAMonitor(variance=0.90, alpha=0.99).fit(data.values(normal))

    prueba = data.load_classic(1, "test")
    t2, spe = modelo.score(data.values(prueba))
    print(metrics.evaluate_run(t2, modelo.t2_limit_, onset=160))

Autor: Pablo Alberto Duque Marin
"""

__version__ = "0.1.0"

from . import data, limits, metrics, plots, preprocessing, stats  # noqa: F401
from .pca import PCAMonitor  # noqa: F401

__all__ = [
    "data",
    "limits",
    "metrics",
    "plots",
    "preprocessing",
    "stats",
    "PCAMonitor",
]
