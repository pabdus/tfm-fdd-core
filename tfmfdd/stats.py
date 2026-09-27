"""Contrastes estadisticos para comparar metodos.

Una diferencia entre dos metodos no es un resultado hasta que se comprueba que
no es ruido. Como evaluamos ambos metodos sobre LAS MISMAS simulaciones, el
contraste correcto es pareado: se compara la diferencia dentro de cada
simulacion, no las medias sueltas.

Se usa Wilcoxon de rangos con signo en vez de la t de Student porque no exige
normalidad, y las metricas de FDD rara vez son normales (el FDR se acumula
contra el 1, el retardo esta acotado por abajo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps


def paired_test(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> dict:
    """Wilcoxon pareado entre dos metodos sobre las mismas simulaciones."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Longitudes distintas: {a.shape} y {b.shape}")

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    if a.size < 5:
        return {"n": int(a.size), "statistic": np.nan, "p_value": np.nan,
                "effect_size": np.nan, "median_difference": np.nan}

    if np.allclose(a, b):
        return {"n": int(a.size), "statistic": 0.0, "p_value": 1.0,
                "effect_size": 0.0, "median_difference": 0.0}

    est, p = sps.wilcoxon(a, b, alternative=alternative, zero_method="zsplit")

    # Tamano de efecto r = Z / sqrt(n), con Z reconstruido del valor p.
    z = sps.norm.ppf(1 - p / 2) if 0 < p < 1 else np.nan
    r = z / np.sqrt(a.size) if np.isfinite(z) else np.nan

    return {
        "n": int(a.size),
        "statistic": float(est),
        "p_value": float(p),
        "effect_size": float(r) if np.isfinite(r) else np.nan,
        "median_difference": float(np.median(a - b)),
    }


def holm_correction(p_values: list[float], alpha: float = 0.05) -> pd.DataFrame:
    """Correccion de Holm-Bonferroni para comparaciones multiples.

    Comparar cinco metodos entre si son diez contrastes; con alpha = 0.05 cabe
    esperar medio falso positivo solo por azar. Holm ajusta el umbral y es menos
    conservador que Bonferroni sin perder control del error.
    """
    p = np.asarray(p_values, dtype=float)
    n = p.size
    orden = np.argsort(p)

    umbral = alpha / (n - np.arange(n))
    significativo_ordenado = np.zeros(n, dtype=bool)
    for i in range(n):
        if p[orden][i] <= umbral[i]:
            significativo_ordenado[i] = True
        else:
            break  # Holm se detiene en el primer no significativo

    significativo = np.zeros(n, dtype=bool)
    significativo[orden] = significativo_ordenado

    return pd.DataFrame({
        "p_value": p,
        "rank": np.argsort(orden) + 1,
        "threshold": umbral[np.argsort(orden)],
        "significant": significativo,
    })


def compare_methods(results: pd.DataFrame, metric: str = "fdr",
                    method_col: str = "method", run_col: str = "simulationRun",
                    fault_col: str | None = "faultNumber") -> pd.DataFrame:
    """Compara todos los pares de metodos sobre una metrica, con correccion de Holm.

    `results` debe tener una fila por metodo y simulacion (y fallo, si hay
    varios). Los pares se emparejan por simulacion Y por fallo: si `fault_col`
    esta en la tabla, el contraste se hace dentro de cada fallo, con su propia
    correccion de Holm entre los pares de metodos de ese fallo. Antes se
    indexaba solo por simulacion y, con varios fallos en la misma tabla, los
    indices se duplicaban y los pares se desalineaban en silencio.
    """
    if fault_col is not None and fault_col in results:
        tablas = []
        for f, g in results.groupby(fault_col):
            t = compare_methods(g, metric, method_col, run_col, fault_col=None)
            t.insert(0, fault_col, f)
            tablas.append(t)
        return pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()

    metodos = sorted(results[method_col].unique())
    filas = []

    for i, m1 in enumerate(metodos):
        for m2 in metodos[i + 1:]:
            d1 = results[results[method_col] == m1].set_index(run_col)[metric]
            d2 = results[results[method_col] == m2].set_index(run_col)[metric]
            if d1.index.has_duplicates or d2.index.has_duplicates:
                raise ValueError(
                    "Hay varias filas por simulacion para un mismo metodo; pasa "
                    "fault_col para emparejar por fallo o filtra un solo fallo."
                )
            comunes = d1.index.intersection(d2.index)
            r = paired_test(d1.loc[comunes].to_numpy(), d2.loc[comunes].to_numpy())
            filas.append({"method_a": m1, "method_b": m2, **r})

    tabla = pd.DataFrame(filas)
    if not tabla.empty:
        corr = holm_correction(tabla["p_value"].tolist())
        tabla["significant"] = corr["significant"].to_numpy()
    return tabla
