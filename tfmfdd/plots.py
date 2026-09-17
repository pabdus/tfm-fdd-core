"""Estilo grafico unico para toda la memoria.

Todas las figuras del TFM salen de aqui, para que las de los tres bloques se
vean iguales. Formato pensado para la plantilla de UNIR: fuente sobria, tamano
legible al reducir, y exportacion a PDF vectorial para que no pixele al
imprimir.

El titulo NO va dentro de la figura: va en la memoria, encima, con el formato
"Figura 1. Nombre" que exigen las instrucciones. La fuente va debajo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "normal": "#3b6ea5",
    "fault": "#c1440e",
    "limit": "#444444",
    "alt1": "#4a7c59",
    "alt2": "#8a6d3b",
    "alt3": "#6b4c7a",
}


def use_style() -> None:
    """Aplica el estilo comun. Llamar una vez al principio de cada script."""
    plt.rcParams.update({
        "figure.figsize": (7.0, 3.6),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
    })


def control_chart(statistic, limit, onset=None, ylabel="T$^2$",
                  log=True, ax=None, sample_minutes=3.0):
    """Carta de control: estadistico frente al tiempo, con limite e inicio del fallo.

    Es la figura mas importante del bloque de deteccion: de un vistazo se ve si
    el metodo detecta, cuanto tarda y si genera falsas alarmas antes del fallo.
    """
    stat = np.asarray(statistic, dtype=float)
    if ax is None:
        _, ax = plt.subplots()

    t = np.arange(len(stat)) * sample_minutes / 60.0  # horas
    ax.plot(t, stat, color=COLORS["normal"], label=ylabel)
    ax.axhline(limit, color=COLORS["limit"], ls="--", lw=1.0,
               label=f"Limite ({limit:.1f})")

    if onset is not None:
        ax.axvline(onset * sample_minutes / 60.0, color=COLORS["fault"],
                   ls=":", lw=1.2, label="Inicio del fallo")

    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Tiempo (h)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", ncols=3, fontsize=8)
    return ax


def fdr_by_fault(agg, value="fdr_mean", err_low="fdr_ci_low", err_high="fdr_ci_high",
                 hard=(3, 9, 15), ax=None):
    """Barras de FDR por fallo, con intervalo de confianza.

    Los fallos dificiles se colorean distinto porque son los que discriminan
    entre metodos; promediarlos con el resto esconde la unica diferencia real.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.0))

    d = agg[agg["faultNumber"] > 0].sort_values("faultNumber")
    x = np.arange(len(d))
    colores = [COLORS["fault"] if f in hard else COLORS["normal"]
               for f in d["faultNumber"]]

    yerr = None
    if err_low in d and err_high in d:
        yerr = np.vstack([
            (d[value] - d[err_low]).clip(lower=0).to_numpy(),
            (d[err_high] - d[value]).clip(lower=0).to_numpy(),
        ])

    ax.bar(x, d[value], color=colores, yerr=yerr, capsize=2, error_kw={"lw": 0.7})
    ax.set_xticks(x)
    ax.set_xticklabels(d["faultNumber"].astype(int), fontsize=7)
    ax.set_xlabel("Fallo (IDV)")
    ax.set_ylabel("Tasa de deteccion")
    ax.set_ylim(0, 1.02)
    return ax


def contribution_plot(contributions, var_names, top=10, ax=None):
    """Grafico de contribucion: que variables disparan el residuo."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.0))

    c = np.asarray(contributions, dtype=float)
    idx = np.argsort(c)[::-1][:top]
    ax.barh(range(len(idx))[::-1], c[idx], color=COLORS["normal"])
    ax.set_yticks(range(len(idx))[::-1])
    ax.set_yticklabels([var_names[i] for i in idx], fontsize=8)
    ax.set_xlabel("Contribucion al SPE")
    return ax


def save(fig, path, formats=("pdf", "png")):
    """Guarda la figura en PDF (para la memoria) y PNG (para revisar rapido)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(path.with_suffix(f".{fmt}"))
    plt.close(fig)
