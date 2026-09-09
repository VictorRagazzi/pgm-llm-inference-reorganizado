"""Plots for network- and variable-level structural analysis."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .structural_metrics import (
    STRUCTURAL_COLUMNS,
    compute_variable_structural_correlations,
)
from .style import (
    ACCENT_COLOR,
    ACCENT_EDGE,
    AXIS_FS,
    BAR_COLOR,
    EDGE_COLOR,
    LABEL_FS,
    TITLE_FS,
)


def plot_error_vs_structure(
    merged: pd.DataFrame, error_metric: str = "accuracy"
) -> None:
    metric_label = {
        "accuracy": "Acurácia",
        "joint_match": "Acerto Conjunto",
        "exact_match": "Acerto Exato",
    }.get(error_metric, error_metric)
    figure, axes = plt.subplots(
        1,
        len(STRUCTURAL_COLUMNS),
        figsize=(5 * len(STRUCTURAL_COLUMNS), 4.5),
        sharey=True,
    )

    for axis, structure_column in zip(axes, STRUCTURAL_COLUMNS):
        values = merged[[structure_column, error_metric]].dropna()
        axis.scatter(
            values[structure_column],
            values[error_metric],
            color=BAR_COLOR,
            edgecolor=EDGE_COLOR,
            s=60,
            zorder=3,
        )
        title = structure_column
        if len(values) >= 3:
            rho, p_value = spearmanr(values[structure_column], values[error_metric])
            title += f"\n(ρ={rho:.2f}, p={p_value:.3f})"
        axis.set_title(title, fontsize=TITLE_FS, fontweight="bold")
        axis.set_xlabel(structure_column, fontsize=AXIS_FS)
        axis.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", labelsize=LABEL_FS)

    axes[0].set_ylabel(metric_label, fontsize=AXIS_FS)
    figure.suptitle(
        f"{metric_label} vs. Características Estruturais da Rede",
        fontsize=TITLE_FS + 1,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()


def _boxplot(
    data: pd.DataFrame,
    category: str,
    categories: list,
    labels: list[str],
    x_label: str,
    title: str,
    show_outliers: bool,
) -> None:
    _, axis = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=data,
        x=category,
        y="accuracy",
        order=categories,
        color=BAR_COLOR,
        ax=axis,
        showfliers=show_outliers,
        boxprops={"edgecolor": EDGE_COLOR},
        medianprops={"color": EDGE_COLOR, "linewidth": 1.5},
        whiskerprops={"color": EDGE_COLOR},
        capprops={"color": EDGE_COLOR},
        flierprops={
            "markerfacecolor": ACCENT_COLOR,
            "markeredgecolor": ACCENT_EDGE,
            "markersize": 5,
        },
    )
    axis.set_xticks(range(len(categories)), labels, fontsize=LABEL_FS)
    axis.set(xlabel=x_label, ylabel="Acurácia por variável", ylim=(-0.05, 1.05))
    axis.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10)
    axis.yaxis.grid(True, linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    plt.tight_layout()
    plt.show()


def plot_accuracy_by_cardinality(
    merged: pd.DataFrame,
    min_n: int = 4,
    show_outliers: bool = False,
) -> None:
    if "cardinality" not in merged.columns:
        print("[AVISO] Coluna 'cardinality' ausente; gráfico não gerado.")
        return

    counts = merged["cardinality"].value_counts()
    categories = sorted(value for value, count in counts.items() if count >= min_n)
    if not categories:
        print(f"[AVISO] Nenhuma cardinalidade possui n ≥ {min_n}.")
        return

    correlations = compute_variable_structural_correlations(merged)
    rows = correlations[correlations["structural_metric"] == "cardinality"]
    suffix = ""
    if not rows.empty and not math.isnan(rows.iloc[0]["p_value"]):
        row = rows.iloc[0]
        significance = " *" if row["p_value"] < 0.05 else ""
        suffix = f"  (ρ={row['rho']:.3f}, p={row['p_value']:.3f}{significance})"

    _boxplot(
        merged[merged["cardinality"].isin(categories)],
        "cardinality",
        categories,
        [f"{value}\n(n={counts[value]})" for value in categories],
        "Cardinalidade da variável (nº de estados)",
        f"Acurácia por Variável vs. Cardinalidade{suffix}",
        show_outliers,
    )


def plot_accuracy_by_n_parents(
    merged: pd.DataFrame,
    min_n: int = 4,
    show_outliers: bool = False,
) -> None:
    counts = merged["n_parents"].value_counts()
    categories = sorted(value for value, count in counts.items() if count >= min_n)
    _boxplot(
        merged[merged["n_parents"].isin(categories)],
        "n_parents",
        categories,
        [f"{value}\n(n={counts[value]})" for value in categories],
        "Número de pais da variável",
        "Acurácia por Variável vs. Número de Pais",
        show_outliers,
    )
