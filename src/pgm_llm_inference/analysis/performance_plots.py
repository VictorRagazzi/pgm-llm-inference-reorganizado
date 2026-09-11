"""Plots for aggregate experiment performance."""

import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .logs import translate_dataset, translate_sampling
from .metrics import filter_by_sampling
from .style import (
    ACCENT_COLOR,
    ACCENT_EDGE,
    AXIS_FS,
    BAR_COLOR,
    EDGE_COLOR,
    GRID_COLOR,
    LABEL_FS,
    TITLE_FS,
)


def _bar_with_labels(axis, data, x_column, y_column, fmt="%.1f%%") -> None:
    sns.barplot(
        data=data,
        x=x_column,
        y=y_column,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        ax=axis,
    )
    for container in axis.containers:
        axis.bar_label(container, fmt=fmt, padding=3, fontsize=LABEL_FS)
    axis.yaxis.grid(True, linestyle="--", alpha=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=LABEL_FS)


def plot_accuracy_by_dataset(
    grouped: pd.DataFrame,
    datasets_per_img: int = 10,
    evidence_sampling: str | None = None,
) -> None:
    subset = filter_by_sampling(grouped, evidence_sampling).copy()
    subset["accuracy_pct"] = subset["accuracy"] * 100
    datasets = subset["dataset"].unique()

    for index in range(math.ceil(len(datasets) / datasets_per_img)):
        chunk = datasets[index * datasets_per_img : (index + 1) * datasets_per_img]
        overall = (
            subset[subset["dataset"].isin(chunk)]
            .groupby("dataset")["accuracy_pct"]
            .mean()
            .reindex(chunk)
            .reset_index()
        )
        overall["dataset"] = overall["dataset"].apply(translate_dataset)
        _, axis = plt.subplots(figsize=(12, 5))
        _bar_with_labels(axis, overall, "dataset", "accuracy_pct")
        axis.set_title(
            "Acurácia dos Experimentos por Dataset",
            fontsize=TITLE_FS,
            fontweight="bold",
            pad=10,
        )
        axis.set(ylabel="Acurácia (%)", xlabel="Dataset", ylim=(0, 115))
        axis.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.show()


def plot_accuracy_by_evidence_ratio(
    grouped: pd.DataFrame,
    evidence_sampling: str | None = None,
    joint_match_column: str = "joint_match",
) -> None:
    subset = filter_by_sampling(grouped, evidence_sampling)
    datasets = sorted(subset["dataset"].unique())
    columns = 5
    rows = math.ceil(len(datasets) / columns) if len(datasets) else 1
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.5 * columns, 4 * rows),
        sharey=True,
        squeeze=False,
    )

    has_joint_match = joint_match_column in subset.columns

    for index, dataset in enumerate(datasets):
        axis = axes[index // columns][index % columns]
        by_dataset = subset[subset["dataset"] == dataset]

        agg_columns = ["accuracy"] + ([joint_match_column] if has_joint_match else [])
        values = (
            by_dataset.groupby("evidence_ratio")[agg_columns]
            .mean()
            .mul(100)
            .reset_index()
            .sort_values("evidence_ratio")
        )
        axis.plot(
            values["evidence_ratio"],
            values["accuracy"],
            color=BAR_COLOR,
            marker="o",
            linewidth=1.3,
            markersize=5,
            alpha=0.85,
            zorder=3,
            label="Acurácia por variável",
        )
        if has_joint_match:
            axis.plot(
                values["evidence_ratio"],
                values[joint_match_column],
                color=ACCENT_COLOR,
                marker="s",
                linewidth=0.75,
                markersize=2,
                alpha=0.85,
                zorder=3,
                label="Exact match",
            )
        axis.set_title(
            translate_dataset(dataset), fontsize=TITLE_FS, fontweight="bold", pad=2
        )
        ticks = list(range(5, 55, 5))
        axis.set_xticks(ticks, [str(tick) for tick in ticks], fontsize=LABEL_FS)
        axis.set(xlim=(0, 55), ylim=(0, 115))
        axis.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    for index in range(len(datasets), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    if has_joint_match:
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.85, 0.96),
            ncol=2,
            fontsize=LABEL_FS,
            frameon=False,
            # bbox_to_anchor=(0.5, -0.02),
        )

    figure.supxlabel("Proporção de Evidência (%)", fontsize=AXIS_FS, fontweight="bold")
    figure.supylabel("Média (%)", fontsize=AXIS_FS, fontweight="bold")
    figure.suptitle(
        "Acurácia e Exact Match x Proporção de Evidência",
        fontsize=TITLE_FS + 2,
        fontweight="bold",
        y=0.96,
    )
    plt.tight_layout()  # reserva mais espaço à direita
    plt.show()

def plot_accuracy_by_sampling(grouped: pd.DataFrame) -> None:
    data = grouped.copy()
    data["sampling_label"] = (
        data["evidence_sampling"] + " / " + data["evidence_layout"]
    )
    aggregate = (
        data.groupby("sampling_label")[["accuracy", "joint_match"]]
        .mean()
        .mul(100)
        .reset_index()
    )
    _, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, metric, title in (
        (axes[0], "accuracy", "Acurácia por Variável por Amostragem de Evidência"),
        (axes[1], "joint_match", "Taxa de Acerto Conjunto do MPE Completo"),
    ):
        _bar_with_labels(axis, aggregate, "sampling_label", metric)
        axis.set_title(title, fontsize=TITLE_FS, fontweight="bold")
        axis.set(xlabel="", ylabel="%", ylim=(0, 115))
    plt.tight_layout()
    plt.show()


def plot_sampling_comparison(
    grouped: pd.DataFrame, layout: str | None = "nested"
) -> None:
    subset = grouped if layout is None else grouped[grouped["evidence_layout"] == layout]
    aggregate = (
        subset.groupby("evidence_sampling")[["accuracy", "joint_match"]].mean().mul(100)
    )
    order = [
        name
        for name in ("mpe_consistent", "mpe_inconsistent")
        if name in aggregate.index
    ]
    aggregate = aggregate.reindex(order)
    x_values = list(range(len(aggregate)))
    width = 0.35
    _, axis = plt.subplots(figsize=(7, 5))
    accuracy_bars = axis.bar(
        [x - width / 2 for x in x_values],
        aggregate["accuracy"],
        width,
        label="Acurácia por variável",
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        zorder=3,
    )
    joint_bars = axis.bar(
        [x + width / 2 for x in x_values],
        aggregate["joint_match"],
        width,
        label="Exact match (configuração conjunta)",
        color=ACCENT_COLOR,
        edgecolor=ACCENT_EDGE,
        linewidth=0.6,
        zorder=3,
    )
    for bars in (accuracy_bars, joint_bars):
        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=LABEL_FS)
    axis.set_xticks(x_values, [translate_sampling(name) for name in aggregate.index])
    axis.set(ylabel="%", ylim=(0, 115))
    axis.set_title(
        "Impacto do tipo de evidência no desempenho",
        fontsize=TITLE_FS,
        fontweight="bold",
        pad=10,
    )
    axis.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    axis.legend(fontsize=LABEL_FS, frameon=False, loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_joint_match_by_dataset(
    grouped: pd.DataFrame,
    datasets_per_img: int = 9,
    evidence_sampling: str | None = None,
) -> None:
    subset = filter_by_sampling(grouped, evidence_sampling).copy()
    subset["joint_match_pct"] = subset["joint_match"] * 100
    datasets = sorted(subset["dataset"].unique())
    suffix = f" — {translate_sampling(evidence_sampling)}" if evidence_sampling else ""

    for index in range(math.ceil(len(datasets) / datasets_per_img)):
        chunk = datasets[index * datasets_per_img : (index + 1) * datasets_per_img]
        overall = (
            subset[subset["dataset"].isin(chunk)]
            .groupby("dataset")["joint_match_pct"]
            .mean()
            .reindex(chunk)
            .reset_index()
        )
        overall["dataset"] = overall["dataset"].apply(translate_dataset)
        _, axis = plt.subplots(figsize=(12, 5))
        _bar_with_labels(axis, overall, "dataset", "joint_match_pct")
        axis.set_title(
            f"Taxa de Acerto Exato do MPE Completo por Dataset{suffix}",
            fontsize=TITLE_FS,
            fontweight="bold",
            pad=10,
        )
        axis.set(ylabel="Taxa de Acerto Conjunto (%)", xlabel="Dataset", ylim=(0, 115))
        axis.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.show()


def plot_accuracy_vs_joint_match(grouped: pd.DataFrame) -> None:
    _, axis = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=grouped,
        x="accuracy",
        y="joint_match",
        hue="evidence_sampling",
        style="evidence_layout",
        s=60,
        ax=axis,
        palette="deep",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color=GRID_COLOR, linewidth=1, zorder=0)
    axis.set(
        xlim=(-0.05, 1.05),
        ylim=(-0.05, 1.05),
        xlabel="Acurácia por Variável",
        ylabel="Acerto Conjunto do MPE Completo",
    )
    axis.set_title(
        "Acurácia por Variável vs. Recuperação do MPE Completo",
        fontsize=TITLE_FS,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()
