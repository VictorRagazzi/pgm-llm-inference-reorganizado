"""Post-hoc plots for compiled decision uncertainty."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def plot_compiled_row_entropy(table: pd.DataFrame, output_path: Path) -> None:
    """Show every context row without conflating domain and token entropy."""
    if table.empty:
        raise ValueError("No compiled decision rows to plot.")

    variables = sorted(table["variable"].unique())
    positions = {variable: index for index, variable in enumerate(variables)}
    x = table["variable"].map(positions)
    fig = Figure(figsize=(max(9, len(variables) * 0.5), 10))
    axes = fig.subplots(3, 1)
    columns = (
        ("domain_entropy", "Entropia do domínio (nats)"),
        ("token_entropy_top_k", "Entropia condicional do top-k (nats)"),
        ("token_top_k_mass", "Massa capturada pelo top-k"),
    )
    for axis, (column, label) in zip(axes, columns):
        axis.scatter(x, table[column], alpha=0.45, s=18)
        axis.set_ylabel(label)
        axis.set_xticks(range(len(variables)), variables, rotation=90)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("Variável (cada ponto = uma context row)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
