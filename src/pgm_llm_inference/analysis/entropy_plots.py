"""Post-hoc plots for compiled decision uncertainty."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
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


def token_vector_filename(variable: str) -> str:
    """Return a deterministic, collision-resistant SVG name for a variable."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", variable).strip("._")
    digest = hashlib.sha256(variable.encode("utf-8")).hexdigest()[:8]
    return f"token_vector_{safe_name or 'variable'}_{digest}.svg"


def plot_token_vector(table: pd.DataFrame, variable: str, output_path: Path) -> None:
    """Plot each context row's raw next-token log-probability vector."""
    rows = table.loc[table["variable"] == variable]
    if rows.empty:
        raise ValueError(f"No token alternatives for variable {variable!r}.")

    row_ids = sorted(rows["row_index"].unique())
    row_positions = {row_id: index for index, row_id in enumerate(row_ids)}
    max_rank = int(rows["rank"].max())
    values = np.full((len(row_ids), max_rank), np.nan)
    labels: dict[tuple[int, int], tuple[str, bool]] = {}
    for item in rows.itertuples(index=False):
        y = row_positions[item.row_index]
        x = item.rank - 1
        values[y, x] = item.logprob
        labels[y, x] = (item.token, item.is_generated)

    cmap = colormaps["viridis"].copy()
    cmap.set_bad("#eeeeee")
    fig = Figure(figsize=(max(13, 0.9 * max_rank + 4), max(4, 0.5 * len(row_ids) + 2)))
    axis = fig.subplots()
    image = axis.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=min(float(np.nanmin(values)), -1e-9),
        vmax=0,
    )
    axis.set_xticks(range(max_rank), [str(rank) for rank in range(1, max_rank + 1)])
    axis.set_yticks(
        range(len(row_ids)),
        [
            f"r{row_id}: {rows.loc[rows['row_index'] == row_id, 'selected_value'].iloc[0]}"
            for row_id in row_ids
        ],
    )
    axis.set_xlabel("Alternativas ordenadas por log-probabilidade (top-k + gerado)")
    axis.set_ylabel("Context row (índice: estado escolhido)")
    axis.set_title(f"{variable}: vetor bruto de log-probs por contexto")
    for (y, x), (token, is_generated) in labels.items():
        rendered = json.dumps(token, ensure_ascii=True)
        if len(rendered) > 16:
            rendered = rendered[:15] + "…"
        prefix = "*" if is_generated else ""
        color = "white" if values[y, x] < image.norm.vmin / 2 else "black"
        axis.text(
            x,
            y,
            f"{prefix}{rendered}\n{values[y, x]:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color=color,
        )
    fig.colorbar(image, ax=axis, label="Log-probabilidade bruta")
    fig.tight_layout()
    fig.savefig(output_path)
