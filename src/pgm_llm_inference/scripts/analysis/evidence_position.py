"""
Análise do experimento de posição da evidência.

Reaproveita load_logs, translate_dataset e as constantes de estilo
(BAR_COLOR, EDGE_COLOR, LABEL_FS, AXIS_FS, TITLE_FS) já definidas em
analysis.py, para manter a mesma paleta/tamanhos usados no resto do
artigo. Ajuste o caminho do import abaixo conforme o nome real do módulo.
"""

import math
from pathlib import Path
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from pgm_llm_inference.scripts.analysis.metrics import (
    load_logs, translate_dataset,
    BAR_COLOR, EDGE_COLOR, ACCENT_COLOR, ACCENT_EDGE,
    LABEL_FS, AXIS_FS, TITLE_FS,
)

EXPERIMENT_TAG = "evidence_position"  # deve casar com o usado no script de execução
REQUIRED_COLUMNS = {
    "dataset", "evidence_node", "node_depth_normalized", "exact_match",
}


# ─────────────────────────────────────────────
#  CARGA / PREPARO
# ─────────────────────────────────────────────

def load_evidence_position_logs(path: str) -> pd.DataFrame:
    """
    Carrega o(s) log(s) e filtra apenas as linhas deste experimento
    (campo "experiment" == "evidence_position"). Se o log estiver
    misturado com outros experimentos (proporção de evidência etc.), o
    filtro evita contaminar a análise.
    """
    df = load_logs(path)
    if "experiment" in df.columns:
        df = df[df["experiment"] == EXPERIMENT_TAG].copy()
    if df.empty:
        print(f"[AVISO] Nenhuma linha com experiment == '{EXPERIMENT_TAG}' encontrada em {path}.")
        return df

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes no log: {sorted(missing)}")

    # run_experiment grava a métrica numérica em log_accuracy e uma
    # versão formatada (por exemplo, "83.33%") em accuracy. A análise
    # estatística deve sempre trabalhar na escala numérica [0, 1].
    if "log_accuracy" in df.columns:
        df["accuracy"] = pd.to_numeric(df["log_accuracy"], errors="coerce")
    elif "accuracy" in df.columns:
        accuracy = df["accuracy"]
        if accuracy.dtype == object:
            accuracy = accuracy.astype(str).str.rstrip("%")
            df["accuracy"] = pd.to_numeric(accuracy, errors="coerce") / 100.0
        else:
            df["accuracy"] = pd.to_numeric(accuracy, errors="coerce")
    else:
        raise ValueError("O log não contém 'log_accuracy' nem 'accuracy'.")

    for column in ("node_depth_normalized", "exact_match"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_depth = df["node_depth_normalized"].notna() & ~df["node_depth_normalized"].between(0, 1)
    if invalid_depth.any():
        raise ValueError(
            f"Há {int(invalid_depth.sum())} profundidade(s) normalizada(s) fora de [0, 1]."
        )

    duplicates = df.duplicated(["dataset", "evidence_node"], keep=False)
    if duplicates.any():
        print(
            f"[AVISO] {int(duplicates.sum())} linhas pertencem a pares "
            "(dataset, evidence_node) duplicados; as correlações usarão todas elas."
        )
    return df


def bin_evidence_position(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Agrupa node_depth_normalized (contínuo em [0, 1]) em n_bins faixas de
    largura igual. Necessário porque a posição varia de rede para rede
    (nº de níveis diferente) — não dá para comparar profundidades
    absolutas entre redes distintas, só a versão normalizada e binada.
    """
    if n_bins < 1:
        raise ValueError("n_bins deve ser pelo menos 1.")

    df = df.copy()
    edges = [i / n_bins for i in range(n_bins + 1)]
    labels = [f"{edges[i]:.1f}–{edges[i + 1]:.1f}" for i in range(n_bins)]
    df["position_bin"] = pd.cut(
        df["node_depth_normalized"],
        bins=edges,
        labels=labels,
        include_lowest=True,
    )
    return df


# ─────────────────────────────────────────────
#  PLOT PRINCIPAL — mesmo estilo de plot_accuracy_by_n_parents
# ─────────────────────────────────────────────

def plot_accuracy_by_evidence_position(
    df: pd.DataFrame,
    n_bins: int = 5,
    min_n: int = 5,
    show_outliers: bool = False,
    save_path: Optional[Path] = None,
):
    """
    Boxplot de accuracy e exact_match por bin de posição normalizada da
    evidência (0 = raiz, 1 = folha mais funda), lado a lado — mesmo
    estilo/paleta de plot_accuracy_by_n_parents em analysis.py.

    `min_n`: bins com menos observações que isso são descartados do
    gráfico (mesma lógica de min_n usada nos demais boxplots do paper).
    """
    binned = bin_evidence_position(df, n_bins=n_bins)
    counts = binned["position_bin"].value_counts(sort=False)
    valid = [str(b) for b, n in counts.items() if n >= min_n]

    if not valid:
        print(f"[AVISO] Nenhum bin com n >= {min_n}. Reduza min_n ou n_bins.")
        return

    subset = binned.copy()
    subset["position_bin"] = subset["position_bin"].astype("string")
    subset = subset[subset["position_bin"].isin(valid)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, metric, title, color, edge in [
        (axes[0], "accuracy", "Acurácia", BAR_COLOR, EDGE_COLOR),
        (axes[1], "exact_match", "Proporção de Exact Match", ACCENT_COLOR, ACCENT_EDGE),
    ]:
        plot_df = subset.dropna(subset=[metric])
        if metric == "accuracy":
            sns.boxplot(
                data=plot_df, x="position_bin", y=metric, order=valid,
                color=color, ax=ax, showfliers=show_outliers,
                boxprops=dict(edgecolor=edge),
                medianprops=dict(color=edge, linewidth=1.5),
                whiskerprops=dict(color=edge),
                capprops=dict(color=edge),
                flierprops=dict(markerfacecolor=color, markeredgecolor=edge, markersize=5),
            )
        else:
            # Exact match é binário por execução; a média por bin é a
            # proporção de configurações completamente corretas. Um boxplot
            # de zeros e uns seria pouco informativo, por isso usamos barras + IC95%.
            sns.barplot(
                data=plot_df, x="position_bin", y=metric, order=valid,
                color=color, edgecolor=edge, ax=ax,
                errorbar=("ci", 95), n_boot=2000, seed=42,
            )
        count_by_label = {str(k): int(v) for k, v in counts.items()}
        ax.set_xticks(range(len(valid)))
        ax.set_xticklabels(
            [f"{b}\n(n={count_by_label[b]})" for b in valid],
            fontsize=LABEL_FS,
        )
        ax.set_xlabel("Posição normalizada da evidência (0=raiz, 1=folha)", fontsize=AXIS_FS)
        ax.set_ylabel(title, fontsize=AXIS_FS)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{title} vs. Posição da Evidência", fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")
    plt.show()
    return fig, axes


# ─────────────────────────────────────────────
#  CORRELAÇÃO DE SPEARMAN
# ─────────────────────────────────────────────

def compute_evidence_position_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman entre node_depth_normalized e accuracy / exact_match, no
    nível de execução (n = nº total de nós avaliados nas 10 redes, não nº
    de redes — mesmo ganho de poder estatístico da análise de nº de pais).
    """
    records = []
    for metric in ["accuracy", "exact_match"]:
        sub = df[["node_depth_normalized", metric]].dropna()
        if len(sub) < 3 or any(sub[column].nunique() < 2 for column in sub.columns):
            rho, pval = float("nan"), float("nan")
        else:
            rho, pval = spearmanr(sub["node_depth_normalized"], sub[metric])
        records.append({"metric": metric, "n": len(sub), "rho": rho, "p_value": pval})
    return pd.DataFrame(records)


def print_evidence_position_correlations(corr_df: pd.DataFrame):
    W = 50
    print("=" * W)
    print(f"{'POSIÇÃO DA EVIDÊNCIA × DESEMPENHO':^{W}}")
    print("=" * W)
    print(f"{'Métrica':<14} {'n':>5} {'rho':>9} {'p-valor':>10}")
    print("-" * W)
    for _, row in corr_df.iterrows():
        if math.isnan(row["p_value"]):
            print(f"{row['metric']:<14} {row['n']:>5}    N/A       N/A")
        else:
            sig = " *" if row["p_value"] < 0.05 else ""
            print(
                f"{row['metric']:<14} {row['n']:>5} "
                f"{row['rho']:>9.3f} {row['p_value']:>10.3g}{sig}"
            )
    print("-" * W)
    print("* p < 0.05  |  n < 3 -> N/A")
    print("=" * W)
    print()


# ─────────────────────────────────────────────
#  CASOS EXTREMOS: RAIZ vs. FOLHA MAIS FUNDA
# ─────────────────────────────────────────────

def summarize_evidence_position_extremes(df: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """
    Resumo isolado (fora do binning) dos dois casos extremos mais
    interpretáveis fisicamente: evidência = raiz da rede
    (node_depth_normalized == 0) vs. evidência = folha mais funda
    (node_depth_normalized == 1). Uma linha por dataset x extremo, com
    accuracy e exact_match médios — para comentário textual no artigo.
    """
    root = df[df["node_depth_normalized"] <= tol].copy()
    leaf = df[df["node_depth_normalized"] >= 1 - tol].copy()
    root["extremo"] = "raiz (depth=0)"
    leaf["extremo"] = "folha mais funda (depth=1)"

    combined = pd.concat([root, leaf])
    if combined.empty:
        print("[AVISO] Nenhum registro com posição exatamente 0 ou 1 encontrado.")
        return combined

    summary = (
        combined.groupby(["dataset", "extremo"], observed=True)
        .agg(
            n=("evidence_node", "size"),
            accuracy=("accuracy", "mean"),
            exact_match=("exact_match", "mean"),
        )
        .reset_index()
    )
    return summary


def print_evidence_position_extremes(summary: pd.DataFrame):
    if summary.empty:
        return
    W = 70
    print("=" * W)
    print(f"{'CASOS EXTREMOS: EVIDÊNCIA = RAIZ vs. FOLHA MAIS FUNDA':^{W}}")
    print("=" * W)
    print(f"{'Dataset':<20} {'Extremo':<26} {'n':>4} {'Acc':>8} {'Exact':>8}")
    print("-" * W)
    for _, row in summary.iterrows():
        print(
            f"{translate_dataset(row['dataset']):<20} {row['extremo']:<26} {int(row['n']):>4} "
            f"{row['accuracy'] * 100:>7.1f}% {row['exact_match'] * 100:>7.1f}%"
        )
    print("-" * W)
    for extremo, rows in summary.groupby("extremo", observed=True):
        total = int(rows["n"].sum())
        accuracy = (rows["accuracy"] * rows["n"]).sum() / total
        exact_match = (rows["exact_match"] * rows["n"]).sum() / total
        print(
            f"{'[GLOBAL]':<20} {extremo:<26} {total:>4} "
            f"{accuracy * 100:>7.1f}% {exact_match * 100:>7.1f}%"
        )
    print("=" * W)
    print()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from pgm_llm_inference.core.config import InferenceConfig
    config = InferenceConfig()

    df = load_evidence_position_logs(config.log_file_name)

    if df.empty:
        raise SystemExit(1)

    figure_path = Path(config.log_file_name).with_name("evidence_position.png")
    plot_accuracy_by_evidence_position(df, n_bins=5, min_n=5, save_path=figure_path)

    corr = compute_evidence_position_correlations(df)
    print_evidence_position_correlations(corr)

    extremes = summarize_evidence_position_extremes(df)
    print_evidence_position_extremes(extremes)
