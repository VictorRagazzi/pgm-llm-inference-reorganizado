"""
Análise dos experimentos de inferência MPE via LLM.

Métricas por variável (accuracy, F1, kappa, consenso) e por configuração
conjunta (exact/joint match), com quebra por tipo de amostragem de
evidência (mpe_consistent vs random) e por layout (nested vs independent).
"""

import json
import math
from collections import Counter
from typing import Optional

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, cohen_kappa_score

from pgm_llm_inference.core.config import InferenceConfig

config = InferenceConfig()

# Paleta consistente para o paper
BAR_COLOR  = "#2C6E9E"
EDGE_COLOR = "#1A3F5C"
GRID_COLOR = "#CCCCCC"
LABEL_FS   = 8
TITLE_FS   = 11
AXIS_FS    = 9

sns.set_theme(style="whitegrid", rc={
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
})


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_logs(path: str) -> pd.DataFrame:
    logs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    df = pd.DataFrame(logs)

    # Compatibilidade com logs antigos, gerados antes das colunas novas
    if "evidence_sampling" not in df.columns:
        df["evidence_sampling"] = "mpe_consistent"
    if "evidence_layout" not in df.columns:
        df["evidence_layout"] = "nested"
    if "exact_match" not in df.columns:
        df["exact_match"] = None

    df["evidence_sampling"] = df["evidence_sampling"].fillna("mpe_consistent")
    df["evidence_layout"]   = df["evidence_layout"].fillna("nested")

    return df


def frozen_evidence(ev: dict) -> str:
    """Transforma o dicionário de evidências em uma string imutável e ordenada."""
    if not ev:
        return "{}"
    return json.dumps(dict(sorted(ev.items())), sort_keys=True)


# ─────────────────────────────────────────────
#  VOTAÇÃO (moda simples — sem pesos; AGENT_WEIGHTS removido por não ter uso)
# ─────────────────────────────────────────────

def _mode_predictions(rows: list[dict]) -> dict[str, str]:
    """Para cada variável prevista, retorna o valor mais votado entre as rows do grupo."""
    all_keys = set(k for r in rows for k in r.get("llm_predictions", {}))
    preds = {}
    for k in all_keys:
        values = [r["llm_predictions"][k] for r in rows if k in r.get("llm_predictions", {})]
        if values:
            preds[k] = Counter(values).most_common(1)[0][0]
    return preds


# ─────────────────────────────────────────────
#  MÉTRICAS POR GRUPO (mesma evidência exata)
# ─────────────────────────────────────────────

def calc_group_metrics(rows: list[dict]) -> dict:
    """
    Calcula métricas de classificação multiclasse e de correspondência
    conjunta para um grupo de linhas que compartilham a mesma evidência
    exata (mesmo dataset + mesmo dict de evidência + mesmo tipo de
    amostragem/layout).
    """
    rep = rows[0]
    ma  = rep.get("map_assignment", {})
    targets = [k for k in ma if k != "_scalar"]

    if not targets:
        return {}

    mode_preds = _mode_predictions(rows)

    y_true = [ma[k] for k in targets if k in mode_preds]
    y_pred = [mode_preds[k] for k in targets if k in mode_preds]

    if not y_true:
        return {}

    n    = len(y_true)
    hits = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = hits / n

    labels = sorted(set(y_true + y_pred))
    if len(labels) == 1:
        f1_macro, f1_weighted = (1.0, 1.0) if y_true == y_pred else (0.0, 0.0)
        kappa = None
    else:
        f1_macro    = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        try:
            kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
        except Exception:
            kappa = None

    # Consenso: quão concordes estão as rows do grupo entre si, por variável
    consensus_scores = []
    for k in targets:
        preds = [r["llm_predictions"][k] for r in rows if k in r.get("llm_predictions", {})]
        if preds:
            top_count = Counter(preds).most_common(1)[0][1]
            consensus_scores.append(top_count / len(preds))
    avg_consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.0

    # Correspondência da configuração conjunta: só é 1.0 se TODAS as
    # variáveis-alvo tiverem sido previstas e todas baterem com o MPE.
    # Complementa a acurácia por variável, que pode ser alta mesmo quando
    # o MPE completo nunca é recuperado.
    joint_match = 1.0 if (len(mode_preds) == n and hits == n) else 0.0

    # exact_match já vem calculado por experimento (em run_experiment); com
    # uma única row por grupo coincide com joint_match. Mantemos os dois:
    # exact_match é o valor "cru" por experimento, joint_match é o valor
    # pós-votação (podem divergir quando há múltiplas rows por grupo).
    reported_exact = [r.get("exact_match") for r in rows if r.get("exact_match") is not None]
    mean_exact = sum(reported_exact) / len(reported_exact) if reported_exact else None

    return {
        "accuracy":     accuracy,
        "f1_macro":     f1_macro,
        "f1_weighted":  f1_weighted,
        "kappa":        kappa,
        "consensus":    avg_consensus,
        "joint_match":  joint_match,
        "exact_match":  mean_exact,
        "hits":         hits,
        "total_nodes":  n,
    }


# ─────────────────────────────────────────────
#  AGRUPAMENTO
# ─────────────────────────────────────────────

GROUP_COLUMNS = ["dataset", "evidence_sampling", "evidence_layout"]


def _group_rows(df: pd.DataFrame) -> dict[tuple, list[dict]]:
    """Agrupa por dataset + tipo de amostragem/layout + evidência exata."""
    groups: dict[tuple, list[dict]] = {}
    for row in df.to_dict("records"):
        key = tuple(row.get(c) for c in GROUP_COLUMNS) + (frozen_evidence(row.get("evidence")),)
        groups.setdefault(key, []).append(row)
    return groups


def compute_grouped_table(df: pd.DataFrame, resolution: int = 5) -> pd.DataFrame:
    """
    Uma linha por combinação (dataset, evidence_sampling, evidence_layout,
    evidência exata), com todas as métricas de calc_group_metrics mais a
    proporção de evidência (binned) — base para todos os plots e tabelas.
    """
    groups = _group_rows(df)
    records = []

    for key, rows in groups.items():
        dataset, sampling, layout, _ = key
        m = calc_group_metrics(rows)
        if not m:
            continue

        rep = rows[0]
        ev_len   = rep.get("evidence_length", 0)
        eval_len = rep.get("evaluated_length", 0)
        total    = ev_len + eval_len
        raw_ratio    = (ev_len / total * 100) if total > 0 else 0
        binned_ratio = round(raw_ratio / resolution) * resolution

        records.append({
            "dataset":           dataset,
            "evidence_sampling": sampling,
            "evidence_layout":   layout,
            "evidence_length":   ev_len,
            "evidence_ratio":    binned_ratio,
            **m,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────

def _bar_with_labels(ax, data, x_col, y_col, fmt="%.1f%%"):
    sns.barplot(data=data, x=x_col, y=y_col, color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=LABEL_FS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=LABEL_FS)


def plot_accuracy_by_dataset(grouped: pd.DataFrame, datasets_per_img: int = 9):
    """Acurácia média por dataset (mesmo gráfico do script original)."""
    grouped = grouped.copy()
    grouped["accuracy_pct"] = grouped["accuracy"] * 100

    datasets   = sorted(grouped["dataset"].unique())
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    for i in range(num_chunks):
        chunk  = datasets[i * datasets_per_img:(i + 1) * datasets_per_img]
        subset = grouped[grouped["dataset"].isin(chunk)]
        overall = subset.groupby("dataset")["accuracy_pct"].mean().reindex(chunk).reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        _bar_with_labels(ax, overall, "dataset", "accuracy_pct")
        ax.set_title("Accuracy of the Experiments by Dataset", fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.set_ylabel("Accuracy (%)", fontsize=AXIS_FS)
        ax.set_xlabel("Dataset", fontsize=AXIS_FS)
        ax.set_ylim(0, 115)
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.show()


def plot_accuracy_by_evidence_ratio(grouped: pd.DataFrame, datasets_per_img: int = 5):
    """Acurácia média por ratio de evidência, um subplot por dataset (original)."""
    datasets   = sorted(grouped["dataset"].unique())
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    for i in range(num_chunks):
        chunk = datasets[i * datasets_per_img:(i + 1) * datasets_per_img]
        n     = len(chunk)
        ncols = min(n, 2)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), sharey=True, squeeze=False)

        for j, ds in enumerate(chunk):
            ax = axes[j // ncols][j % ncols]
            ds_data = grouped[grouped["dataset"] == ds]
            ds_agg = (
                ds_data.groupby("evidence_ratio")["accuracy"]
                .mean().mul(100).reset_index().sort_values("evidence_ratio")
            )
            ratios = ds_agg["evidence_ratio"].astype(str).tolist()
            values = ds_agg["accuracy"].tolist()
            x_pos  = range(len(ratios))

            bars = ax.bar(x_pos, values, color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, width=0.6, zorder=3)
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=7, color="#222222")

            ax.set_title(ds, fontsize=TITLE_FS, fontweight="bold", pad=6)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(ratios, fontsize=LABEL_FS)
            if j % ncols == 0:
                ax.set_ylabel("Mean Accuracy (%)", fontsize=AXIS_FS)
            ax.set_ylim(0, 115)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

        for k in range(len(chunk), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)

        fig.suptitle("Mean Accuracy by Evidence Ratio per Dataset", fontsize=TITLE_FS + 1, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.show()


def plot_accuracy_by_sampling(grouped: pd.DataFrame):
    """
    NOVO: acurácia por variável e taxa de match conjunto (full MPE),
    comparando os modos de amostragem de evidência (mpe_consistent vs
    random) e layout (nested vs independent).
    """
    grouped = grouped.copy()
    grouped["sampling_label"] = grouped["evidence_sampling"] + " / " + grouped["evidence_layout"]

    agg = (
        grouped.groupby("sampling_label")[["accuracy", "joint_match"]]
        .mean().mul(100).reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    _bar_with_labels(axes[0], agg, "sampling_label", "accuracy")
    axes[0].set_title("Per-Variable Accuracy by Evidence Sampling", fontsize=TITLE_FS, fontweight="bold")
    axes[0].set_ylabel("Accuracy (%)", fontsize=AXIS_FS)
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 115)

    _bar_with_labels(axes[1], agg, "sampling_label", "joint_match")
    axes[1].set_title("Full-MPE Match Rate by Evidence Sampling", fontsize=TITLE_FS, fontweight="bold")
    axes[1].set_ylabel("Joint Match Rate (%)", fontsize=AXIS_FS)
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 115)

    plt.tight_layout()
    plt.show()


def plot_joint_match_by_dataset(grouped: pd.DataFrame, datasets_per_img: int = 9):
    """NOVO: taxa de recuperação exata do MPE completo, por dataset."""
    grouped = grouped.copy()
    grouped["joint_match_pct"] = grouped["joint_match"] * 100

    datasets   = sorted(grouped["dataset"].unique())
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    for i in range(num_chunks):
        chunk  = datasets[i * datasets_per_img:(i + 1) * datasets_per_img]
        subset = grouped[grouped["dataset"].isin(chunk)]
        overall = subset.groupby("dataset")["joint_match_pct"].mean().reindex(chunk).reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        _bar_with_labels(ax, overall, "dataset", "joint_match_pct")
        ax.set_title("Full-MPE Exact Match Rate by Dataset", fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.set_ylabel("Joint Match Rate (%)", fontsize=AXIS_FS)
        ax.set_xlabel("Dataset", fontsize=AXIS_FS)
        ax.set_ylim(0, 115)
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.show()


def plot_accuracy_vs_joint_match(grouped: pd.DataFrame):
    """
    NOVO: dispersão acurácia por variável x taxa de match conjunto, por
    grupo (dataset, evidência) — evidencia visualmente o gap entre as
    duas métricas (alta acurácia por variável não implica MPE completo).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=grouped, x="accuracy", y="joint_match",
        hue="evidence_sampling", style="evidence_layout",
        s=60, ax=ax, palette="deep",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRID_COLOR, linewidth=1, zorder=0)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Per-Variable Accuracy", fontsize=AXIS_FS)
    ax.set_ylabel("Full-MPE Joint Match", fontsize=AXIS_FS)
    ax.set_title("Per-Variable Accuracy vs. Full-MPE Recovery", fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  TABELA DE RESULTADOS
# ─────────────────────────────────────────────

def print_results_table(grouped: pd.DataFrame, evidence_length: Optional[int] = None):
    """
    Tabela única cobrindo dataset, tipo de amostragem/layout de evidência,
    acurácia por variável, F1, kappa, consenso e taxa de match conjunto —
    substitui as antigas print_evidence_accuracy_table / print_paper_table.
    """
    df = grouped if evidence_length is None else grouped[grouped["evidence_length"] == evidence_length]
    if df.empty:
        msg = "Nenhum dado encontrado"
        if evidence_length is not None:
            msg += f" com evidence_length = {evidence_length}"
        print(msg)
        return

    W = 112
    title = "RESULTADOS"
    title += f"  (evidence_length = {evidence_length})" if evidence_length is not None else "  (todas as evidence_length)"
    print("=" * W)
    print(f"{title:^{W}}")
    print("=" * W)
    header = (
        f"{'Dataset':<18} {'Sampling':<15} {'Layout':<12} {'Ev.Ratio':>8} "
        f"{'Acc':>7} {'F1-W':>7} {'F1-M':>7} {'Kappa':>7} {'Consenso':>9} {'Joint':>7}"
    )
    print(header)
    print("-" * W)

    group_cols = ["dataset", "evidence_sampling", "evidence_layout", "evidence_ratio"]
    for keys, rows in df.groupby(group_cols):
        dataset, sampling, layout, ratio = keys
        kappa_vals = rows["kappa"].dropna()
        kappa_str  = f"{kappa_vals.mean():>6.3f}" if len(kappa_vals) else "   N/A"
        print(
            f"{dataset:<18} {sampling:<15} {layout:<12} {ratio:>7.0f}% "
            f"{rows['accuracy'].mean()*100:>6.1f}% "
            f"{rows['f1_weighted'].mean():>6.3f} "
            f"{rows['f1_macro'].mean():>6.3f} "
            f"{kappa_str} "
            f"{rows['consensus'].mean()*100:>8.1f}% "
            f"{rows['joint_match'].mean()*100:>6.1f}%"
        )

    print("-" * W)
    print(
        f"{'[MÉDIA GERAL]':<18} {'':<15} {'':<12} {'':>8} "
        f"{df['accuracy'].mean()*100:>6.1f}% "
        f"{df['f1_weighted'].mean():>6.3f} "
        f"{df['f1_macro'].mean():>6.3f} "
        f"{'':>7} "
        f"{df['consensus'].mean()*100:>8.1f}% "
        f"{df['joint_match'].mean()*100:>6.1f}%"
    )
    print("=" * W)
    print()


def print_sampling_summary(grouped: pd.DataFrame):
    """NOVO: resumo agregado só por (evidence_sampling, evidence_layout), cruzando datasets."""
    W = 70
    print("=" * W)
    print(f"{'RESUMO POR TIPO DE AMOSTRAGEM DE EVIDÊNCIA':^{W}}")
    print("=" * W)
    print(f"{'Sampling':<15} {'Layout':<12} {'Acc':>7} {'Joint':>7} {'Consenso':>9}")
    print("-" * W)
    for keys, rows in grouped.groupby(["evidence_sampling", "evidence_layout"]):
        sampling, layout = keys
        print(
            f"{sampling:<15} {layout:<12} "
            f"{rows['accuracy'].mean()*100:>6.1f}% "
            f"{rows['joint_match'].mean()*100:>6.1f}% "
            f"{rows['consensus'].mean()*100:>8.1f}%"
        )
    print("=" * W)
    print()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    LOG_FILE = config.log_file_name
    df = load_logs(LOG_FILE)
    grouped = compute_grouped_table(df)

    plot_accuracy_by_dataset(grouped, datasets_per_img=9)
    plot_accuracy_by_evidence_ratio(grouped, datasets_per_img=5)
    plot_accuracy_by_sampling(grouped)
    plot_joint_match_by_dataset(grouped, datasets_per_img=9)
    plot_accuracy_vs_joint_match(grouped)

    print_sampling_summary(grouped)
    print_results_table(grouped, evidence_length=1)
    print_results_table(grouped)  # todas as evidence_length juntas