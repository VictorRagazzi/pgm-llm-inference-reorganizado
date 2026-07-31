import json
import pandas as pd
from pgm_llm_inference.core.config import InferenceConfig
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import Counter
import math
from typing import Optional
from sklearn.metrics import f1_score, cohen_kappa_score

config = InferenceConfig()

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def load_logs(path):
    logs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return pd.DataFrame(logs)


def frozen_evidence(ev: dict) -> str:
    """Transforma o dicionário de evidências em uma string imutável e ordenada."""
    if not ev:
        return "{}"
    return json.dumps(dict(sorted(ev.items())), sort_keys=True)


# ─────────────────────────────────────────────
#  VOTING HELPERS
# ─────────────────────────────────────────────

def compute_mode(values: list):
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def compute_weighted_mode(values: list, weights: list[float]):
    """
    Votação ponderada: cada voto vale o peso do agente que o emitiu.
    Em caso de empate, cai no fallback de moda simples.
    """
    if not values:
        return None

    scores: dict = {}
    for val, w in zip(values, weights):
        scores[val] = scores.get(val, 0.0) + w

    max_score = max(scores.values())
    winners = [label for label, score in scores.items() if score == max_score]

    if len(winners) == 1:
        return winners[0]

    # Fallback: moda simples entre os empatados
    return Counter(values).most_common(1)[0][0]


def _resolve_prediction(preds_with_idx: list[tuple], norm_weights: Optional[list[float]]):
    """
    Dado uma lista de (valor, índice_do_agente) e os pesos normalizados,
    retorna a predição final (ponderada ou por moda simples).
    """
    preds   = [p for p, _ in preds_with_idx]
    indices = [i for _, i in preds_with_idx]

    if norm_weights is not None:
        weights = [norm_weights[i] for i in indices]
        return compute_weighted_mode(preds, weights)

    return compute_mode(preds)


def _normalize_weights(agent_weights: Optional[list[float]], n_rows: int) -> Optional[list[float]]:
    """Valida e normaliza os pesos. Retorna None se não houver pesos."""
    if agent_weights is None:
        return None
    assert len(agent_weights) == n_rows, (
        f"agent_weights deve ter {n_rows} elementos, recebeu {len(agent_weights)}"
    )
    total = sum(agent_weights)
    return [w / total for w in agent_weights]


# ─────────────────────────────────────────────
#  CORE METRICS
# ─────────────────────────────────────────────

def calc_group_metrics(rows: list[dict], agent_weights: Optional[list[float]] = None) -> dict:
    """
    Calcula métricas de classificação multiclasse para um grupo de experimentos.

    Args:
        rows:          Lista de resultados por agente/experimento (mesma ordem dos pesos).
        agent_weights: Pesos manuais por agente. Se None, usa moda simples (comportamento original).
    """
    rep = rows[0]
    ma  = rep.get("map_assignment", {})
    targets = [k for k in ma if k != "_scalar"]

    if not targets:
        return {}

    norm_weights = _normalize_weights(agent_weights, len(rows))

    mode_preds = {}
    for k in targets:
        preds_with_idx = [
            (r["llm_predictions"][k], i)
            for i, r in enumerate(rows)
            if k in r.get("llm_predictions", {})
        ]
        if preds_with_idx:
            mode_preds[k] = _resolve_prediction(preds_with_idx, norm_weights)

    y_true = [ma[k] for k in targets if k in mode_preds]
    y_pred = [mode_preds[k] for k in targets if k in mode_preds]

    if not y_true:
        return {}

    n    = len(y_true)
    hits = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = hits / n

    labels = sorted(set(y_true + y_pred))

    if len(labels) == 1:
        f1_macro    = 1.0 if y_true == y_pred else 0.0
        f1_weighted = f1_macro
        kappa       = None
    else:
        f1_macro    = f1_score(y_true, y_pred, average="macro",    labels=labels, zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        try:
            kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
        except Exception:
            kappa = None

    consensus_scores = []
    for k in targets:
        preds = [r["llm_predictions"][k] for r in rows if k in r.get("llm_predictions", {})]
        if preds:
            top_count = Counter(preds).most_common(1)[0][1]
            consensus_scores.append(top_count / len(preds))
    avg_consensus = sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.0

    return {
        "accuracy":    accuracy,
        "f1_macro":    f1_macro,
        "f1_weighted": f1_weighted,
        "kappa":       kappa,
        "consensus":   avg_consensus,
        "hits":        hits,
        "total_nodes": n,
    }


def calc_group_accuracy(rows: list[dict], agent_weights: Optional[list[float]] = None) -> float:
    """
    Calcula a acurácia baseada em votação ponderada (ou moda simples se sem pesos).

    Args:
        rows:          Lista de resultados por agente/experimento.
        agent_weights: Pesos manuais por agente. Se None, usa moda simples.
    """
    rep     = rows[0]
    ma      = rep.get("map_assignment", {})
    targets = [k for k in ma if k != "_scalar"]

    if not targets:
        return 0.0
    
    norm_weights     = _normalize_weights(agent_weights, len(rows))
    all_pred_keys    = set(k for r in rows for k in r.get("llm_predictions", {}))

    mode_preds = {}
    for k in all_pred_keys:
        preds_with_idx = [
            (r["llm_predictions"][k], i)
            for i, r in enumerate(rows)
            if k in r.get("llm_predictions", {})
        ]
        if preds_with_idx:
            mode_preds[k] = _resolve_prediction(preds_with_idx, norm_weights)

    hits = sum(1 for k in targets if mode_preds.get(k) == ma[k])
    return hits / len(targets)


# ─────────────────────────────────────────────
#  GROUPED ANALYSIS
# ─────────────────────────────────────────────

def compute_grouped_accuracies(
    df: pd.DataFrame,
    resolution: int = 5,
    agent_weights: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Agrupa por dataset e evidência, aplicando intervalos (binning) no ratio."""
    records = df.to_dict("records")
    groups: dict[tuple, list] = {}

    for row in records:
        key = (row["dataset"], frozen_evidence(row.get("evidence")))
        groups.setdefault(key, []).append(row)

    results = []
    for (dataset, _), rows in groups.items():
        acc = calc_group_accuracy(rows, agent_weights=agent_weights)
        rep = rows[0]

        ev_len   = rep.get("evidence_length", 0)
        eval_len = rep.get("evaluated_length", 0)

        total       = ev_len + eval_len
        raw_ratio   = (ev_len / total * 100) if total > 0 else 0
        binned_ratio = round(raw_ratio / resolution) * resolution

        results.append({
            "dataset":        dataset,
            "evidence_ratio": binned_ratio,
            "accuracy":       acc,
        })

    return pd.DataFrame(results)

def _pipeline_column(df: pd.DataFrame) -> pd.Series:
    """Fallback pra logs antigos que ainda não tinham o campo 'pipeline'."""
    if "pipeline" in df.columns:
        return df["pipeline"].fillna("part1")
    return pd.Series(["part1"] * len(df), index=df.index)

def compute_grouped_accuracies_by_pipeline(
    df: pd.DataFrame,
    resolution: int = 5,
    agent_weights: Optional[list[float]] = None,
) -> pd.DataFrame:
    """Igual a compute_grouped_accuracies, mas agrupando por pipeline em vez de dataset."""
    df = df.copy()
    df["_pipeline"] = _pipeline_column(df)
    records = df.to_dict("records")
    groups: dict[tuple, list] = {}

    for row in records:
        key = (row["_pipeline"], frozen_evidence(row.get("evidence")))
        groups.setdefault(key, []).append(row)

    results = []
    for (pipeline, _), rows in groups.items():
        acc = calc_group_accuracy(rows, agent_weights=agent_weights)
        rep = rows[0]

        ev_len   = rep.get("evidence_length", 0)
        eval_len = rep.get("evaluated_length", 0)
        total     = ev_len + eval_len
        raw_ratio = (ev_len / total * 100) if total > 0 else 0
        binned_ratio = round(raw_ratio / resolution) * resolution

        results.append({
            "pipeline":       pipeline,
            "evidence_ratio": binned_ratio,
            "accuracy":       acc,
        })

    return pd.DataFrame(results)

# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────

def plot_grouped_accuracy(
    df: pd.DataFrame,
    datasets_per_img: int = 5,
    agent_weights: Optional[list[float]] = None,
):
    """Gera visualizações focadas em dataset e ratio de evidência."""
    grouped = compute_grouped_accuracies(df, agent_weights=agent_weights)
    grouped["accuracy_pct"] = grouped["accuracy"] * 100

    datasets   = sorted(grouped["dataset"].unique())
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    # Paleta consistente para paper
    BAR_COLOR    = "#2C6E9E"
    EDGE_COLOR   = "#1A3F5C"
    GRID_COLOR   = "#CCCCCC"
    LABEL_FS     = 8
    TITLE_FS     = 11
    AXIS_FS      = 9

    sns.set_theme(style="whitegrid", rc={
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.color":         GRID_COLOR,
        "grid.linewidth":     0.6,
    })

    for i in range(num_chunks):
        chunk_datasets = datasets[i * datasets_per_img:(i + 1) * datasets_per_img]
        subset = grouped[grouped["dataset"].isin(chunk_datasets)].copy()

        # ── Gráfico 1: Acurácia Média por Dataset ───────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        overall = (
            subset.groupby("dataset")["accuracy_pct"]
            .mean()
            .reindex(chunk_datasets)
            .reset_index()
        )

        sns.barplot(
            data=overall, x="dataset", y="accuracy_pct",
            color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, ax=ax
        )

        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=LABEL_FS)

        ax.set_title(
            "Accuracy of the Experiments by dataset",
            fontsize=TITLE_FS, fontweight="bold", pad=10
        )
        ax.set_ylabel("Accuracy (%)", fontsize=AXIS_FS)
        ax.set_xlabel("Dataset", fontsize=AXIS_FS)
        ax.set_ylim(0, 115)
        ax.tick_params(axis="both", labelsize=LABEL_FS)
        ax.tick_params(axis="x", rotation=15)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.show()

        # ── Gráfico 2: Barras por Evidence Ratio, subplots por Dataset ───────
        n     = len(chunk_datasets)
        ncols = min(n, 2)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(5.5 * ncols, 4 * nrows),
            sharey=True,
            squeeze=False
        )

        for j, ds in enumerate(chunk_datasets):
            ax = axes[j // ncols][j % ncols]

            ds_data = subset[subset["dataset"] == ds].copy()
            ds_agg = (
                ds_data.groupby("evidence_ratio")["accuracy_pct"]
                .mean()
                .reset_index()
                .sort_values("evidence_ratio")
            )

            ratios = ds_agg["evidence_ratio"].astype(str).tolist()
            values = ds_agg["accuracy_pct"].tolist()
            x_pos  = range(len(ratios))

            bars = ax.bar(
                x_pos, values,
                color=BAR_COLOR, edgecolor=EDGE_COLOR,
                linewidth=0.6, width=0.6, zorder=3
            )

            # Rótulos sobre as barras
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}%",
                    ha="center", va="bottom",
                    fontsize=7, color="#222222"
                )

            ax.set_title(ds, fontsize=TITLE_FS, fontweight="bold", pad=6)
            ax.set_xticks(list(x_pos))
            ax.set_xticklabels(ratios, fontsize=LABEL_FS, rotation=0)
            # ax.set_xlabel("Evidence Ratio (%)", fontsize=AXIS_FS)

            if j % ncols == 0:
                ax.set_ylabel("Mean Accuracy (%)", fontsize=AXIS_FS)
            else:
                ax.set_ylabel("")

            ax.set_ylim(0, 115)
            ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
            ax.set_axisbelow(True)
            ax.tick_params(axis="y", labelsize=LABEL_FS)

        # Remove subplots vazios
        for k in range(len(chunk_datasets), nrows * ncols):
            axes[k // ncols][k % ncols].set_visible(False)

        fig.suptitle(
            "Mean Accuracy by Evidence Ratio per Dataset",
            fontsize=TITLE_FS + 1, fontweight="bold", y=1.02
        )

        plt.tight_layout()
        plt.show()

def plot_pipeline_accuracy(
    df: pd.DataFrame,
    agent_weights: Optional[list[float]] = None,
):
    """
    Mesmo estilo de plot_grouped_accuracy, mas comparando PIPELINES entre
    si (part1 vs part2_forward vs part2_twopass) em vez de datasets —
    a média de cada execução, lado a lado (E2.3).
    """
    grouped = compute_grouped_accuracies_by_pipeline(df, agent_weights=agent_weights)
    grouped["accuracy_pct"] = grouped["accuracy"] * 100
    pipelines = sorted(grouped["pipeline"].unique())

    BAR_COLOR, EDGE_COLOR, GRID_COLOR = "#2C6E9E", "#1A3F5C", "#CCCCCC"
    LABEL_FS, TITLE_FS, AXIS_FS = 8, 11, 9

    sns.set_theme(style="whitegrid", rc={
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID_COLOR, "grid.linewidth": 0.6,
    })

    # ── Gráfico 1: acurácia média geral por pipeline ────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(pipelines) * 2), 5))
    overall = (
        grouped.groupby("pipeline")["accuracy_pct"]
        .mean()
        .reindex(pipelines)
        .reset_index()
    )
    sns.barplot(
        data=overall, x="pipeline", y="accuracy_pct",
        color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=LABEL_FS)

    ax.set_title("Mean Accuracy by Pipeline", fontsize=TITLE_FS, fontweight="bold", pad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=AXIS_FS)
    ax.set_xlabel("Pipeline", fontsize=AXIS_FS)
    ax.set_ylim(0, 115)
    ax.tick_params(axis="both", labelsize=LABEL_FS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()

    # ── Gráfico 2: acurácia média por evidence_ratio, uma linha por pipeline ──
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("tab10", n_colors=len(pipelines))

    for color, pipeline in zip(palette, pipelines):
        sub = grouped[grouped["pipeline"] == pipeline]
        agg = (
            sub.groupby("evidence_ratio")["accuracy_pct"]
            .mean()
            .reset_index()
            .sort_values("evidence_ratio")
        )
        ax.plot(agg["evidence_ratio"], agg["accuracy_pct"],
                marker="o", label=pipeline, color=color, linewidth=2)

    ax.set_title("Mean Accuracy by Evidence Ratio, per Pipeline",
                 fontsize=TITLE_FS, fontweight="bold", pad=10)
    ax.set_xlabel("Evidence Ratio (%)", fontsize=AXIS_FS)
    ax.set_ylabel("Mean Accuracy (%)", fontsize=AXIS_FS)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(title="Pipeline", fontsize=9, title_fontsize=9)
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
#  TABLES
# ─────────────────────────────────────────────

def print_evidence_accuracy_table(
    df: pd.DataFrame,
    evidence_length: int = 1,
    agent_weights: Optional[list[float]] = None,
):
    df_filtered = df[df["evidence_length"] == evidence_length].copy()

    if df_filtered.empty:
        print(f"Nenhum dado encontrado com evidence_length = {evidence_length}")
        return

    df_filtered["evidence_key"] = df_filtered["evidence"].apply(
        lambda ev: ", ".join(f"{k}={v}" for k, v in sorted(ev.items()))
    )

    datasets = sorted(df_filtered["dataset"].unique())
    W        = 100
    print("=" * W)
    print(f"{'ACURÁCIA POR EVIDÊNCIA  (evidence_length = ' + str(evidence_length) + ')':^{W}}")
    print("=" * W)

    for ds in datasets:
        ds_data        = df_filtered[df_filtered["dataset"] == ds]
        evidence_groups = sorted(ds_data["evidence_key"].unique())

        print(f"\n┌─ Dataset: {ds}")
        print(f"│  {'Evidência':<28} {'Acertos':>8} {'Acc':>7} {'F1 Mac':>8} {'F1 Wgt':>8} {'Kappa':>8} {'Consenso':>9}")
        print(f"│  " + "─" * 82)

        all_metrics = []
        
        for ev_key in evidence_groups:
            rows = ds_data[ds_data["evidence_key"] == ev_key].to_dict("records")
            m    = calc_group_metrics(rows, agent_weights=agent_weights)
            if not m:
                continue

            all_metrics.append(m)
            kappa_str = f"{m['kappa']:>8.3f}" if m["kappa"] is not None else "     N/A"

            print(
                f"│  {ev_key:<28} "
                f"{m['hits']:>3}/{m['total_nodes']:<3}  "
                f"{m['accuracy']*100:>6.1f}%  "
                f"{m['f1_macro']:>7.3f}  "
                f"{m['f1_weighted']:>7.3f}  "
                f"{kappa_str}  "
                f"{m['consensus']*100:>7.1f}%"
            )

        if all_metrics:
            print(f"│  " + "─" * 82)
            mean        = lambda key: sum(m[key] for m in all_metrics) / len(all_metrics)
            valid_kappas = [m["kappa"] for m in all_metrics if m["kappa"] is not None]
            mean_kappa  = sum(valid_kappas) / len(valid_kappas) if valid_kappas else None
            kappa_str   = f"{mean_kappa:>8.3f}" if mean_kappa is not None else "     N/A"
            total_hits  = sum(m["hits"] for m in all_metrics)
            total_nodes = sum(m["total_nodes"] for m in all_metrics)

            print(
                f"│  {'[MÉDIA DO DATASET]':<28} "
                f"{total_hits:>3}/{total_nodes:<3}  "
                f"{mean('accuracy')*100:>6.1f}%  "
                f"{mean('f1_macro'):>7.3f}  "
                f"{mean('f1_weighted'):>7.3f}  "
                f"{kappa_str}  "
                f"{mean('consensus')*100:>7.1f}%"
            )

        print(f"└" + "─" * 84)

    print()
    print("=" * W)

def print_pipeline_accuracy_table(
    df: pd.DataFrame,
    evidence_length: int = 1,
    agent_weights: Optional[list[float]] = None,
):
    df = df.copy()
    df["_pipeline"] = _pipeline_column(df)
    df_filtered = df[df["evidence_length"] == evidence_length].copy()

    if df_filtered.empty:
        print(f"Nenhum dado encontrado com evidence_length = {evidence_length}")
        return

    df_filtered["evidence_key"] = df_filtered["evidence"].apply(
        lambda ev: ", ".join(f"{k}={v}" for k, v in sorted(ev.items()))
    )

    pipelines = sorted(df_filtered["_pipeline"].unique())
    W = 100
    print("=" * W)
    print(f"{'ACURÁCIA POR PIPELINE  (evidence_length = ' + str(evidence_length) + ')':^{W}}")
    print("=" * W)

    for pipeline in pipelines:
        p_data = df_filtered[df_filtered["_pipeline"] == pipeline]
        evidence_groups = sorted(p_data["evidence_key"].unique())

        print(f"\n┌─ Pipeline: {pipeline}")
        print(f"│  {'Evidência':<28} {'Acertos':>8} {'Acc':>7} {'F1 Mac':>8} {'F1 Wgt':>8} {'Kappa':>8} {'Consenso':>9}")
        print(f"│  " + "─" * 82)

        all_metrics = []
        for ev_key in evidence_groups:
            rows = p_data[p_data["evidence_key"] == ev_key].to_dict("records")
            m = calc_group_metrics(rows, agent_weights=agent_weights)
            if not m:
                continue
            all_metrics.append(m)
            kappa_str = f"{m['kappa']:>8.3f}" if m["kappa"] is not None else "     N/A"
            print(
                f"│  {ev_key:<28} "
                f"{m['hits']:>3}/{m['total_nodes']:<3}  "
                f"{m['accuracy']*100:>6.1f}%  "
                f"{m['f1_macro']:>7.3f}  "
                f"{m['f1_weighted']:>7.3f}  "
                f"{kappa_str}  "
                f"{m['consensus']*100:>7.1f}%"
            )

        if all_metrics:
            print(f"│  " + "─" * 82)
            mean = lambda key: sum(m[key] for m in all_metrics) / len(all_metrics)
            valid_kappas = [m["kappa"] for m in all_metrics if m["kappa"] is not None]
            mean_kappa = sum(valid_kappas) / len(valid_kappas) if valid_kappas else None
            kappa_str = f"{mean_kappa:>8.3f}" if mean_kappa is not None else "     N/A"
            total_hits = sum(m["hits"] for m in all_metrics)
            total_nodes = sum(m["total_nodes"] for m in all_metrics)

            print(
                f"│  {'[MÉDIA DO PIPELINE]':<28} "
                f"{total_hits:>3}/{total_nodes:<3}  "
                f"{mean('accuracy')*100:>6.1f}%  "
                f"{mean('f1_macro'):>7.3f}  "
                f"{mean('f1_weighted'):>7.3f}  "
                f"{kappa_str}  "
                f"{mean('consensus')*100:>7.1f}%"
            )

        print(f"└" + "─" * 84)

    print()
    print("=" * W)

# ─────────────────────────────────────────────
#  ENSEMBLE
# ─────────────────────────────────────────────

import warnings 

def print_prompt_table(df: pd.DataFrame):
    # --- Por prompt_type: média de log_accuracy ---
    prompt_accuracies = (
        df.groupby("prompt_type")["log_accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"log_accuracy": "mean_accuracy"})
    )

    # --- Ensemble (moda) ---
    def make_evidence_key(evidence: dict) -> str:
        return "&".join(f"{k}={v}" for k, v in sorted(evidence.items()))

    df = df.copy()
    df["_evidence_key"] = df["evidence"].apply(make_evidence_key)

    prompt_types = df["prompt_type"].unique().tolist()
    evidence_keys = df["_evidence_key"].unique().tolist()

    ensemble_accuracies = []

    for ev_key in evidence_keys:
        ev_df = df[df["_evidence_key"] == ev_key]
        # Um representante por prompt_type
        rows = []
        for pt in prompt_types:
            subset = ev_df[ev_df["prompt_type"] == pt]
            if not subset.empty:
                rows.append(subset.iloc[0].to_dict())

        if not rows:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics = calc_group_metrics(rows)

        ensemble_accuracies.append(metrics["accuracy"])

    ensemble_mean = sum(ensemble_accuracies) / len(ensemble_accuracies) if ensemble_accuracies else 0.0

    # --- Montar e imprimir tabela ---
    rows_table = []
    for _, row in prompt_accuracies.iterrows():
        rows_table.append((row["prompt_type"], row["mean_accuracy"]))
    rows_table.append(("Ensemble (moda)", ensemble_mean))

    col1_w = max(len(r[0]) for r in rows_table)
    col1_w = max(col1_w, len("Prompt Type"))
    col2_w = max(len("Accuracy"), 10)

    sep = f"+{'-' * (col1_w + 2)}+{'-' * (col2_w + 2)}+"
    header = f"| {'Prompt Type':<{col1_w}} | {'Accuracy':>{col2_w}} |"

    print(sep)
    print(header)
    print(sep)
    for name, acc in rows_table[:-1]:
        print(f"| {name:<{col1_w}} | {acc:>{col2_w}.4f} |")
    print(sep)
    # Linha ensemble destacada
    name, acc = rows_table[-1]
    print(f"| {name:<{col1_w}} | {acc:>{col2_w}.4f} |")
    print(sep)

def plot_prompt_by_dataset(df: pd.DataFrame):
    import warnings
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from collections import Counter
    from typing import Optional

    # ---- helpers (mesmos de antes) ----
    def make_evidence_key(evidence: dict) -> str:
        return "&".join(f"{k}={v}" for k, v in sorted(evidence.items()))

    def get_ensemble_accuracy(sub: pd.DataFrame) -> float:
        sub = sub.copy()
        sub["_ev_key"] = sub["evidence"].apply(make_evidence_key)
        prompt_types = sub["prompt_type"].unique().tolist()
        accs = []
        for ev_key in sub["_ev_key"].unique():
            ev_df = sub[sub["_ev_key"] == ev_key]
            rows = []
            for pt in prompt_types:
                s = ev_df[ev_df["prompt_type"] == pt]
                if not s.empty:
                    rows.append(s.iloc[0].to_dict())
            if not rows:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                accs.append(calc_group_metrics(rows)["accuracy"])
        return float(np.mean(accs)) if accs else 0.0

    # ---- agregar dados ----
    datasets = sorted(df["dataset"].dropna().unique())
    prompt_types = sorted(df["prompt_type"].dropna().unique())

    # matriz: dataset x prompt_type (+ ensemble)
    all_labels = list(prompt_types) + ["Ensemble (moda)"]
    data = {label: [] for label in all_labels}

    for ds in datasets:
        sub = df[df["dataset"] == ds]
        for pt in prompt_types:
            mean_acc = sub[sub["prompt_type"] == pt]["log_accuracy"].mean()
            data[pt].append(float(mean_acc) if not np.isnan(mean_acc) else 0.0)
        data["Ensemble (moda)"].append(get_ensemble_accuracy(sub))

    # ---- plot ----
    n_datasets = len(datasets)
    n_labels = len(all_labels)
    x = np.arange(n_datasets)
    bar_w = 0.72 / n_labels

    palette = plt.cm.get_cmap("tab10")
    colors = [palette(i / max(n_labels - 1, 1)) for i in range(n_labels - 1)]
    ensemble_color = "#2d2d2d"

    fig, ax = plt.subplots(figsize=(max(7, n_datasets * 1.8), 5))
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    for i, label in enumerate(all_labels):
        vals = data[label]
        offset = (i - n_labels / 2 + 0.5) * bar_w
        is_ensemble = label == "Ensemble (moda)"
        color = ensemble_color if is_ensemble else colors[i]
        hatch = "//" if is_ensemble else None
        bars = ax.bar(
            x + offset, vals, width=bar_w * 0.9,
            color=color, alpha=0.88 if is_ensemble else 0.78,
            hatch=hatch, edgecolor="white", linewidth=0.5,
            label=label, zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{val:.2f}", ha="center", va="bottom",
                fontsize=7.5, color="#333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, min(1.0, max(max(v) for v in data.values()) * 1.18))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    ax.legend(
        loc="upper right", frameon=True, framealpha=0.9,
        fontsize=9, title="Prompt type", title_fontsize=9,
    )

    fig.tight_layout()
    plt.show()

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

LOG_FILE = config.log_file_name
df = load_logs(LOG_FILE)

# ── Defina os pesos aqui (um valor por agente, na mesma ordem do DataFrame) ──
# Exemplo: primeiro agente vale o dobro dos demais
# AGENT_WEIGHTS = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
#
# Para usar votação simples (comportamento original), deixe como None:
AGENT_WEIGHTS = None

# print_prompt_table(df)
# plot_prompt_by_dataset(df)
print_pipeline_accuracy_table(df, agent_weights=AGENT_WEIGHTS)

plot_grouped_accuracy(df, datasets_per_img=9, agent_weights=AGENT_WEIGHTS)
plot_pipeline_accuracy(df, agent_weights=AGENT_WEIGHTS)
