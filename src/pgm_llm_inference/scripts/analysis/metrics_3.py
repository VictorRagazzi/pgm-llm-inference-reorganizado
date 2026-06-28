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
            "Accuracy of the Experiments by Dataset",
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


def plot_dataset_accuracy(df, datasets_per_img=5):
    df = df.copy()

    df['has_context'] = df['context_type'].apply(
        lambda x: 'Com contexto' if pd.notnull(x) and x != 'null' else 'Sem contexto'
    )
    df['prompt_group'] = df['prompt_type'] + " (" + df['has_context'] + ")"

    base_orders    = ['expert_1', 'expert_2', 'expert_3', 'expert_4', 'expert_5', 'expert_6']
    context_orders = ['Com contexto', 'Sem contexto']
    desired_order  = [f"{p} ({c})" for c in context_orders for p in base_orders]
    existing_order = [item for item in desired_order if item in df['prompt_group'].unique()]

    df['prompt_group'] = pd.Categorical(df['prompt_group'], categories=existing_order, ordered=True)

    datasets   = df['dataset'].unique()
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    for i in range(num_chunks):
        start          = i * datasets_per_img
        chunk_datasets = datasets[start:start + datasets_per_img]
        subset         = df[df['dataset'].isin(chunk_datasets)].copy()

        plt.figure(figsize=(14, 7))
        sns.set_theme(style="whitegrid", palette="muted")

        ax = sns.barplot(
            data=subset,
            x='dataset', y='log_accuracy',
            hue='prompt_group', hue_order=existing_order,
            palette='viridis', edgecolor='black', linewidth=0.5, errorbar=None
        )

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3, fontsize=9)

        plt.title('Acurácia média por dataset', fontsize=18, fontweight='bold', pad=25)
        plt.ylabel('Acurácia (Log Scale)', fontsize=13)
        plt.xlabel('Dataset', fontsize=13)
        plt.ylim(0, 1.15)
        plt.legend(title='Configuração de Prompt', title_fontsize=12, loc='upper right', framealpha=0.9)
        plt.tight_layout()
        plt.show()


def plot_global_average_accuracy(df):
    df = df.copy()

    df['has_context']  = df['context_type'].apply(
        lambda x: 'With Context' if pd.notnull(x) and x != 'null' else 'No Context'
    )
    df['config_label'] = df['prompt_type'] + " (" + df['has_context'] + ")"

    df_balanced = df.groupby(['dataset', 'config_label'])['log_accuracy'].mean().reset_index()

    base_orders    = ['expert_1', 'expert_2', 'expert_3', 'expert_4', 'expert_5', 'expert_6']
    context_orders = ['With Context', 'No Context']
    ordered_labels = [f"{p} ({c})" for c in context_orders for p in base_orders]
    existing_labels = [l for l in ordered_labels if l in df_balanced['config_label'].unique()]

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="white")

    ax = sns.barplot(
        data=df_balanced,
        x='config_label', y='log_accuracy',
        order=existing_labels, palette='viridis',
        capsize=.05, errorbar=('ci', 95)
    )

    plt.title('Balanced Global Accuracy (Mean of Dataset Means)', fontsize=16, fontweight='bold', pad=25)
    plt.ylabel('Balanced Accuracy', fontsize=12)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=8, fontweight='bold')

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_accuracy_by_evidence_per_dataset(df, datasets_per_img=6):
    df = df.copy()
    df['evidence_pct_num'] = 100 * df['evidence_length'] / (df['evidence_length'] + df['evaluated_length'])

    datasets   = df['dataset'].unique()
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    sns.set_theme(style="white")

    for i in range(num_chunks):
        start          = i * datasets_per_img
        chunk_datasets = datasets[start:start + datasets_per_img]

        cols      = 2
        rows      = math.ceil(len(chunk_datasets) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows),
                                 gridspec_kw={'hspace': 0.4, 'wspace': 0.2})
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, ds_name in enumerate(chunk_datasets):
            ax     = axes_flat[idx]
            subset = df[df['dataset'] == ds_name].sort_values('evidence_pct_num')

            sns.lineplot(
                data=subset, x='evidence_pct_num', y='log_accuracy',
                marker='o', markersize=8, linewidth=3, ax=ax,
                color='#1a5276', label='Mean Accuracy' if idx == 0 else ""
            )

            summary = subset.groupby('evidence_pct_num')['log_accuracy'].mean().reset_index()
            ax.fill_between(summary['evidence_pct_num'], summary['log_accuracy'], color='#1a5276', alpha=0.1)

            ax.set_title(f'Dataset: {ds_name}', fontsize=14, fontweight='bold', pad=12)
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
            sns.despine(ax=ax)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_xlabel('')
            ax.set_ylabel('')

        for j in range(idx + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        fig.text(0.04, 0.5, 'Accuracy (Log)', va='center', rotation='vertical', fontsize=15, fontweight='bold')
        fig.text(0.5, 0.05, 'Evidence Proportion (Context / Total Length)', ha='center', fontsize=15, fontweight='bold')
        fig.suptitle(f'Impact of Evidence Length on Performance - Batch {i+1}',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.90)
        plt.show()


def plot_global_accuracy_by_evidence(df):
    df = df.copy()
    df['evidence_pct_num'] = 100 * df['evidence_length'] / (df['evidence_length'] + df['evaluated_length'])

    bins   = np.arange(0, 101, 10)
    labels = [f'{i}-{i+10}%' for i in range(0, 91, 10)]
    df['evidence_range'] = pd.cut(df['evidence_pct_num'], bins=bins, labels=labels, include_lowest=True)

    df_weighted = df.groupby(['dataset', 'evidence_range'], observed=True)['log_accuracy'].mean().reset_index()

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="white")
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    sns.lineplot(
        data=df_weighted, x='evidence_range', y='log_accuracy',
        marker='o', markersize=10, linewidth=3,
        color='#e67e22', errorbar=('ci', 95)
    )

    plt.title('Balanced Trend: Accuracy vs Evidence (Normalized by Dataset)', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Mean of Dataset Accuracies', fontsize=12)
    plt.xlabel('Evidence Intervals', fontsize=12)
    plt.ylim(0, 1.1)
    sns.despine()
    plt.show()


def plot_accuracy_by_confidence_per_dataset(df, datasets_per_img=5):
    rows = []

    for _, row in df.iterrows():
        preds     = row['llm_predictions']
        target    = row['map_assignment']
        conf_dict = row['confidence']
        dataset   = row['dataset']

        for var, pred_val in preds.items():
            if var in target and var in conf_dict:
                conf_level = conf_dict[var][1].capitalize()
                hit        = 1 if pred_val == target[var] else 0
                rows.append({'dataset': dataset, 'confidence': conf_level, 'hit': hit})

    analysis_df = pd.DataFrame(rows)
    datasets    = analysis_df['dataset'].unique()
    conf_order  = ['Low', 'Medium', 'High']
    num_chunks  = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    for i in range(num_chunks):
        chunk_datasets = datasets[i * datasets_per_img:(i + 1) * datasets_per_img]
        subset         = analysis_df[analysis_df['dataset'].isin(chunk_datasets)].copy()
        subset['confidence'] = pd.Categorical(subset['confidence'], categories=conf_order, ordered=True)

        plt.figure(figsize=(16, 8))
        sns.set_theme(style="white")
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        ax = sns.barplot(
            data=subset, x='dataset', y='hit',
            hue='confidence', hue_order=conf_order,
            palette='Blues', edgecolor='0.3', errorbar=None
        )

        for container in ax.containers:
            ax.bar_label(container,
                         labels=[f'{b.get_height():.2f}' for b in container],
                         padding=3, fontsize=10, fontweight='bold')

        plt.title(f'Model Calibration: Accuracy vs. Confidence Level - Batch {i+1}',
                  fontsize=18, fontweight='bold', pad=25)
        plt.ylabel('Accuracy Rate', fontsize=13)
        plt.xlabel('', fontsize=13)
        plt.ylim(0, 1.15)
        plt.legend(title='LLM Confidence', title_fontsize='12', bbox_to_anchor=(1.02, 1), loc='upper left')
        sns.despine()
        plt.tight_layout()
        plt.show()


def plot_global_accuracy_by_confidence(df):
    rows = []

    for _, row in df.iterrows():
        preds     = row['llm_predictions']
        target    = row['map_assignment']
        conf_dict = row['confidence']
        ds_name   = row['dataset']

        for var, pred_val in preds.items():
            if var in target and var in conf_dict:
                conf_level = conf_dict[var][1].capitalize()
                hit        = 1 if pred_val == target[var] else 0
                rows.append({'dataset': ds_name, 'confidence': conf_level, 'hit': hit})

    analysis_df  = pd.DataFrame(rows)
    conf_order   = ['Low', 'Medium', 'High']
    df_balanced  = analysis_df.groupby(['dataset', 'confidence'], observed=True)['hit'].mean().reset_index()

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="white")

    ax = sns.barplot(
        data=df_balanced, x='confidence', y='hit',
        order=conf_order, palette='Blues', edgecolor='black',
        linewidth=1.5, capsize=.05, errorbar=('ci', 95)
    )

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=10, fontsize=12, fontweight='bold')

    plt.title('Global Model Calibration: Accuracy vs. Reported Confidence', fontsize=16, fontweight='bold', pad=25)
    plt.ylabel('Balanced Accuracy (Success Rate)', fontsize=13, labelpad=10)
    plt.xlabel('LLM Confidence Level', fontsize=13, labelpad=10)
    plt.ylim(0, 1.15)
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_calibration_curve(df):
    rows = []
    for _, row in df.iterrows():
        preds, target, conf_dict = row['llm_predictions'], row['map_assignment'], row['confidence']
        conf_map = {'Low': 0.33, 'Medium': 0.66, 'High': 1.0}
        for var, pred_val in preds.items():
            if var in target and var in conf_dict:
                conf_str = conf_dict[var][1].capitalize()
                hit      = 1 if pred_val == target[var] else 0
                rows.append({'conf_val': conf_map[conf_str], 'conf_label': conf_str, 'hit': hit})

    calib_df = pd.DataFrame(rows)
    summary  = calib_df.groupby('conf_label').agg({'hit': 'mean', 'conf_val': 'mean'}).sort_values('conf_val')

    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Calibração Perfeita')
    plt.plot(summary['conf_val'], summary['hit'], marker='s', markersize=10, linewidth=2, label='GPT-5.4-mini')
    plt.title('Reliability Diagram (Calibração do Modelo)', fontsize=14)
    plt.xlabel('Confiança Média Declarada', fontsize=12)
    plt.ylabel('Acurácia Real (Fração de Acertos)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_variable_difficulty_analysis(df, top_n=10):
    rows = []
    for _, row in df.iterrows():
        preds   = row['llm_predictions']
        target  = row['map_assignment']
        dataset = row['dataset']
        for var, pred_val in preds.items():
            if var in target:
                hit = 1 if pred_val == target[var] else 0
                rows.append({'variable': var, 'hit': hit, 'dataset': dataset})

    var_df  = pd.DataFrame(rows)
    var_acc = var_df.groupby('variable')['hit'].mean().sort_values()

    worst_vars = var_acc.head(top_n)
    best_vars  = var_acc.tail(top_n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    sns.set_theme(style="whitegrid")

    sns.barplot(x=worst_vars.values, y=worst_vars.index, ax=ax1, palette="Reds_r")
    ax1.set_title(f'Top {top_n} Variáveis Mais Difíceis (Menor Acurácia)', fontsize=14)
    ax1.set_xlabel('Acurácia Média')

    sns.barplot(x=best_vars.values, y=best_vars.index, ax=ax2, palette="Greens_r")
    ax2.set_title(f'Top {top_n} Variáveis Mais Fáceis (Maior Acurácia)', fontsize=14)
    ax2.set_xlabel('Acurácia Média')

    plt.tight_layout()
    plt.show()


def plot_error_cascade_per_dataset(df, datasets_per_img=6):
    results = []
    for _, row in df.iterrows():
        preds   = row['llm_predictions']
        target  = row['map_assignment']
        dataset = row['dataset']
        for i, (var, pred_val) in enumerate(preds.items()):
            if var in target:
                hit = 1 if pred_val == target[var] else 0
                results.append({'position': i + 1, 'hit': hit, 'dataset': dataset})

    cascade_df = pd.DataFrame(results)
    datasets   = cascade_df['dataset'].unique()
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    sns.set_theme(style="white")

    for i in range(num_chunks):
        start          = i * datasets_per_img
        chunk_datasets = datasets[start:start + datasets_per_img]

        cols      = 2
        rows      = math.ceil(len(chunk_datasets) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows),
                                 gridspec_kw={'hspace': 0.5, 'wspace': 0.15})
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for idx, ds_name in enumerate(chunk_datasets):
            ax     = axes_flat[idx]
            subset = cascade_df[cascade_df['dataset'] == ds_name]

            sns.lineplot(
                data=subset, x='position', y='hit',
                marker='o', markersize=9, linewidth=3, ax=ax,
                color='#2c3e50', errorbar=('ci', 95)
            )

            ax.set_title(f'Dataset: {ds_name}', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylim(-0.05, 1.1)
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticks(range(1, int(subset['position'].max()) + 1))
            ax.yaxis.grid(True, linestyle='--', alpha=0.6)
            sns.despine(ax=ax)
            ax.set_xlabel('')
            ax.set_ylabel('')

        for j in range(idx + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.text(0.04, 0.5, 'Mean Success Rate (Hit)', va='center', rotation='vertical', fontsize=15, fontweight='bold')
        fig.text(0.5, 0.04, 'Variable Order in Prediction Sequence', ha='center', fontsize=15, fontweight='bold')
        fig.suptitle(f'Sequential Error Cascade Analysis (Batch {i+1})',
                     fontsize=20, fontweight='bold', y=0.98)
        plt.subplots_adjust(left=0.08, bottom=0.12, right=0.95, top=0.90)
        plt.show()


def plot_error_cascade_analysis(df):
    results = []
    for _, row in df.iterrows():
        preds   = row['llm_predictions']
        target  = row['map_assignment']
        dataset = row['dataset']
        for i, (var, pred_val) in enumerate(preds.items()):
            if var in target:
                hit = 1 if pred_val == target[var] else 0
                results.append({'position': i + 1, 'hit': hit, 'dataset': dataset})

    cascade_df       = pd.DataFrame(results)
    balanced_cascade = cascade_df.groupby(['dataset', 'position'], observed=True)['hit'].mean().reset_index()

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="white")

    ax = sns.lineplot(
        data=balanced_cascade, x='position', y='hit',
        marker='o', markersize=10, linewidth=4,
        color='#c0392b', errorbar=('ci', 95), err_style='band'
    )

    plt.title('Sequential Error Cascade: Global Performance Trend', fontsize=18, fontweight='bold', pad=25)
    plt.xlabel('Variable Position in Inference Sequence', fontsize=13, labelpad=15)
    plt.ylabel('Balanced Success Rate (Macro-Avg)', fontsize=13, labelpad=15)
    plt.xticks(range(1, int(balanced_cascade['position'].max()) + 1))
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    sns.despine(trim=True)
    plt.axhline(0.5, color='black', linestyle=':', alpha=0.3, label='Random Baseline')
    plt.annotate('Shaded area: 95% CI across datasets', xy=(0.02, 0.05), xycoords='axes fraction', fontsize=10, alpha=0.7)
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


def print_paper_table(
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
    W        = 88

    print("=" * W)
    print(f"  Evidence Length = {evidence_length}")
    print("=" * W)
    print(f"  {'Dataset / Evidence':<32} {'Acc':>7} {'F1-W':>7} {'F1-M':>7} {'Consensus':>10}")
    print("=" * W)

    for ds in datasets:
        ds_data        = df_filtered[df_filtered["dataset"] == ds]
        evidence_groups = sorted(ds_data["evidence_key"].unique())

        all_metrics = []
        for ev_key in evidence_groups:
            rows = ds_data[ds_data["evidence_key"] == ev_key].to_dict("records")
            m    = calc_group_metrics(rows, agent_weights=agent_weights)
            if not m:
                continue
            all_metrics.append(m)

        if not all_metrics:
            continue

        mean = lambda key: sum(m[key] for m in all_metrics) / len(all_metrics)

        print(
            f"  {ds:<32} "
            f"{mean('accuracy')*100:>6.1f}%  "
            f"{mean('f1_weighted'):>6.3f}  "
            f"{mean('f1_macro'):>6.3f}  "
            f"{mean('consensus')*100:>8.1f}%"
        )

        for ev_key, m in zip(evidence_groups, all_metrics):
            print(
                f"    ↳ {ev_key:<30} "
                f"{m['accuracy']*100:>6.1f}%  "
                f"{m['f1_weighted']:>6.3f}  "
                f"{m['f1_macro']:>6.3f}  "
                f"{m['consensus']*100:>8.1f}%"
            )

        print("-" * W)

    print("=" * W)
    print()


def print_all_paper_tables(df: pd.DataFrame, agent_weights: Optional[list[float]] = None):
    for length in sorted(df["evidence_length"].unique()):
        print_paper_table(df, evidence_length=length, agent_weights=agent_weights)


# ─────────────────────────────────────────────
#  ENSEMBLE
# ─────────────────────────────────────────────

def compute_expert_ensemble_accuracy(
    df: pd.DataFrame,
    agent_weights: Optional[list[float]] = None,
) -> pd.DataFrame:
    """
    Para cada (dataset, evidence, prompt_type):
      - Calcula a moda (ou votação ponderada) das predições entre os trials
      - Compara com o map_assignment para obter acurácia do especialista

    Depois, para cada (dataset, evidence):
      - Faz a média das acurácias entre os especialistas (prompt_types)
    """
    df = df.copy()
    df["evidence_key"] = df["evidence"].apply(frozen_evidence)

    records = []
    for (dataset, evidence_key, prompt_type), group in df.groupby(
        ["dataset", "evidence_key", "prompt_type"]
    ):
        rows = group.to_dict("records")
        acc  = calc_group_accuracy(rows, agent_weights=agent_weights)
        records.append({
            "dataset":      dataset,
            "evidence_key": evidence_key,
            "prompt_type":  prompt_type,
            "accuracy":     acc,
            "n_trials":     len(rows),
        })

    per_expert = pd.DataFrame(records)

    ensemble = (
        per_expert
        .groupby(["dataset", "evidence_key"])
        .agg(
            accuracy=("accuracy", "mean"),
            n_experts=("prompt_type", "nunique"),
            n_trials=("n_trials", "sum"),
        )
        .reset_index()
    )

    dataset_summary = (
        ensemble
        .groupby("dataset")
        .agg(
            accuracy=("accuracy", "mean"),
            n_evidence_groups=("evidence_key", "nunique"),
        )
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )

    return dataset_summary, ensemble, per_expert


def plot_expert_ensemble_accuracy(
    df: pd.DataFrame,
    agent_weights: Optional[list[float]] = None,
):
    dataset_summary, ensemble, _ = compute_expert_ensemble_accuracy(df, agent_weights=agent_weights)

    order = dataset_summary.sort_values("accuracy", ascending=False)["dataset"].tolist()
    ensemble["accuracy_pct"]       = ensemble["accuracy"] * 100
    dataset_summary["accuracy_pct"] = dataset_summary["accuracy"] * 100

    n_evidence = ensemble["evidence_key"].nunique() // len(order)

    weight_label = f" | weights={agent_weights}" if agent_weights else ""
    fig, ax      = plt.subplots(figsize=(max(8, len(order) * 1.5), 5))

    sns.barplot(
        data=dataset_summary, x="dataset", y="accuracy_pct",
        order=order, palette="Blues_d", edgecolor="black", linewidth=0.6, ax=ax,
    )

    sns.stripplot(
        data=ensemble, x="dataset", y="accuracy_pct",
        order=order, color="black", size=6, jitter=True, alpha=0.7, ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    ax.set_title(
        f"Expert Ensemble Accuracy by Dataset ({n_evidence} experiments per dataset){weight_label}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


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
plot_grouped_accuracy(df, datasets_per_img=9, agent_weights=AGENT_WEIGHTS)
# plot_dataset_accuracy(df, datasets_per_img=8)
# plot_global_average_accuracy(df)

# plot_accuracy_by_evidence_per_dataset(df, datasets_per_img=5)

# plot_expert_ensemble_accuracy(df, agent_weights=AGENT_WEIGHTS)
# print_paper_table(df, agent_weights=AGENT_WEIGHTS)
# print_evidence_accuracy_table(df, agent_weights=AGENT_WEIGHTS)
# plot_global_accuracy_by_evidence(df)
# plot_accuracy_by_confidence_per_dataset(df, datasets_per_img=5)
# plot_global_accuracy_by_confidence(df)
# plot_calibration_curve(df)
# plot_variable_difficulty_analysis(df, top_n=10)
# plot_error_cascade_per_dataset(df, datasets_per_img=5)
# plot_error_cascade_analysis(df)