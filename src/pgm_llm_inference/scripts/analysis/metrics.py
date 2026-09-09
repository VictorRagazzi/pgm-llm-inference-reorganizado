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
from scipy.stats import spearmanr

from pgm_llm_inference.core.config import InferenceConfig

config = InferenceConfig()

# Paleta consistente para o paper
BAR_COLOR    = "#2C6E9E"
EDGE_COLOR   = "#1A3F5C"
# Tom mais claro da mesma família azul, usado só quando um gráfico precisa
# distinguir duas séries (ex.: accuracy vs joint_match) sem fugir da paleta
# já usada no restante do paper.
ACCENT_COLOR = "#8FB8D9"
ACCENT_EDGE  = "#4A7A9E"
GRID_COLOR   = "#CCCCCC"
# =======================================================
# PARÂMETROS DE TAMANHO DE FONTE
# Altere estes valores para ajustar os tamanhos nos plots
# (ideais para artigos no Overleaf)
# =======================================================
LABEL_FS     = 10
TITLE_FS     = 14
AXIS_FS      = 12
# =======================================================

sns.set_theme(style="whitegrid", rc={
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
})

# ─────────────────────────────────────────────
#  TRADUÇÃO DE NOMES DE DATASETS
#  Chave: nome do arquivo .bif (como aparece nos logs).
#  Valor: nome legível para exibição nos plots.
#  Se um dataset não estiver no dicionário, o nome original é mantido.
# ─────────────────────────────────────────────

DATASET_LABELS: dict[str, str] = {
    "adhd_cbeb.bif":         "TDAH",
    "covid1_cbeb.bif":       "Sintomas de Covid - 1",
    "covid3_cbeb.bif":       "Sintomas de Covid - 2",
    "gonorrhoeae_cbeb.bif":  "Gonorreia",
    "hepar2_cbeb.bif":       "Hepatite",
    "alarm_cbeb.bif":        "Monitoramento de UTI",
    "foodallergy3_cbeb.bif": "Alergia - 2",
    "foodallergy1_cbeb.bif": "Alergia - 1",
    "diabets_cbeb.bif":      "Diabetes",
    "child_cbeb.bif":        "Doenças pediátricas",
}

# Rótulos legíveis para os tipos de amostragem de evidência, usados nos
# títulos/eixos dos gráficos que filtram ou comparam por evidence_sampling.
SAMPLING_LABELS: dict[str, str] = {
    "mpe_consistent":   "Evidência MPE-consistente",
    "mpe_inconsistent": "Evidência MPE-inconsistente",
}


def translate_dataset(name: str) -> str:
    """Retorna o nome legível do dataset se houver mapeamento; caso contrário,
    retorna o nome original sem alteração."""
    return DATASET_LABELS.get(name, name)


def translate_sampling(name: str) -> str:
    """Retorna o nome legível do tipo de amostragem, se houver mapeamento."""
    return SAMPLING_LABELS.get(name, name)


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


def calc_per_variable_hits(rows: list[dict]) -> list[dict]:
    """
    Mesmo grupo de calc_group_metrics (mesmo dataset + mesma evidência
    exata + mesmo sampling/layout), mas retorna um registro por variável
    avaliada em vez de uma métrica agregada: 1 se a predição pós-votação
    bateu com o MPE naquela variável, 0 caso contrário.

    Granularidade de nó, não de grupo — é a base para correlacionar erro
    com o número de pais de cada variável especificamente, em vez de com
    características agregadas da rede inteira.
    """
    rep = rows[0]
    ma  = rep.get("map_assignment", {})
    targets = [k for k in ma if k != "_scalar"]
    if not targets:
        return []

    mode_preds = _mode_predictions(rows)
    dataset = rep.get("dataset")
    sampling = rep.get("evidence_sampling", "mpe_consistent")
    layout = rep.get("evidence_layout", "nested")

    records = []
    for var in targets:
        if var not in mode_preds:
            continue
        records.append({
            "dataset":           dataset,
            "evidence_sampling": sampling,
            "evidence_layout":   layout,
            "variable":          var,
            "correct":           int(mode_preds[var] == ma[var]),
        })
    return records


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

    Como o agrupamento é feito por evidence_sampling, esta tabela já
    funciona igual para um log único (mpe_consistent + mpe_inconsistent
    misturados) ou para logs separados por tipo — basta concatenar os
    DataFrames de load_logs antes de chamar esta função. Os plots abaixo
    filtram por evidence_sampling a partir desta mesma tabela.
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


def _filter_by_sampling(grouped: pd.DataFrame, evidence_sampling: Optional[str]) -> pd.DataFrame:
    """Aplica o filtro por evidence_sampling, se um valor for passado.

    Centralizado aqui porque vários plots precisam do mesmo comportamento:
    None ou "" -> mantém tudo (dados misturados); qualquer outro valor ->
    filtra exatamente esse tipo (ex.: "mpe_consistent", "mpe_inconsistent").
    """
    if not evidence_sampling:
        return grouped
    return grouped[grouped["evidence_sampling"] == evidence_sampling]


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


def plot_accuracy_by_dataset(
    grouped: pd.DataFrame,
    datasets_per_img: int = 10,
    evidence_sampling: Optional[str] = None,
):
    """Acurácia média por dataset.

    Se `evidence_sampling` for informado ("mpe_consistent" ou
    "mpe_inconsistent"), filtra a tabela antes de plotar e adiciona o tipo
    ao título — chame a função duas vezes, uma por tipo, para comparar os
    dois cenários lado a lado. Se omitido, usa todos os dados juntos
    (comportamento antigo).
    """
    subset = _filter_by_sampling(grouped, evidence_sampling).copy()
    subset["accuracy_pct"] = subset["accuracy"] * 100

    datasets   = subset["dataset"].unique()
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    # title_suffix = f" — {translate_sampling(evidence_sampling)}" if evidence_sampling else ""

    for i in range(num_chunks):
        chunk  = datasets[i * datasets_per_img:(i + 1) * datasets_per_img]
        chunk_subset = subset[subset["dataset"].isin(chunk)]
        overall = chunk_subset.groupby("dataset")["accuracy_pct"].mean().reindex(chunk).reset_index()
        overall["dataset"] = overall["dataset"].apply(translate_dataset)

        fig, ax = plt.subplots(figsize=(12, 5))
        _bar_with_labels(ax, overall, "dataset", "accuracy_pct")
        ax.set_title(f"Acurácia dos Experimentos por Dataset", fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.set_ylabel("Acurácia (%)", fontsize=AXIS_FS)
        ax.set_xlabel("Dataset", fontsize=AXIS_FS)
        ax.set_ylim(0, 115)
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.show()


def plot_accuracy_by_evidence_ratio(
    grouped: pd.DataFrame,
    evidence_sampling: Optional[str] = None,
):
    """Acurácia média por ratio de evidência, um subplot por dataset.
    Exibe a evolução usando um gráfico de linha. 
    """
    subset = _filter_by_sampling(grouped, evidence_sampling)
    datasets   = sorted(subset["dataset"].unique())
    n = len(datasets)
    ncols = 5
    nrows = math.ceil(n / ncols) if n > 0 else 1

    suptitle_suffix = f" — {translate_sampling(evidence_sampling)}" if evidence_sampling else ""

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), sharey=True, squeeze=False)

    for j, ds in enumerate(datasets):
        ax = axes[j // ncols][j % ncols]
        ds_data = subset[subset["dataset"] == ds]
        ds_agg = (
            ds_data.groupby("evidence_ratio")["accuracy"]
            .mean().mul(100).reset_index().sort_values("evidence_ratio")
        )
        ratios = ds_agg["evidence_ratio"].tolist()
        values = ds_agg["accuracy"].tolist()

        # Plota usando o valor numérico da razão de evidência no eixo X
        ax.plot(ratios, values, color=BAR_COLOR, marker='o', linewidth=2, markersize=8, zorder=3)
        
        ax.set_title(translate_dataset(ds), fontsize=TITLE_FS, fontweight="bold", pad=2)
        
        # Configuração do eixo X para ser igual em todos (5 a 50)
        ticks = list(range(5, 55, 5))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t}" for t in ticks], fontsize=LABEL_FS)
        ax.set_xlim(0, 55)
        
        ax.set_ylim(0, 115)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)

    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    fig.supxlabel("Proporção de Evidência (%)", fontsize=AXIS_FS, fontweight="bold")
    fig.supylabel("Acurácia Média (%)", fontsize=AXIS_FS, fontweight="bold")
    
    fig.suptitle(f"Acurácia Média x Proporção de Evidência ", fontsize=TITLE_FS + 2, fontweight="bold", y=0.96)
    plt.tight_layout()
    plt.show()


def plot_accuracy_by_sampling(grouped: pd.DataFrame):
    """
    Acurácia por variável e taxa de match conjunto (full MPE), comparando
    os modos de amostragem de evidência (mpe_consistent vs mpe_inconsistent)
    e layout (nested vs independent), em dois subplots separados.
    """
    grouped = grouped.copy()
    grouped["sampling_label"] = grouped["evidence_sampling"] + " / " + grouped["evidence_layout"]

    agg = (
        grouped.groupby("sampling_label")[["accuracy", "joint_match"]]
        .mean().mul(100).reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    _bar_with_labels(axes[0], agg, "sampling_label", "accuracy")
    axes[0].set_title("Acurácia por Variável por Amostragem de Evidência", fontsize=TITLE_FS, fontweight="bold")
    axes[0].set_ylabel("Acurácia (%)", fontsize=AXIS_FS)
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 115)

    _bar_with_labels(axes[1], agg, "sampling_label", "joint_match")
    axes[1].set_title("Taxa de Acerto Conjunto do MPE Completo por Amostragem", fontsize=TITLE_FS, fontweight="bold")
    axes[1].set_ylabel("Taxa de Acerto Conjunto (%)", fontsize=AXIS_FS)
    axes[1].set_xlabel("")
    axes[1].set_ylim(0, 115)

    plt.tight_layout()
    plt.show()


def plot_sampling_comparison(grouped: pd.DataFrame, layout: Optional[str] = "nested"):
    """
    NOVO: gráfico único com barras agrupadas comparando accuracy e
    joint_match lado a lado, para mpe_consistent vs mpe_inconsistent — a
    mesma visão condensada usada na conversa com o orientador/revisor para
    mostrar de forma direta o gap entre acurácia por variável e recuperação
    do MPE completo, e a queda de desempenho quando a evidência não segue
    o MPE exato.

    Usa a mesma paleta do resto do paper: BAR_COLOR para accuracy e um tom
    mais claro da mesma família (ACCENT_COLOR) para joint_match, em vez de
    introduzir cores novas.

    `layout`: filtra por evidence_layout antes de agregar (default "nested",
    já que é o único layout usado até agora); passe None para não filtrar.
    """
    subset = grouped if layout is None else grouped[grouped["evidence_layout"] == layout]

    agg = (
        subset.groupby("evidence_sampling")[["accuracy", "joint_match"]]
        .mean().mul(100)
    )

    # Ordem fixa (consistent antes de inconsistent) independente da ordem
    # de aparição nos dados.
    order = [s for s in ["mpe_consistent", "mpe_inconsistent"] if s in agg.index]
    agg = agg.reindex(order)

    x_labels = [translate_sampling(s) for s in agg.index]
    x = list(range(len(agg)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    bars_acc = ax.bar(
        [i - width / 2 for i in x], agg["accuracy"], width,
        label="Acurácia por variável",
        color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, zorder=3,
    )
    bars_joint = ax.bar(
        [i + width / 2 for i in x], agg["joint_match"], width,
        label="Exact match (configuração conjunta)",
        color=ACCENT_COLOR, edgecolor=ACCENT_EDGE, linewidth=0.6, zorder=3,
    )

    for bars in (bars_acc, bars_joint):
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=LABEL_FS)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=AXIS_FS)
    ax.set_ylabel("%", fontsize=AXIS_FS)
    ax.set_ylim(0, 115)
    ax.set_title("Impacto do tipo de evidência no desempenho", fontsize=TITLE_FS, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LABEL_FS, frameon=False, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_joint_match_by_dataset(
    grouped: pd.DataFrame,
    datasets_per_img: int = 9,
    evidence_sampling: Optional[str] = None,
):
    """Taxa de recuperação exata do MPE completo, por dataset.

    Mesmo filtro opcional `evidence_sampling` das demais funções por
    dataset, para comparar os dois tipos de amostragem chamando a função
    duas vezes.
    """
    subset = _filter_by_sampling(grouped, evidence_sampling).copy()
    subset["joint_match_pct"] = subset["joint_match"] * 100

    datasets   = sorted(subset["dataset"].unique())
    num_chunks = (len(datasets) + datasets_per_img - 1) // datasets_per_img

    title_suffix = f" — {translate_sampling(evidence_sampling)}" if evidence_sampling else ""

    for i in range(num_chunks):
        chunk  = datasets[i * datasets_per_img:(i + 1) * datasets_per_img]
        chunk_subset = subset[subset["dataset"].isin(chunk)]
        overall = chunk_subset.groupby("dataset")["joint_match_pct"].mean().reindex(chunk).reset_index()
        overall["dataset"] = overall["dataset"].apply(translate_dataset)

        fig, ax = plt.subplots(figsize=(12, 5))
        _bar_with_labels(ax, overall, "dataset", "joint_match_pct")
        ax.set_title(f"Taxa de Acerto Exato do MPE Completo por Dataset{title_suffix}", fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.set_ylabel("Taxa de Acerto Conjunto (%)", fontsize=AXIS_FS)
        ax.set_xlabel("Dataset", fontsize=AXIS_FS)
        ax.set_ylim(0, 115)
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.show()


def plot_accuracy_vs_joint_match(grouped: pd.DataFrame):
    """
    Dispersão acurácia por variável x taxa de match conjunto, por grupo
    (dataset, evidência) — evidencia visualmente o gap entre as duas
    métricas (alta acurácia por variável não implica MPE completo).
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
    ax.set_xlabel("Acurácia por Variável", fontsize=AXIS_FS)
    ax.set_ylabel("Acerto Conjunto do MPE Completo", fontsize=AXIS_FS)
    ax.set_title("Acurácia por Variável vs. Recuperação do MPE Completo", fontsize=TITLE_FS, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  ANÁLISE ESTRUTURAL
#  Relaciona o erro do método (accuracy, joint_match, exact_match) com
#  características estruturais da rede (nodes, edges, max_degree, depth),
#  vindas de get_structure_table (get_table.py). Adicionado em resposta
#  ao pedido do revisor do CBEB por uma análise quantitativa que isole
#  os fatores estruturais, em vez de conclusões só a partir de inspeção
#  de poucas redes.
# ─────────────────────────────────────────────

# Métricas estruturais consideradas. Posição topológica da evidência e uma
# métrica de "ambiguidade semântica" ficaram de fora por decisão de projeto
# (a primeira não é variada nos experimentos; a segunda não tem uma forma
# confiável de cálculo disponível no momento).
STRUCTURAL_COLUMNS = ["nodes", "edges", "max_degree", "depth",
                      "mean_cardinality", "max_cardinality"]   # [NOVO]

ERROR_COLUMNS = ["accuracy", "joint_match", "exact_match"]


def _normalize_dataset_name(name: str) -> str:
    """
    Remove o sufixo '.bif' do nome do dataset, se houver. get_structure_table
    retorna o nome do arquivo .bif original, mas nos logs (dependendo de como
    foram gerados) o dataset pode aparecer sem esse sufixo — normalizamos os
    dois lados antes do merge para não perder linhas silenciosamente.
    """
    if isinstance(name, str) and name.endswith(".bif"):
        return name[:-len(".bif")]
    return name


def build_structural_merge(grouped: pd.DataFrame, structure_table: list[dict]) -> pd.DataFrame:
    """
    Agrega compute_grouped_table por dataset (média de accuracy, joint_match
    e exact_match, ignorando evidence_sampling/layout/ratio) e faz merge com
    a tabela estrutural retornada por get_structure_table (nodes, edges,
    max_degree, depth), casando pelo nome do dataset já normalizado.

    `structure_table` é a lista de dicts que get_structure_table(datasets)
    retorna — passe o resultado pronto dessa chamada.
    """
    per_dataset = (
        grouped.groupby("dataset")[ERROR_COLUMNS]
        .mean()
        .reset_index()
    )
    per_dataset["dataset_norm"] = per_dataset["dataset"].apply(_normalize_dataset_name)

    struct_df = pd.DataFrame(structure_table)
    struct_df["dataset_norm"] = struct_df["dataset"].apply(_normalize_dataset_name)

    merged = per_dataset.merge(
        struct_df[["dataset_norm", "label", *STRUCTURAL_COLUMNS]],
        on="dataset_norm",
        how="inner",
    )

    # Aviso se algum dataset dos logs não casou com a tabela estrutural —
    # melhor avisar do que descartar silenciosamente na análise do paper.
    logged_names = set(per_dataset["dataset_norm"])
    merged_names = set(merged["dataset_norm"])
    missing = logged_names - merged_names
    if missing:
        print(f"[AVISO] {len(missing)} dataset(s) dos logs não encontrados na tabela estrutural: {sorted(missing)}")

    return merged.drop(columns=["dataset_norm"])


def compute_structural_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Correlação de Spearman (coeficiente + p-valor) entre cada métrica de
    erro (accuracy, joint_match, exact_match) e cada métrica estrutural
    (nodes, edges, max_degree, depth), uma linha por par.

    Spearman em vez de Pearson: com amostra pequena (8–11 redes) e sem
    garantia de relação linear, Spearman é mais robusto (só assume relação
    monotônica e é menos sensível a outliers/redes atípicas).
    """
    records = []
    for error_col in ERROR_COLUMNS:
        if error_col not in merged.columns:
            continue
        for struct_col in STRUCTURAL_COLUMNS:
            sub = merged[[error_col, struct_col]].dropna()
            if len(sub) < 3:
                # Amostra pequena demais até para uma correlação bem definida
                rho, pval = float("nan"), float("nan")
            else:
                rho, pval = spearmanr(sub[error_col], sub[struct_col])
            records.append({
                "error_metric":      error_col,
                "structural_metric": struct_col,
                "n":                 len(sub),
                "rho":               rho,
                "p_value":           pval,
            })
    return pd.DataFrame(records)


def print_structural_correlation_table(corr_df: pd.DataFrame):
    """Tabela de correlações de Spearman erro × estrutura, no estilo de print_results_table."""
    W = 72
    print("=" * W)
    print(f"{'CORRELAÇÃO (SPEARMAN): ERRO DO MÉTODO × ESTRUTURA DA REDE':^{W}}")
    print("=" * W)
    print(f"{'Erro':<14} {'Estrutural':<12} {'n':>4} {'rho':>9} {'p-valor':>10}  ")
    print("-" * W)

    for _, row in corr_df.iterrows():
        if math.isnan(row["p_value"]):
            rho_str, p_str, sig = "   N/A", "   N/A", ""
        else:
            rho_str = f"{row['rho']:>9.3f}"
            p_str   = f"{row['p_value']:>10.3f}"
            sig     = " *" if row["p_value"] < 0.05 else ""
        print(f"{row['error_metric']:<14} {row['structural_metric']:<12} {row['n']:>4} {rho_str} {p_str}{sig}")

    print("-" * W)
    print("* p < 0.05  |  n < 3 -> correlação não computada (N/A)")
    print("=" * W)
    print()


def plot_error_vs_structure(merged: pd.DataFrame, error_metric: str = "accuracy"):
    """
    Dispersão de error_metric (accuracy, joint_match ou exact_match) vs.
    cada métrica estrutural (nodes, edges, max_degree, depth), um subplot
    por métrica estrutural — inclui rho/p de Spearman no título de cada
    subplot.
    """
    error_metric_pt = {
        "accuracy": "Acurácia",
        "joint_match": "Acerto Conjunto",
        "exact_match": "Acerto Exato"
    }.get(error_metric, error_metric)

    fig, axes = plt.subplots(1, len(STRUCTURAL_COLUMNS), figsize=(5 * len(STRUCTURAL_COLUMNS), 4.5), sharey=True)

    for ax, struct_col in zip(axes, STRUCTURAL_COLUMNS):
        sub = merged[[struct_col, error_metric]].dropna()

        ax.scatter(sub[struct_col], sub[error_metric], color=BAR_COLOR, edgecolor=EDGE_COLOR, s=60, zorder=3)

        if len(sub) >= 3:
            rho, pval = spearmanr(sub[struct_col], sub[error_metric])
            ax.set_title(f"{struct_col}\n(ρ={rho:.2f}, p={pval:.3f})", fontsize=TITLE_FS, fontweight="bold")
        else:
            ax.set_title(struct_col, fontsize=TITLE_FS, fontweight="bold")

        ax.set_xlabel(struct_col, fontsize=AXIS_FS)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", labelsize=LABEL_FS)

    axes[0].set_ylabel(error_metric_pt, fontsize=AXIS_FS)
    fig.suptitle(f"{error_metric_pt} vs. Características Estruturais da Rede", fontsize=TITLE_FS + 1, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  ANÁLISE POR VARIÁVEL (NÚMERO DE PAIS)
#  Complementa a análise estrutural em nível de rede: em vez de agregar
#  accuracy/joint_match por rede (n = nº de redes, hoje 10), aqui cada
#  variável avaliada é uma observação (n = soma de nós avaliados em todos
#  os experimentos), correlacionada com o nº de pais daquela variável
#  específica. Resolve a crítica de baixo poder estatístico sem precisar
#  rodar mais redes. Requer get_variable_structure_table (get_table.py).
# ─────────────────────────────────────────────

VARIABLE_STRUCTURAL_COLUMNS = ["n_parents", "cardinality"]    # [NOVO]


def build_variable_level_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uma linha por variável avaliada em cada (dataset, evidência exata,
    sampling, layout) — construída sobre os mesmos grupos de
    compute_grouped_table, mas sem agregar accuracy por grupo.
    """
    groups = _group_rows(df)
    records = []
    for _, rows in groups.items():
        records.extend(calc_per_variable_hits(rows))
    return pd.DataFrame(records)


def build_variable_structural_merge(
    variable_table: pd.DataFrame,
    variable_structure_table: list[dict],
    evidence_sampling: Optional[str] = None,
) -> pd.DataFrame:
    """
    [MODIFICADO] Agora também traz 'cardinality' por variável além de
    'n_parents'.  O resto do comportamento é idêntico à versão original.
    """
    subset = _filter_by_sampling(variable_table, evidence_sampling)

    per_var = (
        subset.groupby(["dataset", "variable"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    per_var["dataset_norm"] = per_var["dataset"].apply(_normalize_dataset_name)

    struct_df = pd.DataFrame(variable_structure_table)
    struct_df["dataset_norm"] = struct_df["dataset"].apply(_normalize_dataset_name)

    # [NOVO] cardinality adicionado às colunas puxadas do struct_df.
    # Se get_variable_structure_table ainda não retornar a coluna, o merge
    # vai falhar aqui com KeyError — melhor falhar cedo do que silenciosamente.
    var_struct_cols = ["dataset_norm", "variable"] + VARIABLE_STRUCTURAL_COLUMNS
    available_cols = [c for c in var_struct_cols if c in struct_df.columns]

    merged = per_var.merge(
        struct_df[available_cols],
        on=["dataset_norm", "variable"],
        how="inner",
    )

    # [NOVO] Avisa se cardinality não veio da tabela estrutural, para o
    # usuário saber que precisa atualizar get_variable_structure_table.
    for col in VARIABLE_STRUCTURAL_COLUMNS:
        if col not in merged.columns:
            print(f"[AVISO] Coluna '{col}' ausente em variable_structure_table — "
                  f"adicione o cálculo em get_variable_structure_table().")

    logged = set(zip(per_var["dataset_norm"], per_var["variable"]))
    matched = set(zip(merged["dataset_norm"], merged["variable"]))
    missing = logged - matched
    if missing:
        n = len(missing)
        sample = sorted(missing)[:10]
        print(f"[AVISO] {n} variável(is) dos logs não encontradas na tabela "
              f"estrutural: {sample}{' ...' if n > 10 else ''}")

    return merged.drop(columns=["dataset_norm"])

def compute_variable_structural_correlations(
    merged: pd.DataFrame,
    evidence_sampling: Optional[str] = None,
) -> pd.DataFrame:
    """
    [NOVO] Spearman entre accuracy por variável e cada métrica estrutural
    de nível de variável presente em VARIABLE_STRUCTURAL_COLUMNS
    (n_parents e cardinality).

    Análogo a compute_structural_correlations, mas operando sobre a
    granularidade de nó, onde n é muito maior do que o número de redes.
    """
    records = []
    for col in VARIABLE_STRUCTURAL_COLUMNS:
        if col not in merged.columns:
            continue
        sub = merged[["accuracy", col]].dropna()
        if len(sub) < 3:
            rho, pval = float("nan"), float("nan")
        else:
            rho, pval = spearmanr(sub["accuracy"], sub[col])
        records.append({
            "structural_metric": col,
            "n":                 len(sub),
            "rho":               rho,
            "p_value":           pval,
        })
    return pd.DataFrame(records)

def print_variable_structural_correlations(
    corr_df: pd.DataFrame,
    evidence_sampling: Optional[str] = None,
):
    """[NOVO] Tabela de Spearman accuracy × métricas estruturais por variável."""
    label = (f" ({translate_sampling(evidence_sampling)})"
             if evidence_sampling else " (todos os tipos de evidência)")
    W = 60
    print("=" * W)
    print(f"{'ACURÁCIA POR VARIÁVEL × ESTRUTURA' + label:^{W}}")
    print("=" * W)
    print(f"{'Métrica estrutural':<20} {'n':>5} {'rho':>9} {'p-valor':>10}")
    print("-" * W)
    for _, row in corr_df.iterrows():
        if math.isnan(row["p_value"]):
            rho_str, p_str, sig = "   N/A", "      N/A", ""
        else:
            rho_str = f"{row['rho']:>9.3f}"
            p_str   = f"{row['p_value']:>10.3f}"
            sig     = " *" if row["p_value"] < 0.05 else ""
        print(f"{row['structural_metric']:<20} {row['n']:>5} {rho_str} {p_str}{sig}")
    print("-" * W)
    print("* p < 0.05  |  n < 3 -> N/A")
    print("=" * W)
    print()

def plot_accuracy_by_cardinality(
    merged: pd.DataFrame,
    min_n: int = 4,
    show_outliers: bool = False,
):
    """
    [NOVO] Boxplot de accuracy por cardinalidade da variável (número de
    estados possíveis).  Espelho direto de plot_accuracy_by_n_parents —
    mesma lógica de filtragem por min_n e mesma paleta.

    Requer que 'cardinality' esteja em `merged`; se não estiver (porque
    get_variable_structure_table ainda não a retorna), imprime um aviso e
    retorna sem plotar.
    """
    if "cardinality" not in merged.columns:
        print("[AVISO] plot_accuracy_by_cardinality: coluna 'cardinality' "
              "ausente — adicione o cálculo em get_variable_structure_table().")
        return

    counts = merged["cardinality"].value_counts()
    valid   = sorted(c for c, n in counts.items() if n >= min_n)
    dropped = sorted(c for c, n in counts.items() if n < min_n)

    if not valid:
        print("[AVISO] plot_accuracy_by_cardinality: nenhuma categoria com "
              f"n ≥ {min_n}. Reduza min_n ou verifique os dados.")
        return

    subset = merged[merged["cardinality"].isin(valid)]

    rho_result = compute_variable_structural_correlations(
        merged[["accuracy", "cardinality"]].rename(
            columns={"cardinality": "cardinality"}   # mantém nome
        ).assign(n_parents=merged.get("n_parents"))  # evita erro se n_parents ausente
        if "n_parents" not in merged.columns
        else merged
    )
    # Recupera só a linha de cardinality para o subtítulo
    rho_row = (
        rho_result[rho_result["structural_metric"] == "cardinality"]
        .iloc[0] if not rho_result.empty else None
    )
    rho_label = ""
    if rho_row is not None and not math.isnan(rho_row["p_value"]):
        sig = " *" if rho_row["p_value"] < 0.05 else ""
        rho_label = f"  (ρ={rho_row['rho']:.3f}, p={rho_row['p_value']:.3f}{sig})"

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=subset, x="cardinality", y="accuracy", order=valid,
        color=BAR_COLOR, ax=ax, showfliers=show_outliers,
        boxprops=dict(edgecolor=EDGE_COLOR),
        medianprops=dict(color=EDGE_COLOR, linewidth=1.5),
        whiskerprops=dict(color=EDGE_COLOR),
        capprops=dict(color=EDGE_COLOR),
        flierprops=dict(markerfacecolor=ACCENT_COLOR,
                        markeredgecolor=ACCENT_EDGE, markersize=5),
    )

    ax.set_xticklabels(
        [f"{c}\n(n={counts.get(c, 0)})" for c in valid],
        fontsize=LABEL_FS,
    )
    ax.set_xlabel("Cardinalidade da variável (nº de estados)", fontsize=AXIS_FS)
    ax.set_ylabel("Acurácia por variável", fontsize=AXIS_FS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Acurácia por Variável vs. Cardinalidade{rho_label}",
        fontsize=TITLE_FS, fontweight="bold", pad=10,
    )
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

def compute_parents_correlation(merged: pd.DataFrame) -> dict:
    """
    Spearman entre accuracy por variável e nº de pais daquela variável.
    Retorna também n (nº de variáveis avaliadas) para deixar explícito o
    ganho de poder estatístico em relação à correlação em nível de rede.
    """
    sub = merged[["accuracy", "n_parents"]].dropna()
    if len(sub) < 3:
        return {"n": len(sub), "rho": float("nan"), "p_value": float("nan")}
    rho, pval = spearmanr(sub["accuracy"], sub["n_parents"])
    return {"n": len(sub), "rho": rho, "p_value": pval}


def print_parents_correlation(result: dict, evidence_sampling: Optional[str] = None):
    label = f" ({translate_sampling(evidence_sampling)})" if evidence_sampling else " (todos os tipos de evidência)"
    W = 60
    print("=" * W)
    print(f"{'ACURÁCIA POR VARIÁVEL × Nº DE PAIS' + label:^{W}}")
    print("=" * W)
    if math.isnan(result["p_value"]):
        print(f"n={result['n']} — amostra insuficiente para correlação")
    else:
        sig = " *" if result["p_value"] < 0.05 else ""
        print(f"n={result['n']}  rho={result['rho']:.3f}  p={result['p_value']:.3f}{sig}")
    print("=" * W)
    print()


def plot_accuracy_by_n_parents(merged: pd.DataFrame, min_n: int = 4, show_outliers: bool = False):
    """
    Boxplot de accuracy por nº de pais (variável discreta e geralmente com
    poucos valores distintos — 0, 1, 2, 3... — então boxplot por categoria
    é mais legível que um scatter puro). Mesma paleta do resto do paper.
 
    `min_n`: categorias de n_parents com menos observações que isso são
    descartadas do gráfico (default 4). Com n=1 o "boxplot" é uma única
    observação virando uma linha reta, e com n=3 é uma caixa de 3 pontos —
    nenhum dos dois representa uma distribuição de verdade, então melhor
    omitir e declarar isso (a nota abaixo do gráfico) do que sugerir uma
    variância que os dados não sustentam.
 
    `show_outliers`: desenha ou não os pontos individuais fora do whisker.
    False por default — com os whiskers das categorias mais numerosas já
    descendo perto de 0, os pontos extras acrescentam pouca informação
    nova e poluem o gráfico.
    """
    counts = merged["n_parents"].value_counts()
    valid   = sorted(n for n, c in counts.items() if c >= min_n)
    dropped = sorted(n for n, c in counts.items() if c < min_n)
 
    subset = merged[merged["n_parents"].isin(valid)]
 
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(
        data=subset, x="n_parents", y="accuracy", order=valid,
        color=BAR_COLOR, ax=ax, showfliers=show_outliers,
        boxprops=dict(edgecolor=EDGE_COLOR),
        medianprops=dict(color=EDGE_COLOR, linewidth=1.5),
        whiskerprops=dict(color=EDGE_COLOR),
        capprops=dict(color=EDGE_COLOR),
        flierprops=dict(markerfacecolor=ACCENT_COLOR, markeredgecolor=ACCENT_EDGE, markersize=5),
    )
 
    ax.set_xticklabels([f"{p}\n(n={counts.get(p, 0)})" for p in valid], fontsize=LABEL_FS)
    ax.set_xlabel("Número de pais da variável", fontsize=AXIS_FS)
    ax.set_ylabel("Acurácia por variável", fontsize=AXIS_FS)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Acurácia por Variável vs. Número de Pais", fontsize=TITLE_FS, fontweight="bold", pad=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
 
    # if dropped:
    #     note = f"Categorias com n < {min_n} omitidas: " + ", ".join(f"{d} pais (n={counts[d]})" for d in dropped)
    #     ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center", fontsize=LABEL_FS - 1, color="#555555")
 
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


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    LOG_FILE = config.log_file_name
    df = load_logs(LOG_FILE)
    grouped = compute_grouped_table(df)

    # plot_accuracy_by_dataset(grouped, datasets_per_img=11)
    plot_accuracy_by_evidence_ratio(grouped, evidence_sampling="mpe_consistent")
    # plot_accuracy_by_sampling(grouped)
    # plot_joint_match_by_dataset(grouped, datasets_per_img=9)
    # plot_accuracy_vs_joint_match(grouped)

    # Comparação lado a lado dos dois tipos de amostragem de evidência,
    # chamando a mesma função de plot duas vezes com evidence_sampling
    # diferente (funciona com um único log já mesclado, já que o filtro é
    # aplicado sobre a tabela agrupada, não sobre os arquivos):
    plot_accuracy_by_dataset(grouped, evidence_sampling="mpe_consistent")
    plot_accuracy_by_dataset(grouped, evidence_sampling="mpe_inconsistent")
    plot_joint_match_by_dataset(grouped, evidence_sampling="mpe_consistent")
    plot_joint_match_by_dataset(grouped, evidence_sampling="mpe_inconsistent")

    # Gráfico único comparando accuracy e joint_match entre os dois tipos:
    plot_sampling_comparison(grouped)

    print_results_table(grouped, evidence_length=1)
    print_results_table(grouped)  # todas as evidence_length juntas

    # Análise estrutural em nível de rede (erro médio da rede x nodes/edges/max_degree/depth)
    from pgm_llm_inference.scripts.analysis.get_table import get_structure_table, get_variable_structure_table
    structure_table = get_structure_table(df["dataset"].unique().tolist())
    merged = build_structural_merge(grouped, structure_table)
    plot_error_vs_structure(merged, error_metric="accuracy")
    plot_error_vs_structure(merged, error_metric="joint_match")

    # Análise em nível de variável (erro por nó x nº de pais daquele nó) —
    # n bem maior que o nº de redes, resolve o problema de amostra pequena
    # especificamente para essa correlação.
    variable_structure_table = get_variable_structure_table(df["dataset"].unique().tolist())
    variable_table = build_variable_level_table(df)

    # Todos os tipos de evidência juntos:
    var_merged = build_variable_structural_merge(variable_table, variable_structure_table)
    parents_corr = compute_parents_correlation(var_merged)
    print_parents_correlation(parents_corr)


    # Correlações unificadas (n_parents + cardinality numa tabela só)
    corr_var = compute_variable_structural_correlations(var_merged)
    print_variable_structural_correlations(corr_var)

    # Plots individuais
    plot_accuracy_by_n_parents(var_merged)
    plot_accuracy_by_cardinality(var_merged)        # [NOVO]