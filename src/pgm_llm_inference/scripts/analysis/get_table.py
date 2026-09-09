"""
get_table.py
============
Calcula estatísticas estruturais simples para uma lista de datasets:
Markov Blanket médio, número de raízes (sem pais) e folhas (sem filhos).

Uso:
    python -m pgm_llm_inference.scripts.get_table
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from scipy.stats import spearmanr

from pgm_llm_inference.io.loaders import load_network

# Reaproveita a função de extração de grafo já implementada


DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"

# Nome de exibição (domínio) para cada arquivo .bif
DATASET_LABELS = {
    "gonorrhoeae_cbeb.bif": "Gonorreia",
    "adhd_cbeb.bif": "TDAH",
    "diabets_cbeb.bif": "Diabetes",
    "alarm_cbeb.bif": "Monitoramento de UTI",
    "child_cbeb.bif": "Doenças pediátricas",
    "hepar2_cbeb.bif": "Hepatite",
}

def compute_max_degree(
    variables: List[str],
    parents: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
) -> int:
    """Grau máximo = maior (n_pais + n_filhos) entre todos os nós."""
    return max(len(parents[v]) for v in variables)

def compute_depth(variables: List[str], children: Dict[str, Set[str]]) -> int:
    """
    Profundidade = maior caminho dirigido no DAG, em número de arestas
    (raízes têm profundidade 0 em relação a si mesmas).
    """
    memo: Dict[str, int] = {}

    def longest_path(node: str) -> int:
        if node in memo:
            return memo[node]
        if not children[node]:
            memo[node] = 0
            return 0
        memo[node] = 1 + max(longest_path(c) for c in children[node])
        return memo[node]

    return max(longest_path(v) for v in variables)


def compute_node_depths(
    variables: List[str],
    parents: Dict[str, Set[str]],
    children: Dict[str, Set[str]],
) -> Dict[str, int]:
    """
    Profundidade de cada nó = comprimento do maior caminho dirigido (em
    arestas) desde alguma raiz (nó sem pais) até o nó. Raízes têm
    profundidade 0.
 
    Usa ordenação topológica (Kahn) para computar via programação
    dinâmica ascendente; levanta ValueError se o grafo não for um DAG
    válido (não deveria acontecer com redes bayesianas bem formadas).
    """
    in_degree = {v: len(parents[v]) for v in variables}
    queue = [v for v in variables if in_degree[v] == 0]
    order: List[str] = []
 
    while queue:
        node = queue.pop(0)
        order.append(node)
        for c in children[node]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
 
    if len(order) != len(variables):
        raise ValueError(
            "Grafo não é um DAG válido (ciclo detectado) ou há variáveis "
            "desconectadas do algoritmo de ordenação topológica."
        )
 
    depths: Dict[str, int] = {}
    for node in order:
        if not parents[node]:
            depths[node] = 0
        else:
            depths[node] = 1 + max(depths[p] for p in parents[node])
    return depths
 
 
def get_node_depth_table(
    datasets_to_run=None,
    datasets_dir=None,
) -> List[dict]:
    """
    Opcional: tabela {dataset, variable, depth, max_depth, depth_normalized}
    para todas as variáveis de uma lista de datasets, calculada
    diretamente a partir dos arquivos .bif — útil para conferir offline os
    valores de node_depth_normalized já gravados no log pelo experimento
    de posição da evidência, ou para recalculá-los caso o log não tenha
    esses campos.
 
    datasets_dir: se None, usa a constante DATASETS_DIR já definida no
    arquivo.
    """
    if datasets_dir is None:
        datasets_dir = DATASETS_DIR  # definida no topo de get_table.py
 
    results = []
    for dataset_name in datasets_to_run:
        dataset_path = datasets_dir / dataset_name
        try:
            network, _ = load_network(str(dataset_path))
            parents, children, _ = extract_graph_structure(network)
            node_depths = compute_node_depths(list(network.variables), parents, children)
            max_depth = max(node_depths.values()) if node_depths else 0
        except Exception as e:
            print(f"{dataset_name} -> ERRO: {e}")
            continue
 
        label = DATASET_LABELS.get(dataset_name, dataset_name)
        for var in network.variables:
            d = node_depths[var]
            results.append({
                "dataset": dataset_name,
                "label": label,
                "variable": var,
                "depth": d,
                "max_depth": max_depth,
                # None em redes sem hierarquia (max_depth == 0), em vez de
                # dividir por zero.
                "depth_normalized": (d / max_depth) if max_depth > 0 else None,
            })
    return results
 

def _get_cardinality(var) -> int:
    """
    Retorna o número de estados (cardinalidade) de uma variável do escopo
    do fator. Tenta os atributos mais comuns; levanta AttributeError com
    mensagem descritiva se nenhum funcionar — assim fica fácil de corrigir.
    """
    for attr in ("values", "states", "cardinality", "domain", "k"):
        val = getattr(var, attr, None)
        if val is not None:
            # cardinality pode ser int direto ou uma sequência de estados
            return int(val) if isinstance(val, (int, float)) else len(val)
    raise AttributeError(
        f"Não foi possível determinar a cardinalidade de '{var.name}'. "
        f"Atributos disponíveis: {[a for a in dir(var) if not a.startswith('_')]}"
    )


# ─────────────────────────────────────────────
#  MODIFICADO: extract_graph_structure
#  Agora retorna também um dict {var_name -> cardinalidade}.
# ─────────────────────────────────────────────

def extract_graph_structure(network):
    """
    [MODIFICADO] Retorna (parents, children, cardinalities).

    cardinalities: dict {nome_da_variável -> int} com o número de estados
    de cada variável. Lido diretamente do escopo dos fatores, sem
    dependência de APIs externas.
    """
    parents       = {var: set() for var in network.variables}
    children      = {var: set() for var in network.variables}
    cardinalities = {}          # [NOVO]

    seen_children = set()

    for factor in network.factors:
        scope = factor.scope
        child_var  = scope[0]
        child_name = child_var.name

        if child_name in seen_children:
            raise ValueError(f"Multiple factors for variable '{child_name}'")
        seen_children.add(child_name)

        # [NOVO] Cardinalidade da variável-filho (scope[0])
        cardinalities[child_name] = _get_cardinality(child_var)

        parent_names = [v.name for v in scope[1:]]
        for p in parent_names:
            parents[child_name].add(p)
            children[p].add(child_name)

    if len(seen_children) != len(network.variables):
        missing = set(network.variables) - seen_children
        raise ValueError(f"Missing factors for variables: {missing}")

    return parents, children, cardinalities   # [NOVO] terceiro valor


# ─────────────────────────────────────────────
#  MODIFICADO: compute_structure_metrics
#  Adiciona mean_cardinality e max_cardinality ao retorno.
# ─────────────────────────────────────────────

def compute_structure_metrics(network):
    """
    [MODIFICADO] Retorna (n_nodes, n_edges, max_degree, depth,
    mean_cardinality, max_cardinality).
    """
    parents, children, cardinalities = extract_graph_structure(network)  # [NOVO]

    n_nodes    = len(network.variables)
    n_edges    = sum(len(p) for p in parents.values())
    max_degree = compute_max_degree(network.variables, parents, children)
    depth      = compute_depth(network.variables, children)

    # [NOVO] ──────────────────────────────────────────────────────────────
    card_vals        = list(cardinalities.values())
    mean_cardinality = sum(card_vals) / len(card_vals) if card_vals else 0.0
    max_cardinality  = max(card_vals) if card_vals else 0
    # ─────────────────────────────────────────────────────────────────────

    return n_nodes, n_edges, max_degree, depth, mean_cardinality, max_cardinality


# ─────────────────────────────────────────────
#  MODIFICADO: get_structure_table
# ─────────────────────────────────────────────

def get_structure_table(
    datasets_to_run=None,
    datasets_dir: Path = DATASETS_DIR,
) -> List[dict]:
    results = []

    for idx, dataset_name in enumerate(datasets_to_run, 1):
        dataset_path = datasets_dir / dataset_name

        try:
            network, _ = load_network(str(dataset_path))
            # [NOVO] desempacota os dois novos valores
            n_nodes, n_edges, max_degree, depth, mean_card, max_card = \
                compute_structure_metrics(network)
        except Exception as e:
            print(f"[{idx}] {dataset_name} -> ERRO: {e}")
            continue

        label = DATASET_LABELS.get(dataset_name, dataset_name)

        results.append({
            "dataset":          dataset_name,
            "label":            label,
            "nodes":            n_nodes,
            "edges":            n_edges,
            "max_degree":       max_degree,
            "depth":            depth,
            "mean_cardinality": mean_card,   # [NOVO]
            "max_cardinality":  max_card,    # [NOVO]
        })

        print(
            f"[{idx}] {label} ({dataset_name}) -> "
            f"Nós: {n_nodes}, Arestas: {n_edges}, "
            f"Grau máx.: {max_degree}, Profundidade: {depth}, "
            f"Card. média: {mean_card:.2f}, Card. máx.: {max_card}"  # [NOVO]
        )

    return results


# ─────────────────────────────────────────────
#  MODIFICADO: get_variable_structure_table
#  Adiciona cardinalidade por variável.
# ─────────────────────────────────────────────

def get_variable_structure_table(
    datasets_to_run=None,
    datasets_dir: Path = DATASETS_DIR,
) -> List[dict]:
    results = []

    for dataset_name in datasets_to_run:
        dataset_path = datasets_dir / dataset_name

        try:
            network, _ = load_network(str(dataset_path))
            parents, _, cardinalities = extract_graph_structure(network)  # [NOVO]
        except Exception as e:
            print(f"{dataset_name} -> ERRO: {e}")
            continue

        label = DATASET_LABELS.get(dataset_name, dataset_name)

        for var in network.variables:
            results.append({
                "dataset":     dataset_name,
                "label":       label,
                "variable":    var,
                "n_parents":   len(parents[var]),
                "cardinality": cardinalities.get(var),   # [NOVO]
            })

    return results


def compute_parent_cardinality_correlation(variable_structure_table: List[dict]) -> dict:
    """Spearman entre número de pais e cardinalidade, uma observação por variável."""
    pairs = [
        (row.get("n_parents"), row.get("cardinality"))
        for row in variable_structure_table
        if row.get("n_parents") is not None and row.get("cardinality") is not None
    ]
    n = len(pairs)
    if n < 3 or len({p[0] for p in pairs}) < 2 or len({p[1] for p in pairs}) < 2:
        return {"n": n, "rho": float("nan"), "p_value": float("nan")}

    rho, p_value = spearmanr(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
    )
    return {"n": n, "rho": float(rho), "p_value": float(p_value)}


def print_parent_cardinality_correlation(result: dict) -> None:
    """Imprime a checagem do potencial confundidor pais × cardinalidade."""
    print("\n" + "=" * 65)
    print("  NÚMERO DE PAIS × CARDINALIDADE (SPEARMAN)")
    print("=" * 65)
    if result["p_value"] != result["p_value"]:  # NaN, sem importar math
        print(f"  n={result['n']} | correlação indisponível (amostra/variação insuficiente)")
        return

    rho = result["rho"]
    p_value = result["p_value"]
    strength = "forte" if abs(rho) >= 0.7 else "não forte"
    significance = "significativa" if p_value < 0.05 else "não significativa"
    print(f"  n={result['n']} | rho={rho:.3f} | p={p_value:.3g}")
    print(f"  Associação {strength} e {significance} (limiar de significância: 0,05).")


# ─────────────────────────────────────────────
#  MODIFICADO: to_latex_table
#  Adiciona colunas de cardinalidade.
# ─────────────────────────────────────────────

def to_latex_table(results: List[dict]) -> str:
    rows = "\n".join(
        f"{r['label']:<20} & {r['nodes']:>2} & {r['edges']:>3} & "
        f"{r['max_degree']:>1} & {r['depth']:>2} & "
        f"{r.get('mean_cardinality', float('nan')):>4.1f} & "   # [NOVO]
        f"{r.get('max_cardinality',  0):>2} \\\\"               # [NOVO]
        for r in results
    )

    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Características das redes bayesianas utilizadas.}\n"
        "\\label{tab:dataset}\n"
        "\\begin{tabular}{lrrrrrr}\n"          # [NOVO] duas colunas a mais
        "\\hline\n"
        "Domínio & Nós & Arestas & Grau máx. & Prof. "
        "& Card. média & Card. máx. \\\\\n"    # [NOVO]
        "\\hline\n"
        f"{rows}\n"
        "\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


if __name__ == "__main__":
    datasets = [
        "adhd_cbeb.bif",
        "alarm_cbeb.bif",
        "child_cbeb.bif",
        "diabets_cbeb.bif",
        "gonorrhoeae_cbeb.bif",
        "hepar2_cbeb.bif",
        "foodallergy1_cbeb.bif",
        "foodallergy3_cbeb.bif",
        "covid1_cbeb.bif", # ERRO
        "covid3_cbeb.bif", # ERRO
    ]

    table_results = get_structure_table(datasets)
    variable_results = get_variable_structure_table(datasets)

    print("\n" + to_latex_table(table_results))
    print_parent_cardinality_correlation(
        compute_parent_cardinality_correlation(variable_results)
    )
