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

from pgm_llm_inference.io.loaders import load_network

# Reaproveita a função de extração de grafo já implementada


DATASETS_DIR = Path(__file__).resolve().parents[3] / "datasets"


def extract_graph_structure(network):
    parents = {var: set() for var in network.variables}
    children = {var: set() for var in network.variables}

    seen_children = set()

    for factor in network.factors:
        scope = factor.scope
        child = scope[0].name

        if child in seen_children:
            raise ValueError(f"Multiple factors for variable '{child}'")
        seen_children.add(child)

        parent_names = [v.name for v in scope[1:]]

        for p in parent_names:
            parents[child].add(p)
            children[p].add(child)

    if len(seen_children) != len(network.variables):
        missing = set(network.variables) - seen_children
        raise ValueError(f"Missing factors for variables: {missing}")

    return parents, children


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


def get_variable_structure_table(
    datasets_to_run: Optional[List[str]] = None,
    datasets_dir: Path = DATASETS_DIR,
) -> List[dict]:
    """
    Uma linha por (dataset, variável), com o número de pais daquela
    variável (grau de entrada no DAG).

    Granularidade diferente de get_structure_table: lá é uma linha por
    rede (n = número de redes, hoje 10); aqui é uma linha por nó avaliado
    (n = soma de variáveis de todas as redes, uma ou duas ordens de
    grandeza maior). Usado para correlacionar erro por variável x nº de
    pais daquela variável especificamente, com poder estatístico maior do
    que a correlação em nível de rede.
    """
    results = []

    for dataset_name in datasets_to_run:
        dataset_path = datasets_dir / dataset_name

        try:
            network, _ = load_network(str(dataset_path))
            parents, _ = extract_graph_structure(network)
        except Exception as e:
            print(f"{dataset_name} -> ERRO: {e}")
            continue

        label = DATASET_LABELS.get(dataset_name, dataset_name)

        for var in network.variables:
            results.append(
                {
                    "dataset": dataset_name,
                    "label": label,
                    "variable": var,
                    "n_parents": len(parents[var]),
                }
            )

    return results


def compute_structure_metrics(network) -> Tuple[int, int, int, int]:
    parents, children = extract_graph_structure(network)

    n_nodes = len(network.variables)
    n_edges = sum(len(p) for p in parents.values())
    max_degree = compute_max_degree(network.variables, parents, children)
    depth = compute_depth(network.variables, children)

    return n_nodes, n_edges, max_degree, depth


def get_structure_table(
    datasets_to_run: Optional[List[str]] = None,
    datasets_dir: Path = DATASETS_DIR,
) -> List[dict]:
    results = []

    for idx, dataset_name in enumerate(datasets_to_run, 1):
        dataset_path = datasets_dir / dataset_name

        try:
            network, _ = load_network(str(dataset_path))
            n_nodes, n_edges, max_degree, depth = compute_structure_metrics(network)
        except Exception as e:
            print(f"[{idx}] {dataset_name} -> ERRO: {e}")
            continue

        label = DATASET_LABELS.get(dataset_name, dataset_name)

        results.append(
            {
                "dataset": dataset_name,
                "label": label,
                "nodes": n_nodes,
                "edges": n_edges,
                "max_degree": max_degree,
                "depth": depth,
            }
        )

        print(
            f"[{idx}] {label} ({dataset_name}) -> "
            f"Nós: {n_nodes}, Arestas: {n_edges}, "
            f"Grau máx.: {max_degree}, Profundidade: {depth}"
        )

    return results


def to_latex_table(results: List[dict]) -> str:
    rows = "\n".join(
        f"{r['label']:<20} & {r['nodes']:>2} & {r['edges']:>3} & "
        f"{r['max_degree']:>1} & {r['depth']:>2} \\\\"
        for r in results
    )

    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{Características das redes bayesianas utilizadas.}\n"
        "\\label{tab:dataset}\n"
        "\\begin{tabular}{lrrrr}\n"
        "\\hline\n"
        "Domínio & Nós & Arestas & Grau máx. & Profundidade \\\\\n"
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
        "urinary_cbeb.bif",
    ]

    table_results = get_structure_table(datasets)

    print("\n" + to_latex_table(table_results))