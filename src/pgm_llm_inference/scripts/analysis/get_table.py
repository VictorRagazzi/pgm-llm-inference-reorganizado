"""
get_table.py
============
Calcula estatísticas estruturais simples para uma lista de datasets:
Markov Blanket médio, número de raízes (sem pais) e folhas (sem filhos).

Uso:
    python -m pgm_llm_inference.scripts.get_table
"""

from pathlib import Path
from typing import List, Optional

from pgm_llm_inference.io.loaders import load_network

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"


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


def markov_blanket(node, parents, children):
    mb = set()

    # Parents
    mb.update(parents[node])

    # Children
    mb.update(children[node])

    # Co-parents (excluindo o próprio nó)
    for child in children[node]:
        for p in parents[child]:
            if p != node:
                mb.add(p)

    return mb


def compute_metrics(network):
    parents, children = extract_graph_structure(network)

    for node in network.variables:
        mb = markov_blanket(node, parents, children)
        if node in mb:
            print("BUG:", node)

    # Markov Blanket médio
    total_mb = sum(
        len(markov_blanket(node, parents, children))
        for node in network.variables
    )
    avg_mb = total_mb / len(network.variables)

    # Raízes e folhas
    roots = sum(len(parents[node]) == 0 for node in network.variables)
    leaves = sum(len(children[node]) == 0 for node in network.variables)

    return avg_mb, roots, leaves


def get_table(datasets_to_run: Optional[List[str]] = None, datasets_dir: Path = DATASETS_DIR) -> list[dict]:
    results = []

    for idx, dataset_name in enumerate(datasets_to_run, 1):
        dataset_path = datasets_dir / dataset_name

        network, _ = load_network(str(dataset_path))

        avg_mb, roots, leaves = compute_metrics(network)

        results.append({
            "dataset": dataset_name,
            "avg_markov_blanket": round(avg_mb, 2),
            "roots": roots,
            "leaves": leaves,
        })

        print(
            f"[{idx}] {dataset_name} -> "
            f"AvgMB: {avg_mb:.2f}, Roots: {roots}, Leaves: {leaves}"
        )

    return results


if __name__ == "__main__":
    datasets = [
        "cryptocurrency.xdsl",
        "bankruptcy.bif",
        "coral1.bif",
        "Coronary_Risk.dne",
        "crimescene.bif",
        "sachs.bif",
        "gonorrhoeae.bif",
        "insurance.bif",
        "child.bif.gz",
    ]
    get_table(datasets)
