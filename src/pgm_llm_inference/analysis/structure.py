"""Structural measurements for Bayesian-network datasets."""

from collections import deque
from collections.abc import Iterable, Mapping, Sequence, Set
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.paths import DATASETS_DIR

from .logs import translate_dataset


def compute_max_degree(
    variables: Sequence[str], parents: Mapping[str, Set[str]]
) -> int:
    """Return the largest parent count, preserving the reported metric."""
    return max((len(parents[variable]) for variable in variables), default=0)


def compute_depth(
    variables: Sequence[str], children: Mapping[str, Set[str]]
) -> int:
    """Return the length in edges of the longest directed path."""
    memo: dict[str, int] = {}

    def longest_path(node: str) -> int:
        if node not in memo:
            memo[node] = (
                1 + max(longest_path(child) for child in children[node])
                if children[node]
                else 0
            )
        return memo[node]

    return max((longest_path(variable) for variable in variables), default=0)


def compute_node_depths(
    variables: Sequence[str],
    parents: Mapping[str, Set[str]],
    children: Mapping[str, Set[str]],
) -> dict[str, int]:
    """Return each node's longest distance from a root."""
    in_degree = {variable: len(parents[variable]) for variable in variables}
    queue = deque(variable for variable in variables if in_degree[variable] == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(variables):
        raise ValueError("The network graph is not a valid DAG (cycle detected).")

    depths: dict[str, int] = {}
    for node in order:
        depths[node] = (
            1 + max(depths[parent] for parent in parents[node])
            if parents[node]
            else 0
        )
    return depths


def _get_cardinality(variable: Any) -> int:
    for attribute in ("values", "states", "cardinality", "domain", "k"):
        value = getattr(variable, attribute, None)
        if value is not None:
            return int(value) if isinstance(value, (int, float)) else len(value)
    available = [name for name in dir(variable) if not name.startswith("_")]
    raise AttributeError(
        f"Could not determine the cardinality of '{variable.name}'. "
        f"Available attributes: {available}"
    )


def extract_graph_structure(
    network: Any,
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, int]]:
    """Extract parent, child, and cardinality mappings from network factors."""
    parents = {variable: set() for variable in network.variables}
    children = {variable: set() for variable in network.variables}
    cardinalities: dict[str, int] = {}
    seen_children: set[str] = set()

    for factor in network.factors:
        child_variable = factor.scope[0]
        child_name = child_variable.name
        if child_name in seen_children:
            raise ValueError(f"Multiple factors for variable '{child_name}'")

        seen_children.add(child_name)
        cardinalities[child_name] = _get_cardinality(child_variable)
        for parent_variable in factor.scope[1:]:
            parents[child_name].add(parent_variable.name)
            children[parent_variable.name].add(child_name)

    missing = set(network.variables) - seen_children
    if missing:
        raise ValueError(f"Missing factors for variables: {missing}")

    return parents, children, cardinalities


def compute_structure_metrics(
    network: Any,
) -> tuple[int, int, int, int, float, int]:
    parents, children, cardinalities = extract_graph_structure(network)
    card_values = list(cardinalities.values())
    variables = list(network.variables)

    return (
        len(variables),
        sum(len(parent_set) for parent_set in parents.values()),
        compute_max_degree(variables, parents),
        compute_depth(variables, children),
        sum(card_values) / len(card_values) if card_values else 0.0,
        max(card_values, default=0),
    )


def get_node_depth_table(
    datasets: Iterable[str], datasets_dir: Path = DATASETS_DIR
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset_name in datasets:
        try:
            network = load_network(datasets_dir / dataset_name)
            parents, children, _ = extract_graph_structure(network)
            depths = compute_node_depths(list(network.variables), parents, children)
            max_depth = max(depths.values(), default=0)
        except Exception as error:
            print(f"{dataset_name} -> ERRO: {error}")
            continue

        for variable in network.variables:
            depth = depths[variable]
            results.append(
                {
                    "dataset": dataset_name,
                    "label": translate_dataset(dataset_name),
                    "variable": variable,
                    "depth": depth,
                    "max_depth": max_depth,
                    "depth_normalized": depth / max_depth if max_depth else None,
                }
            )
    return results


def get_structure_table(
    datasets: Iterable[str], datasets_dir: Path = DATASETS_DIR
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index, dataset_name in enumerate(datasets, 1):
        try:
            network = load_network(datasets_dir / dataset_name)
            nodes, edges, max_degree, depth, mean_card, max_card = (
                compute_structure_metrics(network)
            )
        except Exception as error:
            print(f"[{index}] {dataset_name} -> ERRO: {error}")
            continue

        label = translate_dataset(dataset_name)
        results.append(
            {
                "dataset": dataset_name,
                "label": label,
                "nodes": nodes,
                "edges": edges,
                "max_degree": max_degree,
                "depth": depth,
                "mean_cardinality": mean_card,
                "max_cardinality": max_card,
            }
        )
        print(
            f"[{index}] {label} ({dataset_name}) -> "
            f"Nós: {nodes}, Arestas: {edges}, Grau máx.: {max_degree}, "
            f"Profundidade: {depth}, Card. média: {mean_card:.2f}, "
            f"Card. máx.: {max_card}"
        )
    return results


def get_variable_structure_table(
    datasets: Iterable[str], datasets_dir: Path = DATASETS_DIR
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for dataset_name in datasets:
        try:
            network = load_network(datasets_dir / dataset_name)
            parents, _, cardinalities = extract_graph_structure(network)
        except Exception as error:
            print(f"{dataset_name} -> ERRO: {error}")
            continue

        for variable in network.variables:
            results.append(
                {
                    "dataset": dataset_name,
                    "label": translate_dataset(dataset_name),
                    "variable": variable,
                    "n_parents": len(parents[variable]),
                    "cardinality": cardinalities.get(variable),
                }
            )
    return results


def compute_parent_cardinality_correlation(
    variable_structure_table: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    pairs = [
        (row.get("n_parents"), row.get("cardinality"))
        for row in variable_structure_table
        if row.get("n_parents") is not None and row.get("cardinality") is not None
    ]
    count = len(pairs)
    if count < 3 or len({pair[0] for pair in pairs}) < 2 or len(
        {pair[1] for pair in pairs}
    ) < 2:
        return {"n": count, "rho": float("nan"), "p_value": float("nan")}

    rho, p_value = spearmanr(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
    )
    return {"n": count, "rho": float(rho), "p_value": float(p_value)}
