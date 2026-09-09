"""Compare exact MPE with topological greedy decoding."""

import random
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from pgm_llm_inference.experiment.experiment import get_hidden_vars, run_max_product
from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.models import BayesianNetwork
from pgm_llm_inference.mpe.graph import topological_order


def sample_random_evidence(
    network: BayesianNetwork,
    ratio: float,
    rng: random.Random,
) -> dict[str, str]:
    """Sample evidence uniformly while leaving at least one hidden variable."""
    variables = list(network.variables)
    evidence_count = max(0, min(round(ratio * len(variables)), len(variables) - 1))
    return {
        name: rng.choice(network.variables[name].states)
        for name in rng.sample(variables, evidence_count)
    }


def exact_mpe(
    network: BayesianNetwork,
    hidden_variables: list[str],
    evidence: dict[str, str],
) -> dict[str, str]:
    result = run_max_product(
        network=network,
        query_vars=hidden_variables,
        evidence=evidence,
    )
    assignment = result["map_assignment"]
    return {
        variable: assignment[variable]
        for variable in hidden_variables
        if variable in assignment
    }


def greedy_mpe(
    network: BayesianNetwork,
    hidden_variables: list[str],
    evidence: dict[str, str],
) -> dict[str, str]:
    """Choose each local CPD maximum in topological order."""
    assignment = dict(evidence)
    factors = {factor.scope[0].name: factor for factor in network.factors}

    for variable_name in topological_order(network):
        if variable_name in assignment:
            continue

        factor = factors[variable_name]
        values = factor.values
        for axis in range(len(factor.scope) - 1, 0, -1):
            parent = factor.scope[axis]
            state_index = parent.states.index(assignment[parent.name])
            values = np.take(values, state_index, axis=axis)

        best_index = int(np.argmax(values))
        assignment[variable_name] = factor.scope[0].states[best_index]

    return {variable: assignment[variable] for variable in hidden_variables}


def compare_assignments(
    exact: dict[str, str],
    greedy: dict[str, str],
    hidden_variables: list[str],
) -> tuple[int, float]:
    if not hidden_variables:
        return 1, 1.0
    hits = sum(exact.get(variable) == greedy.get(variable) for variable in hidden_variables)
    return int(hits == len(hidden_variables)), hits / len(hidden_variables)


def run_dataset(
    dataset_name: str,
    datasets_dir: Path,
    sample_count: int,
    evidence_ratio: float,
    rng: random.Random,
) -> dict[str, Any] | None:
    """Run the exact-versus-greedy benchmark for one dataset."""
    try:
        network = load_network(datasets_dir / dataset_name)
    except Exception as error:
        print(f"[{dataset_name}] ERRO ao carregar: {error}")
        traceback.print_exc()
        return None

    node_count = len(network.variables)
    print(f"  Rede carregada: {node_count} nós, {len(network.factors)} fatores")
    exact_matches: list[int] = []
    accuracies: list[float] = []

    for sample_index in range(sample_count):
        evidence: dict[str, str] = {}
        try:
            evidence = sample_random_evidence(network, evidence_ratio, rng)
            hidden_variables = get_hidden_vars(network, evidence)
            if not hidden_variables:
                continue

            exact_assignment = exact_mpe(network, hidden_variables, evidence)
            greedy_assignment = greedy_mpe(network, hidden_variables, evidence)
            exact_match, accuracy = compare_assignments(
                exact_assignment, greedy_assignment, hidden_variables
            )
            exact_matches.append(exact_match)
            accuracies.append(accuracy)

            completed = len(accuracies)
            if (sample_index + 1) % 10 == 0 or sample_index == sample_count - 1:
                print(
                    f"  [{sample_index + 1:3d}/{sample_count}] ok={completed} | "
                    f"exact_match={sum(exact_matches) / completed:.3f} | "
                    f"accuracy={sum(accuracies) / completed:.3f} | "
                    f"ev={len(evidence)} | hidden={len(hidden_variables)}"
                )
        except Exception as error:
            print(
                f"  [{dataset_name}] ERRO na amostra {sample_index} "
                f"(evidência={evidence or '?'}): {error}"
            )
            traceback.print_exc()

    completed = len(accuracies)
    if not completed:
        print(f"[{dataset_name}] Nenhuma amostra bem-sucedida — dataset ignorado.")
        return None

    return {
        "dataset": dataset_name,
        "n_nodes": node_count,
        "n_samples": completed,
        "exact_match_mean": sum(exact_matches) / completed,
        "accuracy_mean": sum(accuracies) / completed,
    }
