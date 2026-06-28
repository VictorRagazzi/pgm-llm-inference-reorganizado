from typing import Any
from ..models import BayesianNetwork, Factor, Variable
from ..strategies import EliminationStrategy
from ..core.ordering import DEGREE_ORDERING_MAP
from ..core.factor_ops import multiply_factors, reduce_factor
from ..core.config import InferenceConfig


def run_variable_elimination(
    network: BayesianNetwork,
    strategy: EliminationStrategy,
    query_vars: list[str],
    evidence: dict[str, str],
    elimination_order: list[str] | None,
    config: InferenceConfig,
) -> tuple[Factor, dict[str, Any]]:

    # STEP 1 — apply evidence
    original_evidence = evidence
    evidence = dict(evidence)  # cópia defensiva
    factors = [reduce_factor(f, evidence) for f in network.factors]

    variables = set(network.variables.keys())
    evidence_vars = set(evidence.keys())
    query_set = set(query_vars)

    # STEP 2 — define nuisance vars depending on mode
    if config.mode == "mpe":
        # In MPE, everything not evidence is "query-like"
        nuisance_vars = variables - evidence_vars
    else:
        # Posterior / MAP
        nuisance_vars = variables - query_set - evidence_vars

    # STEP 2.5 — LLM decision phase (optional, MPE only)
    metadata: dict[str, Any] = {}
    metadata["observed_evidence"] = original_evidence

    # STEP 3 — elimination order
    if elimination_order is None:
        elimination_order = (
            DEGREE_ORDERING_MAP[config.default_ordering_heuristic](factors, nuisance_vars)
            if nuisance_vars
            else []
        )

    used_elimination_order: list[str] = []

    # STEP 4 — eliminate vars in order
    for var_name in elimination_order:
        if var_name not in nuisance_vars:
            continue  # safety guard

        used_elimination_order.append(var_name)
        variable = network.variables[var_name]

        factors, step_metadata = eliminate_variable(
            factors=factors,
            variable=variable,
            evidence=evidence,
            strategy=strategy,
            network=network,
            config=config,
        )

        metadata.update(step_metadata)

    metadata["elimination_order"] = used_elimination_order

    # STEP 5 — multiply remaining
    if not factors:
        raise ValueError("No factors remaining after elimination")

    result = factors[0]
    for f in factors[1:]:
        result = multiply_factors(result, f)

    return result, metadata



def eliminate_variable(
    factors: list[Factor],
    variable: Variable,
    evidence: dict[str, str],
    strategy: EliminationStrategy,
    network: BayesianNetwork,
    config: InferenceConfig,
) -> tuple[list[Factor], dict[str, Any]]:

    # Factors containing the variable
    relevant = [
        f for f in factors
        if any(v.name == variable.name for v in f.scope)
    ]

    # Factors that don't contain it
    others = [
        f for f in factors
        if not any(v.name == variable.name for v in f.scope)
    ]

    # Nothing to eliminate
    if not relevant:
        return factors, {}

    # Multiply all relevant factors
    product = relevant[0]
    for f in relevant[1:]:
        product = multiply_factors(product, f)

    # Delegate elimination to strategy
    eliminated, metadata = strategy.eliminate(
        factor=product,
        variable=variable,
        network=network,
        context={
            "evidence": evidence,
            "network": network,
            "mode": config.mode,
        },
    )

    return others + [eliminated], metadata