"""Variable Elimination, numeric reconstruction, and exact inference helpers."""

from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..models import BayesianNetwork, Factor, Variable
from ..core.config import InferenceConfig
from ..core.factor_ops import multiply_factors, reduce_factor, normalize_factor
from ..core.ordering import DEGREE_ORDERING_MAP
from .strategies import EliminationStrategy, MaxProductStrategy


class InferenceEngine(BaseModel):
    network: BayesianNetwork
    strategy: EliminationStrategy
    config: InferenceConfig = Field(default_factory=InferenceConfig)

    model_config = {"arbitrary_types_allowed": True}

    def query(
        self,
        query_vars: list[str],
        evidence: dict[str, str] | None = None,
        elimination_order: list[str] | None = None,
    ) -> dict[str, Any]:
        if evidence is None:
            evidence = {}

        self._validate_query(query_vars, evidence)

        if self.config.verbose:
            print(f"[Inference] query={query_vars}, evidence={evidence}")

        final_factor, metadata = run_variable_elimination(
            network=self.network,
            strategy=self.strategy,
            query_vars=query_vars,
            evidence=evidence,
            elimination_order=elimination_order,
            config=self.config,
        )

        return postprocess_result(
            final_factor=final_factor,
            metadata=metadata,
            strategy=self.strategy,
            config=self.config,
        )

    def _validate_query(self, query_vars: list[str], evidence: dict[str, str]) -> None:
        for var_name in query_vars:
            if var_name not in self.network.variables:
                raise ValueError(
                    f"Query variable '{var_name}' not in network"
                    + f"\nNetwork nodes: {self.network.variables}"
                )

        for var_name, value in evidence.items():
            if var_name not in self.network.variables:
                raise ValueError(f"Evidence variable '{var_name}' not in network")
            if value not in self.network.variables[var_name].states:
                raise ValueError(
                    f"Value '{value}' not in domain {list(self.network.variables[var_name].states)}"
                )


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
    relevant = [f for f in factors if any(v.name == variable.name for v in f.scope)]

    # Factors that don't contain it
    others = [f for f in factors if not any(v.name == variable.name for v in f.scope)]

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


def postprocess_result(
    final_factor: Factor,
    metadata: dict[str, Any],
    strategy,
    config: InferenceConfig,
) -> dict[str, Any]:
    result = {
        "confidence": metadata.get("decisions", None),
        "result_factor": final_factor,
        "strategy_type": strategy.__class__.__name__,
        "elimination_order": metadata.get("elimination_order"),
    }

    if strategy.__class__.__name__ == "SumProductStrategy":
        normalized = normalize_factor(final_factor)
        result["result_factor"] = normalized

        if config.verbose:
            print(f"[Post] normalized sum={float(normalized.values.sum()):.4f}")

    elif isinstance(strategy, MaxProductStrategy):
        max_idx = np.unravel_index(np.argmax(final_factor.values), final_factor.values.shape)

        assignment = {var.name: var.states[max_idx[i]] for i, var in enumerate(final_factor.scope)}

        assignment = strategy.reconstruct_assignment(
            assignment,
            elimination_order=metadata["elimination_order"],
        )

        result["map_assignment"] = assignment
        result["map_probability"] = float(final_factor.values.max())

    return result


def run_max_product(
    network: BayesianNetwork, query_vars: list[str], evidence: dict[str, str]
) -> dict[str, Any]:
    """Compute the exact reference using the configured elimination order."""
    return InferenceEngine(
        network=network,
        strategy=MaxProductStrategy(),
        config=InferenceConfig(verbose=False),
    ).query(query_vars=query_vars, evidence=evidence)


def get_hidden_vars(network: BayesianNetwork, evidence: dict[str, str]) -> list[str]:
    """Return unobserved variables in network order."""
    return [variable for variable in network.variables if variable not in evidence]
