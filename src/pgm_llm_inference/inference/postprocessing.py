from typing import Any
import numpy as np

from ..models import Factor
from ..strategies import MaxProductStrategy
from ..core.factor_ops import normalize_factor
from ..core.config import InferenceConfig


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
        "elimination_order": metadata.get("elimination_order")        
    }

    if strategy.__class__.__name__ == "SumProductStrategy":
        normalized = normalize_factor(final_factor)
        result["result_factor"] = normalized

        if config.verbose:
            print(f"[Post] normalized sum={float(normalized.values.sum()):.4f}")

    elif isinstance(strategy, MaxProductStrategy):
        max_idx = np.unravel_index(
            np.argmax(final_factor.values), final_factor.values.shape
        )

        assignment = {
            var.name: var.domain[max_idx[i]]
            for i, var in enumerate(final_factor.scope)
        }

        assignment = strategy.reconstruct_assignment(
            assignment,
            elimination_order=metadata["elimination_order"],
        )

        result["map_assignment"] = assignment
        result["map_probability"] = float(final_factor.values.max())

    elif hasattr(strategy, "get_predictions"):
        result["llm_predictions"] = strategy.get_predictions()

    return result