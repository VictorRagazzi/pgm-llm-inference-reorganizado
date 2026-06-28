import numpy as np
from typing import Dict, Tuple
from pgm_llm_inference.models import Factor

def count_llm_hits(
    result,
    query_vars,
    map_assignment,
    inference_mode: str = "mpe",
) -> Tuple[int, Dict[str, str]]:
    """
    Count LLM prediction hits against the target assignment and
    also return the predicted assignment.

    Returns:
        hits (int)
        predictions (dict[var -> value])
    """

    if inference_mode not in {"mpe", "map"}:
        raise ValueError("inference_mode must be 'mpe' or 'map'")

    # Variables to compare
    target_vars = (
        list(map_assignment.keys()) if inference_mode == "mpe"
        else query_vars
    )

    hits = 0
    predictions: Dict[str, str] = {}

    # ---------------------------------------------------
    # CASE 1 — MPE semantic → result is dict[var -> value]
    # ---------------------------------------------------
    if isinstance(result, dict):
        for v in target_vars:
            llm_value = result.get(v)
            true_value = map_assignment.get(v)

            if llm_value is not None:
                predictions[v] = llm_value

                if llm_value == true_value:
                    hits += 1

        return hits, predictions

    # ---------------------------------------------------
    # CASE 2 — MAP mode → result is Factor
    # ---------------------------------------------------
    values = result.values
    scope = result.scope

    for target in target_vars:
        # Find variable index in Factor scope
        try:
            var_index = next(i for i, var in enumerate(scope) if var.name == target)
        except StopIteration:
            raise ValueError(f"Variable '{target}' not found in result.scope")

        var = scope[var_index]
        domain = var.domain

        # Marginalize all but this variable
        if values.ndim == 0:
            probs = np.array([values])
        else:
            axes_to_sum = tuple(i for i in range(values.ndim) if i != var_index)
            probs = (
                values.sum(axis=axes_to_sum)
                if axes_to_sum else values.reshape(-1)
            )

        probs = np.asarray(probs).ravel()

        pred_value = domain[int(np.argmax(probs))]
        true_value = map_assignment[target]

        predictions[target] = pred_value

        if pred_value == true_value:
            hits += 1

    return hits, predictions