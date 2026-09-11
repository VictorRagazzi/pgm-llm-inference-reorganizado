import pandas as pd

from pgm_llm_inference.analysis.structural_metrics import (
    build_structural_merge,
    build_variable_structural_merge,
    compute_parents_correlation,
)


def test_build_structural_merge_normalizes_bif_suffix():
    grouped = pd.DataFrame(
        [
            {
                "dataset": "network",
                "accuracy": 0.75,
                "joint_match": 0.5,
                "exact_match": 0.25,
            }
        ]
    )
    structure = [
        {
            "dataset": "network.bif",
            "label": "Network",
            "nodes": 3,
            "edges": 2,
            "max_degree": 1,
            "depth": 2,
            "mean_cardinality": 2.0,
            "max_cardinality": 2,
        }
    ]

    merged = build_structural_merge(grouped, structure)

    assert len(merged) == 1
    assert merged.loc[0, "nodes"] == 3


def test_variable_merge_and_parent_correlation():
    variable_results = pd.DataFrame(
        [
            {"dataset": "n.bif", "variable": "A", "correct": 1},
            {"dataset": "n.bif", "variable": "B", "correct": 1},
            {"dataset": "n.bif", "variable": "C", "correct": 0},
        ]
    )
    structure = [
        {"dataset": "n.bif", "variable": "A", "n_parents": 0, "cardinality": 2},
        {"dataset": "n.bif", "variable": "B", "n_parents": 1, "cardinality": 2},
        {"dataset": "n.bif", "variable": "C", "n_parents": 2, "cardinality": 3},
    ]

    merged = build_variable_structural_merge(variable_results, structure)
    correlation = compute_parents_correlation(merged)

    assert len(merged) == 3
    assert correlation["n"] == 3
    assert correlation["rho"] < 0
