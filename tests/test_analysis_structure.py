from types import SimpleNamespace

from pgm_llm_inference.analysis.structure import (
    compute_node_depths,
    compute_structure_metrics,
    extract_graph_structure,
)


def _network():
    a = SimpleNamespace(name="A", domain=("0", "1"))
    b = SimpleNamespace(name="B", domain=("0", "1", "2"))
    c = SimpleNamespace(name="C", domain=("0", "1"))
    factors = [
        SimpleNamespace(scope=[a]),
        SimpleNamespace(scope=[b, a]),
        SimpleNamespace(scope=[c, b]),
    ]
    return SimpleNamespace(variables=["A", "B", "C"], factors=factors)


def test_extract_graph_structure_and_depths():
    parents, children, cardinalities = extract_graph_structure(_network())

    assert parents == {"A": set(), "B": {"A"}, "C": {"B"}}
    assert children == {"A": {"B"}, "B": {"C"}, "C": set()}
    assert cardinalities == {"A": 2, "B": 3, "C": 2}
    assert compute_node_depths(["A", "B", "C"], parents, children) == {
        "A": 0,
        "B": 1,
        "C": 2,
    }


def test_compute_structure_metrics_preserves_reported_values():
    assert compute_structure_metrics(_network()) == (3, 2, 1, 2, 7 / 3, 3)
