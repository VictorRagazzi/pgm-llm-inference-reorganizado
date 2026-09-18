import gzip
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from pgm_llm_inference import (
    InferenceConfig,
    InferenceEngine,
    MaxProductStrategy,
    SumProductStrategy,
)
from pgm_llm_inference.io.loaders import convert_pgmpy_model
from pgm_llm_inference.io.loaders import load_network, parse_bif
from tests.test_pipeline import make_network


def joint_probability(network, assignment):
    return np.prod(
        [
            factor.values[tuple(v.states.index(assignment[v.name]) for v in factor.scope)]
            for factor in network.factors
        ]
    )


@pytest.mark.parametrize("evidence", [{}, {"A": "on"}, {"D": "on"}, {"B": "off", "C": "on"}])
def test_max_product_matches_enumeration(evidence):
    network = make_network()
    candidates = [
        dict(zip(network.variables, values))
        for values in itertools.product(*(v.states for v in network.variables.values()))
    ]
    candidates = [a for a in candidates if all(a[v] == s for v, s in evidence.items())]
    expected = max(candidates, key=lambda a: joint_probability(network, a))
    result = InferenceEngine(network=network, strategy=MaxProductStrategy()).query(
        query_vars=[v for v in network.variables if v not in evidence],
        evidence=evidence,
    )
    actual = {v: s for v, s in result["map_assignment"].items() if v != "_scalar"}
    assert actual == {v: s for v, s in expected.items() if v not in evidence}
    assert result["map_probability"] == pytest.approx(joint_probability(network, expected))


def test_sum_product_matches_enumeration():
    network = make_network()
    expected = np.array(
        [
            sum(
                joint_probability(network, dict(zip(network.variables, (a, b, c, "on"))))
                for b, c in itertools.product(("off", "on"), repeat=2)
            )
            for a in ("off", "on")
        ]
    )
    result = InferenceEngine(
        network=network, strategy=SumProductStrategy(), config=InferenceConfig(mode="posterior")
    ).query(["A"], {"D": "on"})
    np.testing.assert_allclose(result["result_factor"].values, expected / expected.sum())


@pytest.mark.parametrize(
    "evidence,match", [({"unknown": "off"}, "not in network"), ({"A": "invalid"}, "not in domain")]
)
def test_invalid_evidence(evidence, match):
    with pytest.raises(ValueError, match=match):
        InferenceEngine(network=make_network(), strategy=MaxProductStrategy()).query(
            ["B"], evidence
        )


def test_conversion_preserves_parent_axes_and_states():
    # pgmpy returns evidence in the reverse order of its CPD axes.
    cpds = {
        "A": SimpleNamespace(
            variable="A",
            variables=["A"],
            state_names={"A": ["a0", "a1"]},
            values=np.array([0.2, 0.8]),
            get_evidence=lambda: [],
        ),
        "B": SimpleNamespace(
            variable="B",
            variables=["B"],
            state_names={"B": ["b0", "b1"]},
            values=np.array([0.3, 0.7]),
            get_evidence=lambda: [],
        ),
        "C": SimpleNamespace(
            variable="C",
            variables=["C", "A", "B"],
            state_names={"C": ["c0", "c1"]},
            values=np.arange(8).reshape(2, 2, 2),
            get_evidence=lambda: ["B", "A"],
        ),
    }
    model = SimpleNamespace(
        nodes=lambda: list(cpds),
        get_cpds=lambda name=None: cpds[name] if name else list(cpds.values()),
    )
    network = convert_pgmpy_model(model)
    assert network.variables["A"].states == ("a0", "a1")
    assert [v.name for v in network.factors[-1].scope] == ["C", "B", "A"]
    np.testing.assert_array_equal(network.factors[-1].values, cpds["C"].values.transpose(0, 2, 1))


@pytest.mark.parametrize("suffix", [".bif", ".bif.gz", ".dne", ".dne.gz"])
def test_load_network_formats(tmp_path, suffix):
    if suffix.startswith(".bif"):
        text = """network tiny { }
variable A { type discrete [ 2 ] { off, on }; }
variable B { type discrete [ 2 ] { off, on }; }
probability ( A ) {
  table 0.7, 0.3;
}
probability ( B | A ) {
  (off) 0.9, 0.1;
  (on) 0.2, 0.8;
}
"""
    else:
        text = """bnet tiny {
node A { states = (off, on); probs = (0.7, 0.3); };
node B { states = (off, on); parents = (A); probs = (0.9, 0.1, 0.2, 0.8); };
};"""
    path = tmp_path / f"tiny{suffix}"
    data = text.encode("ISO-8859-1")
    path.write_bytes(gzip.compress(data) if suffix.endswith(".gz") else data)
    network = load_network(path)
    assert network.parents == {"A": (), "B": ("A",)}
    assert network.variables["A"].states == ("off", "on")
    np.testing.assert_allclose(network.factors[1].values, [[0.9, 0.2], [0.1, 0.8]])


def test_semantic_bif_loading_uses_only_structure(tmp_path):
    path = tmp_path / "structure.bif"
    path.write_text(
        """network structure { }
variable Root_node { type discrete [ 2 ] { Off, On }; }
variable Child { type discrete [ 3 ] { low, medium, high }; }
probability ( Root_node ) { ignored numeric content }
probability ( Child | Root_node ) { ignored numeric content }
""",
        encoding="utf-8",
    )
    network = parse_bif(path)
    assert network.name == "structure"
    assert network.variables["Root_node"].states == ("Off", "On")
    assert network.variables["Child"].states == ("low", "medium", "high")
    assert network.parents == {"Root_node": (), "Child": ("Root_node",)}
    assert network.factors == []
