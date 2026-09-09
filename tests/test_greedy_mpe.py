import random

import numpy as np

from pgm_llm_inference.evaluation.greedy_mpe import (
    compare_assignments,
    greedy_mpe,
    sample_random_evidence,
)
from pgm_llm_inference.models import BayesianNetwork, Factor, Variable


def _network() -> BayesianNetwork:
    a = Variable(name="A", states=("no", "yes"))
    b = Variable(name="B", states=("no", "yes"))
    return BayesianNetwork(
        variables={"A": a, "B": b},
        factors=[
            Factor(scope=[a], values=np.array([0.8, 0.2])),
            Factor(scope=[b, a], values=np.array([[0.9, 0.1], [0.1, 0.9]])),
        ],
        explicit_parents={"A": (), "B": ("A",)},
    )


def test_greedy_mpe_uses_parent_assignment():
    assert greedy_mpe(_network(), ["B"], {"A": "yes"}) == {"B": "yes"}


def test_sample_and_compare_helpers():
    network = _network()
    evidence = sample_random_evidence(network, 0.5, random.Random(7))
    assert len(evidence) == 1
    assert compare_assignments({"A": "yes"}, {"A": "yes"}, ["A"]) == (1, 1.0)
    assert compare_assignments({"A": "yes"}, {"A": "no"}, ["A"]) == (0, 0.0)
