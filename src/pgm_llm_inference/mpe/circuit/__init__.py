"""
mpe/circuit
===========
Part 2: semantic maximizing circuit — versão evidence-independent e de
duas passadas do pipeline MPE, formalizada a partir do Part 1 (forward-
only, evidence-conditioned). Ver mpe/circuit/compile.py e
mpe/circuit/evaluate.py para as fases de compilação e avaliação.
"""

from .compile import compile_semantic_circuit, load_or_compile_circuit
from .diagnose import (
    BucketDiagnostics,
    DivergenceSummary,
    compare_forward_vs_twopass,
    evaluate_two_pass_diagnostics,
    print_divergence_report,
)
from .evaluate import evaluate_forward_only, evaluate_two_pass
from .order import build_bucket_structure, constrained_min_degree_order, moralize
from .types import ScoredRow, SemanticCircuit, SemanticFactor

__all__ = [
    "SemanticCircuit",
    "SemanticFactor",
    "ScoredRow",
    "compile_semantic_circuit",
    "load_or_compile_circuit",
    "evaluate_two_pass",
    "evaluate_forward_only",
    "build_bucket_structure",
    "moralize",
    "constrained_min_degree_order",
    "evaluate_two_pass_diagnostics",
    "compare_forward_vs_twopass",
    "print_divergence_report",
    "BucketDiagnostics",
    "DivergenceSummary",
]