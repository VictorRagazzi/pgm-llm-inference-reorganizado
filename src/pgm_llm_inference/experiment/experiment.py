"""
experiment.py
=============
Pipelines de um único experimento de inferência MPE.

run_single_mpe_experiment: roda MPE sobre todas as variáveis ocultas,
    comparando o MPE exato (Max-Product) com a pipeline LLM-MPE
    (bucket elimination semântica + reconstrução + audit).
    Delega internamente para compile_semantic_messages() + infer_from_compiled().

    Para múltiplas evidências sobre o mesmo dataset, é mais eficiente chamar
    compile_semantic_messages() uma vez e infer_from_compiled() por evidência
    diretamente — é o que main.py faz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.config import InferenceConfig
from ..inference import InferenceEngine
from ..mpe.compile import CompiledSemanticMessages, compile_semantic_messages
from ..mpe.infer import infer_from_compiled
from ..strategies import MaxProductStrategy, SumProductStrategy

__all__ = [
    "run_single_mpe_experiment",
    "get_hidden_vars",
    "extract_argmax_from_factor",
    "run_max_product",
    "run_sum_product",
    "CompiledSemanticMessages",
    "compile_semantic_messages",
    "infer_from_compiled",
]


# ---------------------------------------------------------------------------
# Helpers de inferência exata (VE numérico)
# ---------------------------------------------------------------------------

def _build_engine(network, strategy):
    return InferenceEngine(
        network=network,
        strategy=strategy,
        config=InferenceConfig(verbose=False),
    )


def run_sum_product(network, query_vars, evidence):
    return _build_engine(network, SumProductStrategy()).query(
        query_vars=query_vars, evidence=evidence
    )


def run_max_product(network, query_vars, evidence):
    return _build_engine(network, MaxProductStrategy()).query(
        query_vars=query_vars, evidence=evidence
    )


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def get_hidden_vars(network, evidence):
    """Retorna variáveis não observadas (não presentes em evidence)."""
    evidence_vars = set(evidence.keys())
    return [v for v in network.variables.keys() if v not in evidence_vars]


def extract_argmax_from_factor(factor, target_var: str) -> str:
    """Extrai o valor MAP de uma variável a partir de um Factor."""
    scope = factor.scope
    values = factor.values

    var_index = next(i for i, v in enumerate(scope) if v.name == target_var)
    var = scope[var_index]
    domain = var.domain

    axes_to_sum = tuple(i for i in range(values.ndim) if i != var_index)
    probs = values.sum(axis=axes_to_sum) if axes_to_sum else values
    probs = np.asarray(probs).ravel()

    return domain[int(np.argmax(probs))]


# ---------------------------------------------------------------------------
# Pipeline MPE
# ---------------------------------------------------------------------------

def run_single_mpe_experiment(
    *,
    network,
    evidence: dict[str, str],
    llm_fn,
    bif_path: Path,
    query_vars=None,               # mantido para compatibilidade; não usado pelo pipeline MPE
    metadata_path: Path | None = None,
    relationship_path: Path | None = None,
    max_context_rows: int = 96,
    apply_audit_repair_enabled: bool = True,
    max_estimated_llm_calls: int = 30,
    # Parâmetros legados ignorados (eram do pipeline MAP)
    prompt_type=None,
    prompt_critique=None,
    context=None,
):
    """
    Roda um único experimento MPE, comparando Max-Product exato com LLM-MPE.

    Para múltiplas evidências sobre o mesmo dataset, prefira chamar
    compile_semantic_messages() uma vez e infer_from_compiled() por evidência.
    """
    hidden_vars = get_hidden_vars(network, evidence)

    print("\n" + "=" * 60)
    print("run_single_mpe_experiment")
    print(f"Evidence:    {evidence}")
    print(f"Hidden vars: {hidden_vars}")
    print("=" * 60)

    if len(hidden_vars) > max_estimated_llm_calls:
        raise ValueError(
            f"Hidden vars ({len(hidden_vars)}) exceeds max_estimated_llm_calls "
            f"({max_estimated_llm_calls}). Skipping."
        )

    # MPE exato (referência numérica)
    mpe_result = run_max_product(
        network=network,
        query_vars=hidden_vars,
        evidence=evidence,
    )

    # Pipeline LLM-MPE: compilar mensagens + inferir
    compiled = compile_semantic_messages(
        network=network,
        bif_path=bif_path,
        metadata_path=metadata_path,
        relationship_path=relationship_path,
        llm_fn=llm_fn,
        max_context_rows=max_context_rows,
    )

    llm_predictions, confidence_map, llm_cpt = infer_from_compiled(
        compiled=compiled,
        evidence=evidence,
        llm_fn=llm_fn,
        apply_audit_repair_enabled=apply_audit_repair_enabled,
    )

    print("\nRESULTS")
    print("-" * 40)
    for v in hidden_vars:
        llm_val = llm_predictions.get(v, "N/A")
        mpe_val = mpe_result["map_assignment"].get(v, "?")
        match = "✓" if llm_val == mpe_val else "✗"
        print(f"  {match} {v}: LLM={llm_val} | MPE={mpe_val}")

    return {
        "mpe": mpe_result,
        "llm": {
            "llm_predictions": llm_predictions,
            "confidence": confidence_map,
            "llm_cpt": llm_cpt,
        },
    }
