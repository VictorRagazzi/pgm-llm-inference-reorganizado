"""
mpe/infer.py
============
Fase 2 do pipeline MPE: inferência determinística a partir de mensagens
compiladas.

infer_from_compiled() roda para cada configuração de evidência e usa lookup
O(N) nos backpointers produzidos durante a compilação. Não faz chamadas LLM.
"""

from __future__ import annotations

from ..core.config import InferenceConfig
from .compile import CompiledSemanticMessages
from .io import normalize_assignment
from .reconstruction import reconstruct_assignment


def infer_from_compiled(
    *,
    compiled: CompiledSemanticMessages,
    evidence: dict[str, str],
) -> tuple[dict[str, str], dict[str, list], dict[str, list]]:
    """
    Fase 2 — Inferência (roda para CADA configuração de evidência).

    Reutiliza as mensagens compiladas por lookup O(N), sem chamadas LLM.

    Parâmetros
    ----------
    compiled                   : resultado de compile_semantic_messages().
    evidence                   : dict {variable_id_ou_alias: estado}.

    Retorna
    -------
    (hidden_assignment, selected_confidence, llm_cpt)

    hidden_assignment   : {var: estado} para variáveis não observadas.
    selected_confidence : {var: [estado, confiança]} da row selecionada.
    llm_cpt             : {var: lista de rows} para logging / análise.
    """
    config = InferenceConfig()

    bn                 = compiled.bn
    alias_map          = compiled.alias_map
    messages           = compiled.messages
    elimination_order  = compiled.elimination_order

    # Normalizar evidência (aliases → IDs canônicos, estados case-insensitive)
    norm_evidence = normalize_assignment(
        {k: v for k, v in evidence.items()}, bn, alias_map
    )

    if config.show_input_data:
        print(f"\n[INFER] Evidência: {norm_evidence}")
        print("[INFER] Reconstruindo assignment via lookup nas mensagens compiladas...")

    # --- Reconstrução via backpointers (sem LLM) ---
    hidden_assignment, _, selected_confidence = reconstruct_assignment(
        elimination_order=elimination_order,
        evidence=norm_evidence,
        messages=messages,
    )

    # CPT gerada pelo LLM (para logging / análise downstream)
    llm_cpt: dict[str, list] = {
        var: [
            {
                "context": row.context,
                "selected_value": row.selected_value,
                "confidence": row.confidence,
                "rationale": row.rationale,
                "domain_scores": (
                    domain_scores.model_dump()
                    if (domain_scores := getattr(row, "domain_scores", None)) is not None
                    else None
                ),
            }
            for row in msg.rows
        ]
        for var, msg in messages.items()
        if var not in norm_evidence
    }

    return hidden_assignment, selected_confidence, llm_cpt
