"""
mpe/infer.py
============
Fase 2 do pipeline MPE: inferência a partir de mensagens compiladas.

infer_from_compiled() roda para CADA configuração de evidência.
Não faz chamadas LLM no bucket: usa lookup O(N) nas mensagens compiladas.
Só chama o LLM para reconstruction prompt + audit prompt (2 chamadas fixas
por inferência, independente do tamanho da rede).
"""

from __future__ import annotations

from ..core.config import InferenceConfig
from .client import LLMJsonClient
from .compile import CompiledSemanticMessages
from .io import normalize_assignment
from .prompt_builders import build_audit_prompt, build_reconstruction_prompt
from .reconstruction import (
    apply_audit_repair,
    reconstruct_assignment,
    validate_audit_response,
    validate_reconstruction_response,
)
from .types import AuditResponse, ReconstructionResponse


def infer_from_compiled(
    *,
    compiled: CompiledSemanticMessages,
    evidence: dict[str, str],
    llm_fn,
    apply_audit_repair_enabled: bool = True,
    use_real_llm: bool = False,
) -> tuple[dict[str, str], dict[str, list], dict[str, list]]:
    """
    Fase 2 — Inferência (roda para CADA configuração de evidência).

    Reutiliza as mensagens compiladas por lookup O(N) sem novas chamadas LLM
    no bucket. Faz apenas 2 chamadas LLM: reconstruction + audit.

    Parâmetros
    ----------
    compiled                   : resultado de compile_semantic_messages().
    evidence                   : dict {variable_id_ou_alias: estado}.
    llm_fn                     : callable(prompt, response_model) → model.
    apply_audit_repair_enabled : se True, aplica a correção do audit.

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
    metadata           = compiled.metadata
    relationship_notes = compiled.relationship_notes
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
    hidden_assignment, complete_assignment, selected_confidence = reconstruct_assignment(
        elimination_order=elimination_order,
        evidence=norm_evidence,
        messages=messages,
    )

    if config.show_input_data:
        print(f"  ✓ {len(complete_assignment)} variáveis atribuídas")
        print(f"  Hidden: {hidden_assignment}")

    client = LLMJsonClient(config, use_real_llm=use_real_llm)

    # --- Reconstruction prompt (LLM explica o assignment) ---
    if config.show_input_data:
        print("\n[INFER] Reconstruction prompt...")

    reconstruction_prompt = build_reconstruction_prompt(
        hidden_assignment=hidden_assignment,
        complete_assignment=complete_assignment,
        evidence=norm_evidence,
        messages=messages,
        bn=bn,
        metadata=metadata,
        relationship_notes=relationship_notes,
    )
    reconstruction, _ = client.complete_json(
        purpose="reconstruction",
        variable=None,
        prompt=reconstruction_prompt,
        response_model=ReconstructionResponse,
        semantic_validator=lambda model: validate_reconstruction_response(
            model,
            hidden_assignment=hidden_assignment,
            complete_assignment=complete_assignment,
            bn=bn,
            alias_map=alias_map,
        ),
    )

    if config.show_input_data:
        print("  ✓ Reconstruction validada")

    # --- Audit (LLM audita consistência semântica) ---
    if config.show_input_data:
        print("\n[INFER] Auditoria final...")

    audit_prompt = build_audit_prompt(
        complete_assignment=complete_assignment,
        evidence=norm_evidence,
        messages=messages,
        bn=bn,
        metadata=metadata,
        relationship_notes=relationship_notes,
    )
    audit, _ = client.complete_json(
        purpose="final_audit",
        variable=None,
        prompt=audit_prompt,
        response_model=AuditResponse,
        semantic_validator=lambda model: validate_audit_response(
            model,
            bn=bn,
            evidence=norm_evidence,
            complete_assignment=complete_assignment,
            alias_map=alias_map,
        ),
    )

    if config.show_input_data:
        print(f"  ✓ Auditoria concluída — accept={audit.accept}")
        if not audit.accept and getattr(audit, "repair", None):
            print(f"  ⚠ Reparo sugerido: {audit.repair}")

    # --- Audit repair ---
    if apply_audit_repair_enabled:
        complete_assignment = apply_audit_repair(
            audit,
            complete_assignment=complete_assignment,
            evidence=norm_evidence,
            bn=bn,
            alias_map=alias_map,
        )
        hidden_assignment = {
            var: val
            for var, val in complete_assignment.items()
            if var not in norm_evidence
        }

        if config.show_input_data:
            print(f"  ✓ Assignment final: {hidden_assignment}")

    # CPT gerada pelo LLM (para logging / análise downstream)
    llm_cpt: dict[str, list] = {
        var: [
            {
                "context": row.context,
                "selected_value": row.selected_value,
                "confidence": row.confidence,
                "rationale": row.rationale,
            }
            for row in msg.rows
        ]
        for var, msg in messages.items()
        if var not in norm_evidence
    }

    return hidden_assignment, selected_confidence, llm_cpt
