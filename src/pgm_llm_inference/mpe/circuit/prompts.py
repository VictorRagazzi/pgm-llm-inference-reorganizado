"""
mpe/circuit/prompts.py
========================
Prompt de elicitação local (Fase 1 do Part 2, Secao 5 do design doc):
pede ao LLM um ranking COMPLETO dos estados de UMA variável, para CADA
configuração de seus PRÓPRIOS pais — nunca menciona evidência, nunca
menciona filhos ou outras variáveis do circuito. É essa independência
que torna o fator reutilizável para qualquer query futura.
"""

from __future__ import annotations

from typing import Any

from ..io import json_dumps
from ...models import BayesianNetwork
from ..prompt_builders import (
    inference_guidance,
    local_relationship_notes,
    variable_payload,
)
from ..state_semantics import network_state_meanings
from ..types import BriefingResponse, VariableMetadata


def build_factor_prompt(
    variable: str,
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    relationship_notes: dict[str, tuple[str, ...]],
    context_rows: list[dict[str, str]],
    briefing: BriefingResponse | None = None,
) -> str:
    children = bn.children_map()
    states = list(bn.variables[variable].states)

    payload: dict[str, Any] = {
        "task": "semantic_local_factor_ranking",
        "network_briefing": briefing.model_dump(mode="json") if briefing else None,
        "variable": variable_payload(variable, bn, metadata, children),
        "candidate_states": states,
        "state_meanings": network_state_meanings(bn, metadata).get(variable, {}),
        "parent_context_rows": context_rows,
        # Só notas da própria família — nada de filhos, evidência ou
        # mensagens de outros buckets (diferente do Part 1).
        "local_relationship_notes": [
            note
            for note in local_relationship_notes(variable, bn, relationship_notes)
            if note.get("role") == "focus_family"
        ],
        "qualitative_inference_guidance": inference_guidance(metadata),
    }

    instruction = (
        "For EACH row in parent_context_rows, rank ALL of candidate_states "
        "from MOST plausible to LEAST plausible for `variable`, given ONLY "
        "that parent configuration. This is a local, evidence-free judgment "
        "about the variable's own conditional behavior in isolation — do "
        "NOT consider any evidence, any downstream query, or any other "
        "variable beyond the parents listed here. This factor will be "
        "reused for many different future queries, so it must not assume "
        "anything about what will be observed later.\n\n"
        "Return every state in `ranking` exactly once (a full permutation "
        "of candidate_states), most plausible first. If two states are "
        "genuinely indistinguishable given only the parents, keep a fixed "
        "but principled order and say so briefly in the rationale."
    )

    schema = {
        "variable": variable,
        "rows": [
            {
                "context": (
                    "JSON object with parent_id: state pairs, echoing one "
                    "row from parent_context_rows exactly"
                ),
                "ranking": "list with ALL candidate_states, most to least plausible",
                "rationale": "one short sentence",
            }
        ],
    }

    return (
        f"{instruction}\n\n"
        f"DATA:\n{json_dumps(payload)}\n\n"
        f"Respond with JSON matching this schema exactly, with one row per "
        f"requested entry in parent_context_rows:\n{json_dumps(schema)}"
    )