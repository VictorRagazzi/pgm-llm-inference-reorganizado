"""
mpe/prompt_builders.py
======================
Construtores de prompts do pipeline LLM-MPE.

Cada função recebe dados estruturados e retorna uma string de prompt
pronta para enviar ao LLM. Sem side effects, sem chamadas de rede.
"""

from __future__ import annotations

import itertools
from typing import Any

from .graph import (
    evidence_context_payload,
    local_relationship_notes,
    relationship_notes_payload,
    topological_order,
)
from .io import json_dumps
from .state_semantics import resolve_state_meanings, states_pipe, network_state_meanings
from ..models import BayesianNetwork
from .types import (
    BriefingResponse,
    BucketSpec,
    SemanticMessage,
    VariableMetadata,
)


# ---------------------------------------------------------------------------
# Payload helpers (serialização para o corpo do prompt)
# ---------------------------------------------------------------------------

def variable_payload(
    variable: str,
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    children: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    item = metadata.get(variable, VariableMetadata())
    return {
        "id": variable,
        "display_name": item.display_name or variable,
        "description": item.description,
        "expert_note": item.expert_note,
        "aliases": list(item.aliases),
        "states": list(bn.variables[variable].states),
        "parents": list(bn.parents[variable]),
        "children": list(children[variable]),
    }


def inference_guidance(metadata: dict[str, VariableMetadata]) -> list[str]:
    guidance = [
        (
            "Treat this as a learned Bayesian network over discretized measurements, "
            "not as a deterministic diagram."
        ),
        (
            "Edges indicate probabilistic dependence. A parent and child do not "
            "need to have the same state."
        ),
        (
            "Do not use monotonic state matching as the default rule. A low or "
            "extreme child state can be caused by suppressive or dampening "
            "regulation from moderate or high parent contexts."
        ),
        (
            "Intermediate states are real baseline or modulatory states. They are "
            "often preferable to extreme states when the evidence points to "
            "regulated baseline activity."
        ),
        (
            "A moderate parent can suppress or dampen a child to a low state. "
            "Do not assume the child copies the parent state unless the variable "
            "notes support that interpretation."
        ),
        (
            "For evidence variables, judge each context contrastively: ask which "
            "parent context best explains the fixed observation, not which context "
            "has the same state label."
        ),
        (
            "For root variables, choose the state that best explains all downstream "
            "messages. Do not default roots to the lowest state without evidence pressure."
        ),
        (
            "If a variable is in a graph component with no evidence, do not import "
            "semantic pressure from an unrelated component. Use only local baseline "
            "metadata for that component."
        ),
        (
            "For each local decision, prioritize the Markov blanket: parents, "
            "children, and co-parents of children. Evidence outside the Markov "
            "blanket should arrive through incoming messages."
        ),
    ]
    if any(item.expert_note for item in metadata.values()):
        guidance.append(
            "Use variable expert notes as qualitative domain knowledge, but never "
            "invent numeric probabilities."
        )
    return guidance


def network_payload(
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    evidence: dict[str, str],
    relationship_notes: dict[str, tuple[str, ...]] | None,
) -> dict[str, Any]:
    children = bn.children_map()
    topo = topological_order(bn)

    return {
        "network_name": bn.name,
        "variables": [
            variable_payload(variable, bn, metadata, children) for variable in topo
        ],
        "families": [
            {"child": variable, "parents": list(bn.parents[variable])}
            for variable in topo
        ],
        "qualitative_relationship_notes": relationship_notes_payload(bn, relationship_notes),
        "evidence": evidence,
        "evidence_contexts": {
            variable: evidence_context_payload(variable, bn, evidence)
            for variable in topo
        },
        "qualitative_inference_guidance": inference_guidance(metadata),
        "state_meanings": network_state_meanings(bn, metadata),
    }


def generate_context_rows(
    variables: tuple[str, ...],
    bn: BayesianNetwork,
    evidence: dict[str, str],
    max_context_rows: int,
) -> list[dict[str, str]]:
    if not variables:
        return [{}]

    domains: list[tuple[str, ...]] = []
    for variable in variables:
        if variable in evidence:
            domains.append((evidence[variable],))
        else:
            domains.append(bn.variables[variable].states)

    total_rows = 1
    for domain in domains:
        total_rows *= len(domain)
    if total_rows > max_context_rows:
        raise ValueError(
            f"Bucket context would contain {total_rows} rows for variables "
            f"{variables}, exceeding max_context_rows={max_context_rows}."
        )

    return [
        dict(zip(variables, values, strict=True))
        for values in itertools.product(*domains)
    ]


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_intermediate_state_instruction(
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata] | None,
) -> str:
    """
    Gera a instrução sobre estado intermediário adaptada ao domínio real.
    Omite a instrução se a rede não tiver estados intermediários (ex: binária).
    """
    intermediate_labels: set[str] = set()
    for var, variable in bn.variables.items():
        if len(variable.states) < 3:
            continue
        meanings = resolve_state_meanings(var, variable.states, metadata)
        for state, desc in meanings.items():
            if "baseline" in desc or "intermediate" in desc or "average" in desc.lower():
                intermediate_labels.add(state)

    if not intermediate_labels:
        return ""

    canonical = sorted(intermediate_labels)[0]
    return (
        f"- Treat {canonical} (and equivalent intermediate states) as a concrete "
        "baseline/modulatory state, not as a weak fallback.\n"
    )


def build_network_briefing_prompt(
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    evidence: dict[str, str],
    relationship_notes: dict[str, tuple[str, ...]] | None = None,
) -> str:
    payload = network_payload(bn, metadata, evidence, relationship_notes)
    intermediate_instruction = _build_intermediate_state_instruction(bn, metadata)

    return (
        "We are approximating most probable explanation inference in a Bayesian "
        "network using semantic reasoning instead of numeric CPT values.\n\n"
        "The evidence below is fixed. Do not choose alternate values for evidence "
        "variables. Numeric CPT probabilities are intentionally hidden.\n\n"
        "Important inference stance:\n"
        "- This is a learned probabilistic signaling BN, not a deterministic "
        "activation diagram.\n"
        "- Avoid monotonic state matching. A child's state does not directly imply "
        "the same state in parents when a regulator can suppress or dampen a readout.\n"
        f"{intermediate_instruction}"
        "- Use fixed evidence to reason backward through parents and sideways "
        "through shared ancestors.\n"
        "- Legal values for each variable are listed in network.state_meanings. "
        "Always choose from those exact strings.\n\n"
        "Return JSON with:"
        "- network_summary: STRING (single paragraph, no objects)"
        "- important_dependencies: list of strings"
        "- reasoning_rules: list of strings"
        "reasoning_rules.\n\n"
        f"Network data:\n{json_dumps(payload)}"
    )


def build_bucket_prompt(
    bucket: BucketSpec,
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    briefing: BriefingResponse | None,
    evidence: dict[str, str],
    relationship_notes: dict[str, tuple[str, ...]] | None = None,
) -> str:
    children = bn.children_map()
    variable = bucket.variable
    relationship_notes = relationship_notes or {}

    candidate_states = list(bn.variables[variable].states)
    states_str = " | ".join(candidate_states)

    payload = {
        "task": "semantic_bucket_argmax",
        "variable": variable_payload(variable, bn, metadata, children),
        "focus_evidence_context": evidence_context_payload(variable, bn, evidence),
        "local_relationship_notes": local_relationship_notes(
            variable, bn, relationship_notes
        ),
        "is_evidence": bucket.is_evidence,
        "observed_value": bucket.observed_value,
        "candidate_states": candidate_states,
        "local_family_scope": list(bucket.local_scope),
        "separator_variables": [
            variable_payload(item, bn, metadata, children) for item in bucket.separator
        ],
        "separator_evidence_contexts": {
            item: evidence_context_payload(item, bn, evidence)
            for item in bucket.separator
        },
        "context_rows": bucket.context_rows,
        "incoming_semantic_messages": [
            message.model_dump(mode="json") for message in bucket.incoming_messages
        ],
        "network_briefing": briefing.model_dump(mode="json") if briefing else None,
        "qualitative_inference_guidance": inference_guidance(metadata),
    }

    if bucket.is_evidence:
        schema = {
            "variable": variable,
            "observed_value": bucket.observed_value,
            "messages": [
                {
                    "context": "JSON object with variable_id: state pairs",
                    "compatibility": "strong | medium | weak",
                    "rationale": "short reason",
                }
            ],
        }
        instruction = (
            "This variable is observed evidence. Keep the observed value exactly. "
            "For every context row, summarize how compatible that context is with "
            "the observed evidence and its incoming messages. Compatibility is "
            "contrastive: mark the contexts that best explain the evidence as "
            "strong, even when their state labels do not simply match the observed "
            "state."
        )
    else:
        schema = {
            "variable": variable,
            "decisions": [
                {
                    "context": "JSON object with variable_id: state pairs",
                    "selected_value": states_str,
                    "confidence": "high | medium | low",
                    "rationale": "short reason",
                }
            ],
        }
        instruction = (
            f"This variable is hidden. Its legal states are: {states_str}. "
            "For every context row, choose the single state that makes this bucket "
            "most jointly plausible. Use the graph, labels, evidence pressure, and "
            "incoming semantic messages. Compare all candidate states internally "
            "before selecting. Do not invent probabilities."
        )

    return (
        "Perform one local MPE-style max step for this Bayesian-network bucket.\n"
        f"{instruction}\n\n"
        "Expert cautions:\n"
        "- First inspect focus_evidence_context. If component_has_evidence is "
        "false, do not use evidence from another graph component.\n"
        "- Focus on the variable's Markov blanket: parents, children, and "
        "co-parents. Treat other evidence as already summarized by incoming "
        "messages.\n"
        "- Use local_relationship_notes as neighborhood-level field expertise. "
        "They describe reusable mechanisms, not case-specific answers.\n"
        "- For each row, make a compact internal review: first score the focus "
        "family, then check child/co-parent messages, then choose the state that "
        "best balances both.\n"
        "- Incoming messages with evidence_driven=false are weak priors from "
        "unconstrained hidden variables. Use them only as tie-breakers; they must "
        "not outweigh fixed evidence or evidence_driven=true messages.\n"
        "- If this hidden variable has no direct Markov-blanket evidence and no "
        "evidence-driven incoming messages, keep confidence low unless the local "
        "mechanism is exceptionally distinctive.\n"
        "- Do not default to the lowest state merely because an observed descendant "
        "has a low or extreme value.\n"
        "- Do not choose a middle or neutral state as a compromise; select it only "
        "when it is genuinely the best explanation given the evidence.\n"
        "- For root variables, do not soften a coherent extreme state just to appear "
        "cautious — follow the evidence from child messages.\n"
        "- If an expert note says a parent dampens a child, a lower child state "
        "may be more plausible under moderate parent contexts.\n"
        "- Prefer the state that best explains all incoming child messages and "
        "fixed evidence together.\n"
        "- A learned BN can encode suppression, buffering, and non-monotonic "
        "effects.\n\n"
        "Rules:\n"
        "- Return one row for every context row and no extra rows.\n"
        "- Use BIF variable IDs in contexts.\n"
        f"- Use only legal states from candidate_states: {states_str}.\n"
        "- Keep rationales concise; do not reveal hidden step-by-step reasoning.\n"
        "- Return valid JSON only.\n\n"
        f"Required JSON shape:\n{json_dumps(schema)}\n\n"
        f"Bucket data:\n{json_dumps(payload)}"
    )


def build_reconstruction_prompt(
    hidden_assignment: dict[str, str],
    complete_assignment: dict[str, str],
    evidence: dict[str, str],
    messages: dict[str, SemanticMessage],
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    relationship_notes: dict[str, tuple[str, ...]] | None = None,
) -> str:
    # Só a row selecionada de cada hidden var — o resto é ruído para o LLM
    selected_rows = {}
    for var, selected_value in hidden_assignment.items():
        msg = messages.get(var)
        if msg is None:
            continue
        matched = next(
            (r for r in msg.rows if r.selected_value == selected_value),
            msg.rows[0] if msg.rows else None,
        )
        if matched:
            selected_rows[var] = {
                "context": matched.context,
                "selected_value": matched.selected_value,
                "confidence": matched.confidence,
                "rationale": matched.rationale,
            }

    payload = {
        "hidden_assignment": hidden_assignment,
        "complete_assignment": complete_assignment,
        "fixed_evidence": evidence,
        "selected_backpointers": selected_rows,  # só hidden vars, só row escolhida
    }
    schema = {
        "hidden_assignment": hidden_assignment,
        "complete_assignment": complete_assignment,
        "explanation": ["short explanation item"],
    }
    return (
        "Explain the reconstructed MPE assignment from the semantic backpointers. "
        "Do not change the assignment. Return JSON only.\n\n"
        f"Required JSON shape:\n{json_dumps(schema)}\n\n"
        f"Data:\n{json_dumps(payload)}"
    )


def _markov_blanket(var: str, bn: BayesianNetwork) -> set[str]:
    """Pais + filhos + co-pais (pais dos filhos)."""
    parents_map = bn.parents
    children_map = bn.children_map()

    parents = set(parents_map.get(var, ()))
    children = set(children_map.get(var, ()))
    co_parents = {
        p
        for child in children
        for p in parents_map.get(child, ())
        if p != var
    }
    return parents | children | co_parents


def build_audit_prompt(
    complete_assignment: dict[str, str],
    evidence: dict[str, str],
    messages: dict[str, SemanticMessage],
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    relationship_notes: dict[str, tuple[str, ...]] | None = None,
) -> str:
    # Para cada hidden var: só a row selecionada + Markov blanket local
    non_evidence_vars = [v for v in bn.variables if v not in evidence]
    
    audit_entries = {}
    for var in non_evidence_vars:
        blanket = _markov_blanket(var, bn)
        msg = messages.get(var)
        selected_value = complete_assignment.get(var)
        matched_row = None
        if msg and selected_value:
            matched_row = next(
                (r for r in msg.rows if r.selected_value == selected_value),
                msg.rows[0] if msg.rows else None,
            )

        audit_entries[var] = {
            "assigned_value": selected_value,
            "legal_states": list(bn.variables[var].states),
            "markov_blanket_assignment": {
                neighbor: complete_assignment.get(neighbor)
                for neighbor in blanket
            },
            "selected_row": {
                "context": matched_row.context,
                "selected_value": matched_row.selected_value,
                "confidence": matched_row.confidence,
                "rationale": matched_row.rationale,
            } if matched_row else None,
            "relationship_notes": list((relationship_notes or {}).get(var, ())),
        }

    example_var = non_evidence_vars[0] if non_evidence_vars else next(iter(bn.variables))
    example_states = states_pipe(bn.variables[example_var].states)
    repair_example = (
        f'{{"variable": "BIF_ID", "value": "{example_states}", "reason": "..."}}'
    )

    payload = {
        "complete_assignment": complete_assignment,
        "fixed_evidence": evidence,
        "audit_entries": audit_entries,  # grafo local, sem replicar tudo
    }
    schema = {"accept": True, "repair": None, "reason": "short audit reason"}

    return (
        "Audit the complete assignment for semantic consistency with the fixed "
        "evidence and DAG. Re-evaluate the local Markov blankets and qualitative "
        "relationship notes directly; do not accept an assignment merely because "
        "it matches earlier decision messages. Evidence variables cannot be "
        "repaired. If the assignment is acceptable, set accept=true and "
        "repair=null. Do not set accept=false unless the proposed repair changes "
        "one non-evidence variable to a different legal value. If there is one "
        "clear inconsistency, set accept=false and propose exactly one repair as "
        f"{repair_example}. "
        "The value field must be one of the legal_states listed in the "
        "audit_entries for the chosen variable. "
        "Return JSON only.\n\n"
        f"Required JSON shape:\n{json_dumps(schema)}\n\n"
        f"Data:\n{json_dumps(payload)}"
    )