"""Prepare the network context and request its global briefing."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

from ...core.config import InferenceConfig
from ...models import BayesianNetwork
from ..graph import evidence_context_payload, relationship_notes_payload, topological_order
from ..state_semantics import network_state_meanings, resolve_state_meanings
from ..types import BriefingResponse, PromptTrace, VariableMetadata

if TYPE_CHECKING:
    from ..compile_phase.client import LLMJsonClient


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
    relationship_notes: dict[str, list[str]],
) -> dict[str, Any]:
    children = bn.children_map()
    topo = topological_order(bn)

    return {
        "network_name": bn.name,
        "variables": [variable_payload(variable, bn, metadata, children) for variable in topo],
        "families": [
            {"child": variable, "parents": list(bn.parents[variable])} for variable in topo
        ],
        "qualitative_relationship_notes": relationship_notes_payload(bn, relationship_notes),
        "evidence": evidence,
        "evidence_contexts": {
            variable: evidence_context_payload(variable, bn, evidence) for variable in topo
        },
        "qualitative_inference_guidance": inference_guidance(metadata),
        "state_meanings": network_state_meanings(bn, metadata),
    }


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
    relationship_notes: dict[str, list[str]] | None = None,
) -> str:
    payload = network_payload(bn, metadata, evidence, relationship_notes or {})
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
        f"Network data:\n{json.dumps(payload, indent=2)}"
    )


def generate_network_briefing(
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    relationship_notes: dict[str, tuple[str, ...]],
    client: LLMJsonClient,
    config: InferenceConfig,
) -> tuple[BriefingResponse, PromptTrace]:
    """Generate the briefing with empty evidence, preserving its prompt and trace."""
    # Briefing gerado com evidence={}
    if config.show_input_data:
        print("\n[COMPILE] Gerando network briefing via LLM...")

    briefing_prompt = build_network_briefing_prompt(bn, metadata, {}, relationship_notes)
    briefing, trace = client.complete_json(
        purpose="network_briefing",
        variable=None,
        prompt=briefing_prompt,
        response_model=BriefingResponse,
    )
    return briefing, trace
