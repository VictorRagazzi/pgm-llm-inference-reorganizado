"""Build bucket prompts from prepared network context."""

from __future__ import annotations

import json

from ...models import BayesianNetwork
from ..graph import evidence_context_payload, local_relationship_notes
from ..pre_compile_phase.briefing import variable_payload, inference_guidance
from ..types import BriefingResponse, BucketSpec, VariableMetadata
from .domain_scoring import build_state_code_map


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
    state_codes = build_state_code_map(tuple(candidate_states))

    payload = {
        "task": "semantic_bucket_argmax",
        "variable": variable_payload(variable, bn, metadata, children),
        "focus_evidence_context": evidence_context_payload(variable, bn, evidence),
        "local_relationship_notes": local_relationship_notes(variable, bn, relationship_notes),
        "is_evidence": bucket.is_evidence,
        "observed_value": bucket.observed_value,
        "candidate_states": candidate_states,
        "candidate_state_codes": state_codes if not bucket.is_evidence else None,
        "local_family_scope": list(bucket.local_scope),
        "separator_variables": [
            variable_payload(item, bn, metadata, children) for item in bucket.separator
        ],
        "separator_evidence_contexts": {
            item: evidence_context_payload(item, bn, evidence) for item in bucket.separator
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
        codes_str = " | ".join(state_codes)
        schema = {
            "variable": variable,
            "decisions": [
                {
                    "context": "JSON object with variable_id: state pairs",
                    "selected_value": codes_str,
                    "confidence": "high | medium | low",
                    "rationale": "short reason",
                }
            ],
        }
        instruction = (
            f"This variable is hidden. Its legal states are: {states_str}. "
            f"Use candidate_state_codes and return only its code in selected_value "
            f"({codes_str}), never the full state name. "
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
        "- For hidden variables, selected_value must be one exact code from "
        "candidate_state_codes.\n"
        "- Keep rationales concise; do not reveal hidden step-by-step reasoning.\n"
        "- Return valid JSON only.\n\n"
        f"Required JSON shape:\n{json.dumps(schema, indent=2)}\n\n"
        f"Bucket data:\n{json.dumps(payload, indent=2)}"
    )
