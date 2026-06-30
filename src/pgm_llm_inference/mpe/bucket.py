"""
mpe/bucket.py
=============
Lógica de bucket elimination semântico.

Responsável por:
- Construir BucketSpec a partir do estado corrente da eliminação
- Verificar pressão de evidência sobre um bucket
- Normalizar contextos recebidos do LLM (aliases, case)
- Converter BucketResponse → SemanticMessage (com validação completa)
"""

from __future__ import annotations

from .graph import evidence_context_payload
from .io import resolve_variable
from .prompt_builders import generate_context_rows
from ..models import BayesianNetwork
from .types import (
    BucketResponse,
    BucketSpec,
    ContextDecision,
    ContextEvidenceMessage,
    MessageRow,
    SemanticMessage,
    DomainScores,
)


# ---------------------------------------------------------------------------
# Construção do BucketSpec
# ---------------------------------------------------------------------------

def build_bucket_spec(
    variable: str,
    bn: BayesianNetwork,
    evidence: dict[str, str],
    active_messages: list[SemanticMessage],
    order_index: dict[str, int],
    max_context_rows: int,
) -> BucketSpec:
    incoming_messages = [
        message for message in active_messages if variable in message.scope
    ]

    bucket_scope = {variable, *bn.parents[variable]}
    for message in incoming_messages:
        bucket_scope.update(message.scope)

    separator = tuple(
        sorted(bucket_scope - {variable}, key=lambda item: order_index[item])
    )
    context_rows = generate_context_rows(
        separator,
        bn=bn,
        evidence=evidence,
        max_context_rows=max_context_rows,
    )

    return BucketSpec(
        variable=variable,
        is_evidence=variable in evidence,
        observed_value=evidence.get(variable),
        local_scope=tuple(
            sorted({variable, *bn.parents[variable]}, key=lambda item: order_index[item])
        ),
        separator=separator,
        context_rows=context_rows,
        incoming_messages=incoming_messages,
    )


# ---------------------------------------------------------------------------
# Pressão de evidência
# ---------------------------------------------------------------------------

def bucket_has_evidence_pressure(
    bucket: BucketSpec,
    bn: BayesianNetwork,
    evidence: dict[str, str],
) -> bool:
    context = evidence_context_payload(bucket.variable, bn, evidence)
    return (
        bucket.is_evidence
        or bool(context["markov_blanket_evidence"])
        or any(message.evidence_driven for message in bucket.incoming_messages)
    )


# ---------------------------------------------------------------------------
# Normalização de contexto
# ---------------------------------------------------------------------------

def normalize_context(
    raw_context: dict[str, str],
    expected_scope: tuple[str, ...],
    bn: BayesianNetwork,
    alias_map: dict[str, str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    expected_set = set(expected_scope)

    for raw_variable, raw_state in raw_context.items():
        variable = resolve_variable(raw_variable, alias_map)

        if variable not in expected_set:
            raise ValueError(
                f"Unexpected context variable {raw_variable!r}; "
                f"expected {expected_scope}."
            )

        domain = bn.variables[variable].states
        domain_upper = {state.upper(): state for state in domain}
        canonical = domain_upper.get(raw_state.strip().upper())

        if canonical is None:
            allowed = ", ".join(domain)
            raise ValueError(
                f"Illegal state {raw_state!r} for {variable}. "
                f"Allowed states: {allowed}."
            )

        normalized[variable] = canonical

    missing = expected_set - set(normalized)
    if missing:
        raise ValueError(f"Context is missing variables: {sorted(missing)}.")

    return {variable: normalized[variable] for variable in expected_scope}


def context_key(context: dict[str, str], scope: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(context[variable] for variable in scope)


def contexts_match(
    rows: list[dict[str, str]],
    expected_rows: list[dict[str, str]],
    scope: tuple[str, ...],
) -> bool:
    if len(rows) != len(expected_rows):
        return False
    return {context_key(row, scope) for row in rows} == {
        context_key(row, scope) for row in expected_rows
    }


# ---------------------------------------------------------------------------
# BucketResponse → SemanticMessage
# ---------------------------------------------------------------------------

def semantic_message_from_response(
    response: BucketResponse,
    bucket: BucketSpec,
    bn: BayesianNetwork,
    alias_map: dict[str, str],
    evidence: dict[str, str],
    domain_scores_list: list[DomainScores] = [],
) -> SemanticMessage:
    response_variable = resolve_variable(response.variable, alias_map)
    if response_variable != bucket.variable:
        raise ValueError(
            f"Response variable {response.variable!r} does not match bucket "
            f"variable {bucket.variable!r}."
        )

    domain = bn.variables[bucket.variable].states
    domain_upper: dict[str, str] = {s.upper(): s for s in domain}

    if bucket.is_evidence:
        raw_observed = (response.observed_value or "").strip()
        observed_canonical = domain_upper.get(raw_observed.upper())
        if observed_canonical is None:
            raise ValueError(
                f"Evidence response for {bucket.variable} must keep observed value "
                f"{bucket.observed_value!r}; got {response.observed_value!r}."
            )
        if observed_canonical != bucket.observed_value:
            raise ValueError(
                f"Evidence response for {bucket.variable} must keep observed value "
                f"{bucket.observed_value!r}; got {observed_canonical!r}."
            )
        if not response.messages:
            raise ValueError(f"Evidence response for {bucket.variable} has no messages.")

        normalized_messages: list[ContextEvidenceMessage] = []
        for row in response.messages:
            normalized_messages.append(
                ContextEvidenceMessage(
                    context=normalize_context(
                        row.context,
                        expected_scope=bucket.separator,
                        bn=bn,
                        alias_map=alias_map,
                    ),
                    compatibility=row.compatibility,
                    rationale=row.rationale,
                )
            )

        if not contexts_match(
            [row.context for row in normalized_messages],
            bucket.context_rows,
            bucket.separator,
        ):
            raise ValueError(
                f"Evidence response for {bucket.variable} must include exactly one "
                "message for each requested context row."
            )

        return SemanticMessage(
            source_variable=bucket.variable,
            scope=bucket.separator,
            is_evidence=True,
            evidence_driven=True,
            rows=[
                MessageRow(
                    context=row.context,
                    observed_value=bucket.observed_value,
                    compatibility=row.compatibility,
                    rationale=row.rationale,
                )
                for row in sorted(
                    normalized_messages,
                    key=lambda item: context_key(item.context, bucket.separator),
                )
            ],
        )

    # Variável hidden
    if not response.decisions:
        raise ValueError(f"Hidden response for {bucket.variable} has no decisions.")

    if domain_scores_list and len(domain_scores_list) != len(response.decisions):
        raise ValueError(
            f"domain_scores_list length ({len(domain_scores_list)}) does not match "
            f"decisions length ({len(response.decisions)}) for {bucket.variable}."
        )

    normalized_decisions: list[ContextDecision] = []
    decision_scores: list[DomainScores | None] = []  # paralelo a normalized_decisions

    for i, row in enumerate(response.decisions):
        canonical = domain_upper.get(row.selected_value.strip().upper())
        if canonical is None:
            allowed = ", ".join(domain)
            raise ValueError(
                f"Illegal selected value {row.selected_value!r} for "
                f"{bucket.variable}. Allowed states: {allowed}."
            )
        normalized_decisions.append(
            ContextDecision(
                context=normalize_context(
                    row.context,
                    expected_scope=bucket.separator,
                    bn=bn,
                    alias_map=alias_map,
                ),
                selected_value=canonical,
                confidence=row.confidence,
                rationale=row.rationale,
            )
        )
        decision_scores.append(domain_scores_list[i] if domain_scores_list else None)


    if not contexts_match(
        [row.context for row in normalized_decisions],
        bucket.context_rows,
        bucket.separator,
    ):
        raise ValueError(
            f"Hidden response for {bucket.variable} must include exactly one "
            "decision for each requested context row."
        )

    paired = sorted(
        zip(normalized_decisions, decision_scores),
        key=lambda item: context_key(item[0].context, bucket.separator),
    )
    
    return SemanticMessage(
        source_variable=bucket.variable,
        scope=bucket.separator,
        is_evidence=False,
        evidence_driven=bucket_has_evidence_pressure(bucket, bn, evidence),
        rows=[
            MessageRow(
                context=decision.context,
                selected_value=decision.selected_value,
                confidence=decision.confidence,
                rationale=decision.rationale,
                domain_scores=score,
            )
            for decision, score in paired
        ],
    )
