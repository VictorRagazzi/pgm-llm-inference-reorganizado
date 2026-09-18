"""Construct, compile, validate, and propagate semantic bucket messages."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from ...core.config import InferenceConfig
from ...models import BayesianNetwork
from ..graph import evidence_context_payload
from ..normalization import context_key, normalize_context, resolve_variable
from ..types import (
    BriefingResponse,
    BucketResponse,
    BucketSpec,
    ContextDecision,
    ContextEvidenceMessage,
    MessageRow,
    PromptTrace,
    SemanticMessage,
    VariableMetadata,
)
from .prompts import build_bucket_prompt

if TYPE_CHECKING:
    from .client import LLMJsonClient


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

    return [dict(zip(variables, values, strict=True)) for values in itertools.product(*domains)]


def build_bucket_spec(
    variable: str,
    bn: BayesianNetwork,
    evidence: dict[str, str],
    active_messages: list[SemanticMessage],
    order_index: dict[str, int],
    max_context_rows: int,
) -> BucketSpec:
    incoming_messages = [message for message in active_messages if variable in message.scope]

    bucket_scope = {variable, *bn.parents[variable]}
    for message in incoming_messages:
        bucket_scope.update(message.scope)

    separator = tuple(sorted(bucket_scope - {variable}, key=lambda item: order_index[item]))
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


def semantic_message_from_response(
    response: BucketResponse,
    bucket: BucketSpec,
    bn: BayesianNetwork,
    alias_map: dict[str, str],
    evidence: dict[str, str],
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

    normalized_decisions: list[ContextDecision] = []
    for row in response.decisions:
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
                domain_scores=row.domain_scores,
                token_scores=row.token_scores,
            )
        )

    if not contexts_match(
        [row.context for row in normalized_decisions],
        bucket.context_rows,
        bucket.separator,
    ):
        raise ValueError(
            f"Hidden response for {bucket.variable} must include exactly one "
            "decision for each requested context row."
        )

    return SemanticMessage(
        source_variable=bucket.variable,
        scope=bucket.separator,
        is_evidence=False,
        evidence_driven=bucket_has_evidence_pressure(bucket, bn, evidence),
        rows=[
            MessageRow(
                context=row.context,
                selected_value=row.selected_value,
                confidence=row.confidence,
                rationale=row.rationale,
                domain_scores=row.domain_scores,
                token_scores=row.token_scores,
            )
            for row in sorted(
                normalized_decisions,
                key=lambda item: context_key(item.context, bucket.separator),
            )
        ],
    )


def split_bucket_by_context_rows(
    bucket: BucketSpec,
    max_context_rows_per_call: int,
) -> list[BucketSpec]:
    """
    Divide bucket.context_rows em lotes de até max_context_rows_per_call
    linhas, retornando uma cópia do BucketSpec por lote (mesmo variable,
    scope, separator e incoming_messages — só context_rows muda).

    Usado quando o bucket tem mais linhas de contexto do que o permitido
    numa única chamada LLM. Se não precisar dividir, retorna [bucket].
    """
    rows = bucket.context_rows

    if max_context_rows_per_call <= 0:
        raise ValueError("max_context_rows_per_call deve ser positivo.")
    if len(rows) <= max_context_rows_per_call:
        return [bucket]

    batches = [
        rows[i : i + max_context_rows_per_call]
        for i in range(0, len(rows), max_context_rows_per_call)
    ]
    return [bucket.model_copy(update={"context_rows": batch}) for batch in batches]


def merge_bucket_responses(
    responses: list[BucketResponse],
    variable: str,
) -> BucketResponse:
    """
    Combina as respostas parciais de um bucket fatiado (uma chamada LLM
    por lote de context_rows) num único BucketResponse cobrindo todas as
    linhas do bucket completo. Chamado antes de semantic_message_from_response.
    """
    if not responses:
        raise ValueError(f"Nenhuma resposta para combinar no bucket {variable!r}.")

    observed_value = next(
        (r.observed_value for r in responses if r.observed_value is not None),
        None,
    )

    merged_decisions: list[ContextDecision] = []
    merged_messages: list[ContextEvidenceMessage] = []
    for r in responses:
        merged_decisions.extend(r.decisions)
        merged_messages.extend(r.messages)

    return BucketResponse(
        variable=variable,
        decisions=merged_decisions,
        observed_value=observed_value,
        messages=merged_messages,
    )


def compile_bucket_messages(
    *,
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    relationship_notes: dict[str, tuple[str, ...]],
    alias_map: dict[str, str],
    briefing: BriefingResponse,
    elimination_order: list[str],
    client: LLMJsonClient,
    config: InferenceConfig,
    max_context_rows: int,
    max_context_rows_per_call: int,
) -> tuple[dict[str, SemanticMessage], list[PromptTrace]]:
    """Compile all separator rows and preserve reverse-topological propagation."""
    compile_evidence: dict[str, str] = {}
    order_index = {variable: index for index, variable in enumerate(reversed(elimination_order))}
    total_vars = len(elimination_order)
    traces: list[PromptTrace] = []
    messages: dict[str, SemanticMessage] = {}
    active_messages: list[SemanticMessage] = []

    for var_index, variable in enumerate(elimination_order, start=1):
        if config.show_input_data:
            print(f"  [{var_index:02d}/{total_vars:02d}] Bucket: {variable}")

        bucket = build_bucket_spec(
            variable=variable,
            bn=bn,
            evidence=compile_evidence,
            active_messages=active_messages,
            order_index=order_index,
            max_context_rows=max_context_rows,
        )

        def _make_validator(bkt, am):
            def _validate(model: BucketResponse) -> None:
                semantic_message_from_response(
                    model, bkt, bn=bn, alias_map=am, evidence=compile_evidence
                )

            return _validate

        sub_buckets = split_bucket_by_context_rows(bucket, max_context_rows_per_call)

        if config.show_input_data and len(sub_buckets) > 1:
            print(
                f"         → {len(bucket.context_rows)} linhas de contexto, "
                f"dividido em {len(sub_buckets)} chamadas "
                f"(máx. {max_context_rows_per_call} linhas/chamada)"
            )

        responses: list[BucketResponse] = []
        for sub_bucket in sub_buckets:
            prompt = build_bucket_prompt(
                sub_bucket, bn, metadata, briefing, compile_evidence, relationship_notes
            )
            response, trace = client.complete_json(
                purpose="bucket_argmax",
                variable=variable,
                prompt=prompt,
                response_model=BucketResponse,
                semantic_validator=_make_validator(sub_bucket, alias_map),
                candidate_states=(
                    bn.variables[variable].states if not sub_bucket.is_evidence else None
                ),
            )
            traces.append(trace)
            responses.append(response)

        response = (
            responses[0] if len(responses) == 1 else merge_bucket_responses(responses, variable)
        )

        message = semantic_message_from_response(
            response, bucket, bn=bn, alias_map=alias_map, evidence=compile_evidence
        )

        if config.show_input_data:
            print(
                f"         → evidence_driven={message.evidence_driven} "
                f"| scope={message.scope} "
                f"| rows={len(message.rows)}"
            )

        if message.evidence_driven:
            raise RuntimeError(
                f"[COMPILE] Invariante violada: {variable} retornou "
                "evidence_driven=True com evidence={}. "
                "Verifique bucket_has_evidence_pressure."
            )

        messages[variable] = message

        # consumir as mensagens que entraram neste bucket e publicar a nova
        consumed_ids = {id(m) for m in bucket.incoming_messages}
        active_messages = [m for m in active_messages if id(m) not in consumed_ids]
        active_messages.append(message)

    if config.show_input_data:
        print(
            f"\n[COMPILE] ✓ Compilação concluída: "
            f"{len(messages)} mensagens, "
            f"{sum(len(m.rows) for m in messages.values())} rows totais"
        )

    return messages, traces
