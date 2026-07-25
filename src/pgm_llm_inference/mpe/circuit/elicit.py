"""
mpe/circuit/elicit.py
========================
Fase 1 do Part 2 (Secao 5 do design doc) — elicitação de fatores
semânticos locais. Uma ou mais chamadas LLM por variável (chunked se a
tabela de contexto for grande), SEMPRE com evidence={} — o fator
resultante é uma função só da família {variavel} U parents(variavel) e é
reutilizável para qualquer evidência futura, sem nova chamada LLM.

Este módulo NÃO é importado por compile.py's contraparte estrutural
(order.py) nem por evaluate.py — mantém a regra de dependência do design
doc (evaluate/compile não dependem de elicit em runtime de query).
"""

from __future__ import annotations

from ..client import LLMJsonClient
from ..io import resolve_variable
from ...models import BayesianNetwork
from ..prompt_builders import generate_context_rows
from ..types import PromptTrace, VariableMetadata
from .prompts import build_factor_prompt
from .scale import ranking_to_scores
from .types import FactorElicitationResponse, ScoredRow, SemanticFactor, context_key


def _validate_ranking(
    response: FactorElicitationResponse,
    variable: str,
    domain: tuple[str, ...],
    expected_rows: list[dict[str, str]],
    parents: tuple[str, ...],
    alias_map: dict[str, str],
) -> None:
    resolved = resolve_variable(response.variable, alias_map)
    if resolved != variable:
        raise ValueError(
            f"Response variable {response.variable!r} does not match "
            f"expected {variable!r}."
        )

    domain_set = set(domain)
    seen_contexts: set[tuple[str, ...]] = set()
    for row in response.rows:
        if set(row.ranking) != domain_set or len(row.ranking) != len(domain):
            raise ValueError(
                f"Ranking for {variable} must be a permutation of {domain}; "
                f"got {row.ranking!r}."
            )
        seen_contexts.add(context_key(row.context, parents))

    expected_contexts = {context_key(row, parents) for row in expected_rows}
    if seen_contexts != expected_contexts:
        raise ValueError(
            f"Response for {variable} must cover exactly the requested "
            f"parent contexts. Mismatch: "
            f"{expected_contexts.symmetric_difference(seen_contexts)}."
        )


def elicit_semantic_factor(
    *,
    variable: str,
    bn: BayesianNetwork,
    metadata: dict[str, VariableMetadata],
    relationship_notes: dict[str, tuple[str, ...]],
    alias_map: dict[str, str],
    client: LLMJsonClient,
    max_rows_per_call: int = 64,
    max_context_rows: int = 4096,
) -> tuple[SemanticFactor, list[PromptTrace]]:
    """
    Elicita o fator local de `variable`: ranking completo de seus estados
    para cada configuração de seus PRÓPRIOS pais. evidence={} sempre — o
    resultado não depende de nenhuma query específica.
    """
    parents = tuple(bn.parents[variable])
    domain = tuple(bn.variables[variable].states)

    all_context_rows = generate_context_rows(
        parents, bn=bn, evidence={}, max_context_rows=max_context_rows
    )

    traces: list[PromptTrace] = []
    rows: list[ScoredRow] = []

    for start in range(0, len(all_context_rows), max_rows_per_call):
        chunk = all_context_rows[start : start + max_rows_per_call]
        prompt = build_factor_prompt(variable, bn, metadata, relationship_notes, chunk)

        response, trace = client.complete_json(
            purpose="local_factor_ranking",
            variable=variable,
            prompt=prompt,
            response_model=FactorElicitationResponse,
            semantic_validator=lambda model, _chunk=chunk: _validate_ranking(
                model, variable, domain, _chunk, parents, alias_map
            ),
        )
        traces.append(trace)

        row_by_context = {
            context_key(row.context, parents): row for row in response.rows
        }
        for expected in chunk:
            key = context_key(expected, parents)
            row = row_by_context[key]
            rows.append(
                ScoredRow(
                    context=dict(expected),
                    scores=ranking_to_scores(row.ranking),
                    rationale=row.rationale,
                )
            )

    factor = SemanticFactor(variable=variable, parents=parents, states=domain, rows=rows)
    return factor, traces
