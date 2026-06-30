"""
mpe/reconstruction.py
=====================
Fase de reconstrução e validação do pipeline MPE.

reconstruct_assignment agora implementa bucket elimination completo
(upward pass em log-domain + downward pass via backpointers), em vez de
apenas ler o backpointer "local" de cada variável. Isso garante que
evidência observada se propague aritmeticamente por toda a árvore de
buckets antes de qualquer leitura de assignment — não só pai→filho.

- reconstruct_assignment: upward pass (combina fatores semânticos por
  bucket, aplica indicador de evidência, max-elimina hidden vars,
  encaminha mensagens) + downward pass (lê backpointers a partir da raiz).
- apply_audit_repair: aplica a sugestão de reparo do LLM se for válida
  (inalterado — continua sendo um passo de pós-processamento opcional
  sobre o assignment aritmético).
- validate_reconstruction_response: garante que o LLM não alterou o
  assignment na fase de reconstrução (inalterado).
- validate_audit_response: valida a resposta de auditoria antes de
  aplicá-la (inalterado).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from itertools import product as iter_product

from .bucket import context_key
from .io import normalize_assignment, resolve_variable
from ..models import BayesianNetwork
from .types import (
    AuditResponse,
    MessageRow,
    ReconstructionResponse,
    SemanticMessage,
)

# Piso numérico para log(prob) quando prob=0 ou quando uma combinação de
# separador não foi coberta nas rows compiladas (ex.: truncada por
# max_context_rows). Mantém o caminho "vivo" em vez de zerar tudo.
_EPS = 1e-12
_LOG_EPS = math.log(_EPS)


# ---------------------------------------------------------------------------
# Representação interna de fator (log-domain)
# ---------------------------------------------------------------------------

@dataclass
class _Factor:
    scope: tuple[str, ...]                    # variáveis no fator, em qualquer ordem fixa
    table: dict[tuple[str, ...], float]        # key = valores na ordem de `scope` -> log-score


def _domain_log_score(row: MessageRow, value: str) -> float:
    """log P(X=value | contexto da row), a partir de domain_scores (top_logprobs)."""
    if row.domain_scores is not None:
        prob = row.domain_scores.get(value)
        prob = min(max(prob, _EPS), 1.0)
        return math.log(prob)
    # Sem logprobs disponíveis (ex.: LM Studio sem suporte): cai pra indicador
    # duro sobre o selected_value já decidido na compilação.
    return 0.0 if value == row.selected_value else _LOG_EPS


def _build_variable_factor(
    variable: str, message: SemanticMessage, bn: BayesianNetwork
) -> _Factor:
    scope = (variable, *message.scope)
    domain = bn.variables[variable].states
    table: dict[tuple[str, ...], float] = {}
    for row in message.rows:
        context_tuple = tuple(row.context[v] for v in message.scope)
        for value in domain:
            table[(value, *context_tuple)] = _domain_log_score(row, value)
    return _Factor(scope=scope, table=table)


def _factor_value(factor: _Factor, assignment: dict[str, str]) -> float:
    key = tuple(assignment[v] for v in factor.scope)
    return factor.table.get(key, _LOG_EPS)


def _enumerate_separator(bn: BayesianNetwork, separator: tuple[str, ...]):
    if not separator:
        yield ()
        return
    domains = [bn.variables[v].states for v in separator]
    yield from iter_product(*domains)


def _combine_restrict(
    factors: list[_Factor],
    variable: str,
    observed_value: str,
    separator: tuple[str, ...],
    bn: BayesianNetwork,
) -> dict[tuple[str, ...], float]:
    """Bucket de variável de evidência: soma os fatores com X fixo (indicador)."""
    table: dict[tuple[str, ...], float] = {}
    for combo in _enumerate_separator(bn, separator):
        assignment = dict(zip(separator, combo))
        assignment[variable] = observed_value
        table[combo] = sum(_factor_value(f, assignment) for f in factors)
    return table


def _combine_eliminate(
    factors: list[_Factor],
    variable: str,
    separator: tuple[str, ...],
    bn: BayesianNetwork,
) -> tuple[dict[tuple[str, ...], float], dict[tuple[str, ...], str]]:
    """Bucket de variável hidden: max sobre os estados de X, com backpointer."""
    table: dict[tuple[str, ...], float] = {}
    backptr: dict[tuple[str, ...], str] = {}
    domain = bn.variables[variable].states
    for combo in _enumerate_separator(bn, separator):
        base_assignment = dict(zip(separator, combo))
        best_score: float | None = None
        best_value: str | None = None
        for value in domain:
            assignment = dict(base_assignment)
            assignment[variable] = value
            score = sum(_factor_value(f, assignment) for f in factors)
            if best_score is None or score > best_score:
                best_score = score
                best_value = value
        table[combo] = best_score  # type: ignore[assignment]
        backptr[combo] = best_value  # type: ignore[assignment]
    return table, backptr


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def reconstruct_assignment(
    elimination_order: list[str],
    evidence: dict[str, str],
    messages: dict[str, SemanticMessage],
    bn: BayesianNetwork,
) -> tuple[dict[str, str], dict[str, str], dict[str, list]]:
    """
    Bucket elimination completo sobre as mensagens semânticas compiladas.

    Upward pass: para cada variável em elimination_order, combina (soma em
    log) os fatores do bucket — o fator semântico próprio mais o que os
    filhos já encaminharam —, aplica o indicador de evidência se for o
    caso, e max-elimina variáveis hidden gravando backpointer por
    configuração do separador. A mensagem resultante é encaminhada ao
    bucket da variável remanescente mais cedo na ordem de eliminação.

    Downward pass: a partir da raiz, lê os backpointers em sequência
    reversa pra obter o assignment completo.

    Retorna (hidden_assignment, complete_assignment, selected_confidence).
    """
    order_index = {v: i for i, v in enumerate(elimination_order)}

    buckets: dict[str, list[_Factor]] = defaultdict(list)
    for variable in elimination_order:
        buckets[variable].append(
            _build_variable_factor(variable, messages[variable], bn)
        )

    # backpointers[variable] = (separator_scope, {separator_key: melhor valor de X})
    backpointers: dict[str, tuple[tuple[str, ...], dict[tuple[str, ...], str]]] = {}

    for variable in elimination_order:
        factors = buckets.pop(variable, [])
        if not factors:
            continue

        union_scope: list[str] = []
        seen = {variable}
        for f in factors:
            for v in f.scope:
                if v not in seen:
                    seen.add(v)
                    union_scope.append(v)
        separator = tuple(union_scope)

        if variable in evidence:
            new_table = _combine_restrict(
                factors, variable, evidence[variable], separator, bn
            )
        else:
            new_table, backptr = _combine_eliminate(factors, variable, separator, bn)
            backpointers[variable] = (separator, backptr)

        if not separator:
            continue  # raiz isolada — nada a encaminhar

        target = min(separator, key=lambda v: order_index[v])
        buckets[target].append(_Factor(scope=separator, table=new_table))

    # --- downward pass ---
    complete_assignment: dict[str, str] = {}
    for variable in reversed(elimination_order):
        if variable in evidence:
            complete_assignment[variable] = evidence[variable]
            continue

        entry = backpointers.get(variable)
        if entry is None:
            raise ValueError(f"No backpointer computed for {variable}.")
        scope, table = entry
        key = tuple(complete_assignment[v] for v in scope)
        value = table.get(key)
        if value is None:
            raise ValueError(
                f"No backpointer entry for {variable} at separator context "
                f"{dict(zip(scope, key))}."
            )
        complete_assignment[variable] = value

    hidden_assignment = {
        variable: value
        for variable, value in complete_assignment.items()
        if variable not in evidence
    }

    # selected_confidence: melhor esforço — usa a confidence textual da row
    # compilada que corresponde ao contexto de pais final de cada variável.
    # Note: essa confidence reflete o julgamento do LLM sobre a row em si
    # (na compilação, evidence={}), não necessariamente sobre o valor final
    # escolhido após a propagação de evidência, caso o upward pass tenha
    # corrigido a escolha em relação ao argmax local da compilação.
    selected_confidence: dict[str, list] = {}
    for variable, value in hidden_assignment.items():
        message = messages[variable]
        context = {v: complete_assignment[v] for v in message.scope}
        row_by_context = {
            context_key(row.context, message.scope): row for row in message.rows
        }
        row = row_by_context.get(context_key(context, message.scope))
        confidence = row.confidence if row is not None else None
        selected_confidence[variable] = [value, confidence]

    return hidden_assignment, complete_assignment, selected_confidence


def apply_audit_repair(
    audit: AuditResponse,
    complete_assignment: dict[str, str],
    evidence: dict[str, str],
    bn: BayesianNetwork,
    alias_map: dict[str, str],
) -> dict[str, str]:
    """
    Aplica a sugestão de reparo do AuditResponse ao complete_assignment.
    Retorna uma cópia com o reparo aplicado (ou inalterada se não há reparo).

    Silenciosamente ignora reparos inválidos (evidência, estado ilegal).
    A validação prévia via validate_audit_response é recomendada.
    """
    repaired = dict(complete_assignment)
    if audit.accept or audit.repair is None:
        return repaired

    variable = resolve_variable(audit.repair.variable, alias_map)
    if variable in evidence:
        return repaired

    domain_upper = {s.upper(): s for s in bn.variables[variable].states}
    canonical = domain_upper.get(audit.repair.value.strip().upper())
    if canonical is None:
        return repaired

    repaired[variable] = canonical
    return repaired


def validate_reconstruction_response(
    response: ReconstructionResponse,
    hidden_assignment: dict[str, str],
    complete_assignment: dict[str, str],
    bn: BayesianNetwork,
    alias_map: dict[str, str],
) -> None:
    """Garante que o LLM não alterou o assignment na fase de reconstrução."""
    normalized_hidden = normalize_assignment(response.hidden_assignment, bn, alias_map)
    normalized_complete = normalize_assignment(response.complete_assignment, bn, alias_map)
    if normalized_hidden != hidden_assignment:
        raise ValueError("Reconstruction response changed the hidden assignment.")
    if normalized_complete != complete_assignment:
        raise ValueError("Reconstruction response changed the complete assignment.")


def validate_audit_response(
    response: AuditResponse,
    bn: BayesianNetwork,
    evidence: dict[str, str],
    complete_assignment: dict[str, str],
    alias_map: dict[str, str],
) -> None:
    """
    Valida a resposta de auditoria antes de aplicá-la.
    Levanta ValueError se:
    - reject sem repair
    - repair aponta para variável de evidência
    - repair propõe estado ilegal
    - repair não muda nada (o valor já é o atual)
    """
    if response.accept:
        return

    if response.repair is None:
        raise ValueError("Audit reject responses must include one repair suggestion.")

    variable = resolve_variable(response.repair.variable, alias_map)

    if variable in evidence:
        raise ValueError("Audit repair cannot target an evidence variable.")

    domain = bn.variables[variable].states
    domain_upper = {state.upper(): state for state in domain}
    canonical = domain_upper.get(response.repair.value.strip().upper())

    if canonical is None:
        allowed = ", ".join(domain)
        raise ValueError(
            f"Audit repair has illegal value "
            f"{response.repair.value!r} for {variable}. "
            f"Allowed states: {allowed}."
        )

    current_value = complete_assignment.get(variable)
    if current_value is not None and current_value.upper() == canonical.upper():
        raise ValueError(
            "Audit repair must change the assigned value; "
            "return accept=true if no repair is needed."
        )