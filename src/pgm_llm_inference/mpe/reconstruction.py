"""
mpe/reconstruction.py
=====================
Fase de reconstrução e validação do pipeline MPE.

Após a passagem forward de bucket elimination, estas funções:
- reconstruct_assignment: percorre a ordem de eliminação de trás pra frente
  seguindo os backpointers em SemanticMessage para obter o assignment completo
- apply_audit_repair: aplica a sugestão de reparo do LLM se for válida
- validate_reconstruction_response: garante que o LLM não alterou o assignment
- validate_audit_response: valida a resposta de auditoria antes de aplicá-la
"""

from __future__ import annotations

from .bucket import context_key
from .io import normalize_assignment, resolve_variable
from ..models import BayesianNetwork
from .types import (
    AuditResponse,
    ReconstructionResponse,
    SemanticMessage,
)


def reconstruct_assignment(
    elimination_order: list[str],
    evidence: dict[str, str],
    messages: dict[str, SemanticMessage],
) -> tuple[dict[str, str], dict[str, str], dict[str, list]]:
    """
    Percorre elimination_order de trás para frente e lê o backpointer de
    cada variável hidden a partir de messages[variable].

    Retorna (hidden_assignment, complete_assignment, selected_confidence).
    selected_confidence mapeia variável → [selected_value, confidence].
    """
    complete_assignment: dict[str, str] = {}
    selected_confidence: dict[str, list] = {}

    for variable in reversed(elimination_order):
        if variable in evidence:
            complete_assignment[variable] = evidence[variable]
            continue

        message = messages[variable]
        try:
            context = {
                scope_variable: complete_assignment[scope_variable]
                for scope_variable in message.scope
            }
        except KeyError as error:
            missing = error.args[0]
            raise ValueError(
                f"Cannot reconstruct {variable}; separator variable {missing} "
                "has not been assigned."
            ) from error

        row_by_context = {
            context_key(row.context, message.scope): row for row in message.rows
        }
        selected_row = row_by_context.get(context_key(context, message.scope))
        if selected_row is None or selected_row.selected_value is None:
            raise ValueError(
                f"No backpointer decision found for {variable} with context {context}."
            )
        complete_assignment[variable] = selected_row.selected_value
        selected_confidence[variable] = [selected_row.selected_value, selected_row.confidence]

    hidden_assignment = {
        variable: value
        for variable, value in complete_assignment.items()
        if variable not in evidence
    }
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
