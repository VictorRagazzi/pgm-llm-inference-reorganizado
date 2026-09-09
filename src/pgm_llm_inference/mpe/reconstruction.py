"""Deterministic reconstruction from semantic backpointers."""

from .bucket import context_key
from .types import SemanticMessage


def reconstruct_assignment(
    elimination_order: list[str],
    evidence: dict[str, str],
    messages: dict[str, SemanticMessage],
) -> tuple[dict[str, str], dict[str, str], dict[str, list]]:
    """Reconstruct the full assignment in reverse elimination order."""
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

        rows = {context_key(row.context, message.scope): row for row in message.rows}
        selected_row = rows.get(context_key(context, message.scope))
        if selected_row is None or selected_row.selected_value is None:
            raise ValueError(
                f"No backpointer decision found for {variable} with context {context}."
            )

        complete_assignment[variable] = selected_row.selected_value
        selected_confidence[variable] = [
            selected_row.selected_value,
            selected_row.confidence,
        ]

    hidden_assignment = {
        variable: value
        for variable, value in complete_assignment.items()
        if variable not in evidence
    }
    return hidden_assignment, complete_assignment, selected_confidence
