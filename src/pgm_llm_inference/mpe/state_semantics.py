"""
mpe/state_semantics.py
======================
Heurísticas para derivar significado semântico de estados discretos sem LLM.

Usadas como fallback quando metadados gerados por LLM não estão disponíveis
para uma variável, ou para cobrir estados que o LLM omitiu.
"""

from __future__ import annotations

from .types import VariableMetadata


# ---------------------------------------------------------------------------
# Tabela de hints: (conjunto de nomes canônicos, descrição)
# ---------------------------------------------------------------------------

_SEMANTIC_HINTS: list[tuple[set[str], str]] = [
    ({"low", "l"},        "low relative activity or abundance"),
    ({"high", "h"},       "high relative activity or abundance"),
    ({"avg", "med", "medium", "moderate", "mid", "baseline", "normal"},
                          "average or baseline relative activity or abundance"),
    ({"yes", "true", "present", "active", "on"},  "state is active / present"),
    ({"no", "false", "absent", "inactive", "off"}, "state is inactive / absent"),
]


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def infer_state_meanings(states: tuple[str, ...]) -> dict[str, str]:
    """
    Deriva state_meanings heuristicamente a partir dos nomes dos estados.

    Exemplos:
        ("LOW", "AVG", "HIGH")  → heurística cobre tudo
        ("yes", "no")           → heurística cobre tudo
        ("mild", "severe")      → fallback posicional genérico
    """
    result: dict[str, str] = {}
    n = len(states)

    for state in states:
        key = state.lower()
        matched = False
        for hints, description in _SEMANTIC_HINTS:
            if key in hints:
                result[state] = description
                matched = True
                break
        if not matched:
            idx = states.index(state)
            if n == 1:
                result[state] = f"sole state: {state}"
            elif idx == 0:
                result[state] = f"lowest / least-active state ({state})"
            elif idx == n - 1:
                result[state] = f"highest / most-active state ({state})"
            else:
                result[state] = f"intermediate state ({state})"

    return result


def resolve_state_meanings(
    var: str,
    states: tuple[str, ...],
    metadata: dict[str, VariableMetadata] | None,
) -> dict[str, str]:
    """
    Retorna state_meanings para uma variável com prioridade:
      1. metadata[var].state_meanings  (gerado por LLM, completo)
      2. infer_state_meanings(states)  (heurística, sempre disponível)

    Garante que todos os estados estejam cobertos — faz merge caso o LLM
    tenha gerado apenas um subconjunto.
    """
    llm_meanings: dict[str, str] = {}
    if metadata and var in metadata:
        llm_meanings = dict(metadata[var].state_meanings or {})

    heuristic = infer_state_meanings(states)
    return {state: llm_meanings.get(state, heuristic[state]) for state in states}


def states_pipe(states: tuple[str, ...]) -> str:
    """Formata estados como 'A|B|C' para uso em schemas de prompt."""
    return "|".join(states)


def network_state_meanings(
    bn,
    metadata: dict[str, VariableMetadata] | None,
) -> dict[str, dict[str, str]]:
    """
    Constrói o bloco completo de state_meanings para toda a rede.
    `bn` é do tipo mpe.types.BayesianNetwork.
    """
    return {
        var: resolve_state_meanings(var, variable.states, metadata)
        for var, variable in bn.variables.items()
    }
