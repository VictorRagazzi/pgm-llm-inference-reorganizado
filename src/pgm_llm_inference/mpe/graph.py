"""
mpe/graph.py
============
Operações puras sobre o grafo DAG do pipeline MPE.

Todas as funções recebem BayesianNetwork de mpe/types.py (com .variables e
.parents) e retornam estruturas simples (sets, tuples, listas, dicts).
Sem dependências de LLM ou I/O.
"""

from __future__ import annotations

from typing import Any

from ..models import BayesianNetwork
from .types import VariableMetadata


# ---------------------------------------------------------------------------
# Ordenação topológica
# ---------------------------------------------------------------------------

def topological_order(bn: BayesianNetwork) -> list[str]:
    """Retorna as variáveis em ordem topológica (pais antes de filhos)."""
    children = bn.children_map()
    in_degree = {name: len(bn.parents[name]) for name in bn.variables}
    ready = sorted(name for name, degree in in_degree.items() if degree == 0)
    order: list[str] = []

    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                ready.append(child)
        ready.sort()

    if len(order) != len(bn.variables):
        raise ValueError("The parsed graph is not a DAG.")
    return order


# ---------------------------------------------------------------------------
# Ancestrais e descendentes
# ---------------------------------------------------------------------------

def ancestors_of(variable: str, bn: BayesianNetwork) -> set[str]:
    ancestors: set[str] = set()
    stack = list(bn.parents[variable])
    while stack:
        parent = stack.pop()
        if parent in ancestors:
            continue
        ancestors.add(parent)
        stack.extend(bn.parents[parent])
    return ancestors


def descendants_of(variable: str, children: dict[str, tuple[str, ...]]) -> set[str]:
    descendants: set[str] = set()
    stack = list(children[variable])
    while stack:
        child = stack.pop()
        if child in descendants:
            continue
        descendants.add(child)
        stack.extend(children[child])
    return descendants


def undirected_component(
    variable: str,
    bn: BayesianNetwork,
    children: dict[str, tuple[str, ...]],
) -> set[str]:
    component: set[str] = set()
    stack = [variable]
    while stack:
        node = stack.pop()
        if node in component:
            continue
        component.add(node)
        stack.extend(bn.parents[node])
        stack.extend(children[node])
    return component


# ---------------------------------------------------------------------------
# Markov blanket
# ---------------------------------------------------------------------------

def markov_blanket(
    variable: str,
    bn: BayesianNetwork,
    children: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    blanket = set(bn.parents[variable])
    blanket.update(children[variable])
    for child in children[variable]:
        blanket.update(bn.parents[child])
    blanket.discard(variable)
    order_index = {name: index for index, name in enumerate(topological_order(bn))}
    return tuple(sorted(blanket, key=lambda item: order_index[item]))


# ---------------------------------------------------------------------------
# Relação entre variável e evidência
# ---------------------------------------------------------------------------

def evidence_relation(
    variable: str,
    evidence_variable: str,
    bn: BayesianNetwork,
    children: dict[str, tuple[str, ...]],
) -> str:
    if evidence_variable == variable:
        return "self"
    if evidence_variable in bn.parents[variable]:
        return "parent"
    if evidence_variable in children[variable]:
        return "child"
    if evidence_variable in ancestors_of(variable, bn):
        return "ancestor"
    if evidence_variable in descendants_of(variable, children):
        return "descendant"
    if evidence_variable in markov_blanket(variable, bn, children):
        return "markov_blanket"
    if evidence_variable in undirected_component(variable, bn, children):
        return "same_component"
    return "disconnected_component"


def evidence_context_payload(
    variable: str,
    bn: BayesianNetwork,
    evidence: dict[str, str],
) -> dict[str, Any]:
    children = bn.children_map()
    component = undirected_component(variable, bn, children)
    blanket = markov_blanket(variable, bn, children)
    relations = [
        {
            "variable": evidence_variable,
            "value": value,
            "relation_to_focus": evidence_relation(
                variable, evidence_variable, bn, children
            ),
        }
        for evidence_variable, value in evidence.items()
    ]
    component_evidence = {
        evidence_variable: value
        for evidence_variable, value in evidence.items()
        if evidence_variable in component
    }
    blanket_evidence = {
        evidence_variable: value
        for evidence_variable, value in evidence.items()
        if evidence_variable == variable or evidence_variable in blanket
    }

    if variable in evidence:
        instruction = (
            "The focus variable is fixed evidence. It must not be changed; use it "
            "only to score parent and neighboring contexts."
        )
    elif not component_evidence:
        instruction = (
            "This variable is in a disconnected component with no evidence. Do not "
            "import pressure from evidence in other components; use only local "
            "metadata and conservative basal semantics."
        )
    elif blanket_evidence:
        instruction = (
            "There is direct Markov-blanket evidence for this variable. Focus on "
            "the parents, children, and co-parents listed here when judging states."
        )
    else:
        instruction = (
            "Evidence is in the same component but outside this variable's Markov "
            "blanket. Use incoming messages to propagate that evidence rather than "
            "making a direct state-matching assumption."
        )

    return {
        "focus_variable": variable,
        "markov_blanket": list(blanket),
        "component_has_evidence": bool(component_evidence),
        "component_evidence": component_evidence,
        "markov_blanket_evidence": blanket_evidence,
        "all_evidence_relations": relations,
        "local_instruction": instruction,
    }


# ---------------------------------------------------------------------------
# Relationship notes locais (para prompts)
# ---------------------------------------------------------------------------

def local_relationship_notes(
    variable: str,
    bn: BayesianNetwork,
    relationship_notes: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    """
    Retorna as notas de relacionamento relevantes para o vizinho imediato
    de `variable` (a própria família + famílias dos filhos).
    """
    children = bn.children_map()
    notes: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_family(child: str, role: str) -> None:
        family_notes = relationship_notes.get(child)
        if child in seen or family_notes is None:
            return
        if child not in bn.variables:
            return
        seen.add(child)
        notes.append(
            {
                "role": role,
                "child": child,
                "parents": list(bn.parents.get(child, [])),
                "notes": list(family_notes),
            }
        )

    add_family(variable, "focus_family")
    for child in children.get(variable, []):
        add_family(child, "child_family")

    return notes


def relationship_notes_payload(
    bn: BayesianNetwork,
    relationship_notes: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    if relationship_notes:
        return [
            {
                "child": child,
                "parents": list(bn.parents[child]),
                "notes": relationship_notes.get(child, []),
            }
            for child in topological_order(bn)
            if child in relationship_notes
        ]
    return []
