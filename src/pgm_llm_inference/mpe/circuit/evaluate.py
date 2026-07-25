"""
mpe/circuit/evaluate.py
==========================
Fase 3 do Part 2 (Secao 7 do design doc) — avaliação. NENHUMA chamada LLM
acontece aqui. A evidência entra SOMENTE nesta fase, como indicador, e é
o que corrige diagnosticamente ancestrais a partir de evidência em
descendentes (o problema de explaining away que o Part 1 não resolve).

evaluate_two_pass:
    Passada ascendente (max-sum sobre o fator local de cada bucket +
    mensagens já recebidas dos filhos) seguida de reconstrução
    descendente por backpointers. Isso é bucket elimination completo,
    numérico, sobre os fatores compilados na Fase 1.

evaluate_forward_only:
    Modo leve, sem passada ascendente — cada variável usa só seu próprio
    fator dado os pais já resolvidos, na ordem topológica normal. É o
    que o Part 1 faz. Serve como âncora de comparação (E2.1), não como
    método recomendado: não corrige diagnosticamente evidência a jusante.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

from ..io import normalize_assignment
from .types import SemanticCircuit, context_key

NEG_INF = float("-inf")


def _domain_for(variable: str, bn, evidence: dict[str, str]) -> tuple[str, ...]:
    if variable in evidence:
        return (evidence[variable],)
    return tuple(bn.variables[variable].states)


def evaluate_two_pass(
    circuit: SemanticCircuit,
    evidence: dict[str, str],
    max_separator_rows: int = 20_000,
) -> dict[str, str]:
    """
    Retorna o assignment completo (evidência + variáveis hidden) mais
    provável segundo o circuito, para a `evidence` dada. Custo puramente
    numérico e independente do modelo — zero chamadas LLM.
    """
    bn = circuit.bn
    norm_evidence = normalize_assignment(evidence, bn, circuit.alias_map)

    order = circuit.elimination_order
    separators = circuit.bucket_separators
    bucket_target = circuit.bucket_target
    factors = circuit.factors

    # Mensagens recebidas por bucket: alvo -> lista de (separador_da_origem, tabela)
    incoming: dict[str, list[tuple[tuple[str, ...], dict[tuple[str, ...], float]]]] = (
        defaultdict(list)
    )

    message_table: dict[str, dict[tuple[str, ...], float]] = {}
    backpointer: dict[str, dict[tuple[str, ...], str]] = {}

    # --- Passada ascendente (leaves -> raízes, cf. elimination_order) ---
    for variable in order:
        factor = factors[variable]
        separator = separators[variable]
        domains = [_domain_for(v, bn, norm_evidence) for v in separator]

        n_combos = 1
        for d in domains:
            n_combos *= len(d)
        if n_combos > max_separator_rows:
            raise ValueError(
                f"Separator for {variable} would need {n_combos} rows "
                f"(> max_separator_rows={max_separator_rows}); induced "
                "width too large for this evaluation."
            )

        own_domain = _domain_for(variable, bn, norm_evidence)
        own_incoming = incoming.get(variable, [])

        table: dict[tuple[str, ...], float] = {}
        chosen: dict[tuple[str, ...], str] = {}

        for combo in itertools.product(*domains):
            sep_assignment = dict(zip(separator, combo, strict=True))

            best_score = NEG_INF
            best_value = None
            for value in own_domain:
                full = dict(sep_assignment)
                full[variable] = value
                parent_context = {p: full[p] for p in factor.parents}
                score = factor.score(parent_context, value)

                for child_sep, child_table in own_incoming:
                    child_key = context_key(full, child_sep)
                    score += child_table[child_key]

                if score > best_score:
                    best_score, best_value = score, value

            table[combo] = best_score
            chosen[combo] = best_value

        message_table[variable] = table
        backpointer[variable] = chosen

        target = bucket_target[variable]
        if target is not None:
            incoming[target].append((separator, table))

    # --- Passada descendente (reconstrução por backpointers) ---
    assignment: dict[str, str] = {}
    for variable in reversed(order):
        if variable in norm_evidence:
            assignment[variable] = norm_evidence[variable]
            continue
        separator = separators[variable]
        key = tuple(assignment[v] for v in separator)
        assignment[variable] = backpointer[variable][key]

    return assignment


def evaluate_forward_only(
    circuit: SemanticCircuit,
    evidence: dict[str, str],
) -> dict[str, str]:
    """
    Modo leve (Secao 7, "Optional forward-only mode"): sem passada
    ascendente. Cada variável (ordem topológica, raízes primeiro) usa só
    seu próprio fator local dado os pais já resolvidos — mesma lógica do
    Part 1 (reconstruct_assignment). Serve de âncora de comparação
    (E2.1): mostra que Part 2 generaliza Part 1, não que o substitui.
    """
    bn = circuit.bn
    norm_evidence = normalize_assignment(evidence, bn, circuit.alias_map)
    factors = circuit.factors

    topo_order = list(reversed(circuit.elimination_order))  # pais antes de filhos

    assignment: dict[str, str] = {}
    for variable in topo_order:
        if variable in norm_evidence:
            assignment[variable] = norm_evidence[variable]
            continue
        factor = factors[variable]
        parent_context = {p: assignment[p] for p in factor.parents}
        best_value = max(
            factor.states,
            key=lambda value: factor.score(parent_context, value),
        )
        assignment[variable] = best_value

    return assignment
