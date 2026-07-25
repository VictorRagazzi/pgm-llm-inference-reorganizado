"""
mpe/circuit/order.py
=====================
Construção da ESTRUTURA do circuito (Fase 2 do design doc, Secao 6):
grafo moralizado, ordem de eliminação e separadores de bucket. Puramente
estrutural — sem LLM, sem evidência. Roda uma vez por rede.

A ordem de eliminação é "leaves-first restrita": todo filho é eliminado
antes de qualquer um de seus pais (condição necessária para que os fatores
de família {variavel} U parents(variavel) já estejam disponíveis quando
cada bucket é processado). Dentro dessa restrição, usamos min-degree no
grafo moralizado para reduzir a largura induzida (heurística padrão de
bucket elimination).
"""

from __future__ import annotations

from ...models import BayesianNetwork


def moralize(bn: BayesianNetwork) -> dict[str, set[str]]:
    """Grafo não-direcionado: casa co-pais (v-structures) e remove direção."""
    neighbors: dict[str, set[str]] = {v: set() for v in bn.variables}
    for child, parents in bn.parents.items():
        for parent in parents:
            neighbors[child].add(parent)
            neighbors[parent].add(child)
        for i, a in enumerate(parents):
            for b in parents[i + 1:]:
                neighbors[a].add(b)
                neighbors[b].add(a)
    return neighbors


def constrained_min_degree_order(bn: BayesianNetwork) -> list[str]:
    """
    Ordem de eliminação leaves-first, com desempate por grau mínimo no
    grafo moralizado (heurística de largura induzida reduzida).

    "Leaves-first restrita" = a cada passo, só variáveis cujos filhos já
    foram todos eliminados entram na disputa; entre essas, escolhe a de
    menor grau corrente (min-degree), com empate por nome (determinismo).
    """
    children = bn.children_map()
    moral = moralize(bn)
    working = {v: set(n) for v, n in moral.items()}

    remaining = set(bn.variables)
    eliminated: set[str] = set()
    order: list[str] = []

    while remaining:
        ready = [v for v in remaining if all(c in eliminated for c in children[v])]
        if not ready:
            raise ValueError(
                "Could not find a ready variable — is the graph a valid DAG?"
            )
        ready.sort(key=lambda v: (len(working[v]), v))
        chosen = ready[0]
        order.append(chosen)
        eliminated.add(chosen)
        remaining.discard(chosen)

        # Fill-in (triangulação): os vizinhos ainda restantes viram clique.
        still_present = [n for n in working[chosen] if n in remaining]
        for a in still_present:
            for b in still_present:
                if a != b:
                    working[a].add(b)
        for neighbor in working[chosen]:
            working[neighbor].discard(chosen)
        working[chosen] = set()

    return order


def build_bucket_structure(
    bn: BayesianNetwork,
) -> tuple[list[str], dict[str, tuple[str, ...]], dict[str, str | None]]:
    """
    Retorna (elimination_order, separators, bucket_target).

    separators[X]  : variáveis ainda não eliminadas que passam a
                      compartilhar bucket com X depois do fill-in
                      triangular — o "contexto" em que a mensagem de X
                      é indexada.
    bucket_target[X]: membro de separators[X] mais cedo na ordem — é para
                      o bucket dele que a mensagem de X é enviada. None
                      se X é a última variável do seu componente na
                      ordem (mensagem final, já resolvida, sem separador).
    """
    order = constrained_min_degree_order(bn)
    order_index = {v: i for i, v in enumerate(order)}
    moral = moralize(bn)
    working = {v: set(n) for v, n in moral.items()}

    separators: dict[str, tuple[str, ...]] = {}
    bucket_target: dict[str, str | None] = {}

    for variable in order:
        later_neighbors = [
            n for n in working[variable] if order_index[n] > order_index[variable]
        ]
        later_neighbors.sort(key=lambda n: order_index[n])
        separators[variable] = tuple(later_neighbors)
        bucket_target[variable] = later_neighbors[0] if later_neighbors else None

        for a in later_neighbors:
            for b in later_neighbors:
                if a != b:
                    working[a].add(b)
        for neighbor in working[variable]:
            working[neighbor].discard(variable)
        working[variable] = set()

    return order, separators, bucket_target
