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

Desempate (empates no score total, que surgem da escala ordinal
grosseira de ranking_to_scores — ver scale.py): _resolve_best aplica uma
regra determinística de 3 níveis, reprodutível independente de ordem de
iteração do Python — ver docstring de _resolve_best.
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


def _resolve_best(
    candidates: list[tuple[float, float, str]],
    domain_order: dict[str, int],
) -> tuple[str, float, int]:
    """
    candidates: lista de (total_score, own_local_score, value) — total_score
    é o que a passada ascendente maximiza (fator local + mensagens dos
    filhos); own_local_score é a contribuição isolada do fator local,
    mantida no retorno só pra diagnóstico (ver diagnose.py), NÃO usada
    como critério de desempate (ver nota abaixo).

    Desempate determinístico e reprodutível, independente da ordem de
    iteração do Python:
      1. Maior total_score (o critério principal, sempre).
      2. Empate: ordem canônica do domínio da variável (a ordem
         declarada no .bif) — arbitrária, mas fixa e reproduzível.

    Por que NÃO desempatar por own_local_score (tentativa descartada):
    testado empiricamente no caso Burglary/Earthquake -> Alarm, prefer
    o maior own_local_score sistematicamente puxa de volta pro prior
    local — exatamente o viés que causa a falha de explaining away que
    o two-pass existe pra corrigir. Num empate por coincidência
    aritmética (score baixo do Alarm inconsistente + score alto do
    prior cancelando exatamente o score alto do Alarm consistente +
    score baixo do prior "raro"), esse critério reverte para a mesma
    resposta do forward-only, anulando a correção. Fica só a ordem do
    domínio: não finge ser mais "inteligente" do que realmente é --
    um empate aqui é sintoma da escala ordinal grosseira (ranking_to_scores),
    não algo que dê pra resolver com mais lógica em cima do mesmo score.

    Retorna (valor_escolhido, total_score, nº de candidatos que ainda
    empatavam no total_score — >1 significa empate genuíno, resolvido só
    pela ordem do domínio).
    """
    best_total = max(c[0] for c in candidates)
    tied = [c for c in candidates if c[0] == best_total]
    if len(tied) == 1:
        return tied[0][2], best_total, 1

    tied.sort(key=lambda c: domain_order[c[2]])
    return tied[0][2], best_total, len(tied)


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
        domain_order = {state: i for i, state in enumerate(bn.variables[variable].states)}

        table: dict[tuple[str, ...], float] = {}
        chosen: dict[tuple[str, ...], str] = {}

        for combo in itertools.product(*domains):
            sep_assignment = dict(zip(separator, combo, strict=True))

            candidates: list[tuple[float, float, str]] = []
            for value in own_domain:
                full = dict(sep_assignment)
                full[variable] = value
                parent_context = {p: full[p] for p in factor.parents}
                own_score = factor.score(parent_context, value)

                total_score = own_score
                for child_sep, child_table in own_incoming:
                    child_key = context_key(full, child_sep)
                    total_score += child_table[child_key]

                candidates.append((total_score, own_score, value))

            best_value, best_score, _n_tied = _resolve_best(candidates, domain_order)
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

    Não há soma de fatores independentes aqui (só um fator por decisão),
    então não há o problema de empate por escala grosseira que existe em
    evaluate_two_pass — o ranking do próprio fator já é uma permutação
    sem empates por definição (ranking_to_scores nunca repete valor).
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