"""
mpe/circuit/diagnose.py
=========================
Instrumentação pra investigar POR QUE evaluate_two_pass diverge de
evaluate_forward_only em alguns casos e não em outros — sem alterar o
comportamento de produção de evaluate.py. Não é chamada por main.py; é
uma ferramenta de análise separada, pra rodar manualmente sobre um
circuito já compilado.

Usa o MESMO desempate de 3 níveis de evaluate._resolve_best, então os
números aqui refletem exatamente o que evaluate_two_pass decide — a
diferença é que este módulo também guarda o 2º melhor total_score (pra
medir a margem) e quantos candidatos ainda empatavam depois do
desempate por own_score (o "empate genuíno" reportado como is_tie).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

from ..io import normalize_assignment
from .evaluate import _domain_for, _resolve_best, evaluate_forward_only
from .types import SemanticCircuit, context_key


@dataclass
class BucketDiagnostics:
    combo: tuple[str, ...]                # valores do separador nessa linha
    best_value: str
    best_score: float
    second_best_score: float | None
    margin: float | None                  # best_score - second_best_score (por total_score)
    n_genuinely_tied: int                 # >1 = empate não resolvido nem por own_score
    is_tie: bool                          # atalho pra n_genuinely_tied > 1


def evaluate_two_pass_diagnostics(
    circuit: SemanticCircuit,
    evidence: dict[str, str],
    *,
    max_separator_rows: int = 20_000,
) -> tuple[dict[str, str], dict[str, BucketDiagnostics]]:
    """
    Reimplementa evaluate_two_pass (upward + downward, mesmo desempate de
    _resolve_best), mas guarda, para a linha do separador REALMENTE usada
    na reconstrução de cada variável: a margem entre o melhor e o segundo
    melhor total_score, e se a escolha final ainda dependeu do desempate
    genuíno (nível 3 de _resolve_best — ordem canônica do domínio).
    """
    bn = circuit.bn
    norm_evidence = normalize_assignment(evidence, bn, circuit.alias_map)

    order = circuit.elimination_order
    separators = circuit.bucket_separators
    bucket_target = circuit.bucket_target
    factors = circuit.factors

    incoming: dict[str, list[tuple[tuple[str, ...], dict]]] = defaultdict(list)
    backpointer: dict[str, dict[tuple[str, ...], str]] = {}
    # Por variável, todas as linhas do separador com o total_score de CADA
    # valor (não só o vencedor) — necessário pra achar o 2º melhor depois,
    # e quantos candidatos empatavam no nível 3 do desempate.
    all_totals: dict[str, dict[tuple[str, ...], dict[str, float]]] = {}
    all_tie_counts: dict[str, dict[tuple[str, ...], int]] = {}

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
                f"(> max_separator_rows={max_separator_rows})."
            )

        own_domain = _domain_for(variable, bn, norm_evidence)
        own_incoming = incoming.get(variable, [])
        domain_order = {state: i for i, state in enumerate(bn.variables[variable].states)}

        table: dict[tuple[str, ...], float] = {}
        chosen: dict[tuple[str, ...], str] = {}
        totals_by_combo: dict[tuple[str, ...], dict[str, float]] = {}
        tie_by_combo: dict[tuple[str, ...], int] = {}

        for combo in itertools.product(*domains):
            sep_assignment = dict(zip(separator, combo, strict=True))

            candidates: list[tuple[float, float, str]] = []
            totals_by_value: dict[str, float] = {}
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
                totals_by_value[value] = total_score

            best_value, best_score, n_tied = _resolve_best(candidates, domain_order)
            table[combo] = best_score
            chosen[combo] = best_value
            totals_by_combo[combo] = totals_by_value
            tie_by_combo[combo] = n_tied

        backpointer[variable] = chosen
        all_totals[variable] = totals_by_combo
        all_tie_counts[variable] = tie_by_combo

        target = bucket_target[variable]
        if target is not None:
            incoming[target].append((separator, table))

    # --- Downward: reconstrução + diagnóstico na linha realmente usada ---
    assignment: dict[str, str] = {}
    diagnostics: dict[str, BucketDiagnostics] = {}

    for variable in reversed(order):
        if variable in norm_evidence:
            assignment[variable] = norm_evidence[variable]
            continue
        separator = separators[variable]
        key = tuple(assignment[v] for v in separator)
        assignment[variable] = backpointer[variable][key]

        totals_by_value = all_totals[variable][key]
        n_tied = all_tie_counts[variable][key]
        ranked = sorted(totals_by_value.values(), reverse=True)
        best_score = ranked[0]
        second_best = ranked[1] if len(ranked) > 1 else None
        margin = (best_score - second_best) if second_best is not None else None

        diagnostics[variable] = BucketDiagnostics(
            combo=key,
            best_value=assignment[variable],
            best_score=best_score,
            second_best_score=second_best,
            margin=margin,
            n_genuinely_tied=n_tied,
            is_tie=n_tied > 1,
        )

    return assignment, diagnostics


@dataclass
class DivergenceSummary:
    n_evidence_sets: int = 0
    n_hidden_total: int = 0
    n_divergent: int = 0
    n_divergent_ties: int = 0
    margins_at_divergence: list = field(default_factory=list)
    margins_at_agreement: list = field(default_factory=list)

    @property
    def divergence_rate(self) -> float:
        return self.n_divergent / self.n_hidden_total if self.n_hidden_total else 0.0

    @property
    def tie_rate_among_divergent(self) -> float:
        return self.n_divergent_ties / self.n_divergent if self.n_divergent else 0.0

    @property
    def avg_margin_at_divergence(self):
        if not self.margins_at_divergence:
            return None
        return sum(self.margins_at_divergence) / len(self.margins_at_divergence)

    @property
    def avg_margin_at_agreement(self):
        if not self.margins_at_agreement:
            return None
        return sum(self.margins_at_agreement) / len(self.margins_at_agreement)


def compare_forward_vs_twopass(
    circuit: SemanticCircuit,
    evidence_list: list[dict[str, str]],
) -> DivergenceSummary:
    """
    Roda forward-only e two-pass (com diagnóstico) sobre uma lista de
    configurações de evidência e resume: taxa de divergência entre os
    dois modos e, tanto nas divergentes quanto nas concordantes, a
    margem (best - 2º melhor total_score) no ponto usado pela
    reconstrução — já considerando o desempate de 3 níveis.
    """
    summary = DivergenceSummary()

    for evidence in evidence_list:
        summary.n_evidence_sets += 1
        forward = evaluate_forward_only(circuit, evidence)
        twopass, diagnostics = evaluate_two_pass_diagnostics(circuit, evidence)

        hidden_vars = [v for v in twopass if v not in evidence]
        summary.n_hidden_total += len(hidden_vars)

        for variable in hidden_vars:
            diag = diagnostics.get(variable)
            diverged = forward.get(variable) != twopass.get(variable)

            if diverged:
                summary.n_divergent += 1
                if diag is not None:
                    if diag.is_tie:
                        summary.n_divergent_ties += 1
                    if diag.margin is not None:
                        summary.margins_at_divergence.append(diag.margin)
            elif diag is not None and diag.margin is not None:
                summary.margins_at_agreement.append(diag.margin)

    return summary


def print_divergence_report(
    circuit: SemanticCircuit, evidence_list: list[dict[str, str]]
) -> None:
    summary = compare_forward_vs_twopass(circuit, evidence_list)
    print(f"Evidências testadas        : {summary.n_evidence_sets}")
    print(f"Variáveis hidden (total)   : {summary.n_hidden_total}")
    print(
        f"Divergências fwd x 2-pass  : {summary.n_divergent} "
        f"({summary.divergence_rate:.1%} das hidden)"
    )
    if summary.n_divergent:
        print(
            f"  -> empates genuínos na divergência : {summary.n_divergent_ties} "
            f"({summary.tie_rate_among_divergent:.1%} das divergentes)"
        )
        margin_div = summary.avg_margin_at_divergence
        if margin_div is not None:
            print(f"  -> margem média na divergência      : {margin_div:.4f}")
    margin_agree = summary.avg_margin_at_agreement
    if margin_agree is not None:
        print(f"  -> margem média na concordância      : {margin_agree:.4f}")