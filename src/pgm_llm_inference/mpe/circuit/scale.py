"""
mpe/circuit/scale.py
======================
Escala numérica fixa, do lado do código (não do LLM), que converte um
ranking ordinal de estados em scores comparáveis. Mantém o método livre
de probabilidades inventadas pelo modelo (cf. design doc, Secao 5:
"a fixed, code-side scale converts ... qualitative rankings into
comparable scores").
"""

from __future__ import annotations


def ranking_to_scores(ranking: list[str]) -> dict[str, float]:
    """
    ranking[0] é o estado mais plausível, ranking[-1] o menos plausível.

    Score = posição invertida: o mais plausível recebe len(ranking)-1,
    o menos plausível recebe 0. Escala monotônica fixa e determinística —
    nunca vem do LLM, só a ORDEM vem do LLM.
    """
    n = len(ranking)
    return {state: float(n - 1 - index) for index, state in enumerate(ranking)}
