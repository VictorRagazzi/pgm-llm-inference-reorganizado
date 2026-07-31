"""
mpe/circuit/types.py
=====================
Tipos do Part 2 (semantic maximizing circuit):

- Schema de resposta do LLM para a elicitação local (Fase 1): um ranking
  COMPLETO dos estados por linha de contexto, não um único "selected_value"
  como no Part 1 — é isso que permite recombinar numericamente os fatores
  em qualquer evidência depois, sem nova chamada LLM.
- SemanticFactor: o fator local de uma variável (função só dos PRÓPRIOS
  pais), evidência-livre e reutilizável.
- SemanticCircuit: o artefato compilado — fatores + estrutura de buckets
  (separadores, ordem de eliminação) — independente de evidência.

Este módulo é importado por elicit.py, order.py, compile.py e evaluate.py,
mas NUNCA importa `elicit` (regra de dependência do design doc: compile e
evaluate não importam elicit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..types import BriefingResponse, PromptTrace, VariableMetadata


# ---------------------------------------------------------------------------
# Schema de resposta do LLM (elicitação local, por variável, sem evidência)
# ---------------------------------------------------------------------------

class FactorRankingRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    context: dict[str, str]
    ranking: list[str] = Field(
        description=(
            "TODOS os estados da variável, ordenados do mais para o menos "
            "plausível dado este contexto de pais."
        )
    )
    rationale: str = ""


class FactorElicitationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    variable: str
    rows: list[FactorRankingRow] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fator semântico local (produto da elicitação, já com pontuação numérica)
# ---------------------------------------------------------------------------

def context_key(context: dict[str, str], scope: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(context[variable] for variable in scope)


@dataclass
class ScoredRow:
    context: dict[str, str]       # valores dos pais desta variável
    scores: dict[str, float]      # estado -> score numérico (maior = mais plausível)
    rationale: str = ""


@dataclass
class SemanticFactor:
    """
    Fator local de uma variável: função só de seus PRÓPRIOS pais (a
    família), evidência-livre, reutilizável para qualquer query futura.
    Equivalente à "semantic CPT" da Secao 5 do design doc — mas, ao
    contrário do Part 1, guarda um score por VALOR, não só o vencedor.
    """

    variable: str
    parents: tuple[str, ...]
    states: tuple[str, ...]
    rows: list[ScoredRow]

    def __post_init__(self) -> None:
        self._index: dict[tuple[str, ...], ScoredRow] = {
            context_key(row.context, self.parents): row for row in self.rows
        }

    def score(self, parent_context: dict[str, str], value: str) -> float:
        key = context_key(parent_context, self.parents)
        row = self._index.get(key)
        if row is None:
            raise KeyError(
                f"No scored row for {self.variable} with parent context "
                f"{parent_context}."
            )
        try:
            return row.scores[value]
        except KeyError as error:
            raise KeyError(
                f"No score for {self.variable}={value} in context "
                f"{parent_context}."
            ) from error


# ---------------------------------------------------------------------------
# Circuito compilado: fatores + estrutura, independentes de evidência.
# ---------------------------------------------------------------------------

@dataclass
class SemanticCircuit:
    factors: dict[str, SemanticFactor]
    elimination_order: list[str]                   # leaves-first (min-degree restrito à topologia)
    bucket_separators: dict[str, tuple[str, ...]]   # var -> separador estrutural
    bucket_target: dict[str, str | None]            # var -> bucket que recebe sua mensagem
    bn: Any                                         # BayesianNetwork (modo MPE, parse_bif)
    alias_map: dict[str, str]
    metadata: dict[str, VariableMetadata]
    briefing: BriefingResponse | None = None        # gerado 1x, evidence={}, injetado em cada fator
    traces: list[PromptTrace] = field(default_factory=list)