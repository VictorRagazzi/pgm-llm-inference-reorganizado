"""
mpe/types.py
============
Schemas Pydantic exclusivos do pipeline LLM-MPE.

Variable e BayesianNetwork foram unificados em models/ e não vivem mais aqui.
Este módulo contém apenas os tipos que não existem fora do pipeline MPE:
metadados de variáveis, tipos de resposta do LLM e tipos de rastreamento.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Metadados qualitativos de variável (gerados por LLM, usados nos prompts)
# ---------------------------------------------------------------------------

class VariableMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    display_name: str | None = None
    description: str | None = None
    expert_note: str | None = None
    aliases: tuple[str, ...] = ()
    state_meanings: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Tipos de mensagem / decisão do bucket pipeline
# ---------------------------------------------------------------------------

class ContextDecision(BaseModel):
    context: dict[str, str]
    selected_value: str
    confidence: str
    rationale: str

    @field_validator("context", mode="before")
    @classmethod
    def parse_context_string(cls, v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("{"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
            result = {}
            for pair in v.split(","):
                pair = pair.strip()
                if "=" in pair:
                    key, _, value = pair.partition("=")
                    result[key.strip()] = value.strip()
            if result:
                return result
            raise ValueError(f"Cannot parse context string: {v!r}")
        raise ValueError(f"Cannot parse context: {v!r}")


class ContextEvidenceMessage(BaseModel):
    context: dict[str, str]
    compatibility: str
    rationale: str


class MessageRow(BaseModel):
    context: dict[str, str]
    selected_value: str | None = None
    observed_value: str | None = None
    compatibility: str | None = None
    confidence: str | None = None
    domain_scores: DomainScores | None = None
    rationale: str


class SemanticMessage(BaseModel):
    source_variable: str
    scope: tuple[str, ...]
    is_evidence: bool
    evidence_driven: bool = False
    rows: list[MessageRow]


class BucketSpec(BaseModel):
    variable: str
    is_evidence: bool
    observed_value: str | None
    local_scope: tuple[str, ...]
    separator: tuple[str, ...]
    context_rows: list[dict[str, str]]
    incoming_messages: list[SemanticMessage]


class DomainScores(BaseModel):
    """
    Scores (probs) por estado do domínio de uma variável, para uma
    única linha de contexto. Produzido na Fase 1 a partir do top_logprobs
    da API; consumido no upward pass (Fase 2) para max-product aritmético.
    """
    model_config = ConfigDict(frozen=True)

    by_state: dict[str, float]
    default_score: float = -50.0

    def get(self, state: str) -> float:
        """Score do estado, ou default_score se nunca apareceu no top_logprobs."""
        return self.by_state.get(state, self.default_score)

    def missing(self, domain: tuple[str, ...]) -> tuple[str, ...]:
        """Estados do domínio sem score real (vão cair no default_score)."""
        return tuple(state for state in domain if state not in self.by_state)

    def best(self) -> tuple[str, float]:
        return max(self.by_state.items(), key=lambda kv: kv[1])
# ---------------------------------------------------------------------------
# Tipos de resposta LLM
# ---------------------------------------------------------------------------

class BriefingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    network_summary: str
    important_dependencies: list[str] = Field(default_factory=list)
    reasoning_rules: list[str] = Field(default_factory=list)


class BucketResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    variable: str
    decisions: list[ContextDecision] = Field(default_factory=list)
    observed_value: str | None = None
    messages: list[ContextEvidenceMessage] = Field(default_factory=list)


class ReconstructionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hidden_assignment: dict[str, str]
    complete_assignment: dict[str, str]
    explanation: list[str] = Field(default_factory=list)


class RepairSuggestion(BaseModel):
    model_config = ConfigDict(extra="ignore")

    variable: str
    value: str
    reason: str | None = None


class AuditResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    accept: bool
    repair: RepairSuggestion | None = None
    reason: str


# ---------------------------------------------------------------------------
# Tipos de rastreamento de prompt / retry
# ---------------------------------------------------------------------------

class LLMAttempt(BaseModel):
    attempt: int
    response_text: str | None = None
    parsed_response: dict[str, Any] | None = None
    error: str | None = None


class PromptTrace(BaseModel):
    purpose: str
    variable: str | None = None
    prompt: str
    attempts: list[LLMAttempt] = Field(default_factory=list)
