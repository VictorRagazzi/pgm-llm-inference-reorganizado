"""
mpe/client.py
=============
Cliente LLM do pipeline MPE.

LLMJsonClient: envia prompts, faz retry em caso de JSON inválido e rastreia
todas as tentativas em PromptTrace. Usa InferenceConfig para configuração
(sem AppSettings separado).

extract_json_object: extrai o primeiro objeto JSON de uma resposta de texto,
tolerando blocos de código markdown.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError
import httpx

from ..core.config import InferenceConfig
from .types import LLMAttempt, PromptTrace, DomainScores

TModel = TypeVar("TModel", bound=BaseModel)


def extract_json_object(response_text: str) -> dict[str, Any]:
    """Extrai o primeiro objeto JSON de response_text (strip de blocos ```json)."""
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


class LLMJsonClient:
    """
    Cliente que envia um prompt ao LLM e valida a resposta como JSON Pydantic.
 
    Faz até `config.openai_max_retries + 1` tentativas, passando o erro
    de validação de volta ao modelo para autocorreção.
 
    Parâmetros
    ----------
    config       : InferenceConfig com as URLs, modelos e flags de debug.
    use_real_llm : True → OpenAI/proxy (openai_base_url + openai_api_key).
                   False → LM Studio local (local_url, sem api_key).
    dry_run      : Se True, o cliente é criado mas complete_json() lança erro.
                   Útil para testes de construção sem chamar a rede.
    """
 
    def __init__(
        self,
        config: InferenceConfig,
        *,
        use_real_llm: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.use_real_llm = use_real_llm
        self.dry_run = dry_run
        self._client = None
 
        if dry_run:
            return
 
        from openai import OpenAI
 
        if use_real_llm:
            if not config.openai_api_key:
                raise ValueError(
                    "openai_api_key é obrigatório no InferenceConfig "
                    "(ou via variável de ambiente PGM_OPENAI_API_KEY) "
                    "quando use_real_llm=True."
                )
            self._client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_base_url or None,
            )
        else:
            # LM Studio: API compatível com OpenAI, sem api_key obrigatória.
            # O SDK exige algum valor; "lm-studio" é a convenção da documentação
            # oficial do LM Studio.
            self._client = OpenAI(
                api_key="lm-studio",
                base_url=config.local_url.rstrip("/chat/completions").rstrip("/"),
            )
 
    @property
    def _model(self) -> str:
        """Modelo efetivo para a chamada atual."""
        return self.config.openai_model if self.use_real_llm else self.config.local_model
 
    def complete_json(
        self,
        *,
        purpose: str,
        variable: str | None,
        prompt: str,
        response_model: type[TModel],
        semantic_validator: Callable[[TModel], None] | None = None,
        logprobs_field: str | None = None,
        logprobs_states: tuple[str, ...] | None = None,
    ) -> tuple[TModel, PromptTrace, list[DomainScores]]:
        if self.dry_run:
            raise RuntimeError("complete_json não pode ser chamado em modo dry-run.")
        if self._client is None:
            raise RuntimeError("Cliente LLM não foi inicializado.")
 
        trace = PromptTrace(purpose=purpose, variable=variable, prompt=prompt)
        retry_instruction = ""
 
        for attempt_number in range(1, self.config.openai_max_retries + 2):
            request_prompt = prompt + retry_instruction
 
            if self.config.show_llm_prompt:
                print(
                    f"\n=== LLM Prompt (tentativa {attempt_number}) ===\n"
                    f"{request_prompt}\n"
                )
                time.sleep(5)
 
            response_text: str | None = None
            parsed: dict[str, Any] | None = None
            domain_scores_list: list[DomainScores] = []
 
            try:
                call_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a specialist in Bayesian networks "
                                "and probabilistic graphical models. "
                                "Return valid JSON only."
                            ),
                        },
                        {"role": "user", "content": request_prompt},
                    ],
                    "timeout": httpx.Timeout(2400.0, connect=30.0),
                    # "extra_body": {
                    #     "provider": {
                    #         "order": ["DeepInfra", "Together"], 
                    #         "allow_fallbacks": False
                    #     }
                    # },
                    "temperature": self.config.openai_temperature,
                }
                if logprobs_field and logprobs_states:
                    call_kwargs["logprobs"] = True
                    call_kwargs["top_logprobs"] = min(20, max(5, len(logprobs_states) + 3))

 
                # json_object só é solicitado para modelos que suportam — o LM Studio
                # suporta para a maioria dos modelos recentes, mas pode ser desabilitado
                # via config se o modelo local não suportar.
                if self.use_real_llm and self.config.openai_use_json_response_format:
                    call_kwargs["response_format"] = {"type": "json_object"}
 
                completion = self._client.chat.completions.create(**call_kwargs)
                response_text = completion.choices[0].message.content or ""

                if logprobs_field and logprobs_states:
                    lp = getattr(completion.choices[0], "logprobs", None)
                    domain_scores_list = extract_domain_scores(
                        response_text, lp.content if lp else [],
                        field_name=logprobs_field, states=logprobs_states,
                    )
 
                if self.config.show_llm_output:
                    print(
                        f"\n=== LLM Response (tentativa {attempt_number}) ===\n"
                        f"{response_text}\n"
                    )
                    time.sleep(5)
 
                parsed = extract_json_object(response_text)
                model = response_model.model_validate(parsed)
                if semantic_validator is not None:
                    semantic_validator(model)
 
                trace.attempts.append(
                    LLMAttempt(
                        attempt=attempt_number,
                        response_text=response_text,
                        parsed_response=parsed,
                    )
                )
                return model, trace, domain_scores_list
 
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                trace.attempts.append(
                    LLMAttempt(
                        attempt=attempt_number,
                        response_text=response_text,
                        parsed_response=parsed,
                        error=str(error),
                    )
                )
                retry_instruction = (
                    "\n\nYour previous response was invalid for this task. "
                    f"Error: {error}\nReturn only corrected JSON matching the "
                    "requested schema and context rows."
                )
 
        last_error = trace.attempts[-1].error if trace.attempts else "unknown error"
        raise ValueError(f"LLM response failed validation: {last_error}")

def _token_offsets(tokens: list[str]) -> list[tuple[int, int]]:
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok)
    return offsets


def extract_domain_scores(
    response_text: str,
    logprob_content: list[Any],
    *,
    field_name: str,
    states: tuple[str, ...],
    default_score: float = -50.0,
) -> list[DomainScores]:
    """
    Para cada ocorrência de `"<field_name>": "valor"` no texto bruto da
    resposta, retorna {estado: probabilidade_normalizada} a partir do top_logprobs.
    """
    if not logprob_content:
        return []

    import math 
    tokens = [t.token for t in logprob_content]
    offsets = _token_offsets(tokens)
    pattern = re.compile(rf'"{re.escape(field_name)}"\s*:\s*"')

    results: list[DomainScores] = []
    for match in pattern.finditer(response_text):
        value_start = match.end()
        idx = next((i for i, (s, e) in enumerate(offsets) if s <= value_start < e), None)
        if idx is None:
            results.append(DomainScores(by_state={}, default_score=0.0)) # Probabilidade padrão zero
            continue

        entry = logprob_content[idx]
        candidates = {entry.token.strip().upper(): entry.logprob}
        for alt in entry.top_logprobs or []:
            candidates.setdefault(alt.token.strip().upper(), alt.logprob)

        # 1. Coleta os logprobs brutos dos estados
        raw_logprobs: dict[str, float] = {}
        for state in states:
            key = state.strip().upper()
            hit = next((lp for tok, lp in candidates.items()
                        if tok.startswith(key[:len(tok)]) or key.startswith(tok)), None)
            if hit is not None:
                raw_logprobs[state] = hit
            else:
                raw_logprobs[state] = default_score # -50.0 representa um logprob muito baixo

        # 2. Aplica o exponencial e normaliza (com estabilidade numérica)
        scores: dict[str, float] = {}
        if raw_logprobs:
            max_logprob = max(raw_logprobs.values())
            
            # Calcula as exponenciais parciais (subtraindo o max para evitar overflow)
            unnormalized_probs = {
                state: math.exp(lp - max_logprob) 
                for state, lp in raw_logprobs.items()
            }
            
            sum_probs = sum(unnormalized_probs.values())
            
            # Normaliza para que a soma seja 1.0
            if sum_probs > 0:
                scores = {
                    state: p / sum_probs 
                    for state, p in unnormalized_probs.items()
                }
            else:
                # Caso extremo de fallback onde tudo sumou zero
                scores = {state: 1.0 / len(states) for state in states}

        results.append(DomainScores(by_state=scores, default_score=0.0))

    return results