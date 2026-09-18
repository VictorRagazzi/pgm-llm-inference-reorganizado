"""
mpe/compile_phase/client.py
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
import math
import re
import time
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError
import httpx

from ...core.config import InferenceConfig
from .domain_scoring import MAX_TOP_LOGPROBS, build_state_code_map
from ..types import DecisionTokenScores, DomainScores, LLMAttempt, PromptTrace, TokenScore

TModel = TypeVar("TModel", bound=BaseModel)

_SELECTED_VALUE_PATTERN = re.compile(r'"selected_value"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"')


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


_JSON_TOKEN_AFFIXES = frozenset(' \t\r\n"{}[],:')


def _token_affixes(token: str, value: str) -> tuple[str, str] | None:
    """Return JSON punctuation around an exact categorical token value."""
    for start in range(len(token)):
        if not token.startswith(value, start):
            continue
        prefix = token[:start]
        suffix = token[start + len(value) :]
        if all(character in _JSON_TOKEN_AFFIXES for character in prefix + suffix):
            return prefix, suffix
    return None


def _decision_scores_from_logprobs(
    response_text: str,
    token_logprobs: list[Any],
    candidate_states: tuple[str, ...],
) -> list[tuple[DomainScores | None, DecisionTokenScores | None]]:
    """Extract state and raw token alternatives at each JSON decision position."""
    response_bytes = response_text.encode("utf-8")
    try:
        token_bytes = [
            bytes(item.bytes)
            if getattr(item, "bytes", None) is not None
            else item.token.encode("utf-8")
            for item in token_logprobs
        ]
    except (AttributeError, TypeError, ValueError):
        return []
    if b"".join(token_bytes) != response_bytes:
        return []

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for value in token_bytes:
        offsets.append((cursor, cursor + len(value)))
        cursor += len(value)

    state_codes = build_state_code_map(candidate_states)
    canonical = {state.casefold(): state for state in candidate_states}
    canonical.update({code.casefold(): state for code, state in state_codes.items()})
    results: list[tuple[DomainScores | None, DecisionTokenScores | None]] = []
    for match in _SELECTED_VALUE_PATTERN.finditer(response_text):
        start = len(response_text[: match.start("value")].encode("utf-8"))
        end = len(response_text[: match.end("value")].encode("utf-8"))
        covered = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_start < end
        ]
        if not covered:
            results.append((None, None))
            continue

        try:
            selected = json.loads(f'"{match.group("value")}"')
        except json.JSONDecodeError:
            results.append((None, None))
            continue
        selected_state = canonical.get(str(selected).casefold())
        if selected_state is None:
            results.append((None, None))
            continue

        try:
            selected_scores = [float(token_logprobs[index].logprob) for index in covered]
        except (AttributeError, TypeError, ValueError):
            results.append((None, None))
            continue
        if not all(math.isfinite(score) for score in selected_scores):
            results.append((None, None))
            continue
        by_state = {selected_state: sum(selected_scores)}
        token_scores = None

        # Alternativas só são comparáveis diretamente quando o valor ocupa um token.
        # Para candidatos multi-token, o servidor não fornece a continuação condicional
        # após uma alternativa que diverge do texto efetivamente gerado.
        if len(covered) == 1:
            token_logprob = token_logprobs[covered[0]]
            raw_alternatives = []
            for alternative in getattr(token_logprob, "top_logprobs", None) or ():
                try:
                    token = str(alternative.token)
                    score = float(alternative.logprob)
                except (AttributeError, TypeError, ValueError):
                    continue
                if math.isfinite(score):
                    raw_alternatives.append(TokenScore(token=token, logprob=score))
            token_scores = DecisionTokenScores(
                selected_token=TokenScore(token=token_logprob.token, logprob=selected_scores[0]),
                top_logprobs=raw_alternatives,
            )
            affixes = _token_affixes(token_logprob.token, str(selected))
            candidates = (
                state_codes
                if str(selected) in state_codes
                else {state: state for state in candidate_states}
            )
            if affixes is not None:
                prefix, suffix = affixes
                for alternative in raw_alternatives:
                    score = alternative.logprob
                    for candidate, state in candidates.items():
                        if alternative.token == f"{prefix}{candidate}{suffix}":
                            previous = by_state.get(state, -math.inf)
                            by_state[state] = max(previous, score)
                            break

        if set(by_state) == set(candidate_states):
            results.append((DomainScores(by_state=by_state), token_scores))
        else:
            results.append((None, token_scores))
    return results


def _attach_domain_scores(
    parsed: dict[str, Any],
    response_text: str,
    token_logprobs: list[Any] | None,
    candidate_states: tuple[str, ...] | None,
) -> None:
    if not candidate_states:
        return
    decisions = parsed.get("decisions")
    if not isinstance(decisions, list):
        return

    state_codes = build_state_code_map(candidate_states)
    canonical = {state.casefold(): state for state in candidate_states}
    canonical.update({code.casefold(): state for code, state in state_codes.items()})
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        # Telemetry must come from the provider response, never from model JSON.
        decision.pop("domain_scores", None)
        decision.pop("token_scores", None)
        selected = decision.get("selected_value")
        if isinstance(selected, str):
            decoded = canonical.get(selected.strip().casefold())
            if decoded is not None:
                decision["selected_value"] = decoded

    if not token_logprobs:
        return
    try:
        scores = _decision_scores_from_logprobs(response_text, token_logprobs, candidate_states)
    except Exception:
        # Log-probs são telemetria opcional e nunca invalidam a resposta principal.
        return
    if len(scores) != len(decisions):
        return
    for decision, (domain_scores, token_scores) in zip(decisions, scores):
        if not isinstance(decision, dict):
            continue
        if domain_scores is not None:
            decision["domain_scores"] = domain_scores.model_dump()
        if token_scores is not None:
            decision["token_scores"] = token_scores.model_dump()


def _logprobs_are_unsupported(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    return status_code in {400, 422} or (
        isinstance(error, TypeError) and "logprob" in str(error).casefold()
    )


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
        self._logprobs_supported: bool | None = None

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
        candidate_states: tuple[str, ...] | None = None,
    ) -> tuple[TModel, PromptTrace]:
        if self.dry_run:
            raise RuntimeError("complete_json não pode ser chamado em modo dry-run.")
        if self._client is None:
            raise RuntimeError("Cliente LLM não foi inicializado.")

        trace = PromptTrace(purpose=purpose, variable=variable, prompt=prompt)
        retry_instruction = ""

        for attempt_number in range(1, self.config.llm_max_retries + 2):
            request_prompt = prompt + retry_instruction

            if self.config.show_llm_prompt:
                print(f"\n=== LLM Prompt (tentativa {attempt_number}) ===\n{request_prompt}\n")
                time.sleep(5)

            response_text: str | None = None
            parsed: dict[str, Any] | None = None

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
                    "temperature": self.config.openai_temperature,
                }

                # json_object só é solicitado para modelos que suportam — o LM Studio
                # suporta para a maioria dos modelos recentes, mas pode ser desabilitado
                # via config se o modelo local não suportar.
                if self.use_real_llm and self.config.openai_use_json_response_format:
                    call_kwargs["response_format"] = {"type": "json_object"}

                request_logprobs = (
                    bool(candidate_states)
                    and getattr(self, "_logprobs_supported", None) is not False
                )
                if request_logprobs:
                    call_kwargs["logprobs"] = True
                    call_kwargs["top_logprobs"] = MAX_TOP_LOGPROBS

                try:
                    completion = self._client.chat.completions.create(**call_kwargs)
                except Exception as error:
                    if not request_logprobs or not _logprobs_are_unsupported(error):
                        raise
                    self._logprobs_supported = False
                    call_kwargs.pop("logprobs", None)
                    call_kwargs.pop("top_logprobs", None)
                    completion = self._client.chat.completions.create(**call_kwargs)
                response_text = completion.choices[0].message.content or ""

                if self.config.show_llm_output:
                    print(f"\n=== LLM Response (tentativa {attempt_number}) ===\n{response_text}\n")
                    time.sleep(5)

                parsed = extract_json_object(response_text)
                choice_logprobs = getattr(completion.choices[0], "logprobs", None)
                token_logprobs = getattr(choice_logprobs, "content", None)
                if request_logprobs and "logprobs" in call_kwargs:
                    self._logprobs_supported = bool(token_logprobs)
                _attach_domain_scores(
                    parsed,
                    response_text,
                    token_logprobs,
                    candidate_states,
                )
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
                return model, trace

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
