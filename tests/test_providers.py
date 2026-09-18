import json
from types import SimpleNamespace

import openai
import pytest
from pydantic import BaseModel

from pgm_llm_inference.core.config import InferenceConfig
from pgm_llm_inference.llm import providers
from pgm_llm_inference.llm.parsing import extract_last_json_object
from pgm_llm_inference.mpe.compile_phase.client import LLMJsonClient, extract_json_object


class Answer(BaseModel):
    value: str


def test_remote_request_and_last_json_parsing(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"value": "draft"}\n{"value": "final"}')
                )
            ]
        )

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    config = InferenceConfig(openai_api_key="fake-test-key", openai_model="test-model")
    result = providers.create_openai_llm_function(config)("prompt", Answer)
    assert result.value == "final"
    assert calls == [
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a specialist in Bayesian networks, "
                    "probabilistic graphical models, and biological signaling pathways. "
                    "Return valid JSON only.",
                },
                {"role": "user", "content": "prompt"},
            ],
            "max_tokens": 60000,
            "timeout": config.llm_timeout,
            "extra_body": {"reasoning": {"effort": "low"}},
        }
    ]


def test_local_stream_and_request_parameters(monkeypatch):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs))
        chunks = [
            "data: " + json.dumps({"choices": [{"delta": {"content": part}}]})
            for part in ['{"value":', '"final"}']
        ]
        return SimpleNamespace(
            status_code=200, iter_lines=lambda **_: iter([*chunks, "data: [DONE]"])
        )

    monkeypatch.setattr(providers.requests, "post", post)
    assert providers.local_llm_structured("prompt", Answer).value == "final"
    config = InferenceConfig()
    assert calls == [
        (
            config.local_url,
            {
                "json": {
                    "model": config.local_model,
                    "messages": [{"role": "user", "content": "prompt"}],
                    "temperature": 0.01,
                    "stream": True,
                },
                "stream": True,
                "timeout": 60,
            },
        )
    ]


def test_parsers_keep_different_contracts():
    raw = '{"value": "first"}\n{"value": "last"}'
    assert extract_last_json_object(raw) == {"value": "last"}
    with pytest.raises(json.JSONDecodeError):
        extract_json_object(raw)
    assert extract_json_object('```json\n{"value": "only"}\n```') == {"value": "only"}


def test_semantic_client_validation_retry():
    calls = []
    responses = iter(['{"wrong": "schema"}', '{"value": "valid"}'])

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]
        )

    client = LLMJsonClient(InferenceConfig(llm_max_retries=1), dry_run=True)
    client.dry_run = False
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    result, trace = client.complete_json(
        purpose="test", variable=None, prompt="prompt", response_model=Answer
    )
    assert result.value == "valid"
    assert len(trace.attempts) == 2
    assert trace.attempts[0].error and trace.attempts[1].error is None
    assert calls[0]["messages"][1]["content"] == "prompt"
    assert "Your previous response was invalid" in calls[1]["messages"][1]["content"]
