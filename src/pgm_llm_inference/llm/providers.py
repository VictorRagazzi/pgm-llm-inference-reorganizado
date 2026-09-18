"""Provider selection and structured remote/local LLM calls.

Metadata and relationship generation keep their own request and parsing contract.
Bucket compilation uses the specialized client in mpe.compile_phase.client.
"""

import json
from typing import Callable, TypeVar

import requests
from pydantic import BaseModel

from pgm_llm_inference.core.config import InferenceConfig
from .parsing import extract_last_json_object

T = TypeVar("T", bound=BaseModel)


def get_model_name(use_real_llm: bool) -> str:
    config = InferenceConfig()
    if use_real_llm:
        return config.openai_model
    else:
        return config.local_model


def build_llm_fn(*, use_real_llm: bool, use_local_llm: bool):
    if use_real_llm:
        print("🔌 Using REAL LLM")
        config = InferenceConfig()
        return create_openai_llm_function(config)

    if use_local_llm:
        print("🖥️ Using LOCAL LLM")
        return local_llm_structured

    print("🧪 Using MOCK LLM")

    def mock_llm(prompt: str, schema):
        return schema()

    return mock_llm


def create_openai_llm_function(config: InferenceConfig) -> Callable[[str, type[T]], T]:
    """Return a structured remote client using the metadata request contract."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")

    if not config.openai_api_key:
        raise ValueError("OpenAI API key not configured")

    client = OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
    )

    def query_llm(prompt: str, schema: type[T]) -> T:
        try:
            if config.show_llm_prompt:
                print(f"================== LLM Prompt ====================\n{prompt}")

            response = client.chat.completions.create(
                model=config.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a specialist in Bayesian networks, "
                            "probabilistic graphical models, and biological "
                            "signaling pathways. Return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=60000,
                timeout=config.llm_timeout,
                extra_body={
                    "reasoning": {
                        "effort": "low",
                    }
                },
            )
            message = response.choices[0].message
            raw_text = message.content or ""

            if config.show_llm_output:
                print(f"================== LLM Output ====================\n{raw_text}")

            data = extract_last_json_object(raw_text)
            return schema.model_validate(data)

        except Exception as e:
            raise RuntimeError(f"LLM query failed: {e}") from e

    return query_llm


def local_llm_structured(prompt: str, schema: type[T]) -> T:
    raw = local_llm(prompt)
    try:
        data = extract_last_json_object(raw)
    except Exception as e:
        raise RuntimeError(f"Local LLM did not return valid JSON.\nRaw output:\n{raw}") from e

    try:
        return schema.model_validate(data)
    except Exception as e:
        raise RuntimeError(f"JSON does not match schema {schema.__name__}.\n{data}") from e


def local_llm(prompt: str) -> str:
    """Collect streamed text from the configured local chat endpoint."""
    config = InferenceConfig()
    local_url = config.local_url
    local_model = config.local_model

    payload = {
        "model": local_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.01,
        "stream": True,
    }
    try:
        resp = requests.post(local_url, json=payload, stream=True, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Local LLM request failed: {e}")

    if resp.status_code != 200:
        # try to show body for debug
        body = ""
        try:
            body = resp.text
        except Exception:
            body = "<could not read body>"
        raise RuntimeError(f"Local LLM returned {resp.status_code}: {body}")

    # Attempt to consume streaming chunks. Many local servers send "data: ..." lines.
    text_parts = []
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        # Drop SSE prefix if present
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if line in ("[DONE]", ""):
            continue
        # Try parse JSON chunk (OpenAI-like)
        try:
            obj = json.loads(line)
            # OpenAI-style streamed chunk structure: {"choices":[{"delta":{"content":"..."}}, ...]}
            choices = obj.get("choices")
            if choices:
                for c in choices:
                    delta = c.get("delta", {})
                    content = delta.get("content")
                    if content:
                        text_parts.append(content)
                    # some servers return full text in 'message' or 'text'
                    elif "message" in c and isinstance(c["message"], dict):
                        cont = c["message"].get("content")
                        if isinstance(cont, str):
                            text_parts.append(cont)
                    elif "text" in c and isinstance(c["text"], str):
                        text_parts.append(c["text"])
            else:
                # fallback: try top-level text
                if isinstance(obj.get("text"), str):
                    text_parts.append(obj["text"])
        except Exception:
            # Not JSON -- treat as plain text chunk
            text_parts.append(line)

    return "".join(text_parts)
