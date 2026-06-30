from typing import Callable, TypeVar, Type
from pgm_llm_inference.core.config import InferenceConfig
from pgm_llm_inference.utils import _extract_last_json_object, _extract_text_from_message
import time
from pydantic import BaseModel


T = TypeVar("T")

def create_openai_llm_function(config: InferenceConfig) -> Callable[[str], str]:
    """
    Create a wrapper function for querying an OpenAI (or compatible) chat model.

    The returned function is a lightweight callable that:
        - Sends a user prompt to an OpenAI API endpoint
        - Uses deterministic generation (temperature=0)
        - Extracts and returns the model's textual response

    This is mainly intended for injection into high-level inference pipelines.

    Args:
        config: Configuration object containing API key, base URL, model name,
                and timeout settings.

    Returns:
        A function that accepts a prompt string and returns the model output.

    Raises:
        ImportError: If the openai package is not installed.
        ValueError: If the API key is missing.
        RuntimeError: If the API call fails.

    Example:
        >>> config = InferenceConfig(openai_api_key="sk-...")
        >>> llm_fn = create_openai_llm_function(config)
        >>> answer = llm_fn("What is a Bayesian Network?")
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI package not installed. Install with: pip install openai"
        )

    if not config.openai_api_key:
        raise ValueError("OpenAI API key not configured")

    client = OpenAI(
        api_key=config.openai_api_key,
        base_url=config.openai_base_url,
    )

    def query_llm(prompt: str, schema: type[T]) -> T:
        try:
            # request_prompt = prompt + retry_instruction
            if (config.show_llm_prompt):
                print(f"================== LLM Prompt ====================\n{prompt}")
                # time.sleep(8)

            response = client.chat.completions.create(
                model=config.openai_model,
                messages=[{
                            "role": "system",
                            "content": (
                                "You are a specialist in Bayesian networks, "
                                "probabilistic graphical models, and biological "
                                "signaling pathways. Return valid JSON only."
                            ),
                        },
                        {"role": "user", "content": prompt}],
                max_tokens=60000,
                timeout=config.llm_timeout,
                extra_body={
                    "reasoning": {
                        "effort": "low",
                    }
                },
            )
            message  = response.choices[0].message
            raw_text = message.content or ""

            if (config.show_llm_output):
                # time.sleep(2.5)
                print(f"================== LLM Output ====================\n{raw_text}")

            data = _extract_last_json_object(raw_text)
            return schema.model_validate(data)

        except Exception as e:
            raise RuntimeError(f"LLM query failed: {e}") from e

    return query_llm

def pydantic_to_openai_json_schema(schema: Type[BaseModel]) -> dict:
    json_schema = schema.model_json_schema()

    properties = json_schema.get("properties", {})
    required = list(properties.keys())

    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }