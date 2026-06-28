from pgm_llm_inference.core.config import InferenceConfig
from pgm_llm_inference.strategies.llm.openai import create_openai_llm_function
from pgm_llm_inference.strategies.llm.local import local_llm_structured
from pydantic import BaseModel
from typing import Callable, Literal

class SemanticInferenceResult(BaseModel):
    variable: str | None = None
    value: str | None = None
    confidence: Literal["high", "medium", "low"] = ""
    reasoning: str | None = None
    internal_reasoning: str | None = None
    
class InferenceCritiqueResult(BaseModel):
    variable: str
    value: str
    confidence: Literal["high", "medium", "low"] = "medium"
    valid: bool
    reason: str | None

class ContextGenerationResult(BaseModel):
    context: str | dict[str, str] | None = None

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

def run_inference(llm_fn, prompt: str) -> SemanticInferenceResult:
    return llm_fn(prompt, SemanticInferenceResult)

def run_critique(llm_fn, prompt: str) -> InferenceCritiqueResult:
    return llm_fn(prompt, InferenceCritiqueResult)