"""
Utility functions for the library.

Parsing helpers for LLM outputs (extracting JSON objects and raw text from
chat-completion messages) and a shared counter used to track LLM call volume
across an experiment run.
"""

import ast
import json
import re
import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from .mpe.compile import compile_semantic_messages   # ajuste o import

if TYPE_CHECKING:
    from .mpe.compile import CompiledSemanticMessages

# Contador global de chamadas LLM, resetado manualmente entre experimentos
# (ver scripts/main.py e scripts/main_single_run.py).
llm_request_count = 0

TABLES_DIR = Path(__file__).resolve().parents[1] / "tables"

def _compiled_path(dataset_name: str) -> Path:
    stem = Path(dataset_name).stem          # "hepar2.bif" → "hepar2"
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    return TABLES_DIR / f"{stem}.compiled.pkl"


def load_or_compile(
    dataset_name: str,
    *,
    network,
    bif_path: Path,
    metadata_path: Path,
    relationship_path: Path,
    llm_fn,
    use_real_llm: bool,
) -> "CompiledSemanticMessages":
    """
    Tenta carregar CompiledSemanticMessages de tables/<stem>.compiled.pkl.
    Se não existir (ou estiver corrompido), compila e salva.
    """
    path = _compiled_path(dataset_name)

    if path.exists():
        print(f">>> [CACHE] Carregando tabelas compiladas de '{path.name}'...")
        try:
            with path.open("rb") as f:
                compiled = pickle.load(f)
            print(f">>> [CACHE] ✓ {len(compiled.messages)} mensagens carregadas do cache.")
            return compiled
        except Exception as e:
            print(f">>> [CACHE] ⚠ Falha ao carregar cache ({e}), recompilando...")

    compiled = compile_semantic_messages(
        network=network,
        bif_path=bif_path,
        metadata_path=metadata_path,
        relationship_path=relationship_path,
        llm_fn=llm_fn,
        use_real_llm=use_real_llm,
    )
    print(f">>> [COMPILE] ✓ {len(compiled.messages)} mensagens compiladas.")

    print(f">>> [CACHE] Salvando em '{path}'...")
    with path.open("wb") as f:
        pickle.dump(compiled, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f">>> [CACHE] ✓ Salvo.")

    return compiled

def _extract_last_json_object(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Empty LLM output")

    # 1. Encontrar blocos entre chaves { ... } com suporte a aninhamento
    stack = []
    start = None
    results = []

    for i, char in enumerate(raw):
        if char == '{':
            if not stack:
                start = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    results.append(raw[start:i+1])

    if not results:
        raise ValueError(f"No JSON object found. Raw: {raw}")

    # Pegamos o último objeto encontrado
    candidate = results[-1].strip()

    # 2. LIMPEZA DE COMENTÁRIOS
    # Remove comentários de linha única (// ...) que não estejam dentro de URLs
    candidate = re.sub(r'(?<![:/])//.*', '', candidate)

    # 3. Limpeza de quebras de linha dentro de strings
    def replace_newlines(match):
        return match.group(0).replace('\n', '\\n').replace('\r', '\\r')

    candidate = re.sub(r'"(.*?)"', replace_newlines, candidate, flags=re.DOTALL)

    # 4. Tentativa de Parse
    try:
        # Tenta o JSON padrão primeiro (geralmente funciona após remover //)
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Tenta ast.literal_eval como fallback para aspas simples ou formatos Python-like
        try:
            parsed = ast.literal_eval(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        # Última tentativa: limpeza agressiva de caracteres de controle
        try:
            clean_candidate = re.sub(r'[\x00-\x1F]+', ' ', candidate)
            return json.loads(clean_candidate)
        except Exception:
            raise ValueError(f"Failed to parse JSON. Error: {str(e)}. Candidate: {candidate}")


def _extract_text_from_message(msg) -> str:
    """
    Extract textual output from:
    1) legacy string content
    2) multimodal content blocks
    3) GPT-5 reasoning / reasoning_details summaries
    """

    # 1. Legacy: content é string
    if isinstance(msg.content, str):
        text = msg.content.strip()
        if text:
            return text

    # 2. Multimodal: content é lista de blocos
    if isinstance(msg.content, list):
        parts: list[str] = []

        for block in msg.content:
            if isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text", ""))

        text = "".join(parts).strip()
        if text:
            return text

    # 3. GPT-5+: reasoning como string direta
    reasoning = getattr(msg, "reasoning", None)
    if isinstance(reasoning, str):
        text = reasoning.strip()
        if text:
            return text

    # 4. GPT-5+: reasoning_details com summaries
    details = getattr(msg, "reasoning_details", None)
    if isinstance(details, list):
        parts: list[str] = []

        for item in details:
            if (
                isinstance(item, dict)
                and item.get("type") == "reasoning.summary"
                and isinstance(item.get("summary"), str)
            ):
                parts.append(item["summary"])

        text = "\n".join(parts).strip()
        if text:
            return text

    # Nada aproveitável
    raise ValueError(
        "LLM returned no usable textual output. "
        f"Full message: {msg.model_dump()}"
    )
