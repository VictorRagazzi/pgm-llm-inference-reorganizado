import pickle
from pathlib import Path

from pgm_llm_inference.paths import COMPILED_TABLES_DIR

from .compile import (
    COMPILED_SCHEMA_VERSION,
    CompiledSemanticMessages,
    compile_semantic_messages,
)


def compiled_cache_path(dataset_name: str, model_name: str) -> Path:
    """Return the selected cache location for a dataset and model."""
    COMPILED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(dataset_name).stem

    # return COMPILED_TABLES_DIR / f"{stem}.compiled.pkl"
    safe_model = model_name.replace("/", "__").replace(":", "-")
    return COMPILED_TABLES_DIR / f"{stem}.{safe_model}_VE.compiled.pkl"

def load_or_compile(
    dataset_name: str,
    *,
    model_name: str,
    network,
    bif_path: Path,
    metadata_path: Path,
    relationship_path: Path,
    llm_fn,
    use_real_llm: bool,
    max_context_rows_per_call: int | None = None,
) -> CompiledSemanticMessages:
    """Load a compiled MPE table from cache, or compile and persist it."""
    path = compiled_cache_path(dataset_name, model_name)

    if path.exists():
        print(f">>> [CACHE] Carregando tabelas compiladas de '{path.name}'...")
        try:
            with path.open("rb") as file:
                compiled = pickle.load(file)
            if getattr(compiled, "schema_version", None) != COMPILED_SCHEMA_VERSION:
                raise ValueError("formato de cache incompatível")
            print(f">>> [CACHE] OK: {len(compiled.messages)} mensagens carregadas do cache.")
            return compiled
        except Exception as error:
            print(f">>> [CACHE] WARN: falha ao carregar cache ({error}), recompilando...")

    compiled = compile_semantic_messages(
        network=network,
        bif_path=bif_path,
        metadata_path=metadata_path,
        relationship_path=relationship_path,
        llm_fn=llm_fn,
        use_real_llm=use_real_llm,
        max_context_rows_per_call=max_context_rows_per_call,
    )
    print(f">>> [COMPILE] OK: {len(compiled.messages)} mensagens compiladas.")

    print(f">>> [CACHE] Salvando em '{path}'...")
    with path.open("wb") as file:
        pickle.dump(compiled, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(">>> [CACHE] OK: salvo.")

    return compiled
