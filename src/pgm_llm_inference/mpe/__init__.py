"""
pgm_llm_inference.mpe
=====================
Pipeline LLM-MPE (Most Probable Explanation via Semantic Bucket Elimination).

API pública
-----------
Ponto de entrada principal:

    from pgm_llm_inference.mpe import compile_semantic_messages, infer_from_compiled

Uso em dois passos:

    # Passo 1 — roda UMA VEZ por dataset
    compiled = compile_semantic_messages(
        network=bn,
        bif_path=Path("sachs.bif"),
        metadata_path=Path("sachs_metadata.json"),
        relationship_path=Path("sachs_notes.json"),
        llm_fn=my_llm_fn,
    )

    # Passo 2 — roda para CADA configuração de evidência
    hidden, confidence, cpt = infer_from_compiled(
        compiled=compiled,
        evidence={"Akt": "LOW", "Erk": "HIGH"},
        llm_fn=my_llm_fn,
    )

Tipos
-----
Os tipos Pydantic do pipeline estão em mpe.types e são acessíveis diretamente:

    from pgm_llm_inference.mpe import BayesianNetwork, Variable, SemanticMessage
"""

# Fase 1
from .compile import CompiledSemanticMessages, compile_semantic_messages

# Fase 2
from .infer import infer_from_compiled

# Tipos de rede (unificados em models/)
from ..models import BayesianNetwork, Variable

# Tipos exclusivos do pipeline MPE
from .types import (
    AuditResponse,
    BriefingResponse,
    BucketResponse,
    BucketSpec,
    ContextDecision,
    ContextEvidenceMessage,
    LLMAttempt,
    MessageRow,
    PromptTrace,
    ReconstructionResponse,
    RepairSuggestion,
    SemanticMessage,
    VariableMetadata,
)

# I/O
from .io import (
    build_alias_map,
    load_metadata,
    normalize_assignment,
    parse_bif,
    resolve_variable,
)

# Grafo
from .graph import (
    ancestors_of,
    descendants_of,
    markov_blanket,
    topological_order,
)

# Cliente LLM
from .client import LLMJsonClient, extract_json_object

__all__ = [
    # Fase 1 & 2
    "compile_semantic_messages",
    "CompiledSemanticMessages",
    "infer_from_compiled",
    # Tipos
    "BayesianNetwork",
    "Variable",
    "VariableMetadata",
    "SemanticMessage",
    "BucketSpec",
    "BriefingResponse",
    "BucketResponse",
    "ReconstructionResponse",
    "AuditResponse",
    "RepairSuggestion",
    "ContextDecision",
    "ContextEvidenceMessage",
    "MessageRow",
    "PromptTrace",
    "LLMAttempt",
    # I/O
    "parse_bif",
    "load_metadata",
    "build_alias_map",
    "resolve_variable",
    "normalize_assignment",
    # Grafo
    "topological_order",
    "markov_blanket",
    "ancestors_of",
    "descendants_of",
    # Cliente
    "LLMJsonClient",
    "extract_json_object",
]
