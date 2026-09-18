"""Coordinate semantic preparation and compilation; retain the serialized type here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import InferenceConfig
from ..models import BayesianNetwork
from .graph import topological_order
from .normalization import build_alias_map
from .types import BriefingResponse, PromptTrace, SemanticMessage, VariableMetadata

COMPILED_SCHEMA_VERSION = 5


@dataclass
class CompiledSemanticMessages:
    """
    Produto da fase de compilação (Bucket Elimination com evidence={}).

    Campos
    ------
    messages          : dict variable → SemanticMessage com todas as rows
                        do produto cartesiano do separador.
    elimination_order : lista de variáveis na ordem de eliminação (reverso
                        topológico), reutilizada em reconstruct_assignment.
    briefing          : BriefingResponse gerado com evidence={}.
    bn                : BayesianNetwork parseada do BIF.
    alias_map         : mapeamento alias → variable_id (normalização de nomes).
    metadata          : metadados por variável (display_name, expert_note etc.).
    relationship_notes: notas de relacionamento qualitativo por variável.
    traces            : histórico de chamadas LLM da fase de compilação.
    """

    messages: dict[str, SemanticMessage]
    elimination_order: list[str]
    briefing: BriefingResponse
    bn: BayesianNetwork
    alias_map: dict[str, str]
    metadata: dict[str, VariableMetadata]
    relationship_notes: dict[str, tuple[str, ...]]
    traces: list[PromptTrace] = field(default_factory=list)
    schema_version: int = COMPILED_SCHEMA_VERSION


def compile_semantic_messages(
    *,
    network: BayesianNetwork,
    bif_path: Path | None = None,
    metadata_path: Path | None = None,
    relationship_path: Path | None = None,
    llm_fn,
    max_context_rows: int = 2048,
    max_context_rows_per_call: int | None = None,
    use_real_llm: bool = False,
) -> CompiledSemanticMessages:
    """
    Fase 1 — Compilação (roda UMA VEZ por dataset).

    Executa Bucket Elimination com evidence={} para garantir que cada bucket
    receba o produto cartesiano completo dos estados do separador.

    Parâmetros
    ----------
    network           : objeto de rede do pgm_llm_inference. Pode conter apenas
                        variáveis e topologia, sem CPTs.
    bif_path          : caminho opcional para o arquivo .bif. Quando omitido,
                        ``network`` é usado diretamente.
    metadata_path     : caminho para o .json de metadados (gerado se ausente).
    relationship_path : caminho para o .json de notas (gerado se ausente).
    llm_fn            : callable(prompt, response_model) → model.
    max_context_rows  : limite de linhas por bucket.
    max_context_rows_per_call : limite de linhas por chamada LLM (se None, usa max_context_rows).

    Retorna
    -------
    CompiledSemanticMessages com mensagens cobrindo todos os contextos.
    """
    # Keep the online API importable without preparing prompts or LLM clients.
    from .compile_phase.buckets import compile_bucket_messages
    from .compile_phase.client import LLMJsonClient
    from .pre_compile_phase.briefing import generate_network_briefing
    from .pre_compile_phase.metadata import load_or_generate_metadata
    from .pre_compile_phase.relationships import load_or_generate_relationship_notes

    config = InferenceConfig()

    if max_context_rows_per_call is None:
        max_context_rows_per_call = max_context_rows  # sem divisão por padrão

    bn, metadata = load_or_generate_metadata(network, bif_path, metadata_path, llm_fn, config)
    relationship_notes = load_or_generate_relationship_notes(relationship_path, bn, llm_fn, config)

    alias_map = build_alias_map(bn, metadata)

    order = topological_order(bn)
    elimination_order = list(reversed(order))
    total_vars = len(elimination_order)

    if config.show_input_data:
        print(f"\n[COMPILE] Ordem de eliminação: {total_vars} variáveis")
        print("  evidence={} → produto cartesiano completo em cada bucket")

    client = LLMJsonClient(config, use_real_llm=use_real_llm)
    traces: list[PromptTrace] = []

    briefing, trace = generate_network_briefing(bn, metadata, relationship_notes, client, config)
    traces.append(trace)

    if config.show_input_data:
        print("  ✓ Briefing concluído")
        print(f"\n[COMPILE] Bucket elimination — {total_vars} variáveis")

    messages, bucket_traces = compile_bucket_messages(
        bn=bn,
        metadata=metadata,
        relationship_notes=relationship_notes,
        alias_map=alias_map,
        briefing=briefing,
        elimination_order=elimination_order,
        client=client,
        config=config,
        max_context_rows=max_context_rows,
        max_context_rows_per_call=max_context_rows_per_call,
    )
    traces.extend(bucket_traces)

    return CompiledSemanticMessages(
        messages=messages,
        elimination_order=elimination_order,
        briefing=briefing,
        bn=bn,
        alias_map=alias_map,
        metadata=metadata,
        relationship_notes=relationship_notes,
        traces=traces,
    )
