"""
mpe/compile.py
==============
Fase 1 do pipeline MPE: Bucket Elimination com evidence={}.

compile_semantic_messages() roda UMA VEZ por dataset e produz
CompiledSemanticMessages — um snapshot de todas as SemanticMessages
cobrindo o produto cartesiano completo dos estados dos pais de cada variável.

Invariante central (Opção A):
    active_messages permanece [] durante toda a compilação.
    Com evidence={}, bucket_has_evidence_pressure=False para toda variável.
    separator(v) == tuple(sorted(parents(v))).
    Nenhuma mensagem é propagada: produto cartesiano completo garantido.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import InferenceConfig
from .bucket import build_bucket_spec, semantic_message_from_response
from .client import LLMJsonClient
from .io import build_alias_map, load_metadata, parse_bif
from .prompt_builders import build_bucket_prompt, build_network_briefing_prompt
from ..models import BayesianNetwork
from .types import (
    BriefingResponse,
    BucketResponse,
    PromptTrace,
    SemanticMessage,
    VariableMetadata,
)
from .graph import topological_order


# ---------------------------------------------------------------------------
# Tipo público
# ---------------------------------------------------------------------------

@dataclass
class CompiledSemanticMessages:
    """
    Produto da fase de compilação (Bucket Elimination com evidence={}).

    Campos
    ------
    messages          : dict  variable → SemanticMessage com todas as rows
                        do produto cartesiano dos pais.
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


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _load_or_generate_metadata(
    network,
    bif_path: Path,
    metadata_path: Path | None,
    llm_fn,
    config: InferenceConfig,
) -> tuple[BayesianNetwork, dict[str, VariableMetadata]]:
    from .metadata_generation import generate_metadata_with_llm

    bn = parse_bif(bif_path)

    if config.show_input_data:
        print("\n[COMPILE] Carregando metadados da rede...")

    if metadata_path is None or not metadata_path.exists():
        if config.show_input_data:
            print("  → Gerando metadados via LLM (arquivo não encontrado)...")
        generate_metadata_with_llm(bn=network, llm_fn=llm_fn, output_path=metadata_path)

    metadata = load_metadata(metadata_path, bn)

    if config.show_input_data:
        print(f"  ✓ Metadados carregados: {len(metadata)} entradas")

    return bn, metadata


def _load_or_generate_relationship_notes(
    network,
    relationship_path: Path | None,
    bn: BayesianNetwork,
    llm_fn,
    config: InferenceConfig,
) -> dict[str, tuple[str, ...]]:
    from .metadata_generation import (
        generate_relationship_notes_with_llm,
        load_relationship_notes,
    )

    if config.show_input_data:
        print("\n[COMPILE] Carregando relationship notes...")

    if relationship_path is None or not relationship_path.exists():
        if config.show_input_data:
            print("  → Gerando relationship notes via LLM (arquivo não encontrado)...")
        generate_relationship_notes_with_llm(
            bn=network, llm_fn=llm_fn, output_path=relationship_path
        )

    relationship_notes = load_relationship_notes(relationship_path, bn)

    if config.show_input_data:
        print(
            f"  ✓ Relationship notes carregadas: "
            f"{len(relationship_notes)} variáveis anotadas"
        )

    return relationship_notes


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compile_semantic_messages(
    *,
    network,
    bif_path: Path,
    metadata_path: Path | None = None,
    relationship_path: Path | None = None,
    llm_fn,
    max_context_rows: int = 512,
    use_real_llm: bool = False,
) -> CompiledSemanticMessages:
    """
    Fase 1 — Compilação (roda UMA VEZ por dataset).

    Executa Bucket Elimination com evidence={} para garantir que cada bucket
    receba o produto cartesiano completo dos estados dos pais.

    Parâmetros
    ----------
    network           : objeto de rede do pgm_llm_inference (para geração de
                        metadados, caso necessário).
    bif_path          : caminho para o arquivo .bif do dataset.
    metadata_path     : caminho para o .json de metadados (gerado se ausente).
    relationship_path : caminho para o .json de notas (gerado se ausente).
    llm_fn            : callable(prompt, response_model) → model.
    max_context_rows  : limite de linhas por bucket.

    Retorna
    -------
    CompiledSemanticMessages com mensagens cobrindo todos os contextos.
    """
    config = InferenceConfig()

    bn, metadata = _load_or_generate_metadata(
        network, bif_path, metadata_path, llm_fn, config
    )
    relationship_notes = _load_or_generate_relationship_notes(
        network, relationship_path, bn, llm_fn, config
    )

    alias_map = build_alias_map(bn, metadata)
    compile_evidence: dict[str, str] = {}

    order = topological_order(bn)
    order_index = {variable: index for index, variable in enumerate(order)}
    elimination_order = list(reversed(order))
    total_vars = len(elimination_order)

    if config.show_input_data:
        print(f"\n[COMPILE] Ordem de eliminação: {total_vars} variáveis")
        print("  evidence={} → produto cartesiano completo em cada bucket")

    client = LLMJsonClient(config, use_real_llm=use_real_llm)
    traces: list[PromptTrace] = []

    # Briefing gerado com evidence={}
    if config.show_input_data:
        print("\n[COMPILE] Gerando network briefing via LLM...")

    briefing_prompt = build_network_briefing_prompt(
        bn, metadata, compile_evidence, relationship_notes
    )
    briefing, trace = client.complete_json(
        purpose="network_briefing",
        variable=None,
        prompt=briefing_prompt,
        response_model=BriefingResponse,
    )
    traces.append(trace)

    if config.show_input_data:
        print("  ✓ Briefing concluído")
        print(f"\n[COMPILE] Bucket elimination — {total_vars} variáveis")

    messages: dict[str, SemanticMessage] = {}
    active_messages: list[SemanticMessage] = []  # invariante: sempre []

    for var_index, variable in enumerate(elimination_order, start=1):
        if config.show_input_data:
            print(f"  [{var_index:02d}/{total_vars:02d}] Bucket: {variable}")

        bucket = build_bucket_spec(
            variable=variable,
            bn=bn,
            evidence=compile_evidence,
            active_messages=active_messages,
            order_index=order_index,
            max_context_rows=max_context_rows,
        )

        prompt = build_bucket_prompt(
            bucket, bn, metadata, briefing, compile_evidence, relationship_notes
        )

        def _make_validator(bkt, am):
            def _validate(model: BucketResponse) -> None:
                semantic_message_from_response(
                    model, bkt, bn=bn, alias_map=am, evidence=compile_evidence
                )
            return _validate

        response, trace = client.complete_json(
            purpose="bucket_argmax",
            variable=variable,
            prompt=prompt,
            response_model=BucketResponse,
            semantic_validator=_make_validator(bucket, alias_map),
        )
        traces.append(trace)

        message = semantic_message_from_response(
            response, bucket, bn=bn, alias_map=alias_map, evidence=compile_evidence
        )

        if config.show_input_data:
            print(
                f"         → evidence_driven={message.evidence_driven} "
                f"| scope={message.scope} "
                f"| rows={len(message.rows)}"
            )

        if message.evidence_driven:
            raise RuntimeError(
                f"[COMPILE] Invariante violada: {variable} retornou "
                "evidence_driven=True com evidence={}. "
                "Verifique bucket_has_evidence_pressure."
            )

        # active_messages não atualizado — invariante mantida
        messages[variable] = message

    if config.show_input_data:
        print(
            f"\n[COMPILE] ✓ Compilação concluída: "
            f"{len(messages)} mensagens, "
            f"{sum(len(m.rows) for m in messages.values())} rows totais"
        )

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
