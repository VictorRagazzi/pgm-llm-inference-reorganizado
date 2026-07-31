"""
mpe/circuit/compile.py
=========================
Orquestra a compilação do Part 2: elicitação de fatores locais (Fase 1)
+ estrutura de buckets (Fase 2). Roda UMA VEZ por dataset; evaluate.py
responde a qualquer evidência depois, sem novas chamadas LLM.

Cache em disco separado do Part 1 (sufixo .circuit.pkl) — não colide com
tables/<stem>.compiled.pkl e não requer recompilar o Part 1 (regra de
não-contradição: Part 1 fica intacto).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ..client import LLMJsonClient
from ..io import build_alias_map, load_metadata, parse_bif
from ..metadata_generation import (
    generate_metadata_with_llm,
    generate_relationship_notes_with_llm,
    load_relationship_notes,
)
from ..prompt_builders import build_network_briefing_prompt
from ..types import BriefingResponse, PromptTrace
from ...core.config import InferenceConfig
from .elicit import elicit_semantic_factor
from .order import build_bucket_structure
from .types import SemanticCircuit


def _load_or_generate_metadata(bif_path: Path, metadata_path: Path | None, network, llm_fn):
    bn = parse_bif(bif_path)
    if metadata_path is not None and not metadata_path.exists():
        generate_metadata_with_llm(bn=network, llm_fn=llm_fn, output_path=metadata_path)
    metadata = load_metadata(metadata_path, bn)
    return bn, metadata


def _load_or_generate_relationship_notes(relationship_path: Path | None, bn, network, llm_fn):
    if relationship_path is not None and not relationship_path.exists():
        generate_relationship_notes_with_llm(
            bn=network, llm_fn=llm_fn, output_path=relationship_path
        )
    return load_relationship_notes(relationship_path, bn)


def compile_semantic_circuit(
    *,
    network,
    bif_path: Path,
    metadata_path: Path | None = None,
    relationship_path: Path | None = None,
    llm_fn,
    use_real_llm: bool = False,
    max_rows_per_call: int = 64,
    max_context_rows: int = 4096,
    verbose: bool = True,
) -> SemanticCircuit:
    """
    Fase 1 + Fase 2 combinadas: elicita um fator local por variável
    (evidence={} sempre) e constrói a estrutura de buckets (min-degree,
    sem LLM). Nenhuma passada de mensagens acontece aqui — isso é
    responsabilidade de evaluate.py, por query (Fase 3).
    """
    config = InferenceConfig()

    bn, metadata = _load_or_generate_metadata(bif_path, metadata_path, network, llm_fn)
    relationship_notes = _load_or_generate_relationship_notes(
        relationship_path, bn, network, llm_fn
    )
    alias_map = build_alias_map(bn, metadata)

    order, separators, bucket_target = build_bucket_structure(bn)

    if verbose:
        print(f"\n[COMPILE] Ordem de eliminação (min-degree): {len(order)} variáveis")
        print("  fatores locais evidence-free → reutilizáveis para qualquer query")

    client = LLMJsonClient(config, use_real_llm=use_real_llm)

    if verbose:
        print("\n[COMPILE] Gerando network briefing via LLM...")

    briefing_prompt = build_network_briefing_prompt(bn, metadata, {}, relationship_notes)
    briefing, briefing_trace = client.complete_json(
        purpose="network_briefing",
        variable=None,
        prompt=briefing_prompt,
        response_model=BriefingResponse,
    )

    if verbose:
        print("  ✓ Briefing concluído")
        print(f"\n[COMPILE] Elicitação de fatores locais — {len(order)} variáveis")

    factors = {}
    traces: list[PromptTrace] = [briefing_trace]

    for index, variable in enumerate(order, start=1):
        if verbose:
            print(f"  [{index:02d}/{len(order):02d}] Fator local: {variable}")
        factor, factor_traces = elicit_semantic_factor(
            variable=variable,
            bn=bn,
            metadata=metadata,
            relationship_notes=relationship_notes,
            alias_map=alias_map,
            client=client,
            briefing=briefing,
            max_rows_per_call=max_rows_per_call,
            max_context_rows=max_context_rows,
        )
        factors[variable] = factor
        traces.extend(factor_traces)

    return SemanticCircuit(
        factors=factors,
        elimination_order=order,
        bucket_separators=separators,
        bucket_target=bucket_target,
        bn=bn,
        alias_map=alias_map,
        metadata=metadata,
        briefing=briefing,
        traces=traces,
    )


def _circuit_cache_path(dataset_name: str, tables_dir: Path) -> Path:
    stem = Path(dataset_name).stem
    tables_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir / f"{stem}.circuit.pkl"


def load_or_compile_circuit(
    dataset_name: str,
    *,
    network,
    bif_path: Path,
    metadata_path: Path,
    relationship_path: Path,
    llm_fn,
    use_real_llm: bool,
    tables_dir: Path,
) -> SemanticCircuit:
    """Tenta carregar de tables/<stem>.circuit.pkl; senão compila e salva."""
    path = _circuit_cache_path(dataset_name, tables_dir)

    if path.exists():
        try:
            with path.open("rb") as f:
                circuit = pickle.load(f)
            print(f">>> [CACHE] Circuito carregado de '{path.name}' "
                  f"({len(circuit.factors)} fatores).")
            return circuit
        except Exception as error:
            print(f">>> [CACHE] Falha ao carregar cache ({error}), recompilando...")

    circuit = compile_semantic_circuit(
        network=network,
        bif_path=bif_path,
        metadata_path=metadata_path,
        relationship_path=relationship_path,
        llm_fn=llm_fn,
        use_real_llm=use_real_llm,
    )

    with path.open("wb") as f:
        pickle.dump(circuit, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f">>> [CACHE] Circuito salvo em '{path}'.")

    return circuit