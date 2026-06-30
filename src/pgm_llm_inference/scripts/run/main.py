from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pgm_llm_inference.utils as utils
from pgm_llm_inference.utils import load_or_compile
from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.logging.experiment_logger import log_experiment, log_experiment_csv
from pgm_llm_inference.experiment.llm_factory import build_llm_fn
from pgm_llm_inference.experiment.experiment import get_hidden_vars
from pgm_llm_inference.mpe import compile_semantic_messages, infer_from_compiled, CompiledSemanticMessages, parse_bif
from pgm_llm_inference.experiment.batch import run_batch, make_evidence_sizes
from pgm_llm_inference.evaluation.metrics import count_llm_hits
from pgm_llm_inference.experiment.llm_factory import get_model_name

# re-exportado de runners para o TRUE MPE
from pgm_llm_inference.experiment.experiment import run_max_product

MPE = "mpe"


def _notify_beep(frequency: int, duration_ms: int) -> None:
    """
    Toca um beep curto ao final de um dataset/da execução inteira.

    Usa winsound apenas no Windows; em qualquer outro SO (ou se o beep
    falhar por qualquer motivo) a notificação é simplesmente ignorada —
    não deve interromper o experimento.
    """
    import sys

    if sys.platform != "win32":
        return
    try:
        import winsound
        winsound.Beep(frequency, duration_ms)
    except Exception:
        pass


@dataclass
class ExperimentConfig:
    dataset_name: str
    use_real_llm: bool = True
    use_local_llm: bool = True
    inference_mode: str = MPE
    context_type: Optional[str] = None
    prompt_critique: str = "critique_inference"
    prompt_types: List[str] = field(default_factory=lambda: ["simple"])
    evidence_sizes: List[int] = field(default_factory=list)
    query_sizes: List[int] = field(default_factory=lambda: [1])
    n_trials: int = 3
    max_estimated_llm_calls: int = 300

# ---------------------------------------------------------------------------
# Funções de execução
# ---------------------------------------------------------------------------

def run_experiment(
    network,
    batch_cfg: Dict,
    global_cfg: ExperimentConfig,
    llm_fn,
    compiled: CompiledSemanticMessages,
    bif_path: str,
) -> Dict[str, Any]:
    """
    Executa um experimento único usando as mensagens pré-compiladas.

    Fluxo:
      1. TRUE MPE via max-product (numérico, inalterado).
      2. infer_from_compiled — lookup O(N) + reconstruction + audit (2 LLM calls).
      3. Métricas e log.

    Parâmetros
    ----------
    compiled : resultado de compile_semantic_messages(), gerado UMA VEZ por dataset.
               Reutilizado para todas as configurações de evidência do batch.
    """
    evidence = batch_cfg["evidence"]
    hidden_vars = get_hidden_vars(network, evidence)

    print("\n" + "=" * 60)
    print(f"Prompt type : {batch_cfg['prompt_type']}")
    print(f"Evidence    : {evidence}")
    print(f"Evidence len: {len(evidence)}")
    print(f"Hidden vars : {hidden_vars}")
    print("=" * 60)

    # Guard: mantém a proteção contra redes muito grandes
    if len(hidden_vars) > global_cfg.max_estimated_llm_calls:
        raise ValueError(
            f"Maximum number of LLM calls reached ({global_cfg.max_estimated_llm_calls}), "
            "skipping to next experiment"
        )

    # --- TRUE MPE (inalterado) ---
    mpe_result = run_max_product(
        network=network,
        query_vars=hidden_vars,
        evidence=evidence,
    )
    mpe_assignment = mpe_result["map_assignment"]

    # --- LLM MPE via mensagens compiladas ---
    # Não roda Bucket Elimination; apenas lookup + reconstruction + audit.
    llm_predictions, confidence_map, llm_cpt = infer_from_compiled(
        compiled=compiled,
        evidence=evidence,
        llm_fn=llm_fn,
        apply_audit_repair_enabled=True,
        use_real_llm=global_cfg.use_real_llm
    )

    # --- Métricas ---
    if global_cfg.inference_mode == MPE:
        result_hits_input = llm_predictions
        evaluated_vars = [
            v for v in mpe_assignment
            if v not in evidence and v != "_scalar"
        ]
    else:
        result_hits_input = llm_predictions   # MAP não usa compiled, mas mantemos assinatura
        evaluated_vars = batch_cfg["query_vars"]

    llm_hits, llm_predictions_out = count_llm_hits(
        result=result_hits_input,
        query_vars=batch_cfg["query_vars"],
        map_assignment=mpe_assignment,
        inference_mode=global_cfg.inference_mode,
    )

    accuracy = llm_hits / len(evaluated_vars) if evaluated_vars else 0.0

    # --- Print results ---
    print("\nRESULTS")
    print("-" * 40)
    for v in hidden_vars:
        llm_val = llm_predictions.get(v, "N/A")
        mpe_val = mpe_assignment.get(v, "N/A")
        print(f"{v}: LLM={llm_val} | MPE={mpe_val}")

    return {
        "evidence_length": len(evidence),
        "hits": llm_hits,
        "log_accuracy": accuracy,
        "accuracy": f"{accuracy * 100:.2f}%",
        "prompt_type": batch_cfg["prompt_type"],
        "context_type": global_cfg.context_type,
        "dataset": global_cfg.dataset_name,
        "model_name": get_model_name(global_cfg.use_real_llm),
        "evaluated_length": len(evaluated_vars),
        "evidence": evidence,
        "llm_predictions": llm_predictions_out,
        "map_assignment": mpe_assignment,
        "confidence": confidence_map,
        "llm_cpt": llm_cpt,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    datasets = [
        "gonorrhoeae.bif",
        "diabets.bif",
        # "aspergillus.bif",
        # "adhd.bif",
        # "munin1.bif",
        # "hepar2.bif",

        # "cryptocurrency.bif",
        # "sachs.bif",
        # "coronary.bif",
        # "crimescene.bif",
        # "coral1.bif",
        # "asia.bif",
        # "insurance.bif",
    ]

    cfg = ExperimentConfig(
        dataset_name="",
        prompt_types=["variable_assignment"],
    )

    llm_fn = build_llm_fn(use_real_llm=cfg.use_real_llm, use_local_llm=cfg.use_local_llm)

    BASE_DIR = Path(__file__).resolve().parents[3]

    for name in datasets:
        print(f"\n{'=' * 60}")
        print(f">>> Dataset: {name}")
        print(f"{'=' * 60}")
        cfg.dataset_name = name

        # --- Carregamento da rede ---
        path = BASE_DIR / "datasets" / name
        network, context = load_network(str(path), cfg.context_type, llm_fn)

        # --- COMPILAÇÃO: roda UMA vez por dataset ---
        # Executa Bucket Elimination com evidence={} → produto cartesiano completo.
        # Custo: N chamadas LLM (uma por variável).
        # --- Tamanho do batch baseado na rede compilada ---
        num_nodes = len(network.variables.keys())
        limit = int(num_nodes * 0.5)
        current_evidence_sizes = make_evidence_sizes(limit)
        # current_evidence_sizes = list(range(1, limit + 1))
        print(f"\n>>> Testando evidence sizes {current_evidence_sizes}")


        print(f"\n>>> [COMPILE] Compilando mensagens semânticas para '{name}'...")
        compiled = load_or_compile(
            dataset_name=name,
            network=network,
            bif_path=path,
            metadata_path=BASE_DIR / "metadata" / f"{name.split('.')[0]}.jsonl",
            relationship_path=BASE_DIR / "relationships" / f"{name.split('.')[0]}.jsonl",
            llm_fn=llm_fn,
            use_real_llm=cfg.use_real_llm,
        )
        print(f">>> [COMPILE] ✓ {len(compiled.messages)} mensagens compiladas.")


        # --- INFERÊNCIA: roda para cada evidência — sem LLM no bucket ---
        # Custo por inferência: 0 chamadas LLM no bucket + 2 LLM (reconstruction + audit).
        for batch_config in run_batch(
            network=network,
            prompt_types=cfg.prompt_types,
            evidence_sizes=current_evidence_sizes,
            n_trials=cfg.n_trials,
            inference_mode=cfg.inference_mode,
            llm_fn=llm_fn,
        ):
            try:
                utils.llm_request_count = 0  # reset por tentativa

                log_data = run_experiment(
                    network=network,
                    batch_cfg=batch_config,
                    global_cfg=cfg,
                    llm_fn=llm_fn,
                    compiled=compiled,          # ← mensagens pré-compiladas
                    bif_path=path,              # type: ignore
                )

                log_experiment(log_data)

            except Exception as e:
                print(f"❌ Error in {name}: {e}")

        _notify_beep(440, 500)

    _notify_beep(600, 1000)


if __name__ == "__main__":
    main()