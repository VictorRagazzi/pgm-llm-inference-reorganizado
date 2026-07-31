"""
scripts/run/main.py
====================
Ponto de entrada único do pipeline MPE-LLM.

TUDO que costuma mudar de execução pra execução mora em CONFIG, logo
abaixo dos imports: datasets, número de evidências, trials por tamanho
de evidência, e a flag `pipeline` que escolhe qual versão rodar:

    PART1         -> forward-only, evidência entra na reconstrução
                     (mpe.compile / mpe.infer, original).
    PART2_FORWARD -> circuito Part 2 em modo forward-only (mesma lógica
                     do Part 1, mas usando os fatores do circuito — é a
                     âncora de comparação E2.1).
    PART2_TWOPASS -> circuito Part 2 completo: passada ascendente +
                     reconstrução por backpointers, corrige diagnostica-
                     mente ancestrais a partir de evidência a jusante.

Todas as três opções são compiladas UMA VEZ por dataset (cache em
tables/) e depois avaliadas para cada configuração de evidência do
sweep.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pgm_llm_inference.utils as utils
from pgm_llm_inference.utils import load_or_compile, TABLES_DIR
from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.logging.experiment_logger import log_experiment, log_experiment_csv
from pgm_llm_inference.experiment.llm_factory import build_llm_fn, get_model_name
from pgm_llm_inference.experiment.experiment import get_hidden_vars, run_max_product
from pgm_llm_inference.mpe import infer_from_compiled
from pgm_llm_inference.mpe.circuit import (
    evaluate_forward_only,
    evaluate_two_pass,
    load_or_compile_circuit,
)
from pgm_llm_inference.experiment.batch import run_batch, make_evidence_sizes
from pgm_llm_inference.evaluation.metrics import count_llm_hits

MAP = "map"
MPE = "mpe"

PART1 = "part1"
PART2_FORWARD = "part2_forward"
PART2_TWOPASS = "part2_twopass"
PIPELINES = (PART1, PART2_FORWARD, PART2_TWOPASS)


# ===========================================================================
# CONFIG — edite só aqui pra mudar datasets, tamanho do sweep, pipeline, etc.
# ===========================================================================

@dataclass
class ExperimentConfig:
    # Quais redes rodar, em ordem. Descomente/adicione conforme necessário.
    datasets: List[str] = field(default_factory=lambda: [
        "gonorrhoeae.bif",
        # "diabets.bif",
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
    ])

    # A flag pedida: qual pipeline usar. Ver PIPELINES acima.
    pipeline: str = PART1

    use_real_llm: bool = True
    use_local_llm: bool = True
    inference_mode: str = MPE
    context_type: Optional[str] = None
    prompt_types: List[str] = field(default_factory=lambda: ["variable_assignment"])

    # --- Controle do sweep de evidência ---
    # Se evidence_sizes for None, gera automaticamente via make_evidence_sizes
    # a partir de evidence_size_ratio * (nº de variáveis da rede). Para fixar
    # manualmente (ex.: só testar evidência de tamanho 1, 3 e 5), preencha
    # evidence_sizes=[1, 3, 5] e evidence_size_ratio é ignorado.
    evidence_sizes: Optional[List[int]] = None
    evidence_size_ratio: float = 0.5

    query_sizes: List[int] = field(default_factory=lambda: [1, 2])

    # Nº de configurações de evidência sorteadas POR tamanho de evidência.
    n_trials: int = 3

    max_estimated_llm_calls: int = 300

    # Preenchido em runtime (main() seta por dataset, não editar aqui).
    dataset_name: str = ""

    def __post_init__(self):
        if self.inference_mode not in (MAP, MPE):
            raise ValueError(f"Invalid inference_mode: {self.inference_mode}")
        if self.pipeline not in PIPELINES:
            raise ValueError(f"Invalid pipeline: {self.pipeline}. Use one of {PIPELINES}")

    def resolve_evidence_sizes(self, num_nodes: int) -> List[int]:
        if self.evidence_sizes is not None:
            return self.evidence_sizes
        limit = max(1, int(num_nodes * self.evidence_size_ratio))
        return make_evidence_sizes(limit)


CONFIG = ExperimentConfig(
    datasets=[
        # "diabets.bif",
        # "aspergillus.bif",
        # "adhd.bif",
        # "munin1.bif",
        # "hepar2.bif",
        # "asia.bif",
        # "gonorrhoeae.bif",
        # "insurance.bif",
        # "crimescene.bif",
        # "cryptocurrency.bif",
        # "coral1.bif",
        # "sachs.bif",
        "test.bif",
        # "coronary.bif",
        ],
    # pipeline=PART1,
    # pipeline=PART2_FORWARD,
    pipeline=PART2_TWOPASS,
    n_trials=5,
    evidence_sizes=None,     # ou, por ex., [1, 2, 3, 5, 8]
    evidence_size_ratio=0.65,
    query_sizes=[1, 2],
)

# ===========================================================================
# Notificação sonora (opcional, Windows only, nunca quebra o experimento)
# ===========================================================================

def _notify_beep(frequency: int, duration_ms: int) -> None:
    import sys
    if sys.platform != "win32":
        return
    try:
        import winsound
        winsound.Beep(frequency, duration_ms)
    except Exception:
        pass


# ===========================================================================
# Compilação — uma função por pipeline, mesma assinatura de saída (um
# "compiled" opaco que predict_assignment() sabe interpretar).
# ===========================================================================

def compile_pipeline(cfg: ExperimentConfig, *, network, name: str, path: Path,
                      metadata_path: Path, relationship_path: Path, llm_fn) -> Any:
    if cfg.pipeline == PART1:
        compiled = load_or_compile(
            dataset_name=name,
            network=network,
            bif_path=path,
            metadata_path=metadata_path,
            relationship_path=relationship_path,
            llm_fn=llm_fn,
            use_real_llm=cfg.use_real_llm,
        )
        print(f">>> [COMPILE:{cfg.pipeline}] ✓ {len(compiled.messages)} mensagens compiladas.")
        return compiled

    # PART2_FORWARD e PART2_TWOPASS usam o MESMO circuito compilado —
    # só mudam no modo de avaliação (ver predict_assignment).
    circuit = load_or_compile_circuit(
        dataset_name=name,
        network=network,
        bif_path=path,
        metadata_path=metadata_path,
        relationship_path=relationship_path,
        llm_fn=llm_fn,
        use_real_llm=cfg.use_real_llm,
        tables_dir=TABLES_DIR,
    )
    print(f">>> [COMPILE:{cfg.pipeline}] ✓ {len(circuit.factors)} fatores locais compilados.")
    return circuit


def predict_assignment(cfg: ExperimentConfig, compiled: Any, evidence: Dict[str, str], llm_fn):
    """
    Retorna (predictions, confidence_map, llm_cpt) — mesma forma pros três
    pipelines, mesmo que Part 2 não produza confidence/cpt (ficam vazios).
    """
    if cfg.pipeline == PART1:
        return infer_from_compiled(
            compiled=compiled,
            evidence=evidence,
            llm_fn=llm_fn,
            apply_audit_repair_enabled=True,
            use_real_llm=cfg.use_real_llm,
        )
    if cfg.pipeline == PART2_FORWARD:
        return evaluate_forward_only(compiled, evidence), {}, {}
    if cfg.pipeline == PART2_TWOPASS:
        return evaluate_two_pass(compiled, evidence), {}, {}
    raise ValueError(f"Unknown pipeline: {cfg.pipeline}")


# ===========================================================================
# Execução de um experimento único (uma configuração de evidência)
# ===========================================================================

def run_experiment(network, batch_cfg: Dict, cfg: ExperimentConfig, llm_fn, compiled: Any) -> Dict[str, Any]:
    evidence = batch_cfg["evidence"]
    hidden_vars = get_hidden_vars(network, evidence)

    print("\n" + "=" * 60)
    print(f"Pipeline    : {cfg.pipeline}")
    print(f"Prompt type : {batch_cfg['prompt_type']}")
    print(f"Evidence    : {evidence}")
    print(f"Evidence len: {len(evidence)}")
    print(f"Hidden vars : {hidden_vars}")
    print("=" * 60)

    if len(hidden_vars) > cfg.max_estimated_llm_calls:
        raise ValueError(
            f"Maximum number of LLM calls reached ({cfg.max_estimated_llm_calls}), "
            "skipping to next experiment"
        )

    # --- TRUE MPE (oracle numérico, sempre igual independente do pipeline) ---
    mpe_result = run_max_product(network=network, query_vars=hidden_vars, evidence=evidence)
    mpe_assignment = mpe_result["map_assignment"]

    # --- Predição do pipeline escolhido ---
    llm_predictions, confidence_map, llm_cpt = predict_assignment(cfg, compiled, evidence, llm_fn)

    # --- Métricas ---
    if cfg.inference_mode == MPE:
        result_hits_input = llm_predictions
        evaluated_vars = [v for v in mpe_assignment if v not in evidence and v != "_scalar"]
    else:
        result_hits_input = llm_predictions
        evaluated_vars = batch_cfg["query_vars"]

    llm_hits, llm_predictions_out = count_llm_hits(
        result=result_hits_input,
        query_vars=batch_cfg["query_vars"],
        map_assignment=mpe_assignment,
        inference_mode=cfg.inference_mode,
    )
    accuracy = llm_hits / len(evaluated_vars) if evaluated_vars else 0.0

    print("\nRESULTS")
    print("-" * 40)
    for v in hidden_vars:
        print(f"{v}: PRED={llm_predictions.get(v, 'N/A')} | MPE={mpe_assignment.get(v, 'N/A')}")

    return {
        "pipeline": cfg.pipeline,
        "evidence_length": len(evidence),
        "hits": llm_hits,
        "log_accuracy": accuracy,
        "accuracy": f"{accuracy * 100:.2f}%",
        "prompt_type": batch_cfg["prompt_type"],
        "context_type": cfg.context_type,
        "dataset": cfg.dataset_name,
        "model_name": get_model_name(cfg.use_real_llm),
        "evaluated_length": len(evaluated_vars),
        "evidence": evidence,
        "llm_predictions": llm_predictions_out,
        "map_assignment": mpe_assignment,
        "confidence": confidence_map,
        "llm_cpt": llm_cpt,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    cfg = CONFIG
    llm_fn = build_llm_fn(use_real_llm=cfg.use_real_llm, use_local_llm=cfg.use_local_llm)
    BASE_DIR = Path(__file__).resolve().parents[3]

    for name in cfg.datasets:
        print(f"\n{'=' * 60}\n>>> Dataset: {name}  |  Pipeline: {cfg.pipeline}\n{'=' * 60}")
        cfg.dataset_name = name

        path = BASE_DIR / "datasets" / name
        network, context = load_network(str(path), cfg.context_type, llm_fn)

        num_nodes = len(network.variables.keys())
        current_evidence_sizes = cfg.resolve_evidence_sizes(num_nodes)
        print(f"\n>>> Testando evidence sizes {current_evidence_sizes}")

        stem = name.split(".")[0]
        metadata_path = BASE_DIR / "metadata" / f"{stem}.jsonl"
        relationship_path = BASE_DIR / "relationships" / f"{stem}.jsonl"

        print(f"\n>>> [COMPILE:{cfg.pipeline}] Compilando '{name}'...")
        compiled = compile_pipeline(
            cfg, network=network, name=name, path=path,
            metadata_path=metadata_path, relationship_path=relationship_path,
            llm_fn=llm_fn,
        )

        for batch_config in run_batch(
            network=network,
            prompt_types=cfg.prompt_types,
            evidence_sizes=current_evidence_sizes,
            query_sizes=cfg.query_sizes,
            n_trials=cfg.n_trials,
            inference_mode=cfg.inference_mode,
            llm_fn=llm_fn,
        ):
            try:
                utils.llm_request_count = 0

                log_data = run_experiment(
                    network=network,
                    batch_cfg=batch_config,
                    cfg=cfg,
                    llm_fn=llm_fn,
                    compiled=compiled,
                )

                log_experiment(log_data)
                # log_experiment_csv(log_data)

            except Exception as e:
                print(f"❌ Error in {name}: {e}")

        _notify_beep(440, 500)

    _notify_beep(600, 1000)


if __name__ == "__main__":
    main()