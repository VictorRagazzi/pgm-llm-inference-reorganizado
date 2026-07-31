
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from pgm_llm_inference.mpe.circuit import load_or_compile_circuit, print_divergence_report
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

    ])

    # A flag pedida: qual pipeline usar. Ver PIPELINES acima.
    pipeline: str = PART1

    use_real_llm: bool = True
    use_local_llm: bool = True
    inference_mode: str = MPE
    context_type: Optional[str] = None
    prompt_types: List[str] = field(default_factory=lambda: ["variable_assignment"])

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
    datasets=["gonorrhoeae.bif"],
    pipeline=PART1,
    n_trials=5,
    evidence_sizes=None,     # ou, por ex., [1, 2, 3, 5, 8]
    evidence_size_ratio=0.65,
    query_sizes=[1, 2],
)

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

        print(f"\n>>> [COMPILE] Compilando/carregando circuito para '{name}'...")
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

        evidence_list = [
            batch_config["evidence"]
            for batch_config in run_batch(
                network=network,
                prompt_types=cfg.prompt_types,
                evidence_sizes=current_evidence_sizes,
                query_sizes=cfg.query_sizes,
                n_trials=cfg.n_trials,
                inference_mode=cfg.inference_mode,
                llm_fn=llm_fn,
            )
        ]

        print(f"\n>>> {len(evidence_list)} configurações de evidência coletadas.")
        print_divergence_report(circuit, evidence_list)


if __name__ == "__main__":
    main()