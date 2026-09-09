"""
Experimento de posição da evidência (resposta ao revisor do CBEB).

Diferença em relação ao experimento de proporção de evidência já existente:
lá a evidência é amostrada (aleatoriamente, em tamanhos crescentes); aqui é
EXAUSTIVA — para cada rede, cada variável é usada, uma de cada vez, como
único elemento de evidência. Não há amostragem nem repetição de trial: o
número de execuções por rede é exatamente o número de variáveis da rede.

Valor de evidência usado: o valor daquela variável na configuração MPE
completa da rede (evidence={}), ou seja, evidência "MPE-consistente" de
tamanho 1 — mesma convenção já usada no experimento de proporção. Isso
isola o efeito de POSIÇÃO do efeito de CONSISTÊNCIA da evidência (que já
é medido em outro experimento).

Cada execução é logada com os campos padrão de log_experiment mais três
campos novos, usados depois em analysis_evidence_position.py:
    - "experiment"           : "evidence_position"  (para filtrar o log)
    - "evidence_node"        : nome da variável usada como evidência
    - "node_depth"            : profundidade absoluta do nó (raiz = 0)
    - "node_depth_normalized": node_depth / profundidade_máxima_da_rede

IMPORTANTE — ajustar antes de rodar:
    - Os imports abaixo assumem os mesmos caminhos de módulo vistos no
      código colado (`run_experiment`, `ExperimentConfig`, `MPE` no script
      principal de experimentos; `extract_graph_structure` em get_table.py).
      Ajuste os caminhos conforme a localização real desses símbolos no
      seu projeto.
    - Se `log_experiment()` escreve sempre no mesmo arquivo de log
      configurado em InferenceConfig, os logs deste experimento vão ficar
      misturados com os do experimento de proporção no mesmo arquivo — o
      campo "experiment" serve exatamente para separá-los depois. Se
      preferir um arquivo separado, ajuste log_experiment/InferenceConfig
      para apontar para outro arquivo antes de rodar este script.
    - Este script não foi executado (o pacote pgm_llm_inference não está
      disponível neste ambiente) — recomendo testar primeiro com 1 dataset
      pequeno antes de rodar nas 10 redes.
"""

import json
from pathlib import Path

from pgm_llm_inference.core.config import InferenceConfig
import pgm_llm_inference.utils as utils
from pgm_llm_inference.utils import load_or_compile
from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.logging.experiment_logger import LOG_PATH, log_experiment
from pgm_llm_inference.experiment.llm_factory import build_llm_fn
from pgm_llm_inference.experiment.experiment import run_max_product, get_hidden_vars

# Reaproveita a lógica de inferência + métricas já implementada, em vez de
# duplicá-la. Ajuste o caminho do import conforme o nome real do módulo
# que contém run_experiment/ExperimentConfig/MAP/MPE no seu projeto.
from pgm_llm_inference.scripts.run.main import run_experiment, ExperimentConfig, MPE

# extract_graph_structure e compute_node_depths — a segunda função precisa
# ser adicionada a get_table.py (ver get_table_additions.py).
from pgm_llm_inference.scripts.analysis.get_table import (
    extract_graph_structure,
    compute_node_depths,
)

inference_cfg = InferenceConfig()

DATASETS = [
    "adhd_cbeb.bif",
    "alarm_cbeb.bif",
    "child_cbeb.bif",
    "covid1_cbeb.bif",
    "covid3_cbeb.bif",
    "diabets_cbeb.bif",
    "foodallergy1_cbeb.bif",
    "foodallergy3_cbeb.bif",
    "gonorrhoeae_cbeb.bif",
    "hepar2_cbeb.bif",
]

# Rótulo usado para filtrar este experimento dentro do log em analysis.py.
EXPERIMENT_TAG = "evidence_position"
RESUME = True


def completed_evidence_nodes(dataset_name: str) -> set[str]:
    """Pares já logados, para uma retomada não duplicar execuções."""
    if not RESUME or not LOG_PATH.exists():
        return set()

    completed = set()
    with LOG_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("experiment") == EXPERIMENT_TAG and record.get("dataset") == dataset_name:
                node = record.get("evidence_node")
                if node is not None:
                    completed.add(node)
    return completed


def run_evidence_position_for_dataset(
    name: str,
    base_dir: Path,
    llm_fn,
    cfg: ExperimentConfig,
    max_context_rows_per_call: int = 96,
) -> None:
    path = base_dir / "datasets" / name
    network, _ = load_network(str(path), cfg.context_type, llm_fn)

    # --- Profundidade normalizada por nó (0 = raiz, 1 = folha mais funda) ---
    parents, children, _ = extract_graph_structure(network)
    node_depths = compute_node_depths(list(network.variables), parents, children)
    max_depth = max(node_depths.values()) if node_depths else 0

    if max_depth == 0:
        print(f"⚠ '{name}': profundidade máxima = 0 (rede sem hierarquia), pulando.")
        return

    # --- MPE completo (evidence={}) ---
    # Fornece (a) o valor "consistente" usado como evidência de cada nó
    # individualmente, e (b) o gabarito usado nas métricas de acurácia.
    hidden_vars_full = get_hidden_vars(network, {})
    full_mpe = run_max_product(network=network, query_vars=hidden_vars_full, evidence={})
    mpe_assignment = full_mpe["map_assignment"]

    # --- Compilação (uma vez por dataset, igual ao script principal) ---
    model_name = inference_cfg.openai_model if cfg.use_real_llm else inference_cfg.local_model
    compiled = load_or_compile(
        dataset_name=name,
        network=network,
        model_name=model_name,
        bif_path=path,
        metadata_path=base_dir / "metadata" / f"{name.split('.')[0]}.jsonl",
        relationship_path=base_dir / "relationships" / f"{name.split('.')[0]}.jsonl",
        llm_fn=llm_fn,
        use_real_llm=cfg.use_real_llm,
        max_context_rows_per_call=max_context_rows_per_call,
    )

    cfg.dataset_name = name
    # Rótulo apenas informativo no log (run_experiment não usa este campo
    # para decidir comportamento, só o registra na saída).
    cfg.evidence_sampling = EXPERIMENT_TAG

    n_vars = len(network.variables)
    completed = completed_evidence_nodes(name)
    print(f">>> '{name}': {n_vars} execuções exaustivas (uma por variável)")
    if completed:
        print(f"    Retomada: {len(completed)} nó(s) já concluído(s) serão ignorados.")

    for i, node in enumerate(network.variables, 1):
        if node in completed:
            continue
        if node not in mpe_assignment:
            print(f"  ⚠ nó '{node}' sem valor no MPE completo, pulando.")
            continue

        evidence = {node: mpe_assignment[node]}
        batch_cfg = {
            "prompt_type": cfg.prompt_types[0],
            "evidence": evidence,
            "query_vars": None,  # modo MPE
        }

        try:
            utils.llm_request_count = 0  # reset por execução
            log_data = run_experiment(
                network=network,
                batch_cfg=batch_cfg,
                global_cfg=cfg,
                llm_fn=llm_fn,
                compiled=compiled,
                bif_path=path,
            )
        except Exception as e:
            print(f"  ❌ [{i}/{n_vars}] erro no nó evidência '{node}': {e}")
            continue

        log_data["experiment"] = EXPERIMENT_TAG
        log_data["evidence_node"] = node
        log_data["node_depth"] = node_depths[node]
        log_data["node_depth_normalized"] = node_depths[node] / max_depth

        log_experiment(log_data)
        print(
            f"  [{i}/{n_vars}] nó={node:<20} depth_norm={log_data['node_depth_normalized']:.2f} "
            f"acc={log_data['accuracy']} exact={log_data['exact_match']}"
        )


def main():
    cfg = ExperimentConfig(
        dataset_name="",
        prompt_types=["variable_assignment"],
        inference_mode=MPE,
    )
    llm_fn = build_llm_fn(use_real_llm=cfg.use_real_llm, use_local_llm=cfg.use_local_llm)
    # .../src/pgm_llm_inference/scripts/run/run_evidence_pos.py -> raiz do repositório
    base_dir = Path(__file__).resolve().parents[4]

    for name in DATASETS:
        print(f"\n{'=' * 60}\n>>> Posição da evidência — Dataset: {name}\n{'=' * 60}")
        try:
            run_evidence_position_for_dataset(name, base_dir, llm_fn, cfg)
        except Exception as e:
            print(f"❌ Falha em '{name}': {e}")


if __name__ == "__main__":
    main()
