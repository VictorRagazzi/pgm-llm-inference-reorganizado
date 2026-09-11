"""
main_single_run.py
===================
Exemplo mínimo: roda UMA única inferência MPE sobre UM dataset,
com evidência e query fixas no código. Útil para testar rapidamente a
pipeline ou depurar um dataset novo sem rodar o batch completo de main.py.

Para varrer múltiplos datasets e tamanhos de evidência, use main.py.
"""

from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.paths import dataset_path
from pgm_llm_inference.logging.experiment_logger import (
    log_experiment,
    log_experiment_csv,
)
from pgm_llm_inference.experiment.llm_factory import build_llm_fn, get_model_name
from pgm_llm_inference.experiment.experiment import run_single_mpe_experiment
from pgm_llm_inference.evaluation.metrics import count_llm_hits


MPE = "mpe"


def main():

    dataset_name = "earthquake.bif.gz"
    path = dataset_path(dataset_name)

    use_real_llm = False
    use_local_llm = True
    llm_fn = build_llm_fn(
        use_real_llm=use_real_llm,
        use_local_llm=use_local_llm,
    )

    network = load_network(path)

    query_vars = ["JohnCalls", "MaryCalls"]
    evidence = {"Earthquake": "yes"}

    for prompt_type in ["simple"]:

        result = run_single_mpe_experiment(
            network=network,
            evidence=evidence,
            llm_fn=llm_fn,
            bif_path=path,
            max_hidden_variables=5,
        )

        result_hits = result["llm"]["llm_predictions"]
        evaluated_vars = [
            v for v in result[MPE]["map_assignment"].keys()
            if v not in evidence and v != "_scalar"
        ]

        llm_hits, _ = count_llm_hits(
            result=result_hits,
            query_vars=query_vars,
            map_assignment=result[MPE]["map_assignment"],
            inference_mode=MPE,
        )

        evaluated_count = len(evaluated_vars)
        accuracy = llm_hits / evaluated_count if evaluated_count > 0 else 0.0

        log_input = {
            "evidence": evidence,
            "query_vars": query_vars,
            "query_length": len(query_vars),
            "evaluated_vars": evaluated_vars,
            "evaluated_length": evaluated_count,
            "result_hits": result_hits,
            "hits": llm_hits,
            "accuracy": f"{accuracy * 100:.2f}%",
            "dataset": dataset_name,
            "prompt_type": prompt_type,
            "model_name": get_model_name(use_real_llm),
            "inference_mode": MPE,
        }

        log_experiment(log_input)
        log_experiment_csv(log_input)

if __name__ == "__main__":
    main()
