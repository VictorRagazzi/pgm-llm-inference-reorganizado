from typing import Any

from pgm_llm_inference.evaluation.metrics import count_llm_hits
from pgm_llm_inference.experiment.config import MPE, ExperimentConfig
from pgm_llm_inference.experiment.experiment import get_hidden_vars, run_max_product
from pgm_llm_inference.experiment.llm_factory import get_model_name
from pgm_llm_inference.mpe import CompiledSemanticMessages, infer_from_compiled


def run_experiment(
    *,
    network,
    batch_config: dict[str, Any],
    config: ExperimentConfig,
    compiled: CompiledSemanticMessages,
) -> dict[str, Any]:
    """Run one exact-vs-semantic inference comparison for a batch item."""
    evidence = batch_config["evidence"]
    hidden_vars = get_hidden_vars(network, evidence)

    print("\n" + "=" * 60)
    print(f"Prompt type : {batch_config['prompt_type']}")
    print(f"Evidence    : {evidence}")
    print(f"Evidence len: {len(evidence)}")
    print(f"Hidden vars : {hidden_vars}")
    print("=" * 60)

    if len(hidden_vars) > config.max_hidden_variables:
        raise ValueError(
            f"Hidden-variable limit reached ({config.max_hidden_variables}); "
            "skipping experiment"
        )

    mpe_result = run_max_product(
        network=network,
        query_vars=hidden_vars,
        evidence=evidence,
    )
    mpe_assignment = mpe_result["map_assignment"]

    llm_predictions, confidence_map, llm_cpt = infer_from_compiled(
        compiled=compiled,
        evidence=evidence,
    )

    if config.inference_mode == MPE:
        evaluated_vars = [
            variable
            for variable in mpe_assignment
            if variable not in evidence and variable != "_scalar"
        ]
    else:
        evaluated_vars = batch_config["query_vars"]

    llm_hits, logged_predictions = count_llm_hits(
        result=llm_predictions,
        query_vars=batch_config["query_vars"],
        map_assignment=mpe_assignment,
        inference_mode=config.inference_mode,
    )

    accuracy = llm_hits / len(evaluated_vars) if evaluated_vars else 0.0
    exact_match = int(llm_hits == len(evaluated_vars)) if evaluated_vars else None

    print("\nRESULTS")
    print("-" * 40)
    for variable in hidden_vars:
        llm_value = llm_predictions.get(variable, "N/A")
        mpe_value = mpe_assignment.get(variable, "N/A")
        print(f"{variable}: LLM={llm_value} | MPE={mpe_value}")

    return {
        "evidence_length": len(evidence),
        "hits": llm_hits,
        "log_accuracy": accuracy,
        "accuracy": f"{accuracy * 100:.2f}%",
        "prompt_type": batch_config["prompt_type"],
        "context_type": config.context_type,
        "dataset": config.dataset_name,
        "exact_match": exact_match,
        "evidence_sampling": config.evidence_sampling,
        "model_name": get_model_name(config.use_real_llm),
        "evaluated_length": len(evaluated_vars),
        "evidence": evidence,
        "llm_predictions": logged_predictions,
        "map_assignment": mpe_assignment,
        "confidence": confidence_map,
        "llm_cpt": llm_cpt,
    }
