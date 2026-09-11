from pgm_llm_inference.mpe.cache import load_or_compile
from pgm_llm_inference.paths import dataset_path, metadata_path, relationship_path
from pgm_llm_inference.io.loaders import load_network
from pgm_llm_inference.logging.experiment_logger import log_experiment
from pgm_llm_inference.experiment.config import ExperimentConfig
from pgm_llm_inference.experiment.llm_factory import build_llm_fn, get_model_name
from pgm_llm_inference.experiment.batch import run_batch, make_evidence_sizes
from pgm_llm_inference.experiment.runner import run_experiment

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    datasets = [
        # "gonorrhoeae.bif",
        # "adhd.bif",
        # "cryptocurrency.bif",
        # "crimescene.bif",
        # "insurance.bif",
        # "sachs.bif",
        # "arctic_sea.bif",
        # "sachs.bif",

        # "adhd_cbeb.bif",
        # "covid3_cbeb.bif", # ERRO
        "covid3_TESTE.bif", # ERRO
        # "covid1_cbeb.bif", # ERRO
        # "gonorrhoeae_cbeb.bif",
        # "hepar2_cbeb.bif",
        # "alarm_cbeb.bif",
        # "foodallergy3_cbeb.bif",
        # "foodallergy1_cbeb.bif",
        # "diabets_cbeb.bif",
        # "child_cbeb.bif",
        # "urinary_cbeb.bif",

        # "cardiovascular_cbeb.bif", # ERRO
        # "munin1_cbeb.bif",
        # "aspergillus.bif",
        # "coronary.bif",
        # "coral1.bif",
        # "bankruptcy.bif",
        # "asia.bif",
    ]

    cfg = ExperimentConfig(
        dataset_name="",
        prompt_types=["variable_assignment"],
        query_sizes=[1, 2],
    )

    llm_fn = build_llm_fn(use_real_llm=cfg.use_real_llm, use_local_llm=cfg.use_local_llm)
    model_name = get_model_name(cfg.use_real_llm)

    for name in datasets:

        print(f"\n{'=' * 60}")
        print(f">>> Dataset: {name}")
        print(f"{'=' * 60}")
        cfg.dataset_name = name

        # --- Carregamento da rede ---
        path = dataset_path(name)
        network = load_network(path)

        # --- Tamanho do batch baseado na rede ---
        num_nodes = len(network.variables.keys())
        limit = int(num_nodes * 0.5)
        current_evidence_sizes = make_evidence_sizes(limit)
        print(f"\n>>> Testando evidence sizes {current_evidence_sizes}")

        # Tentativas com max_context_rows_per_call decrescente: valor original → 76 → 64
        retry_context_rows = [cfg.max_context_rows_per_call, cfg.max_context_rows_per_call - 20, cfg.max_context_rows_per_call - 30]
        dataset_ok = False

        for attempt, max_rows in enumerate(retry_context_rows):
            if attempt > 0:
                print(f"\n⚠️  Tentativa {attempt + 1}/3 para '{name}' com max_context_rows_per_call={max_rows}...")
            try:
                # --- COMPILAÇÃO: roda UMA vez por dataset ---
                # Executa Bucket Elimination com evidence={} → produto cartesiano completo.
                # Custo: N chamadas LLM (uma por variável).
                print(f"\n>>> [COMPILE] Compilando mensagens semânticas para '{name}'...")

                compiled = load_or_compile(
                    dataset_name=name,
                    model_name=model_name,
                    network=network,
                    bif_path=path,
                    metadata_path=metadata_path(name),
                    relationship_path=relationship_path(name),
                    llm_fn=llm_fn,
                    use_real_llm=cfg.use_real_llm,
                    max_context_rows_per_call=max_rows,
                )
                print(f">>> [COMPILE] ✓ {len(compiled.messages)} mensagens compiladas.")

                # --- INFERÊNCIA: lookup e reconstrução para cada evidência ---
                for batch_config in run_batch(
                    network=network,
                    prompt_types=cfg.prompt_types,
                    evidence_sizes=current_evidence_sizes,
                    query_sizes=cfg.query_sizes,
                    n_trials=cfg.n_trials,
                    inference_mode=cfg.inference_mode,
                    evidence_sampling=cfg.evidence_sampling,
                ):
                    try:
                        log_data = run_experiment(
                            network=network,
                            batch_config=batch_config,
                            config=cfg,
                            compiled=compiled,
                        )

                        log_experiment(log_data)
                        # log_experiment_csv(log_data)

                    except Exception as e:
                        print(f"❌ Error in experiment ({name}): {e}")

                dataset_ok = True
                break  # sucesso — não precisa tentar novamente

            except Exception as e:
                print(f"❌ Falha na tentativa {attempt + 1}/3 para '{name}' (max_context_rows_per_call={max_rows}): {e}")

        if not dataset_ok:
            print(f"\n⛔ Todas as tentativas falharam para '{name}'. Pulando para o próximo dataset.\n")

        _notify_beep(440, 500)

    _notify_beep(600, 1000)


if __name__ == "__main__":
    main()
