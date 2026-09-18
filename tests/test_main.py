import pytest

from tests.test_pipeline import install_fake_llms, make_network
from pgm_llm_inference.mpe import compile as compiler


@pytest.mark.parametrize("failed_attempts", [0, 1, 2, 3])
def test_main_preserves_batch_and_retry_behavior(monkeypatch, failed_attempts):
    from pgm_llm_inference.scripts.run import main as entry

    network = make_network()
    llm_fn, _ = install_fake_llms(monkeypatch, network)
    compiled = compiler.compile_semantic_messages(network=network, llm_fn=llm_fn)
    attempts, records, experiments, beeps = [], [], [], []

    def load_or_compile(**kwargs):
        attempts.append(kwargs["max_context_rows_per_call"])
        if len(attempts) <= failed_attempts:
            raise ValueError("Controlled compilation failure")
        return compiled

    def run_experiment(**kwargs):
        experiments.append(kwargs["batch_config"])
        assert kwargs["config"].n_trials == 5
        if len(experiments) == 1:
            raise ValueError("Controlled failure of one batch item")
        return kwargs["batch_config"]

    monkeypatch.setattr(entry, "load_network", lambda _: network)
    monkeypatch.setattr(entry, "build_llm_fn", lambda **_: llm_fn)
    monkeypatch.setattr(entry, "get_model_name", lambda _: "test-model")
    monkeypatch.setattr(entry, "load_or_compile", load_or_compile)
    monkeypatch.setattr(entry, "run_experiment", run_experiment)
    monkeypatch.setattr(entry, "log_experiment", records.append)
    monkeypatch.setattr(entry, "_notify_beep", lambda *args: beeps.append(args))
    entry.main()
    assert attempts == [32, 12, 2][: min(failed_attempts + 1, 3)]
    assert len(experiments) == (10 if failed_attempts < 3 else 0)
    assert records == experiments[1:]
    assert beeps == [(440, 500), (600, 1000)]
