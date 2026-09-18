"""Behavior captured before consolidation, with synthetic data and fake LLMs."""

import hashlib
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from pgm_llm_inference.experiment.batch import make_evidence_sizes, run_batch
from pgm_llm_inference.experiment.config import ExperimentConfig
from pgm_llm_inference.experiment.runner import run_experiment
from pgm_llm_inference.models import BayesianNetwork, Factor, Variable
from pgm_llm_inference.mpe import cache, compile as compiler
from pgm_llm_inference.mpe.infer import infer_from_compiled
from pgm_llm_inference.mpe.compile_phase import client
from pgm_llm_inference.mpe.types import PromptTrace


def make_network():
    variables = {name: Variable(name=name, states=("off", "on")) for name in "ABCD"}
    return BayesianNetwork(
        name="regression",
        variables=variables,
        factors=[
            Factor(scope=[variables["A"]], values=np.array([0.7, 0.3])),
            Factor(scope=[variables["B"]], values=np.array([0.4, 0.6])),
            Factor(
                scope=[variables[v] for v in "CAB"],
                values=np.array([[[0.9, 0.6], [0.5, 0.1]], [[0.1, 0.4], [0.5, 0.9]]]),
            ),
            Factor(scope=[variables[v] for v in "DC"], values=np.array([[0.8, 0.2], [0.2, 0.8]])),
        ],
    )


def install_fake_llms(monkeypatch, network):
    calls = []

    def metadata_llm(prompt, schema):
        calls.append((schema.__name__, prompt))
        if schema.__name__ == "NetworkMetadataResponse":
            return schema(
                metadata={
                    name: {"description": f"Description of {name}", "aliases": [f"alias_{name}"]}
                    for name in network.variables
                }
            )
        return schema(
            relationships={
                name: {"notes": [f"Parents of {name}: {', '.join(parents)}"]}
                for name, parents in network.parents.items()
                if parents
            }
        )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def complete_json(
            self,
            *,
            purpose,
            variable,
            prompt,
            response_model,
            semantic_validator=None,
            candidate_states=None,
        ):
            calls.append((purpose, prompt))
            if variable is None:
                model = response_model(network_summary="Synthetic regression network")
            else:
                payload = json.loads(prompt.split("Bucket data:\n", 1)[1])
                model = response_model(
                    variable=variable,
                    decisions=[
                        {
                            "context": row,
                            "selected_value": candidate_states[
                                sum(v == "on" for v in row.values()) % 2
                            ],
                            "confidence": "low",
                            "rationale": "Controlled response",
                            "domain_scores": {"by_state": {"off": -0.3, "on": -1.4}},
                            "token_scores": {
                                "selected_token": {"token": "A", "logprob": -0.3},
                                "top_logprobs": [{"token": "B", "logprob": -1.4}],
                            },
                        }
                        for row in payload["context_rows"]
                    ],
                )
            if semantic_validator:
                semantic_validator(model)
            return model, PromptTrace(purpose=purpose, variable=variable, prompt=prompt)

    monkeypatch.setattr(client, "LLMJsonClient", FakeClient)
    return metadata_llm, calls


def capture_pipeline(tmp_path, monkeypatch):
    from pgm_llm_inference.experiment import logging as logger
    from pgm_llm_inference.experiment import runner

    network = make_network()
    llm_fn, calls = install_fake_llms(monkeypatch, network)
    metadata = tmp_path / "metadata.json"
    relationships = tmp_path / "relationships.json"
    compiled = compiler.compile_semantic_messages(
        network=network,
        metadata_path=metadata,
        relationship_path=relationships,
        llm_fn=llm_fn,
        max_context_rows_per_call=2,
    )
    first_calls = calls.copy()
    # Reusing metadata must skip exactly the two generation stages.
    calls.clear()
    loaded_metadata = compiler.compile_semantic_messages(
        network=network,
        metadata_path=metadata,
        relationship_path=relationships,
        llm_fn=llm_fn,
        max_context_rows_per_call=2,
    )
    assert loaded_metadata.messages == compiled.messages
    assert calls == first_calls[2:]

    def no_llm(*args, **kwargs):
        raise AssertionError("Compiled inference must not call an LLM")

    monkeypatch.setattr(client, "LLMJsonClient", no_llm)
    monkeypatch.setattr(runner, "get_model_name", lambda _: "regression-model")
    outputs = [
        infer_from_compiled(compiled=compiled, evidence=evidence)
        for evidence in (
            {},
            {"A": "on"},
            {"D": "on"},
            {"alias_a": " ON "},
            {name: "off" for name in "ABCD"},
        )
    ]
    assert outputs[1] == outputs[3]
    batches = {}
    records = []
    for mode in ("mpe", "map"):
        for sampling in ("mpe_consistent", "mpe_inconsistent", "random"):
            items = list(
                run_batch(
                    network=network,
                    prompt_types=["variable_assignment"],
                    evidence_sizes=[0, 1, 2],
                    query_sizes=[1, 2],
                    n_trials=3,
                    inference_mode=mode,
                    evidence_sampling=sampling,
                )
            )
            batches[f"{mode}/{sampling}"] = items
            config = ExperimentConfig(
                dataset_name="regression.bif", inference_mode=mode, evidence_sampling=sampling
            )
            records.append(
                run_experiment(
                    network=network, batch_config=items[-1], config=config, compiled=compiled
                )
            )

    class FixedDatetime:
        @staticmethod
        def utcnow():
            return datetime(2026, 1, 1)

    monkeypatch.setattr(logger, "datetime", FixedDatetime)
    monkeypatch.setattr(logger, "LOG_PATH", tmp_path / "results.jsonl")
    for record in records:
        logger.log_experiment(record)
    log_lines = logger.LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert all("timestamp" not in record for record in records)
    assert log_lines == [
        json.dumps({**record, "timestamp": "2026-01-01T00:00:00"}, ensure_ascii=False)
        for record in records
    ]
    return {
        "prompt_hashes": [
            (purpose, hashlib.sha256(prompt.encode()).hexdigest())
            for purpose, prompt in first_calls
        ],
        "order": compiled.elimination_order,
        "messages": {
            name: message.model_dump(mode="json") for name, message in compiled.messages.items()
        },
        "inference": outputs,
        "batches": batches,
        "jsonl": [json.loads(line) for line in log_lines],
        "evidence_sizes": {str(n): make_evidence_sizes(n) for n in (0, 1, 10, 11, 30, 100)},
    }


def test_pipeline_matches_before_refactor(tmp_path, monkeypatch):
    actual = capture_pipeline(tmp_path, monkeypatch)
    expected = json.loads((Path(__file__).parent / "fixtures/pipeline.json").read_text())
    assert json.loads(json.dumps(actual)) == expected


def test_cache_hit_and_incompatible_schema(tmp_path, monkeypatch):
    network = make_network()
    llm_fn, _ = install_fake_llms(monkeypatch, network)
    monkeypatch.setattr(cache, "COMPILED_TABLES_DIR", tmp_path)
    kwargs = dict(
        dataset_name="regression.bif",
        model_name="provider/model:tag",
        network=network,
        bif_path=None,
        metadata_path=None,
        relationship_path=None,
        llm_fn=llm_fn,
        use_real_llm=False,
        max_context_rows_per_call=2,
    )
    compiled = cache.load_or_compile(**kwargs)
    path = tmp_path / "regression.provider__model-tag_VE.compiled.pkl"
    assert path.exists()

    def no_compile(**kwargs):
        raise AssertionError("Valid cache must not recompile")

    monkeypatch.setattr(cache, "compile_semantic_messages", no_compile)
    loaded = cache.load_or_compile(**kwargs)
    assert loaded.messages == compiled.messages
    assert type(loaded).__module__ == "pgm_llm_inference.mpe.compile"
    compiled.schema_version = 0
    path.write_bytes(pickle.dumps(compiled))
    with pytest.raises(AssertionError, match="Valid cache"):
        cache.load_or_compile(**kwargs)


def test_compilation_rejects_missing_context_rows(monkeypatch):
    network = make_network()
    llm_fn, _ = install_fake_llms(monkeypatch, network)
    original = client.LLMJsonClient.complete_json

    def incomplete(self, **kwargs):
        if kwargs["variable"] is not None:
            response = kwargs["response_model"](variable=kwargs["variable"], decisions=[])
            kwargs["semantic_validator"](response)
        return original(self, **kwargs)

    monkeypatch.setattr(client.LLMJsonClient, "complete_json", incomplete)
    with pytest.raises(ValueError, match="has no decisions"):
        compiler.compile_semantic_messages(network=network, llm_fn=llm_fn)


@pytest.mark.parametrize(
    "failure,match",
    [
        ("missing_row", "No backpointer decision found"),
        ("unassigned_separator", "has not been assigned"),
    ],
)
def test_reconstruction_rejects_incomplete_tables(monkeypatch, failure, match):
    network = make_network()
    llm_fn, _ = install_fake_llms(monkeypatch, network)
    compiled = compiler.compile_semantic_messages(network=network, llm_fn=llm_fn)
    if failure == "missing_row":
        compiled.messages["A"].rows.clear()
    else:
        compiled.messages["A"].scope = ("D",)
    with pytest.raises(ValueError, match=match):
        infer_from_compiled(compiled=compiled, evidence={})


def test_cached_inference_does_not_import_compilation(tmp_path, monkeypatch):
    network = make_network()
    llm_fn, _ = install_fake_llms(monkeypatch, network)
    compiled = compiler.compile_semantic_messages(network=network, llm_fn=llm_fn)
    path = tmp_path / "synthetic.pkl"
    path.write_bytes(pickle.dumps(compiled))
    # A fresh interpreter catches imports hidden by the test runner's module cache.
    script = """
import pickle
import sys
from pathlib import Path
from pgm_llm_inference.core.config import InferenceConfig
from pgm_llm_inference.mpe import infer_from_compiled

InferenceConfig.model_config.update(env_file=None, env_prefix="PGM_TEST_")
compiled = pickle.loads(Path(sys.argv[1]).read_bytes())
assert type(compiled).__module__ == "pgm_llm_inference.mpe.compile"
assignment, confidence, tables = infer_from_compiled(compiled=compiled, evidence={"A": "on"})
assert set(assignment) == {"B", "C", "D"}
assert set(confidence) == set(tables) == set(assignment)
assert not any(name.startswith(("pgm_llm_inference.mpe.compile_phase",
                                "pgm_llm_inference.mpe.pre_compile_phase"))
               for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(path)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
