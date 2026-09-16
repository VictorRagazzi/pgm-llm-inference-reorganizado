import math
import pickle

import pandas as pd
import pytest

from pgm_llm_inference.analysis.domain_entropy import (
    CompiledMessagesCache,
    attach_domain_entropy,
    compiled_row_entropy_table,
    domain_entropy,
    top_token_statistics,
)
from pgm_llm_inference.analysis.entropy_plots import plot_compiled_row_entropy
from pgm_llm_inference.models import BayesianNetwork
from pgm_llm_inference.mpe.compile import CompiledSemanticMessages
from pgm_llm_inference.mpe.io import build_alias_map
from pgm_llm_inference.mpe.types import (
    BriefingResponse,
    DecisionTokenScores,
    DomainScores,
    MessageRow,
    SemanticMessage,
    TokenScore,
)


def _compiled() -> CompiledSemanticMessages:
    network = BayesianNetwork.from_structure(
        variable_states={"Parent": ("low", "high"), "Child": ("no", "yes")},
        children={"Parent": ("Child",)},
    )
    messages = {
        "Parent": SemanticMessage(
            source_variable="Parent",
            scope=(),
            is_evidence=False,
            rows=[
                MessageRow(
                    context={},
                    selected_value="low",
                    rationale="root",
                    domain_scores=DomainScores(
                        by_state={"low": math.log(0.75), "high": math.log(0.25)}
                    ),
                    token_scores=DecisionTokenScores(
                        selected_token=TokenScore(token="A", logprob=math.log(0.6)),
                        top_logprobs=[
                            TokenScore(token="A", logprob=math.log(0.6)),
                            TokenScore(token="B", logprob=math.log(0.3)),
                        ],
                    ),
                )
            ],
        ),
        "Child": SemanticMessage(
            source_variable="Child",
            scope=("Parent",),
            is_evidence=False,
            rows=[
                MessageRow(
                    context={"Parent": "low"},
                    selected_value="yes",
                    rationale="low parent",
                    domain_scores=DomainScores(
                        by_state={"no": math.log(0.2), "yes": math.log(0.8)}
                    ),
                ),
                MessageRow(
                    context={"Parent": "high"},
                    selected_value="no",
                    rationale="high parent",
                    domain_scores=None,
                ),
            ],
        ),
    }
    return CompiledSemanticMessages(
        messages=messages,
        elimination_order=["Child", "Parent"],
        briefing=BriefingResponse(network_summary="test"),
        bn=network,
        alias_map=build_alias_map(network, {}),
        metadata={},
        relationship_notes={},
    )


def _logs(parent_state: str = "low") -> pd.DataFrame:
    common = {
        "dataset": "network.bif",
        "evidence_sampling": "mpe_consistent",
        "evidence_layout": "nested",
        "evidence": {},
        "map_assignment": {"Parent": parent_state, "Child": "yes"},
    }
    return pd.DataFrame(
        [
            {**common, "llm_predictions": {"Parent": "low", "Child": "yes"}},
            {**common, "llm_predictions": {"Parent": "low", "Child": "yes"}},
            {**common, "llm_predictions": {"Parent": "high", "Child": "no"}},
        ]
    )


def test_domain_entropy_softmaxes_log_scores_and_supports_legacy_probabilities() -> None:
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))

    assert domain_entropy(
        {"low": math.log(0.75), "high": math.log(0.25)}
    ) == pytest.approx(expected)
    assert domain_entropy(
        {"low": 0.75, "high": 0.25}, score_kind="probability"
    ) == pytest.approx(expected)


def test_top_token_entropy_reports_truncation_mass_separately() -> None:
    scores = DecisionTokenScores(
        selected_token=TokenScore(token="A", logprob=math.log(0.6)),
        top_logprobs=[
            TokenScore(token="A", logprob=math.log(0.6)),
            TokenScore(token="B", logprob=math.log(0.3)),
        ],
    )

    entropy, mass, count = top_token_statistics(scores)

    assert entropy == pytest.approx(
        -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    )
    assert mass == pytest.approx(0.9)
    assert count == 2


def test_compiled_table_keeps_each_context_row() -> None:
    table = compiled_row_entropy_table(_compiled())

    assert len(table) == 3
    parent = table.loc[table["variable"] == "Parent"].iloc[0]
    assert parent["context"] == "{}"
    assert parent["token_top_k_mass"] == pytest.approx(0.9)
    child_rows = table.loc[table["variable"] == "Child"]
    assert set(child_rows["context"]) == {
        '{"Parent": "low"}',
        '{"Parent": "high"}',
    }
    assert child_rows["token_entropy_top_k"].isna().all()


def test_compiled_entropy_plot_uses_context_rows(tmp_path) -> None:
    path = tmp_path / "entropy.png"

    plot_compiled_row_entropy(compiled_row_entropy_table(_compiled()), path)

    assert path.is_file()


def test_attaches_context_entropy_next_to_existing_correct_metric() -> None:
    table = attach_domain_entropy(_logs(), {"network.bif": _compiled()})

    parent = table.loc[table["variable"] == "Parent"].iloc[0]
    child = table.loc[table["variable"] == "Child"].iloc[0]
    assert parent["correct"] == 1
    assert parent["parent_context"] == {}
    assert parent["entropy"] == pytest.approx(
        -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    )
    assert parent["token_top_k_mass"] == pytest.approx(0.9)
    assert parent["token_top_k_count"] == 2
    assert parent["token_entropy_top_k"] == pytest.approx(
        -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    )
    assert child["correct"] == 1
    assert child["parent_context"] == {"Parent": "low"}
    assert child["entropy"] == pytest.approx(
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    )
    assert math.isnan(child["token_entropy_top_k"])


def test_missing_optional_scores_produce_nan() -> None:
    table = attach_domain_entropy(_logs("HIGH"), {"network": _compiled()})
    child_entropy = table.loc[table["variable"] == "Child", "entropy"].iloc[0]

    assert math.isnan(child_entropy)


def test_uses_evidence_when_an_observed_parent_is_absent_from_map_assignment() -> None:
    logs = _logs().copy()
    logs["evidence"] = [{"parent": "LOW"}] * len(logs)
    logs["map_assignment"] = [{"Child": "yes"}] * len(logs)

    table = attach_domain_entropy(logs, {"network.bif": _compiled()})

    assert table["variable"].tolist() == ["Child"]
    assert table.loc[0, "parent_context"] == {"Parent": "low"}
    assert table.loc[0, "entropy"] == pytest.approx(
        -(0.8 * math.log(0.8) + 0.2 * math.log(0.2))
    )


def test_compiled_cache_loads_each_dataset_only_once(tmp_path) -> None:
    path = tmp_path / "network.compiled.pkl"
    with path.open("wb") as file:
        pickle.dump(_compiled(), file)
    cache = CompiledMessagesCache(tmp_path)

    first = cache["network.bif"]
    path.unlink()
    second = cache["network.bif"]

    assert second is first
    assert len(cache) == 1
