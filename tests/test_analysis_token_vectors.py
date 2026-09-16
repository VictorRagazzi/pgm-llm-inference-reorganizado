import math
from types import SimpleNamespace

import pytest

from pgm_llm_inference.analysis.entropy_plots import (
    plot_token_vector,
    token_vector_filename,
)
from pgm_llm_inference.analysis.token_vectors import compiled_token_vector_table
from pgm_llm_inference.mpe.types import (
    DecisionTokenScores,
    MessageRow,
    SemanticMessage,
    TokenScore,
)


def _compiled():
    return SimpleNamespace(
        messages={
            "A": SemanticMessage(
                source_variable="A",
                scope=("B",),
                is_evidence=False,
                rows=[
                    MessageRow(
                        context={"B": "1"},
                        selected_value="on",
                        rationale="one",
                        token_scores=DecisionTokenScores(
                            selected_token=TokenScore(token="B", logprob=-0.1),
                            top_logprobs=[
                                TokenScore(token="A", logprob=-2.0),
                                TokenScore(token="B", logprob=-0.1),
                                TokenScore(token=" B", logprob=-3.0),
                            ],
                        ),
                    ),
                    MessageRow(
                        context={"B": "0"},
                        selected_value="off",
                        rationale="zero",
                        token_scores=DecisionTokenScores(
                            selected_token=TokenScore(token="A", logprob=-0.3),
                            top_logprobs=[TokenScore(token="B", logprob=-1.5)],
                        ),
                    ),
                ],
            ),
            "C": SemanticMessage(
                source_variable="C",
                scope=(),
                is_evidence=False,
                rows=[MessageRow(context={}, selected_value="x", rationale="missing")],
            ),
        }
    )


def test_token_table_keeps_every_alternative_and_context() -> None:
    table = compiled_token_vector_table(_compiled())

    assert len(table) == 5
    assert set(table["variable"]) == {"A"}
    first = table.loc[table["row_index"] == 0]
    assert first["token"].tolist() == ["B", "A", " B"]
    assert first["logprob"].tolist() == pytest.approx([-0.1, -2.0, -3.0])
    assert first["context"].unique().tolist() == ['{"B": "1"}']
    second = table.loc[table["row_index"] == 1]
    assert second["token"].tolist() == ["A", "B"]
    assert second["source"].tolist() == ["generated", "top_k"]
    assert second["is_generated"].tolist() == [True, False]


def test_variable_plot_contains_token_vector_not_domain_scores(tmp_path) -> None:
    table = compiled_token_vector_table(_compiled())
    path = tmp_path / token_vector_filename("A")

    plot_token_vector(table, "A", path)

    assert path.is_file()
    assert path.stat().st_size > 0
    with pytest.raises(ValueError, match="No token alternatives"):
        plot_token_vector(table, "C", tmp_path / "missing.svg")


def test_token_vector_filename_is_safe_and_distinguishes_names() -> None:
    assert token_vector_filename("A/B") != token_vector_filename("A:B")
    assert "/" not in token_vector_filename("A/B")


def test_missing_generated_token_is_kept_even_if_top_k_is_empty() -> None:
    compiled = _compiled()
    compiled.messages["A"].rows[0].token_scores.top_logprobs = []

    table = compiled_token_vector_table(compiled)
    selected = table.loc[table["row_index"] == 0].iloc[0]

    assert selected["token"] == "B"
    assert math.isclose(selected["logprob"], -0.1)
    assert selected["source"] == "generated"
