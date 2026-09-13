import pytest
from pydantic import ValidationError

from pgm_llm_inference.mpe.types import ContextDecision, DomainScores, MessageRow


def test_domain_scores_requires_state_scores_and_defaults_missing_state_score() -> None:
    scores = DomainScores(by_state={"Baixa": -1.25, "Média": -0.35, "Alta": -2.0})

    assert scores.by_state == {"Baixa": -1.25, "Média": -0.35, "Alta": -2.0}
    assert scores.default_score == 0.0

    with pytest.raises(ValidationError):
        DomainScores()


def test_context_decision_accepts_nested_domain_scores() -> None:
    decision = ContextDecision.model_validate(
        {
            "context": {},
            "selected_value": "Média",
            "confidence": "high",
            "rationale": "Most likely state.",
            "domain_scores": {
                "by_state": {"Baixa": -1.25, "Média": -0.35, "Alta": -2.0},
                "default_score": -10.0,
            },
        }
    )

    assert decision.domain_scores == DomainScores(
        by_state={"Baixa": -1.25, "Média": -0.35, "Alta": -2.0},
        default_score=-10.0,
    )


def test_message_row_accepts_optional_domain_scores() -> None:
    row = MessageRow(context={}, selected_value="Sim", rationale="Selected state.")
    scored_row = MessageRow.model_validate(
        {
            "context": {},
            "selected_value": "Sim",
            "rationale": "Selected state.",
            "domain_scores": {"by_state": {"Não": -2.0, "Sim": -0.1}},
        }
    )

    assert row.domain_scores is None
    assert scored_row.domain_scores == DomainScores(by_state={"Não": -2.0, "Sim": -0.1})
