import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pgm_llm_inference.mpe.infer import infer_from_compiled
from pgm_llm_inference.mpe.types import DomainScores


class InferFromCompiledTests(unittest.TestCase):
    @patch("pgm_llm_inference.mpe.infer.reconstruct_assignment")
    @patch("pgm_llm_inference.mpe.infer.normalize_assignment")
    def test_reconstructs_and_serializes_non_evidence_messages(
        self,
        normalize_assignment: Mock,
        reconstruct_assignment: Mock,
    ) -> None:
        normalize_assignment.return_value = {"Observed": "yes"}
        reconstruct_assignment.return_value = (
            {"Hidden": "on"},
            {"Observed": "yes", "Hidden": "on"},
            {"Hidden": ["on", 0.9]},
        )
        observed_row = SimpleNamespace(
            context={}, selected_value="yes", confidence=1.0, rationale="observed"
        )
        hidden_row = SimpleNamespace(
            context={"Observed": "yes"},
            selected_value="on",
            confidence=0.9,
            rationale="compiled",
            domain_scores=DomainScores(by_state={"off": -2.0, "on": -0.1}),
        )
        compiled = SimpleNamespace(
            bn=object(),
            alias_map={"observed": "Observed"},
            elimination_order=["Hidden", "Observed"],
            messages={
                "Observed": SimpleNamespace(rows=[observed_row]),
                "Hidden": SimpleNamespace(rows=[hidden_row]),
            },
        )

        with contextlib.redirect_stdout(io.StringIO()):
            hidden, confidence, cpt = infer_from_compiled(
                compiled=compiled,
                evidence={"observed": "YES"},
            )

        normalize_assignment.assert_called_once_with(
            {"observed": "YES"}, compiled.bn, compiled.alias_map
        )
        reconstruct_assignment.assert_called_once_with(
            elimination_order=compiled.elimination_order,
            evidence={"Observed": "yes"},
            messages=compiled.messages,
        )
        self.assertEqual(hidden, {"Hidden": "on"})
        self.assertEqual(confidence, {"Hidden": ["on", 0.9]})
        self.assertEqual(
            cpt,
            {
                "Hidden": [
                    {
                        "context": {"Observed": "yes"},
                        "selected_value": "on",
                        "confidence": 0.9,
                        "rationale": "compiled",
                        "domain_scores": {
                            "by_state": {"off": -2.0, "on": -0.1},
                            "default_score": 0.0,
                        },
                    }
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
