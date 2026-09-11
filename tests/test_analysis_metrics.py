import unittest

import pandas as pd

from pgm_llm_inference.analysis.metrics import (
    build_variable_level_table,
    compute_grouped_table,
)


class AnalysisMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        common = {
            "dataset": "network.bif",
            "evidence_sampling": "mpe_consistent",
            "evidence_layout": "nested",
            "evidence": {"Observed": "yes"},
            "evidence_length": 1,
            "evaluated_length": 2,
            "map_assignment": {"A": "yes", "B": "no"},
        }
        self.logs = pd.DataFrame(
            [
                {
                    **common,
                    "llm_predictions": {"A": "yes", "B": "no"},
                    "exact_match": 1,
                },
                {
                    **common,
                    "llm_predictions": {"A": "yes", "B": "yes"},
                    "exact_match": 0,
                },
                {
                    **common,
                    "llm_predictions": {"A": "no", "B": "no"},
                    "exact_match": 0,
                },
            ]
        )

    def test_groups_votes_and_preserves_existing_metrics(self) -> None:
        grouped = compute_grouped_table(self.logs)

        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped.loc[0, "accuracy"], 1.0)
        self.assertEqual(grouped.loc[0, "joint_match"], 1.0)
        self.assertAlmostEqual(grouped.loc[0, "consensus"], 2 / 3)
        self.assertAlmostEqual(grouped.loc[0, "exact_match"], 1 / 3)
        self.assertEqual(grouped.loc[0, "evidence_ratio"], 35)

    def test_builds_one_result_per_evaluated_variable(self) -> None:
        variables = build_variable_level_table(self.logs)

        self.assertEqual(set(variables["variable"]), {"A", "B"})
        self.assertEqual(variables["correct"].tolist(), [1, 1])
