import json
import tempfile
import unittest
from pathlib import Path

from pgm_llm_inference.analysis.logs import (
    frozen_evidence,
    load_logs,
    translate_dataset,
)


class AnalysisLogsTests(unittest.TestCase):
    def test_load_logs_preserves_legacy_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            log_path = Path(temporary_directory) / "experiments.jsonl"
            log_path.write_text(
                json.dumps({"dataset": "adhd_cbeb.bif", "evidence": {"B": "2"}})
                + "\n",
                encoding="utf-8",
            )

            data = load_logs(log_path)

        self.assertEqual(data.loc[0, "evidence_sampling"], "mpe_consistent")
        self.assertEqual(data.loc[0, "evidence_layout"], "nested")
        self.assertIsNone(data.loc[0, "exact_match"])

    def test_normalizes_labels_and_evidence_keys(self) -> None:
        self.assertEqual(translate_dataset("adhd_cbeb.bif"), "TDAH")
        self.assertEqual(translate_dataset("unknown.bif"), "unknown.bif")
        self.assertEqual(frozen_evidence({"B": "2", "A": "1"}), '{"A": "1", "B": "2"}')
