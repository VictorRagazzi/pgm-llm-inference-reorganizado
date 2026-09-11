import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pgm_llm_inference.experiment.config import ExperimentConfig
from pgm_llm_inference.experiment.runner import run_experiment


class ExperimentConfigTests(unittest.TestCase):
    def test_rejects_unknown_inference_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid mode"):
            ExperimentConfig(dataset_name="network.bif", inference_mode="unknown")


class RunExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.network = SimpleNamespace(variables={"Observed": object(), "Hidden": object()})
        self.batch_config = {
            "prompt_type": "variable_assignment",
            "evidence": {"Observed": "yes"},
            "query_vars": None,
        }
        self.config = ExperimentConfig(dataset_name="network.bif")
        self.compiled = object()

    @patch("pgm_llm_inference.experiment.runner.get_model_name", return_value="model")
    @patch("pgm_llm_inference.experiment.runner.count_llm_hits")
    @patch("pgm_llm_inference.experiment.runner.infer_from_compiled")
    @patch("pgm_llm_inference.experiment.runner.run_max_product")
    def test_builds_the_existing_mpe_log_record(
        self,
        run_max_product: Mock,
        infer_from_compiled: Mock,
        count_llm_hits: Mock,
        _get_model_name: Mock,
    ) -> None:
        run_max_product.return_value = {"map_assignment": {"Hidden": "on"}}
        infer_from_compiled.return_value = (
            {"Hidden": "on"},
            {"Hidden": ["on", 0.9]},
            {"Hidden": []},
        )
        count_llm_hits.return_value = (1, {"Hidden": "on"})

        with contextlib.redirect_stdout(io.StringIO()):
            result = run_experiment(
                network=self.network,
                batch_config=self.batch_config,
                config=self.config,
                compiled=self.compiled,
            )

        run_max_product.assert_called_once_with(
            network=self.network,
            query_vars=["Hidden"],
            evidence={"Observed": "yes"},
        )
        infer_from_compiled.assert_called_once_with(
            compiled=self.compiled,
            evidence={"Observed": "yes"},
        )
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["log_accuracy"], 1.0)
        self.assertEqual(result["accuracy"], "100.00%")
        self.assertEqual(result["exact_match"], 1)
        self.assertEqual(result["llm_predictions"], {"Hidden": "on"})
        self.assertEqual(result["map_assignment"], {"Hidden": "on"})

    @patch("pgm_llm_inference.experiment.runner.infer_from_compiled")
    @patch("pgm_llm_inference.experiment.runner.run_max_product")
    def test_rejects_experiments_above_the_configured_limit(
        self,
        run_max_product: Mock,
        infer_from_compiled: Mock,
    ) -> None:
        self.config.max_hidden_variables = 0

        with self.assertRaisesRegex(ValueError, "Hidden-variable limit"):
            with contextlib.redirect_stdout(io.StringIO()):
                run_experiment(
                    network=self.network,
                    batch_config=self.batch_config,
                    config=self.config,
                    compiled=self.compiled,
                )

        run_max_product.assert_not_called()
        infer_from_compiled.assert_not_called()


if __name__ == "__main__":
    unittest.main()
