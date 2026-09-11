import unittest
from unittest.mock import patch

from pgm_llm_inference import BayesianNetwork
from pgm_llm_inference.mpe import compile_semantic_messages
from pgm_llm_inference.mpe.types import PromptTrace


class CptlessNetworkTests(unittest.TestCase):
    def test_builds_parent_map_without_factors(self) -> None:
        network = BayesianNetwork.from_structure(
            name="example",
            variable_states={
                "Weather": ("sunny", "rainy"),
                "Traffic": ("light", "heavy"),
                "Delay": ("no", "yes"),
            },
            children={
                "Weather": ("Traffic",),
                "Traffic": ("Delay",),
            },
        )

        self.assertEqual(network.name, "example")
        self.assertEqual(network.factors, [])
        self.assertEqual(network.parents["Weather"], ())
        self.assertEqual(network.parents["Traffic"], ("Weather",))
        self.assertEqual(network.parents["Delay"], ("Traffic",))
        self.assertEqual(network.children_map()["Traffic"], ("Delay",))

    def test_rejects_unknown_edge_variable(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown child"):
            BayesianNetwork.from_structure(
                variable_states={"A": ("no", "yes")},
                children={"A": ("Missing",)},
            )

    def test_rejects_cycles(self) -> None:
        with self.assertRaisesRegex(ValueError, "cycle detected"):
            BayesianNetwork.from_structure(
                variable_states={
                    "A": ("no", "yes"),
                    "B": ("no", "yes"),
                },
                children={"A": ("B",), "B": ("A",)},
            )

    @patch("pgm_llm_inference.mpe.compile.LLMJsonClient")
    def test_semantic_compilation_accepts_an_in_memory_network(self, client_class) -> None:
        network = BayesianNetwork.from_structure(
            variable_states={"A": ("no", "yes")},
            children={},
        )

        def fake_llm(_prompt, schema):
            if schema.__name__ == "NetworkMetadataResponse":
                return schema.model_validate(
                    {
                        "metadata": {
                            "A": {
                                "display_name": "Variable A",
                                "state_meanings": {"no": "absent", "yes": "present"},
                            }
                        }
                    }
                )
            return schema.model_validate({"relationships": {}})

        def fake_complete(*, purpose, variable, prompt, response_model, **_kwargs):
            if purpose == "network_briefing":
                response = response_model.model_validate(
                    {
                        "network_summary": "One binary variable.",
                        "important_dependencies": [],
                        "reasoning_rules": [],
                    }
                )
            else:
                response = response_model.model_validate(
                    {
                        "variable": variable,
                        "decisions": [
                            {
                                "context": {},
                                "selected_value": "no",
                                "confidence": "medium",
                                "rationale": "Basal state.",
                            }
                        ],
                    }
                )
            return response, PromptTrace(purpose=purpose, variable=variable, prompt=prompt)

        client_class.return_value.complete_json.side_effect = fake_complete

        compiled = compile_semantic_messages(
            network=network,
            llm_fn=fake_llm,
        )

        self.assertIs(compiled.bn, network)
        self.assertEqual(compiled.metadata["A"].display_name, "Variable A")
        self.assertEqual(compiled.messages["A"].rows[0].selected_value, "no")


if __name__ == "__main__":
    unittest.main()
