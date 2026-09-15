import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pgm_llm_inference.mpe.cache import compiled_cache_path, load_or_compile
from pgm_llm_inference.mpe.compile import COMPILED_SCHEMA_VERSION


class CompiledMessagesCacheTests(unittest.TestCase):
    def test_active_switch_keeps_one_cache_file_per_dataset_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            with patch(
                "pgm_llm_inference.mpe.cache.COMPILED_TABLES_DIR",
                cache_directory,
            ):
                path = compiled_cache_path("network.bif", "provider/model")

        self.assertEqual(path.name, "network.provider__model_VE.compiled.pkl")

    @patch("pgm_llm_inference.mpe.cache.compile_semantic_messages")
    def test_compiles_once_and_reuses_the_dataset_cache(self, compile_messages: Mock) -> None:
        compile_messages.return_value = SimpleNamespace(
            messages={"Node": "message"},
            schema_version=COMPILED_SCHEMA_VERSION,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_directory = Path(temporary_directory)
            with patch(
                "pgm_llm_inference.mpe.cache.COMPILED_TABLES_DIR",
                cache_directory,
            ):
                arguments = {
                    "dataset_name": "network.bif",
                    "model_name": "provider/model",
                    "network": object(),
                    "bif_path": Path("network.bif"),
                    "metadata_path": Path("metadata.jsonl"),
                    "relationship_path": Path("relationships.jsonl"),
                    "llm_fn": object(),
                    "use_real_llm": False,
                    "max_context_rows_per_call": 12,
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    first = load_or_compile(**arguments)
                    second = load_or_compile(**arguments)

        self.assertEqual(first.messages, second.messages)
        compile_messages.assert_called_once_with(
            network=arguments["network"],
            bif_path=arguments["bif_path"],
            metadata_path=arguments["metadata_path"],
            relationship_path=arguments["relationship_path"],
            llm_fn=arguments["llm_fn"],
            use_real_llm=False,
            max_context_rows_per_call=12,
        )
