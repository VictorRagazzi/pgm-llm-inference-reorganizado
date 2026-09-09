import unittest

from pgm_llm_inference.paths import (
    DATASETS_DIR,
    METADATA_DIR,
    RELATIONSHIPS_DIR,
    SOURCE_ROOT,
    dataset_path,
    metadata_path,
    relationship_path,
)


class ProjectPathsTests(unittest.TestCase):
    def test_runtime_directories_share_the_source_root(self) -> None:
        self.assertEqual(DATASETS_DIR, SOURCE_ROOT / "datasets")
        self.assertEqual(METADATA_DIR, SOURCE_ROOT / "metadata")
        self.assertEqual(RELATIONSHIPS_DIR, SOURCE_ROOT / "relationships")

    def test_artifact_paths_use_the_dataset_stem(self) -> None:
        self.assertEqual(dataset_path("network.bif"), DATASETS_DIR / "network.bif")
        self.assertEqual(metadata_path("network.bif.gz"), METADATA_DIR / "network.jsonl")
        self.assertEqual(
            relationship_path("network.bif.gz"),
            RELATIONSHIPS_DIR / "network.jsonl",
        )
