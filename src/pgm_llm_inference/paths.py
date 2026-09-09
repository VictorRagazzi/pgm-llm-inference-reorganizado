"""Canonical locations for datasets and generated experiment artifacts."""

from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = SOURCE_ROOT / "datasets"
METADATA_DIR = SOURCE_ROOT / "metadata"
RELATIONSHIPS_DIR = SOURCE_ROOT / "relationships"
COMPILED_TABLES_DIR = SOURCE_ROOT / "tables"


def dataset_path(dataset_name: str) -> Path:
    return DATASETS_DIR / dataset_name


def metadata_path(dataset_name: str) -> Path:
    stem = Path(dataset_name).name.split(".", 1)[0]
    return METADATA_DIR / f"{stem}.jsonl"


def relationship_path(dataset_name: str) -> Path:
    stem = Path(dataset_name).name.split(".", 1)[0]
    return RELATIONSHIPS_DIR / f"{stem}.jsonl"
