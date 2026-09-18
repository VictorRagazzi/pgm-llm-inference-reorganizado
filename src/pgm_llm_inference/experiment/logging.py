"""Append experiment results to the configured JSONL log."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pgm_llm_inference.core.config import InferenceConfig

LOG_PATH = Path(InferenceConfig().log_file_name)


def log_experiment(record: dict[str, Any]) -> None:
    record = {
        **record,
        "timestamp": datetime.utcnow().isoformat(),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
