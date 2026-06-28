import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from pgm_llm_inference.core.config import InferenceConfig

config = InferenceConfig()

LOG_PATH = Path(config.log_file_name)
CSV_LOG_PATH = Path("logs/experiments.csv")

def log_experiment(record: Dict[str, Any]) -> None:
    record = {
        **record,
        "timestamp": datetime.utcnow().isoformat(),
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_experiment_csv(record: Dict[str, Any]) -> None:
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **record,
    }

    CSV_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = CSV_LOG_PATH.exists()

    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)