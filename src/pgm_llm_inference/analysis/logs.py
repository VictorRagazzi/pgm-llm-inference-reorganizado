import json
from pathlib import Path

import pandas as pd


DATASET_LABELS: dict[str, str] = {
    "adhd_cbeb.bif": "TDAH",
    "covid1_cbeb.bif": "Sintomas de Covid - 1",
    "covid3_cbeb.bif": "Sintomas de Covid - 2",
    "gonorrhoeae_cbeb.bif": "Gonorreia",
    "hepar2_cbeb.bif": "Hepatite",
    "alarm_cbeb.bif": "Monitoramento de UTI",
    "foodallergy3_cbeb.bif": "Alergia - 2",
    "foodallergy1_cbeb.bif": "Alergia - 1",
    "diabets_cbeb.bif": "Diabetes",
    "child_cbeb.bif": "Doenças pediátricas",
}

SAMPLING_LABELS: dict[str, str] = {
    "mpe_consistent": "Evidência MPE-consistente",
    "mpe_inconsistent": "Evidência MPE-inconsistente",
}


def translate_dataset(name: str) -> str:
    return DATASET_LABELS.get(name, name)


def translate_sampling(name: str) -> str:
    return SAMPLING_LABELS.get(name, name)


def load_logs(path: str | Path) -> pd.DataFrame:
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    data = pd.DataFrame(records)
    if "evidence_sampling" not in data.columns:
        data["evidence_sampling"] = "mpe_consistent"
    if "evidence_layout" not in data.columns:
        data["evidence_layout"] = "nested"
    if "exact_match" not in data.columns:
        data["exact_match"] = None

    data["evidence_sampling"] = data["evidence_sampling"].fillna("mpe_consistent")
    data["evidence_layout"] = data["evidence_layout"].fillna("nested")
    return data


def frozen_evidence(evidence: dict) -> str:
    if not evidence:
        return "{}"
    return json.dumps(dict(sorted(evidence.items())), sort_keys=True)
