from collections import Counter

import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

from .logs import frozen_evidence


GROUP_COLUMNS = ["dataset", "evidence_sampling", "evidence_layout"]


def _mode_predictions(rows: list[dict]) -> dict[str, str]:
    all_keys = {key for row in rows for key in row.get("llm_predictions", {})}
    predictions = {}
    for key in all_keys:
        values = [
            row["llm_predictions"][key]
            for row in rows
            if key in row.get("llm_predictions", {})
        ]
        if values:
            predictions[key] = Counter(values).most_common(1)[0][0]
    return predictions


def calc_group_metrics(rows: list[dict]) -> dict:
    representative = rows[0]
    assignment = representative.get("map_assignment", {})
    targets = [key for key in assignment if key != "_scalar"]
    if not targets:
        return {}

    mode_predictions = _mode_predictions(rows)
    y_true = [assignment[key] for key in targets if key in mode_predictions]
    y_pred = [mode_predictions[key] for key in targets if key in mode_predictions]
    if not y_true:
        return {}

    count = len(y_true)
    hits = sum(expected == predicted for expected, predicted in zip(y_true, y_pred))
    labels = sorted(set(y_true + y_pred))
    if len(labels) == 1:
        f1_macro, f1_weighted = (1.0, 1.0) if y_true == y_pred else (0.0, 0.0)
        kappa = None
    else:
        f1_macro = f1_score(
            y_true, y_pred, average="macro", labels=labels, zero_division=0
        )
        f1_weighted = f1_score(
            y_true, y_pred, average="weighted", labels=labels, zero_division=0
        )
        try:
            kappa = cohen_kappa_score(y_true, y_pred, labels=labels)
        except Exception:
            kappa = None

    consensus_scores = []
    for key in targets:
        predictions = [
            row["llm_predictions"][key]
            for row in rows
            if key in row.get("llm_predictions", {})
        ]
        if predictions:
            top_count = Counter(predictions).most_common(1)[0][1]
            consensus_scores.append(top_count / len(predictions))

    average_consensus = (
        sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.0
    )
    joint_match = float(len(mode_predictions) == count and hits == count)
    reported_exact = [
        row.get("exact_match") for row in rows if row.get("exact_match") is not None
    ]
    mean_exact = sum(reported_exact) / len(reported_exact) if reported_exact else None

    return {
        "accuracy": hits / count,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "kappa": kappa,
        "consensus": average_consensus,
        "joint_match": joint_match,
        "exact_match": mean_exact,
        "hits": hits,
        "total_nodes": count,
    }


def calc_per_variable_hits(rows: list[dict]) -> list[dict]:
    representative = rows[0]
    assignment = representative.get("map_assignment", {})
    targets = [key for key in assignment if key != "_scalar"]
    if not targets:
        return []

    mode_predictions = _mode_predictions(rows)
    records = []
    for variable in targets:
        if variable not in mode_predictions:
            continue
        records.append(
            {
                "dataset": representative.get("dataset"),
                "evidence_sampling": representative.get(
                    "evidence_sampling", "mpe_consistent"
                ),
                "evidence_layout": representative.get("evidence_layout", "nested"),
                "variable": variable,
                "correct": int(mode_predictions[variable] == assignment[variable]),
            }
        )
    return records


def _group_rows(data: pd.DataFrame) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for row in data.to_dict("records"):
        key = tuple(row.get(column) for column in GROUP_COLUMNS) + (
            frozen_evidence(row.get("evidence")),
        )
        groups.setdefault(key, []).append(row)
    return groups


def compute_grouped_table(data: pd.DataFrame, resolution: int = 5) -> pd.DataFrame:
    records = []
    for key, rows in _group_rows(data).items():
        dataset, sampling, layout, _ = key
        metrics = calc_group_metrics(rows)
        if not metrics:
            continue

        representative = rows[0]
        evidence_length = representative.get("evidence_length", 0)
        evaluated_length = representative.get("evaluated_length", 0)
        total = evidence_length + evaluated_length
        raw_ratio = evidence_length / total * 100 if total > 0 else 0
        records.append(
            {
                "dataset": dataset,
                "evidence_sampling": sampling,
                "evidence_layout": layout,
                "evidence_length": evidence_length,
                "evidence_ratio": round(raw_ratio / resolution) * resolution,
                **metrics,
            }
        )
    return pd.DataFrame(records)


def filter_by_sampling(
    data: pd.DataFrame, evidence_sampling: str | None
) -> pd.DataFrame:
    if not evidence_sampling:
        return data
    return data[data["evidence_sampling"] == evidence_sampling]


def build_variable_level_table(data: pd.DataFrame) -> pd.DataFrame:
    records = []
    for rows in _group_rows(data).values():
        records.extend(calc_per_variable_hits(rows))
    return pd.DataFrame(records)
