"""Statistical joins and correlations for network structure analysis."""

from typing import Any

import pandas as pd
from scipy.stats import spearmanr

from .metrics import filter_by_sampling

STRUCTURAL_COLUMNS = [
    "nodes",
    "edges",
    "max_degree",
    "depth",
    "mean_cardinality",
    "max_cardinality",
]
ERROR_COLUMNS = ["accuracy", "joint_match", "exact_match"]
VARIABLE_STRUCTURAL_COLUMNS = ["n_parents", "cardinality"]


def normalize_dataset_name(name: str) -> str:
    return name.removesuffix(".bif")


def build_structural_merge(
    grouped: pd.DataFrame,
    structure_table: list[dict[str, Any]],
) -> pd.DataFrame:
    per_dataset = grouped.groupby("dataset")[ERROR_COLUMNS].mean().reset_index()
    per_dataset["dataset_norm"] = per_dataset["dataset"].apply(normalize_dataset_name)

    structure = pd.DataFrame(structure_table)
    structure["dataset_norm"] = structure["dataset"].apply(normalize_dataset_name)
    merged = per_dataset.merge(
        structure[["dataset_norm", "label", *STRUCTURAL_COLUMNS]],
        on="dataset_norm",
        how="inner",
    )

    missing = set(per_dataset["dataset_norm"]) - set(merged["dataset_norm"])
    if missing:
        print(
            f"[AVISO] {len(missing)} dataset(s) dos logs não encontrados "
            f"na tabela estrutural: {sorted(missing)}"
        )
    return merged.drop(columns=["dataset_norm"])


def build_variable_structural_merge(
    variable_table: pd.DataFrame,
    variable_structure_table: list[dict[str, Any]],
    evidence_sampling: str | None = None,
) -> pd.DataFrame:
    subset = filter_by_sampling(variable_table, evidence_sampling)
    per_variable = (
        subset.groupby(["dataset", "variable"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    per_variable["dataset_norm"] = per_variable["dataset"].apply(
        normalize_dataset_name
    )

    structure = pd.DataFrame(variable_structure_table)
    structure["dataset_norm"] = structure["dataset"].apply(normalize_dataset_name)
    requested = ["dataset_norm", "variable", *VARIABLE_STRUCTURAL_COLUMNS]
    available = [column for column in requested if column in structure.columns]
    merged = per_variable.merge(
        structure[available],
        on=["dataset_norm", "variable"],
        how="inner",
    )

    for column in VARIABLE_STRUCTURAL_COLUMNS:
        if column not in merged.columns:
            print(
                f"[AVISO] Coluna '{column}' ausente em variable_structure_table."
            )

    logged = set(zip(per_variable["dataset_norm"], per_variable["variable"]))
    matched = set(zip(merged["dataset_norm"], merged["variable"]))
    missing = logged - matched
    if missing:
        sample = sorted(missing)[:10]
        suffix = " ..." if len(missing) > 10 else ""
        print(
            f"[AVISO] {len(missing)} variável(is) dos logs não encontradas "
            f"na tabela estrutural: {sample}{suffix}"
        )
    return merged.drop(columns=["dataset_norm"])


def compute_variable_structural_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in VARIABLE_STRUCTURAL_COLUMNS:
        if column not in merged.columns:
            continue
        values = merged[["accuracy", column]].dropna()
        rho, p_value = (
            spearmanr(values["accuracy"], values[column])
            if len(values) >= 3
            else (float("nan"), float("nan"))
        )
        records.append(
            {
                "structural_metric": column,
                "n": len(values),
                "rho": rho,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(records)


def compute_parents_correlation(merged: pd.DataFrame) -> dict[str, float | int]:
    values = merged[["accuracy", "n_parents"]].dropna()
    if len(values) < 3:
        return {"n": len(values), "rho": float("nan"), "p_value": float("nan")}
    rho, p_value = spearmanr(values["accuracy"], values["n_parents"])
    return {"n": len(values), "rho": rho, "p_value": p_value}
