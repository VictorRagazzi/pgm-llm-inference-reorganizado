"""Join compiled per-context domain scores to variable-level log metrics."""

from __future__ import annotations

import math
import pickle
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pandas as pd

from pgm_llm_inference.mpe.compile import (
    COMPILED_SCHEMA_VERSION,
    CompiledSemanticMessages,
)
from pgm_llm_inference.mpe.normalization import normalize_assignment
from pgm_llm_inference.mpe.types import DecisionTokenScores
from pgm_llm_inference.paths import COMPILED_TABLES_DIR

from .metrics import _group_rows, calc_per_variable_hits


ScoreKind = Literal["logprob", "probability"]


class CompiledMessagesCache:
    """Read-only, in-memory cache for trusted compiled pickle artifacts.

    By default, ``dataset.bif`` resolves to
    ``src/tables/dataset.compiled.pkl``. ``dataset_paths`` makes variants such
    as model-specific cache names explicit instead of choosing one by fuzzy
    matching.

    Pickle files can execute code while loading and must therefore come from a
    trusted source.
    """

    def __init__(
        self,
        tables_dir: str | Path = COMPILED_TABLES_DIR,
        *,
        dataset_paths: Mapping[str, str | Path] | None = None,
    ) -> None:
        self.tables_dir = Path(tables_dir)
        self.dataset_paths = {
            dataset: Path(path) for dataset, path in (dataset_paths or {}).items()
        }
        self._items: dict[str, CompiledSemanticMessages] = {}

    def __getitem__(self, dataset: str) -> CompiledSemanticMessages:
        if dataset not in self._items:
            path = self.dataset_paths.get(dataset)
            if path is None:
                path = self.tables_dir / f"{Path(dataset).stem}.compiled.pkl"
            if not path.exists():
                raise FileNotFoundError(
                    f"No compiled messages found for {dataset!r}: {path}"
                )
            with path.open("rb") as file:
                compiled = pickle.load(file)
            if getattr(compiled, "schema_version", None) != COMPILED_SCHEMA_VERSION:
                raise ValueError(
                    f"Compiled cache for {dataset!r} has schema version "
                    f"{getattr(compiled, 'schema_version', None)!r}; expected "
                    f"{COMPILED_SCHEMA_VERSION}."
                )
            self._items[dataset] = compiled
        return self._items[dataset]

    def __len__(self) -> int:
        return len(self._items)


def domain_entropy(
    by_state: Mapping[str, float],
    *,
    score_kind: ScoreKind = "logprob",
) -> float:
    """Return Shannon entropy in nats for the available state scores.

    Current compilations store token log-probabilities, so ``logprob`` applies
    a numerically stable softmax. ``probability`` exists for explicitly
    identified legacy artifacts and normalizes their non-negative values.
    Only states present in ``by_state`` participate; ``default_score`` is not
    a score for a named state and is intentionally ignored.
    """

    if not by_state:
        raise ValueError("Cannot compute entropy from an empty by_state mapping.")
    values = [float(value) for value in by_state.values()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Domain scores must all be finite.")

    if score_kind == "logprob":
        maximum = max(values)
        weights = [math.exp(value - maximum) for value in values]
    elif score_kind == "probability":
        if any(value < 0 for value in values):
            raise ValueError("Probability scores cannot be negative.")
        weights = values
    else:
        raise ValueError(f"Unknown score kind: {score_kind!r}.")

    total = sum(weights)
    if total <= 0:
        raise ValueError("Domain scores must have positive total weight.")
    probabilities = [weight / total for weight in weights]
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    )


def top_token_statistics(scores: DecisionTokenScores) -> tuple[float, float, int]:
    """Return top-k conditional entropy, captured mass, and token count.

    The entropy is conditional on the returned alternatives; it is not the
    full-vocabulary entropy and must not be compared with domain entropy.
    """
    logprobs = [item.logprob for item in scores.top_logprobs]
    if not logprobs:
        return math.nan, math.nan, 0
    if not all(math.isfinite(value) for value in logprobs):
        raise ValueError("Top-token log-probabilities must be finite.")
    probabilities = [math.exp(value) for value in logprobs]
    mass = sum(probabilities)
    if mass <= 0:
        return math.nan, mass, len(logprobs)
    conditional = [value / mass for value in probabilities]
    entropy = -sum(value * math.log(value) for value in conditional if value > 0)
    return entropy, mass, len(logprobs)


def compiled_row_entropy_table(compiled: CompiledSemanticMessages) -> pd.DataFrame:
    """Summarize both entropy measures for every compiled context row."""
    records = []
    for variable, message in compiled.messages.items():
        domain = compiled.bn.variables[variable].states
        for row in message.rows:
            if row.selected_value is None:
                continue
            scores = getattr(row, "domain_scores", None)
            if scores is not None and set(scores.by_state) != set(domain):
                raise ValueError(f"Incomplete domain scores for {variable!r}.")
            token_scores = getattr(row, "token_scores", None)
            token_entropy, token_mass, token_count = (
                top_token_statistics(token_scores)
                if token_scores is not None
                else (math.nan, math.nan, 0)
            )
            records.append(
                {
                    "variable": variable,
                    "context": json.dumps(row.context, ensure_ascii=False, sort_keys=True),
                    "selected_value": row.selected_value,
                    "domain_size": len(domain),
                    "domain_entropy": (
                        domain_entropy(scores.by_state)
                        if scores is not None
                        else math.nan
                    ),
                    "token_entropy_top_k": token_entropy,
                    "token_top_k_mass": token_mass,
                    "token_top_k_count": token_count,
                }
            )
    return pd.DataFrame.from_records(records)


def _compiled_for_dataset(
    compiled_cache: CompiledMessagesCache | Mapping[str, CompiledSemanticMessages],
    dataset: str,
) -> CompiledSemanticMessages:
    try:
        return compiled_cache[dataset]
    except KeyError:
        stem = Path(dataset).stem
        try:
            return compiled_cache[stem]
        except KeyError as error:
            raise KeyError(
                f"No compiled messages registered for dataset {dataset!r}."
            ) from error


def _parents_for(
    compiled: CompiledSemanticMessages,
    variable: str,
) -> tuple[str, ...]:
    try:
        parents = tuple(compiled.bn.parents[variable])
    except AttributeError:
        # Pickles created before explicit_parents was introduced retain the
        # old _parents field. Reading it here preserves scientific artifacts
        # without mutating or reserializing them.
        legacy_parents = compiled.bn.__dict__.get("_parents")
        if not isinstance(legacy_parents, Mapping) or variable not in legacy_parents:
            raise
        parents = tuple(legacy_parents[variable])

    message_scope = tuple(compiled.messages[variable].scope)
    if set(parents) != set(message_scope):
        raise ValueError(
            f"Parent/message scope mismatch for {variable!r}: "
            f"parents={parents!r}, message.scope={message_scope!r}."
        )
    return parents


def _row_entropy(
    compiled: CompiledSemanticMessages,
    variable: str,
    map_assignment: Mapping[str, str],
    evidence: Mapping[str, str],
    *,
    score_kind: ScoreKind,
) -> tuple[dict[str, str], float, float, float, int]:
    raw_map_assignment = {
        key: value for key, value in map_assignment.items() if key != "_scalar"
    }
    normalized_map = normalize_assignment(
        raw_map_assignment, compiled.bn, compiled.alias_map
    )
    normalized_evidence = normalize_assignment(
        dict(evidence), compiled.bn, compiled.alias_map
    )
    conflicts = {
        variable
        for variable in normalized_map.keys() & normalized_evidence.keys()
        if normalized_map[variable] != normalized_evidence[variable]
    }
    if conflicts:
        raise ValueError(
            "Conflicting values between evidence and map_assignment for: "
            f"{sorted(conflicts)}."
        )
    assignment = {**normalized_evidence, **normalized_map}
    parents = _parents_for(compiled, variable)
    try:
        context = {parent: assignment[parent] for parent in parents}
    except KeyError as error:
        raise ValueError(
            f"map_assignment is missing parent {error.args[0]!r} of {variable!r}."
        ) from error

    matching_rows = [
        row for row in compiled.messages[variable].rows if row.context == context
    ]
    if len(matching_rows) != 1:
        raise ValueError(
            f"Expected exactly one row for {variable!r} with context {context!r}; "
            f"found {len(matching_rows)}."
        )

    row = matching_rows[0]
    scores = getattr(row, "domain_scores", None)
    entropy = (
        domain_entropy(scores.by_state, score_kind=score_kind)
        if scores is not None
        else math.nan
    )
    token_scores = getattr(row, "token_scores", None)
    token_entropy, token_mass, token_count = (
        top_token_statistics(token_scores)
        if token_scores is not None
        else (math.nan, math.nan, 0)
    )
    return context, entropy, token_entropy, token_mass, token_count


def attach_domain_entropy(
    log_rows: pd.DataFrame,
    compiled_cache: CompiledMessagesCache
    | Mapping[str, CompiledSemanticMessages]
    | None = None,
    *,
    score_kind: ScoreKind = "logprob",
) -> pd.DataFrame:
    """Build the existing variable-level table and append row entropy.

    The grouping and ``correct`` value are exactly those used by
    :func:`analysis.metrics.build_variable_level_table`. ``parent_context`` is
    included so sampled values can be audited before downstream statistics.
    Missing optional ``domain_scores`` produce ``NaN`` entropy while lookup
    inconsistencies fail loudly.
    """

    cache = compiled_cache if compiled_cache is not None else CompiledMessagesCache()
    records: list[dict] = []
    for rows in _group_rows(log_rows).values():
        representative = rows[0]
        dataset = representative.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("Every log group must identify a non-empty dataset.")
        compiled = _compiled_for_dataset(cache, dataset)
        assignment = representative.get("map_assignment", {})
        evidence = representative.get("evidence", {})

        for record in calc_per_variable_hits(rows):
            context, entropy, token_entropy, token_mass, token_count = _row_entropy(
                compiled,
                record["variable"],
                assignment,
                evidence,
                score_kind=score_kind,
            )
            record["entropy"] = entropy
            record["token_entropy_top_k"] = token_entropy
            record["token_top_k_mass"] = token_mass
            record["token_top_k_count"] = token_count
            record["parent_context"] = context
            records.append(record)

    return pd.DataFrame(records)
