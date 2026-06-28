"""
mpe/io.py
=========
I/O e normalização de nomes do pipeline MPE.

- parse_bif: lê um arquivo .bif e retorna BayesianNetwork (MPE)
- load_metadata: carrega metadados de variáveis de um .json
- build_alias_map / resolve_variable: normalização de nomes vindos do LLM
- normalize_assignment / parse_assignment_arg: normalização de evidence
- json_dumps: helper de serialização compartilhado
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..models import BayesianNetwork, Variable
from .types import VariableMetadata


# ---------------------------------------------------------------------------
# Serialização
# ---------------------------------------------------------------------------

def json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2)


# ---------------------------------------------------------------------------
# Parser BIF
# ---------------------------------------------------------------------------

def parse_bif(path: Path) -> BayesianNetwork:
    """Lê um arquivo .bif e retorna BayesianNetwork do pipeline MPE."""
    text = path.read_text(encoding="utf-8")
    network_match = re.search(r"\bnetwork\s+([A-Za-z0-9_]+)", text)
    network_name = network_match.group(1) if network_match else path.stem

    variables: dict[str, Variable] = {}
    variable_pattern = re.compile(
        r"variable\s+([A-Za-z0-9_]+)\s*\{\s*"
        r"type\s+discrete\s*\[\s*\d+\s*\]\s*\{([^}]*)\}\s*;\s*\}",
        re.DOTALL,
    )
    for match in variable_pattern.finditer(text):
        name = match.group(1)
        states = tuple(part.strip() for part in match.group(2).split(",") if part.strip())
        if not states:
            raise ValueError(f"Variable {name} has no parsed states.")
        variables[name] = Variable(name=name, states=states)

    if not variables:
        raise ValueError(f"No variables were parsed from {path}.")

    parents: dict[str, tuple[str, ...]] = {name: () for name in variables}
    probability_pattern = re.compile(
        r"probability\s*\(\s*([A-Za-z0-9_]+)\s*(?:\|\s*([^)]+?))?\s*\)\s*\{",
        re.DOTALL,
    )
    for match in probability_pattern.finditer(text):
        child = match.group(1).strip()
        if child not in variables:
            raise ValueError(f"Probability block references unknown variable {child}.")
        parent_text = match.group(2) or ""
        parsed_parents = tuple(
            parent.strip() for parent in parent_text.split(",") if parent.strip()
        )
        for parent in parsed_parents:
            if parent not in variables:
                raise ValueError(
                    f"Probability block for {child} references unknown parent {parent}."
                )
        parents[child] = parsed_parents

    return BayesianNetwork(name=network_name, variables=variables, parents=parents)


# ---------------------------------------------------------------------------
# Metadata loader
# ---------------------------------------------------------------------------

def load_metadata(path: Path | None, bn: BayesianNetwork) -> dict[str, VariableMetadata]:
    if path is None:
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Metadata JSON must be an object keyed by BIF variable ID.")

    metadata: dict[str, VariableMetadata] = {}
    for variable, raw_metadata in payload.items():
        if variable not in bn.variables:
            raise ValueError(f"Metadata references unknown variable {variable}.")
        metadata[variable] = VariableMetadata.model_validate(raw_metadata)
    return metadata


# ---------------------------------------------------------------------------
# Alias map: normaliza nomes vindos do LLM para IDs BIF canônicos
# ---------------------------------------------------------------------------

def canonical_key(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value).lower()


def build_alias_map(
    bn: BayesianNetwork, metadata: dict[str, VariableMetadata]
) -> dict[str, str]:
    alias_map: dict[str, str] = {}

    def add(alias: str, variable: str) -> None:
        candidates = {alias, canonical_key(alias)}
        for candidate in candidates:
            existing = alias_map.get(candidate)
            if existing is not None and existing != variable:
                raise ValueError(
                    f"Alias {alias!r} maps to both {existing!r} and {variable!r}."
                )
            alias_map[candidate] = variable

    for variable in bn.variables:
        add(variable, variable)
        item = metadata.get(variable)
        if item is None:
            continue
        if item.display_name:
            add(item.display_name, variable)
        for alias in item.aliases:
            add(alias, variable)

    return alias_map


def resolve_variable(raw_name: str, alias_map: dict[str, str]) -> str:
    if raw_name in alias_map:
        return alias_map[raw_name]
    canonical = canonical_key(raw_name)
    if canonical in alias_map:
        return alias_map[canonical]
    raise ValueError(f"Unknown variable or alias: {raw_name}")


# ---------------------------------------------------------------------------
# Normalização de assignments (evidence ou resposta do LLM)
# ---------------------------------------------------------------------------

def parse_assignment_arg(value: str | None) -> dict[str, str]:
    """Converte string 'VAR=STATE,VAR2=STATE2' ou JSON em dict."""
    if value is None or not value.strip():
        return {}

    stripped = value.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("Evidence JSON must be an object.")
        return {str(key): str(raw_value) for key, raw_value in parsed.items()}

    assignment: dict[str, str] = {}
    for chunk in stripped.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Evidence item {item!r} must use KEY=VALUE syntax."
            )
        key, raw_state = item.split("=", 1)
        assignment[key.strip()] = raw_state.strip()
    return assignment


def normalize_assignment(
    raw_assignment: dict[str, str],
    bn: BayesianNetwork,
    alias_map: dict[str, str],
) -> dict[str, str]:
    """
    Normaliza um assignment: resolve aliases e canonicaliza estados
    (case-insensitive) contra o domínio real da rede.
    """
    normalized: dict[str, str] = {}
    for raw_variable, raw_state in raw_assignment.items():
        variable = resolve_variable(raw_variable, alias_map)

        domain = bn.variables[variable].states
        domain_upper = {s.upper(): s for s in domain}
        canonical = domain_upper.get(raw_state.strip().upper())

        if canonical is None:
            allowed = ", ".join(domain)
            raise ValueError(
                f"Illegal state {raw_state!r} for {variable}. Allowed states: {allowed}."
            )

        existing = normalized.get(variable)
        if existing is not None and existing != canonical:
            raise ValueError(
                f"Conflicting assignments for {variable}: {existing} and {canonical}."
            )
        normalized[variable] = canonical
    return normalized
