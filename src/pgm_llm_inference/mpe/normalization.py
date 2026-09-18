"""Canonical variable/state names and context keys shared by both phases."""

from __future__ import annotations

import re

from ..models import BayesianNetwork
from .types import VariableMetadata


def canonical_key(value: str) -> str:
    return re.sub(r"[\s_\-]+", "", value).lower()


def build_alias_map(bn: BayesianNetwork, metadata: dict[str, VariableMetadata]) -> dict[str, str]:
    alias_map: dict[str, str] = {}

    def add(alias: str, variable: str) -> None:
        candidates = {alias, canonical_key(alias)}
        for candidate in candidates:
            existing = alias_map.get(candidate)
            if existing is not None and existing != variable:
                raise ValueError(f"Alias {alias!r} maps to both {existing!r} and {variable!r}.")
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
            raise ValueError(f"Conflicting assignments for {variable}: {existing} and {canonical}.")
        normalized[variable] = canonical
    return normalized


def normalize_context(
    raw_context: dict[str, str],
    expected_scope: tuple[str, ...],
    bn: BayesianNetwork,
    alias_map: dict[str, str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    expected_set = set(expected_scope)

    for raw_variable, raw_state in raw_context.items():
        variable = resolve_variable(raw_variable, alias_map)

        if variable not in expected_set:
            raise ValueError(
                f"Unexpected context variable {raw_variable!r}; expected {expected_scope}."
            )

        domain = bn.variables[variable].states
        domain_upper = {state.upper(): state for state in domain}
        canonical = domain_upper.get(raw_state.strip().upper())

        if canonical is None:
            allowed = ", ".join(domain)
            raise ValueError(
                f"Illegal state {raw_state!r} for {variable}. Allowed states: {allowed}."
            )

        normalized[variable] = canonical

    missing = expected_set - set(normalized)
    if missing:
        raise ValueError(f"Context is missing variables: {sorted(missing)}.")

    return {variable: normalized[variable] for variable in expected_scope}


def context_key(context: dict[str, str], scope: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(context[variable] for variable in scope)
