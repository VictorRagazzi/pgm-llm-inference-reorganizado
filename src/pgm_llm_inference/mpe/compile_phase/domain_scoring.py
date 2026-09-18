"""Stable single-token labels for comparable per-state LLM scores."""

from __future__ import annotations


MAX_TOP_LOGPROBS = 20
_STATE_CODES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def build_state_code_map(states: tuple[str, ...]) -> dict[str, str]:
    """Return short categorical labels without changing canonical state names."""
    codes = [
        _STATE_CODES[index] if index < len(_STATE_CODES) else f"S{index + 1}"
        for index in range(len(states))
    ]
    return dict(zip(codes, states))
