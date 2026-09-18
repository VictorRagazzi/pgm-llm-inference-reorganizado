"""Numeric elimination strategies and Max-Product backpointers."""

from abc import ABC, abstractmethod
from typing import Any

from ..models import BayesianNetwork, Factor, Variable
from ..core.factor_ops import marginalize_factor, maximize_factor


class EliminationStrategy(ABC):
    @abstractmethod
    def eliminate(
        self,
        factor: Factor,
        variable: Variable,
        context: dict[str, Any],
    ) -> tuple[Factor, dict[str, Any]]:
        pass

    def reset(self) -> None:
        """Optional hook called before a new inference query."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class SumProductStrategy(EliminationStrategy):
    def eliminate(
        self,
        factor: Factor,
        variable: Variable,
        network: BayesianNetwork,
        context: dict[str, Any],
    ) -> tuple[Factor, dict[str, Any]]:
        return marginalize_factor(factor, variable), {}


class MaxProductStrategy(EliminationStrategy):
    def __init__(self):
        self.argmax_store: dict[str, dict[tuple, str]] = {}
        self.argmax_scope: dict[str, list[Variable]] = {}

    def eliminate(
        self,
        factor: Factor,
        variable: Variable,
        network: BayesianNetwork,
        context: dict[str, Any],
    ) -> tuple[Factor, dict[str, Any]]:
        result, argmax_map = maximize_factor(factor, variable)

        if argmax_map:
            self.argmax_store[variable.name] = argmax_map
            self.argmax_scope[variable.name] = result.scope.copy()

        return result, {"argmax": argmax_map}

    def reconstruct_assignment(
        self,
        final_assignment: dict[str, str],
        elimination_order: list[str],
    ) -> dict[str, str]:
        result = final_assignment.copy()

        for var_name in reversed(elimination_order):
            if var_name not in self.argmax_store:
                continue

            argmax_map = self.argmax_store[var_name]
            scope = self.argmax_scope[var_name]

            if len(argmax_map) == 1:
                result[var_name] = next(iter(argmax_map.values()))
                continue

            key = tuple(
                scope[i].states.index(result[scope[i].name])
                for i in range(len(scope))
                if scope[i].name in result
            )

            if key in argmax_map:
                result[var_name] = argmax_map[key]

        return result

    def reset(self) -> None:
        self.argmax_store.clear()
        self.argmax_scope.clear()
