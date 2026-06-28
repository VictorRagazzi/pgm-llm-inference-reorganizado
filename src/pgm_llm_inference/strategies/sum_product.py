from typing import Any
from .base import EliminationStrategy
from ..models import Factor, Variable, BayesianNetwork
from ..core.factor_ops import marginalize_factor


class SumProductStrategy(EliminationStrategy):
    def eliminate(
        self,
        factor: Factor,
        variable: Variable,
        network: BayesianNetwork,
        context: dict[str, Any],
    ) -> tuple[Factor, dict[str, Any]]:
        return marginalize_factor(factor, variable), {}