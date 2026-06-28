from abc import ABC, abstractmethod
from typing import Any
from ..models import Factor, Variable


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