import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from .variable import Variable


class Factor(BaseModel):
    """
    Represents a multi-dimensional probability table over a set of variables.
    """

    scope: list[Variable] = Field(...)
    values: NDArray[np.float64] = Field(...)

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("values", mode="before")
    @classmethod
    def convert_to_numpy(cls, v) -> NDArray[np.float64]:
        if not isinstance(v, np.ndarray):
            v = np.array(v, dtype=np.float64)
        return v.astype(np.float64)

    @field_validator("values")
    @classmethod
    def validate_probabilities(cls, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if np.any(v < 0):
            raise ValueError("All probability values must be >= 0")
        return v

    def model_post_init(self, __context) -> None:
        expected_shape = tuple(var.cardinality for var in self.scope)
        if self.values.shape != expected_shape:
            raise ValueError(
                f"Values shape {self.values.shape} doesn't match "
                f"scope shape {expected_shape}"
            )

    def __repr__(self) -> str:
        return f"Factor(scope={[v.name for v in self.scope]}, shape={self.values.shape})"