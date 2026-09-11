from pydantic import BaseModel, Field, field_validator


class Variable(BaseModel):
    """
    Representa uma variável aleatória discreta na rede Bayesiana.

    O domínio discreto é representado pelo campo canônico
    `states: tuple[str, ...]`.
    """

    name: str = Field(..., min_length=1)
    states: tuple[str, ...] = Field(..., min_length=2)

    @field_validator("states", mode="before")
    @classmethod
    def coerce_to_tuple(cls, v):
        if isinstance(v, (list, tuple)):
            v = tuple(v)
        if len(v) != len(set(v)):
            raise ValueError("States must be unique.")
        return v

    @property
    def cardinality(self) -> int:
        return len(self.states)

    def __hash__(self) -> int:
        return hash((self.name, self.states))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and (
            self.name == other.name and self.states == other.states
        )

    def __repr__(self) -> str:
        return f"Variable(name='{self.name}', states={list(self.states)})"
