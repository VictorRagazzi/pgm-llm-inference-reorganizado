from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from .variable import Variable
from .factor import Factor


class BayesianNetwork(BaseModel):
    """
    Rede Bayesiana: variáveis discretas + (opcionalmente) fatores e topologia.

    Dois modos de uso
    -----------------
    1. VE numérico (via loaders.py → convert_pgmpy_model):
       - `factors` preenchidos com CPTs numéricas.
       - `parents` derivado automaticamente de `factors` (scope[0]=filho, scope[1:]=pais).
       - `name` é opcional (default "").

    2. Pipeline MPE (via mpe/io.py → parse_bif):
       - `factors` vazio (sem CPTs numéricas).
       - `parents` fornecido explicitamente via `_parents`.
       - `name` preenchido com o nome da rede do arquivo .bif.

    A property `parents` resolve isso: retorna `_parents` se fornecido,
    caso contrário deriva dos `factors`.
    """

    name: str = ""
    variables: dict[str, Variable] = Field(default_factory=dict)
    factors: list[Factor] = Field(default_factory=list)
    # Topologia explícita — usada pelo pipeline MPE (parse_bif).
    # Quando None, parents é derivado de factors (modo VE numérico).
    _parents: dict[str, tuple[str, ...]] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def model_post_init(self, __context) -> None:
        # Captura o campo `parents` passado no construtor antes que Pydantic o descarte
        pass

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        return super().model_validate(obj, *args, **kwargs)

    def __init__(self, *, parents: dict[str, tuple[str, ...]] | None = None, **data):
        super().__init__(**data)
        if parents is not None:
            object.__setattr__(self, "_parents", parents)

    @model_validator(mode="after")
    def validate_factor_scopes(self) -> "BayesianNetwork":
        for i, factor in enumerate(self.factors):
            for var in factor.scope:
                if var.name not in self.variables:
                    raise ValueError(
                        f"Factor {i} references undefined variable: {var.name}"
                    )
        return self

    @model_validator(mode="after")
    def validate_cardinality_consistency(self) -> "BayesianNetwork":
        for factor in self.factors:
            for var in factor.scope:
                network_var = self.variables[var.name]
                if var.states != network_var.states:
                    raise ValueError(
                        f"States mismatch for variable '{var.name}': "
                        f"factor has {var.states}, network has {network_var.states}"
                    )
        return self

    # ------------------------------------------------------------------
    # Mutação (usada por convert_pgmpy_model)
    # ------------------------------------------------------------------

    def add_variable(self, variable: Variable) -> None:
        if variable.name in self.variables:
            raise ValueError(f"Variable '{variable.name}' already exists.")
        self.variables[variable.name] = variable

    def add_factor(self, factor: Factor) -> None:
        for var in factor.scope:
            if var.name not in self.variables:
                raise ValueError(f"Undefined variable '{var.name}' in factor.")
        self.factors.append(factor)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_variable(self, name: str) -> Variable:
        return self.variables[name]

    def get_factors_with_variable(self, var_name: str) -> list[Factor]:
        return [f for f in self.factors if any(v.name == var_name for v in f.scope)]

    # ------------------------------------------------------------------
    # Topologia
    # ------------------------------------------------------------------

    @property
    def parents(self) -> dict[str, tuple[str, ...]]:
        """
        Mapa de pais de cada variável.

        Prioridade:
        1. `_parents` fornecido explicitamente (parse_bif / pipeline MPE).
        2. Derivado dos `factors` (convention: scope[0]=filho, scope[1:]=pais).
        """
        if self._parents is not None:
            return self._parents
        result: dict[str, tuple[str, ...]] = {name: () for name in self.variables}
        for factor in self.factors:
            if len(factor.scope) > 1:
                child = factor.scope[0].name
                result[child] = tuple(v.name for v in factor.scope[1:])
        return result

    def children_map(self) -> dict[str, tuple[str, ...]]:
        """Mapa inverso: nome → tuple de filhos."""
        children: dict[str, list[str]] = {name: [] for name in self.variables}
        for child, pars in self.parents.items():
            for parent in pars:
                children[parent].append(child)
        return {name: tuple(sorted(values)) for name, values in children.items()}

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"BayesianNetwork("
            f"name={self.name!r}, "
            f"variables={list(self.variables.keys())}, "
            f"num_factors={len(self.factors)})"
        )
