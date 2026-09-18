from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

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

    2. Pipeline MPE (via io/loaders.py → parse_bif):
       - `factors` vazio (sem CPTs numéricas).
       - `explicit_parents` fornecido pelo parser.
       - `name` preenchido com o nome da rede do arquivo .bif.

    A property `parents` resolve isso: retorna `_parents` se fornecido,
    caso contrário deriva dos `factors`.
    """

    name: str = ""
    variables: dict[str, Variable] = Field(default_factory=dict)
    factors: list[Factor] = Field(default_factory=list)
    explicit_parents: dict[str, tuple[str, ...]] | None = Field(
        default=None,
        exclude=True,
    )

    model_config = {"arbitrary_types_allowed": True}

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

    @classmethod
    def from_structure(
        cls,
        *,
        variable_states: Mapping[str, Sequence[str]],
        children: Mapping[str, Iterable[str]],
        name: str = "",
    ) -> "BayesianNetwork":
        """Create a discrete Bayesian network without numeric CPTs.

        ``children`` uses the presentation-friendly ``parent -> children``
        representation. Variables that have no children may be omitted from
        that mapping, but every parent and child referenced by an edge must be
        declared in ``variable_states``.
        """
        if not variable_states:
            raise ValueError("A network must contain at least one variable.")

        variables: dict[str, Variable] = {}
        for variable_name, states in variable_states.items():
            if isinstance(states, str):
                raise ValueError(
                    f"States for '{variable_name}' must be a sequence, not a string."
                )
            variables[variable_name] = Variable(
                name=variable_name,
                states=tuple(states),
            )

        parent_lists: dict[str, list[str]] = {variable: [] for variable in variables}
        child_sets: dict[str, set[str]] = {variable: set() for variable in variables}

        for parent, raw_children in children.items():
            if parent not in variables:
                raise ValueError(f"Unknown parent variable: {parent!r}.")
            if isinstance(raw_children, str):
                raise ValueError(
                    f"Children of '{parent}' must be a sequence, not a string."
                )

            for child in raw_children:
                if child not in variables:
                    raise ValueError(
                        f"Edge {parent!r} -> {child!r} references an unknown child."
                    )
                if child == parent:
                    raise ValueError(f"Self edge is not allowed: {parent!r} -> {child!r}.")
                if child in child_sets[parent]:
                    continue
                child_sets[parent].add(child)
                parent_lists[child].append(parent)

        # Fail early with a useful error instead of finding a cycle only when
        # the semantic compilation has already started.
        in_degree = {variable: len(parents) for variable, parents in parent_lists.items()}
        ready = [variable for variable in variables if in_degree[variable] == 0]
        visited = 0
        while ready:
            current = ready.pop(0)
            visited += 1
            for child in child_sets[current]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    ready.append(child)

        if visited != len(variables):
            cyclic = sorted(variable for variable, degree in in_degree.items() if degree)
            raise ValueError(
                "The network structure must be a DAG; cycle detected involving: "
                + ", ".join(cyclic)
            )

        return cls(
            name=name,
            variables=variables,
            explicit_parents={
                variable: tuple(parents) for variable, parents in parent_lists.items()
            },
        )

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
        1. `explicit_parents` fornecido por parse_bif.
        2. Derivado dos `factors` (convention: scope[0]=filho, scope[1:]=pais).
        """
        if self.explicit_parents is not None:
            return self.explicit_parents
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
