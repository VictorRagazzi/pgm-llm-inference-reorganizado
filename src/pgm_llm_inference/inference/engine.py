from pydantic import BaseModel, Field
from typing import Any

from ..models import BayesianNetwork
from ..strategies import EliminationStrategy
from ..core.config import InferenceConfig
from .ve_algorithm import run_variable_elimination
from .postprocessing import postprocess_result


class InferenceEngine(BaseModel):
    network: BayesianNetwork
    strategy: EliminationStrategy
    config: InferenceConfig = Field(default_factory=InferenceConfig)

    model_config = {"arbitrary_types_allowed": True}

    def query(
        self,
        query_vars: list[str],
        evidence: dict[str, str] | None = None,
        elimination_order: list[str] | None = None,
    ) -> dict[str, Any]:
        if evidence is None:
            evidence = {}

        self._validate_query(query_vars, evidence)

        if self.config.verbose:
            print(f"[Inference] query={query_vars}, evidence={evidence}")

        final_factor, metadata = run_variable_elimination(
            network=self.network,
            strategy=self.strategy,
            query_vars=query_vars,
            evidence=evidence,
            elimination_order=elimination_order,
            config=self.config,
        )

        return postprocess_result(
            final_factor=final_factor,
            metadata=metadata,
            strategy=self.strategy,
            config=self.config,
        )

    def _validate_query(self, query_vars: list[str], evidence: dict[str, str]) -> None:
        for var_name in query_vars:
            if var_name not in self.network.variables:
                raise ValueError(
                    f"Query variable '{var_name}' not in network"
                    + f"\nNetwork nodes: {self.network.variables}"
                )


        for var_name, value in evidence.items():
            if var_name not in self.network.variables:
                raise ValueError(f"Evidence variable '{var_name}' not in network")
            if value not in self.network.variables[var_name].domain:
                raise ValueError(
                    f"Value '{value}' not in domain {self.network.variables[var_name].domain}"
                )