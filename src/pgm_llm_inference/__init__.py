"""
PGM-LLM Inference Library

Biblioteca modular para inferência em Redes Bayesianas com estratégias plugáveis.
Suporta inferência numérica clássica (Sum-Product, Max-Product) e pipeline
LLM-MPE (Most Probable Explanation via Semantic Bucket Elimination).
"""

__version__ = "0.1.0"

from .models import Variable, Factor, BayesianNetwork
from .core.config import InferenceConfig
from .strategies import (
    EliminationStrategy,
    SumProductStrategy,
    MaxProductStrategy,
)
from .inference import InferenceEngine

__all__ = [
    "Variable",
    "Factor",
    "BayesianNetwork",
    "InferenceConfig",
    "EliminationStrategy",
    "SumProductStrategy",
    "MaxProductStrategy",
    "InferenceEngine",
]
