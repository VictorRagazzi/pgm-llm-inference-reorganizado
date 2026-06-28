import random

from pgm_llm_inference.models import Factor

def min_degree_ordering(factors: list[Factor], nuisance_vars: set[str]) -> list[str]:
    """
    Compute elimination ordering using min-degree heuristic.

    The min-degree heuristic is a greedy algorithm that always eliminates
    the variable that appears in the fewest factors. This tends to minimize
    the size of intermediate factors during Variable Elimination.

    Algorithm:
    1. Count how many factors each nuisance variable appears in
    2. Pick the variable with minimum count
    3. Remove factors containing that variable
    4. Repeat until all nuisance variables are ordered

    Args:
        factors: List of factors in the network
        nuisance_vars: Set of variable names to eliminate (not query or evidence)

    Returns:
        List of variable names in elimination order

    Example:
        >>> # Network with factors over (A,B), (B,C), (C,D)
        >>> # To compute P(A), eliminate B, C, D
        >>> ordering = min_degree_ordering(factors, {'B', 'C', 'D'})
        >>> # Might return ['D', 'C', 'B'] or similar
    """
    ordering = []
    remaining_factors = factors.copy()
    remaining_vars = nuisance_vars.copy()

    while remaining_vars:
        # Count appearances of each variable in remaining factors
        var_counts = {
            var: sum(1 for f in remaining_factors if any(v.name == var for v in f.scope))
            for var in remaining_vars
        }

        # Pick variable with minimum count
        # If tie, pick alphabetically first for determinism
        next_var = min(var_counts.keys(), key=lambda v: (var_counts[v], v))
        ordering.append(next_var)
        remaining_vars.remove(next_var)

        # Simulate elimination: remove factors containing next_var
        remaining_factors = [
            f for f in remaining_factors if not any(v.name == next_var for v in f.scope)
        ]

    return ordering

def max_influence_ordering(
    factors: list[Factor],
    nuisance_vars: set[str]
) -> list[str]:
    """
    Compute an inference ordering optimized for explanatory construction
    (LLM-based MPE), not for variable elimination.

    Heuristic:
    - Variables that appear in MORE factors are inferred FIRST.
    - This prioritizes structurally influential / upstream variables.
    - Factors are NOT removed after selection, since influence should persist.

    Args:
        factors: List of factors in the network
        nuisance_vars: Set of variable names to infer (not query or evidence)

    Returns:
        List of variable names in inference order
    """
    ordering = []
    remaining_vars = nuisance_vars.copy()

    # Precompute factor participation counts
    var_counts = {
        var: sum(
            1 for f in factors
            if any(v.name == var for v in f.scope)
        )
        for var in remaining_vars
    }

    while remaining_vars:
        # Pick variable with MAX participation
        # Tie-break alphabetically for determinism
        next_var = max(
            remaining_vars,
            key=lambda v: (var_counts.get(v, 0), -ord(v[0]))
        )

        ordering.append(next_var)
        remaining_vars.remove(next_var)

    return ordering

def topological_ordering(factors: list[Factor], nuisance_vars: set[str]) -> list[str]:
    """
    Compute elimination ordering using topological sort (causal order).
    
    Parents always come before children, following the causal flow of the
    Bayesian Network. The first element of each factor's scope is the child,
    the remaining are parents.
    
    Args:
        factors: List of factors in the network
        nuisance_vars: Set of variable names to eliminate
    
    Returns:
        List of variable names in topological order
    """
    # Extrai pais de cada variável a partir dos fatores
    parents: dict[str, set[str]] = {var: set() for var in nuisance_vars}
    for factor in factors:
        child = factor.scope[0].name
        if child in nuisance_vars:
            for parent in factor.scope[1:]:
                if parent.name in nuisance_vars:
                    parents[child].add(parent.name)

    ordering = []
    remaining = set(nuisance_vars)

    while remaining:
        ready = {v for v in remaining if not (parents[v] & remaining)}

        if not ready:
            # Ciclo detectado — adiciona o restante em ordem alfabética
            ordering.extend(sorted(remaining))
            break

        next_var = min(ready)  # determinismo
        ordering.append(next_var)
        remaining.remove(next_var)

    return ordering

def reverse_topological_ordering(factors: list[Factor], nuisance_vars: set[str]) -> list[str]:
    """
    Reverse topological order — children before parents.
    
    Serves as a negative baseline: if topological ordering helps,
    this should hurt, validating the causal hypothesis.
    """
    return topological_ordering(factors, nuisance_vars)[::-1]


def central_ordering(factors: list[Factor], nuisance_vars: set[str]) -> list[str]:
    """
    Order by factor centrality — most connected variables first.
    
    Resolves hub nodes first, maximizing information propagation
    for subsequent inferences.
    """
    ordering = []
    remaining = set(nuisance_vars)
    remaining_factors = list(factors)

    while remaining:
        degree = {
            v: sum(1 for f in remaining_factors if any(s.name == v for s in f.scope))
            for v in remaining
        }

        # Maior grau primeiro, alfabético para determinismo
        next_var = max(remaining, key=lambda v: (degree[v], v))
        ordering.append(next_var)
        remaining.remove(next_var)

        remaining_factors = [
            f for f in remaining_factors
            if not any(s.name == next_var for s in f.scope)
        ]

    return ordering


def random_ordering(factors: list[Factor], nuisance_vars: set[str]) -> list[str]:
    """
    Random ordering with fixed seed — pure statistical baseline.
    
    Measures how much any structured heuristic helps over chance.
    """
    ordering = sorted(nuisance_vars)  # determinismo antes do shuffle
    random.shuffle(ordering)
    return ordering


DEGREE_ORDERING_MAP = {
    "min_degree": min_degree_ordering,
    "max_degree": max_influence_ordering,
    "topological": topological_ordering,
    "reverse_topological": reverse_topological_ordering,
    "central": central_ordering,
    "random": random_ordering
}

