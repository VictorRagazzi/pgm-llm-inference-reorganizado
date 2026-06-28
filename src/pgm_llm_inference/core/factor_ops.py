"""
NumPy operations for factor manipulation.

This module contains pure functions for factor operations:
- Multiplication (using einsum)
- Marginalization (sum-product)
- Maximization (max-product with argmax tracking)
- Evidence reduction
- Normalization

These operations are the mathematical core of the Variable Elimination algorithm.
Correctness of these operations is CRITICAL for correct inference.
"""

import numpy as np
from numpy.typing import NDArray

from ..models import Factor, Variable


def multiply_factors(factor1: Factor, factor2: Factor) -> Factor:
    """
    Multiply two factors using Einstein summation for efficiency.

    This computes the pointwise product of two factors over the union of their scopes.
    Uses np.einsum for efficient multi-dimensional array multiplication.

    Mathematical operation:
        φ₁(X₁) × φ₂(X₂) = φ₃(X₁ ∪ X₂)
        where φ₃(x₁, x₂) = φ₁(x₁) × φ₂(x₂)

    Args:
        factor1: First factor
        factor2: Second factor

    Returns:
        New factor over the union of scopes with pointwise product values

    Example:
        >>> # Factor(A,B) × Factor(B,C) → Factor(A,B,C)
        >>> # If A,B,C are binary:
        >>> # result[i,j,k] = factor1[i,j] * factor2[j,k]
    """
    # Build output scope (union, preserving order from factor1 then factor2)
    result_scope = factor1.scope.copy()
    for var in factor2.scope:
        # Check if variable already in scope by name
        if not any(v.name == var.name for v in result_scope):
            result_scope.append(var)

    # Generate einsum subscript string
    # Use letters a-z for variable indices (supports up to 26 variables)
    var_to_char = {var.name: chr(97 + i) for i, var in enumerate(result_scope)}

    factor1_subscript = "".join(var_to_char[v.name] for v in factor1.scope)
    factor2_subscript = "".join(var_to_char[v.name] for v in factor2.scope)
    result_subscript = "".join(var_to_char[v.name] for v in result_scope)

    einsum_str = f"{factor1_subscript},{factor2_subscript}->{result_subscript}"

    # Perform multiplication using einsum
    result_values = np.einsum(einsum_str, factor1.values, factor2.values)

    return Factor(scope=result_scope, values=result_values)



def marginalize_factor(factor: Factor, variable: Variable) -> Factor:
    """
    Sum out a variable from a factor (marginalization).

    This operation is used in Sum-Product Variable Elimination to eliminate
    nuisance variables by summing over all possible values.

    Mathematical operation:
        φ'(Y) = Σₓ φ(X, Y)
        where X is the variable to eliminate and Y are the remaining variables

    Args:
        factor: The factor to marginalize
        variable: The variable to sum out

    Returns:
        New factor with the variable removed from scope

    Example:
        >>> # Factor(A,B) marginalize B → Factor(A)
        >>> # result[i] = sum over j of factor[i,j]
    """
    # If variable not in scope, return unchanged
    if not any(v.name == variable.name for v in factor.scope):
        return factor

    # Find axis corresponding to variable
    axis = None
    for i, v in enumerate(factor.scope):
        if v.name == variable.name:
            axis = i
            break

    if axis is None:
        return factor

    # Sum along that axis
    result_values = np.sum(factor.values, axis=axis)

    # Remove variable from scope
    result_scope = [v for v in factor.scope if v.name != variable.name]

    # Handle edge case: if scope becomes empty, return scalar factor
    if len(result_scope) == 0:
        # Create a dummy variable for scalar factors (needs at least 2 domain values)
        scalar_var = Variable(name="_scalar", states=("value", "_unused"))
        result_scope = [scalar_var]
        # Put the scalar value in first position, 0 in second
        result_values = np.array([result_values, 0.0])

    return Factor(scope=result_scope, values=result_values)



def maximize_factor(factor: Factor, variable: Variable) -> tuple[Factor, dict[tuple, str]]:
    """
    Maximize over a variable, returning reduced factor and argmax mapping.

    This operation is used in Max-Product Variable Elimination for MAP inference.
    It finds the maximum value for each configuration of remaining variables and
    tracks which value of the eliminated variable achieved that maximum.

    Mathematical operation:
        φ'(Y) = maxₓ φ(X, Y)
        argmax(Y) = argmaxₓ φ(X, Y)

    Args:
        factor: The factor to maximize
        variable: The variable to maximize over

    Returns:
        Tuple of:
        - New factor with maximum values (variable removed from scope)
        - Dictionary mapping remaining variable configurations to optimal values
          Keys are tuples of indices, values are domain values

    Example:
        >>> # Factor(A,B) maximize B → (Factor(A), {(0,): 'b1', (1,): 'b2'})
        >>> # For each value of A, store which value of B was optimal
    """
    # If variable not in scope, return unchanged with empty argmax
    if not any(v.name == variable.name for v in factor.scope):
        return factor, {}

    # Find axis corresponding to variable
    axis = None
    for i, v in enumerate(factor.scope):
        if v.name == variable.name:
            axis = i
            break

    if axis is None:
        return factor, {}

    # Get maximum values
    max_values = np.max(factor.values, axis=axis)

    # Get argmax indices
    argmax_indices = np.argmax(factor.values, axis=axis)

    # Build argmax mapping: (other_var_indices) -> optimal_value
    result_scope = [v for v in factor.scope if v.name != variable.name]
    argmax_map = {}

    # Iterate over all configurations of remaining variables
    for idx in np.ndindex(argmax_indices.shape):
        optimal_idx = argmax_indices[idx]
        optimal_value = variable.domain[optimal_idx]
        argmax_map[idx] = optimal_value

    # Handle edge case: if scope becomes empty, return scalar factor
    if len(result_scope) == 0:
        scalar_var = Variable(name="_scalar", states=("value", "_unused"))
        result_scope = [scalar_var]
        max_values = np.array([max_values, 0.0])
        # Argmax map for scalar: just store the optimal value
        argmax_map = {(0,): variable.domain[argmax_indices.item()]}

    return Factor(scope=result_scope, values=max_values), argmax_map



def reduce_factor(factor: Factor, evidence: dict[str, str]) -> Factor:
    """
    Reduce a factor by fixing evidence variables to observed values.

    This operation conditions the factor on observed evidence by selecting
    only the rows/columns corresponding to the observed values and removing
    the evidence variables from the scope.

    Args:
        factor: The factor to reduce
        evidence: Dictionary mapping variable names to observed values

    Returns:
        New factor with evidence variables fixed and removed from scope

    Example:
        >>> # Factor(A,B) with evidence B='b1' → Factor(A)
        >>> # Select only the column where B='b1'
    """
    if not evidence:
        return factor

    result_values = factor.values
    result_scope = list(factor.scope)

    # Process each evidence variable
    for var_name, observed_value in evidence.items():
        # Find if this variable is in the factor scope
        var_idx = None
        var_obj = None
        for i, v in enumerate(result_scope):
            if v.name == var_name:
                var_idx = i
                var_obj = v
                break

        if var_idx is None:
            # Variable not in this factor's scope, skip
            continue

        # Find the index of the observed value in the domain
        try:
            value_idx = var_obj.domain.index(observed_value)
        except ValueError:
            raise ValueError(
                f"Evidence value '{observed_value}' not in domain {var_obj.domain} "
                f"for variable '{var_name}'"
            )

        # Select the slice corresponding to the observed value
        # Use advanced indexing to select along the appropriate axis
        indices = [slice(None)] * result_values.ndim
        indices[var_idx] = value_idx
        result_values = result_values[tuple(indices)]

        # Remove the variable from scope
        result_scope.pop(var_idx)

    # Handle edge case: if scope becomes empty, return scalar factor
    if len(result_scope) == 0:
        scalar_var = Variable(name="_scalar", states=("value", "_unused"))
        result_scope = [scalar_var]
        result_values = np.array([result_values.item(), 0.0])

    return Factor(scope=result_scope, values=result_values)


def normalize_factor(factor: Factor) -> Factor:
    """
    Normalize a factor so that its values sum to 1.0.

    This is used to convert unnormalized factors into proper probability
    distributions after inference.

    Args:
        factor: The factor to normalize

    Returns:
        New factor with values scaled to sum to 1.0

    Raises:
        ValueError: If factor values sum to zero (cannot normalize)
    """
    total = np.sum(factor.values)

    if total == 0:
        raise ValueError("Cannot normalize factor: values sum to zero")

    normalized_values = factor.values / total

    return Factor(scope=factor.scope, values=normalized_values)


def get_factor_value(factor: Factor, assignment: dict[str, str]) -> float:
    """
    Retrieve the probability value for a specific variable assignment.

    Args:
        factor: The factor to query
        assignment: Dictionary mapping variable names to values

    Returns:
        The probability value for the given assignment

    Raises:
        ValueError: If assignment doesn't cover all scope variables or
                   contains invalid values

    Example:
        >>> # Factor(A,B) query with A='a1', B='b2'
        >>> value = get_factor_value(factor, {'A': 'a1', 'B': 'b2'})
    """
    # Build index tuple for array lookup
    indices = []
    for var in factor.scope:
        if var.name not in assignment:
            raise ValueError(
                f"Assignment missing variable '{var.name}'. "
                f"Required: {[v.name for v in factor.scope]}"
            )

        value = assignment[var.name]
        try:
            idx = var.domain.index(value)
        except ValueError:
            raise ValueError(
                f"Value '{value}' not in domain {var.domain} for variable '{var.name}'"
            )

        indices.append(idx)

    return float(factor.values[tuple(indices)])
