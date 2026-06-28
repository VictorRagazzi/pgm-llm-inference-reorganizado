# Design Document: PGM-LLM Inference Library

## Overview

This library implements Variable Elimination (VE) for Bayesian Networks with a novel architecture that supports pluggable elimination strategies. The core innovation is treating VE as a meta-algorithm where the elimination operation can be swapped between traditional numerical methods (Sum-Product, Max-Product) and a novel LLM-driven semantic approach.

The architecture follows the Strategy Pattern, allowing researchers to compare:
- **Sum-Product VE**: Standard posterior inference via marginalization
- **Max-Product VE**: MAP inference via numerical optimization with argmax tracking
- **LLM Semantic VE**: MAP inference via language model reasoning over variable semantics

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│  (Define Network, Set Evidence, Run Inference)          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 BayesianNetwork                          │
│  - Variables (name, cardinality, domain)                │
│  - Factors (CPTs)                                       │
│  - Network validation                                    │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              InferenceEngine                             │
│  - Variable Elimination orchestration                    │
│  - Elimination ordering                                  │
│  - Factor multiplication                                 │
│  - Strategy delegation                                   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┬──────────────┐
        │                         │              │
┌───────▼────────┐   ┌───────────▼──────┐  ┌───▼──────────────┐
│ SumProduct     │   │  MaxProduct      │  │ LLMSemantic      │
│ Strategy       │   │  Strategy        │  │ Strategy         │
│                │   │                  │  │                  │
│ - Summation    │   │ - Maximization   │  │ - LLM Query      │
│                │   │ - Argmax track   │  │ - Prompt build   │
└────────────────┘   └──────────────────┘  └──────────────────┘
```

### Core Components

1. **Variable**: Represents a discrete random variable with name, cardinality, and domain values
2. **Factor**: Encapsulates a multi-dimensional probability table with scope and values
3. **BayesianNetwork**: Container for variables and factors with validation
4. **InferenceEngine**: Implements the VE algorithm skeleton
5. **EliminationStrategy** (Abstract): Interface for elimination operations
6. **Concrete Strategies**: SumProductStrategy, MaxProductStrategy, LLMSemanticStrategy

## Components and Interfaces

### 1. Variable Model (Pydantic)

```python
from pydantic import BaseModel, Field, field_validator

class Variable(BaseModel):
    """
    Represents a discrete random variable in a Bayesian Network.
    
    Attributes:
        name: Unique identifier for the variable
        domain: List of possible values (e.g., ['true', 'false'])
        cardinality: Number of possible values (computed from domain)
    """
    name: str = Field(..., min_length=1)
    domain: list[str] = Field(..., min_length=2)
    
    @property
    def cardinality(self) -> int:
        return len(self.domain)
    
    @field_validator('domain')
    @classmethod
    def validate_unique_domain(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("Domain values must be unique")
        return v
```

**Responsibilities:**
- Store variable metadata with automatic validation
- Compute cardinality from domain
- Provide immutable, validated data structure

### 2. Factor Class

```python
import numpy as np
from pydantic import BaseModel, Field, field_validator
from numpy.typing import NDArray

class Factor(BaseModel):
    """
    Represents a multi-dimensional probability table over a set of variables.
    Uses NumPy arrays for efficient computation with Pydantic validation.
    
    Attributes:
        scope: List of Variable objects this factor is defined over
        values: Probability values as NumPy array
    """
    scope: list[Variable]
    values: NDArray[np.float64] = Field(...)
    
    model_config = {"arbitrary_types_allowed": True}
    
    @field_validator('values')
    @classmethod
    def validate_values_shape(cls, v: NDArray, info) -> NDArray:
        if 'scope' in info.data:
            expected_shape = tuple(var.cardinality for var in info.data['scope'])
            if v.shape != expected_shape:
                raise ValueError(f"Values shape {v.shape} doesn't match scope shape {expected_shape}")
        return v
    
    @field_validator('values')
    @classmethod
    def validate_probabilities(cls, v: NDArray) -> NDArray:
        if np.any(v < 0) or np.any(v > 1):
            raise ValueError("All probability values must be in [0, 1]")
        return v
```

**Key Methods:**
- `multiply(other: Factor) -> Factor`: Pointwise product using `np.einsum` for efficiency
- `reduce(evidence: dict[str, str]) -> Factor`: Fix evidence variables to observed values
- `marginalize(variable: Variable) -> Factor`: Sum out a variable using `np.sum`
- `maximize(variable: Variable) -> tuple[Factor, dict]`: Max out variable using `np.max` and `np.argmax`
- `get_value(assignment: dict[str, str]) -> float`: Retrieve probability for specific assignment
- `normalize() -> Factor`: Scale values to sum to 1.0

**Design Decision - NumPy with Best Practices:**
The Factor class uses NumPy arrays with optimized operations:
- **`np.einsum`** for flexible, efficient factor multiplication
- **Advanced indexing** for evidence reduction
- **Vectorized operations** for marginalization and maximization
- **Memory-efficient views** where possible (avoid unnecessary copies)
- **Numerical stability** using log-space computation for very small probabilities (optional)

### 3. BayesianNetwork Model (Pydantic)

```python
from pydantic import BaseModel, Field, model_validator

class BayesianNetwork(BaseModel):
    """
    Container for variables and factors representing a Bayesian Network.
    Provides validation and consistency checking.
    
    Attributes:
        variables: Dict mapping variable names to Variable objects
        factors: List of Factor objects (CPTs)
    """
    variables: dict[str, Variable] = Field(default_factory=dict)
    factors: list[Factor] = Field(default_factory=list)
    
    @model_validator(mode='after')
    def validate_factor_scopes(self) -> 'BayesianNetwork':
        """Ensure all factor scope variables are defined in the network."""
        for factor in self.factors:
            for var in factor.scope:
                if var.name not in self.variables:
                    raise ValueError(f"Factor references undefined variable: {var.name}")
        return self
    
    @model_validator(mode='after')
    def validate_cardinality_consistency(self) -> 'BayesianNetwork':
        """Ensure same variable has consistent cardinality across factors."""
        for factor in self.factors:
            for var in factor.scope:
                network_var = self.variables[var.name]
                if var.cardinality != network_var.cardinality:
                    raise ValueError(
                        f"Cardinality mismatch for {var.name}: "
                        f"network has {network_var.cardinality}, "
                        f"factor has {var.cardinality}"
                    )
        return self
```

**Key Methods:**
- `add_variable(variable: Variable) -> None`: Register a variable
- `add_factor(factor: Factor) -> None`: Add a CPT with automatic validation
- `get_variable(name: str) -> Variable`: Retrieve variable by name
- `get_factors_with_variable(var_name: str) -> list[Factor]`: Get all factors containing a variable

### 4. EliminationStrategy (Abstract Base Class)

```python
class EliminationStrategy(ABC):
    """
    Abstract interface for variable elimination operations.
    """
    
    @abstractmethod
    def eliminate(self, factor: Factor, variable: Variable, 
                  context: Dict) -> Union[Factor, Tuple[Factor, Dict]]:
        """
        Eliminate a variable from a factor.
        
        Args:
            factor: The factor to eliminate from
            variable: The variable to eliminate
            context: Additional context (evidence, network, etc.)
            
        Returns:
            For Sum/Max: New factor with variable eliminated
            For Max: (new_factor, argmax_assignments)
        """
        pass
```

### 5. SumProductStrategy

```python
class SumProductStrategy(EliminationStrategy):
    """
    Elimination via summation (marginalization).
    Used for computing posterior marginals.
    """
```

**Implementation:**
- Calls `factor.marginalize(variable)`
- Returns reduced factor
- No argmax tracking needed

### 6. MaxProductStrategy

```python
class MaxProductStrategy(EliminationStrategy):
    """
    Elimination via maximization.
    Used for MAP inference with numerical optimization.
    """
```

**Implementation:**
- Calls `factor.maximize(variable)`
- Returns (reduced_factor, argmax_dict)
- Stores argmax values for backtracking
- After all eliminations, reconstructs full MAP assignment

**Critical Design Note:**
The MaxProduct strategy must track argmax values at each elimination step. The argmax for variable X may depend on the values of remaining variables Y, Z. We store: `argmax[X] = {(y_val, z_val): x_val, ...}`. During backtracking, we use the MAP values of Y and Z to look up the optimal X value.

### 7. LLMSemanticStrategy

```python
class LLMSemanticStrategy(EliminationStrategy):
    """
    Elimination via LLM semantic reasoning.
    Queries an LLM to predict the most plausible value.
    """
```

**Implementation:**
- Constructs a textual prompt with:
  - Variable name and domain
  - Current evidence assignments
  - Names of related variables (from factor scope)
- Calls injected `llm_query_function(prompt) -> value`
- Validates returned value is in domain
- Reduces factor by fixing variable to LLM's prediction
- Returns reduced factor

**Prompt Template Example:**
```
Given the following information about a Bayesian Network:

Variable to predict: Sprinkler
Possible values: ['on', 'off']

Current evidence:
- Rain: 'yes'

Related variables in this context: Rain, Sprinkler

Based on semantic reasoning, what is the most plausible value for Sprinkler?
Respond with only the value.
```

### 8. InferenceEngine Class

```python
from pydantic import BaseModel, Field

class InferenceEngine(BaseModel):
    """
    Orchestrates Variable Elimination algorithm.
    Uses dependency injection for strategy pattern.
    
    Attributes:
        network: BayesianNetwork instance
        strategy: EliminationStrategy instance
        config: Library-wide configuration settings
    """
    network: BayesianNetwork
    strategy: EliminationStrategy
    config: 'InferenceConfig' = Field(default_factory=lambda: InferenceConfig())
    
    model_config = {"arbitrary_types_allowed": True}
```

**Key Methods:**
- `query(query_vars: list[str], evidence: dict[str, str], elimination_order: list[str] | None = None)`: Main inference entry point
- `_get_elimination_order(query_vars: set[str], evidence: set[str]) -> list[str]`: Compute heuristic ordering
- `_multiply_factors(factors: list[Factor]) -> Factor`: Pointwise product using efficient einsum
- `_eliminate_variable(factors: list[Factor], variable: Variable) -> Factor`: Multiply and eliminate

**Algorithm Flow:**

```
1. Input: query_vars, evidence, strategy
2. Apply evidence to all factors (reduce)
3. Identify nuisance variables (not query, not evidence)
4. Determine elimination ordering
5. For each variable in ordering:
   a. Collect all factors mentioning this variable
   b. Multiply these factors together
   c. Call strategy.eliminate(product_factor, variable, context)
   d. Add resulting factor back to factor list
6. Multiply remaining factors (over query variables)
7. Post-process based on strategy:
   - SumProduct: Normalize
   - MaxProduct: Backtrack argmax values
   - LLMSemantic: Return LLM-predicted assignment
8. Return result
```

## Configuration Management

### InferenceConfig (Pydantic Settings)

```python
from pydantic_settings import BaseSettings

class InferenceConfig(BaseSettings):
    """
    Library-wide configuration using Pydantic Settings.
    Can be configured via environment variables or .env file.
    """
    # Logging
    verbose: bool = False
    log_level: str = "INFO"
    
    # Numerical stability
    epsilon: float = 1e-10  # Small value to avoid division by zero
    use_log_space: bool = False  # Use log probabilities for numerical stability
    
    # Elimination ordering
    default_ordering_heuristic: str = "min_degree"  # or "min_fill"
    
    # LLM settings
    llm_timeout: float = 30.0  # seconds
    llm_max_retries: int = 3
    
    model_config = {
        "env_prefix": "PGM_",  # Environment variables like PGM_VERBOSE=true
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }
```

**Usage:**
```python
# From environment or .env file
config = InferenceConfig()

# Or explicit configuration
config = InferenceConfig(verbose=True, use_log_space=True)

# Pass to inference engine
engine = InferenceEngine(network=bn, strategy=strategy, config=config)
```

## Data Models

### Factor Storage: NumPy Array Representation

**Structure:**
```python
class Factor:
    scope: List[Variable]  # Ordered list of variables
    values: np.ndarray     # Multi-dimensional array
    
    # Example: Factor over (Rain, Sprinkler)
    # scope = [rain_var, sprinkler_var]
    # values = np.array([[0.8, 0.2],   # Rain=F: Sprinkler=F, Sprinkler=T
    #                    [0.1, 0.9]])  # Rain=T: Sprinkler=F, Sprinkler=T
```

**Indexing Convention:**
- Variables in scope are ordered
- Array dimensions correspond to scope order
- Index i for variable V corresponds to V.domain[i]

**Operations:**
- **Multiplication**: Use `np.einsum` or explicit broadcasting
- **Marginalization**: Use `np.sum(axis=...)`
- **Maximization**: Use `np.max(axis=...)` and `np.argmax(axis=...)`

### Assignment Representation

**Type-Safe Dictionary:**
```python
# Type alias for clarity
Assignment = dict[str, str]

# Example
assignment: Assignment = {
    'Rain': 'yes',
    'Sprinkler': 'on',
    'GrassWet': 'yes'
}
```

Used for:
- Evidence specification
- MAP results
- LLM context

**Note:** Using Python 3.12+ native type hints (dict, list, tuple) instead of typing.Dict, typing.List, etc.

### Argmax Tracking for MaxProduct

**Structure:**
```python
argmax_store = {
    'Sprinkler': {
        ('yes',): 'on',    # If Rain='yes', Sprinkler='on' is optimal
        ('no',): 'off'     # If Rain='no', Sprinkler='off' is optimal
    }
}
```

**Key:** Tuple of values for remaining variables in scope after elimination
**Value:** Optimal value for the eliminated variable

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property Reflection

Before defining the final properties, I've analyzed the prework to eliminate redundancy:

**Consolidations:**
- Properties 1.1 and 2.1 (variable/factor storage) can be combined into a general "data persistence" property
- Properties 4.2 and 4.5 (normalization) are redundant - both test that output sums to 1
- Properties 7.2 and scope reduction are covered by individual strategy tests
- Properties 10.1-10.5 (logging) can be combined into one comprehensive logging property

**Unique Value Properties:**
Each remaining property provides distinct validation:
- Factor operations (multiply, reduce, marginalize, maximize) each test different mathematical operations
- Strategy interface compliance vs. strategy correctness are separate concerns
- Error handling for different input types (evidence, LLM response, queries) test different validation paths

### Core Properties

**Property 1: Variable round-trip consistency**
*For any* variable with a name, cardinality, and domain, creating the variable and then retrieving it from the network should return the same name, cardinality, and domain values.
**Validates: Requirements 1.1**

**Property 2: Factor scope validation**
*For any* factor with a given scope, adding it to a network should succeed if and only if all variables in the scope are already defined in the network.
**Validates: Requirements 1.2**

**Property 3: Factor dimension consistency**
*For any* factor with scope variables, the factor's value array dimensions should match the product of the cardinalities of the variables in the scope.
**Validates: Requirements 1.3**

**Property 4: Network container completeness**
*For any* network with added variables and factors, querying the network structure should return all added variables and factors with their complete definitions.
**Validates: Requirements 1.4**

**Property 5: Factor multiplication commutativity**
*For any* two factors F1 and F2, multiplying F1 × F2 should produce the same result as F2 × F1 (same scope and equivalent values).
**Validates: Requirements 2.2**

**Property 6: Factor multiplication scope union**
*For any* two factors F1 and F2, the scope of F1 × F2 should equal the union of the scopes of F1 and F2.
**Validates: Requirements 2.2**

**Property 7: Factor value lookup correctness**
*For any* factor and any valid assignment to its scope variables, querying the factor with that assignment should return a probability value between 0 and 1.
**Validates: Requirements 2.3**

**Property 8: Evidence reduction scope**
*For any* factor and any evidence assignment, applying the evidence should produce a new factor whose scope excludes all evidence variables.
**Validates: Requirements 2.5**

**Property 9: Evidence reduction consistency**
*For any* factor F and evidence E, reducing F by E and then querying with assignment A should equal querying F with assignment (A ∪ E).
**Validates: Requirements 2.5**

**Property 10: Strategy injection persistence**
*For any* inference engine initialized with a strategy S, all subsequent elimination operations should use strategy S.
**Validates: Requirements 3.1**

**Property 11: Elimination ordering compliance**
*For any* provided elimination ordering, the inference engine should eliminate variables in exactly that order.
**Validates: Requirements 3.3**

**Property 12: Heuristic ordering validity**
*For any* query with nuisance variables, when no ordering is provided, the computed heuristic ordering should contain each nuisance variable exactly once.
**Validates: Requirements 3.4**

**Property 13: Sum-Product normalization invariant**
*For any* posterior query using Sum-Product strategy, the returned probability distribution should sum to 1.0 (within numerical tolerance).
**Validates: Requirements 4.2, 4.5**

**Property 14: Posterior scope correctness**
*For any* posterior query with query variables Q, the returned factor should have scope exactly equal to Q.
**Validates: Requirements 4.4**

**Property 15: MAP output completeness**
*For any* MAP query using Max-Product strategy, the result should contain both a maximum probability value and a complete assignment to all query variables.
**Validates: Requirements 5.2**

**Property 16: Argmax reconstruction consistency**
*For any* MAP query, the reconstructed assignment should be consistent with all stored argmax values at each elimination step.
**Validates: Requirements 5.4**

**Property 17: Local maximization efficiency**
*For any* MAP query with N variables, no intermediate factor during elimination should have a scope size equal to N (ensuring local rather than global maximization).
**Validates: Requirements 5.5**

**Property 18: LLM prompt excludes probabilities**
*For any* variable elimination using LLM Semantic Strategy, the constructed prompt should not contain any numerical probability values.
**Validates: Requirements 6.3**

**Property 19: LLM response validation**
*For any* LLM prediction for a variable V, if the predicted value is not in V's domain, the system should raise a validation error.
**Validates: Requirements 6.4**

**Property 20: Strategy interface compliance**
*For any* elimination strategy, calling the `eliminate` method should return a factor whose scope does not contain the eliminated variable.
**Validates: Requirements 7.2**

**Property 21: Strategy interchangeability**
*For any* inference query, switching between different strategies should not require any changes to the Variable Elimination algorithm code.
**Validates: Requirements 7.5**

**Property 22: Invalid evidence error handling**
*For any* evidence assignment containing a variable not in the network or a value not in that variable's domain, the system should raise a descriptive error before inference begins.
**Validates: Requirements 9.1**

**Property 23: LLM invalid response error handling**
*For any* LLM response that returns a value outside the variable's domain, the system should raise an error that includes both the invalid response and the valid domain.
**Validates: Requirements 9.3**

**Property 24: Inconsistent cardinality detection**
*For any* two factors that reference the same variable but with different cardinalities, the system should detect and report the inconsistency during network validation.
**Validates: Requirements 9.4**

**Property 25: Non-existent variable query error**
*For any* query requesting a variable that doesn't exist in the network, the system should raise a descriptive error.
**Validates: Requirements 9.5**

**Property 26: Verbose logging completeness**
*For any* inference query with verbose mode enabled, the system should log all elimination steps, factor operations, and strategy-specific actions (LLM prompts, argmax tracking).
**Validates: Requirements 10.1, 10.2, 10.3, 10.4**

## Error Handling

### Input Validation Errors

1. **Invalid Variable Reference**: Raised when a factor references undefined variables
   - Error message includes: factor scope, undefined variable names
   - Raised at: Factor addition to network

2. **Dimension Mismatch**: Raised when factor values don't match scope cardinalities
   - Error message includes: expected dimensions, actual dimensions
   - Raised at: Factor creation

3. **Invalid Evidence**: Raised when evidence contains invalid variables or values
   - Error message includes: invalid variable/value, valid options
   - Raised at: Query execution start

4. **Invalid Query Variable**: Raised when query requests non-existent variable
   - Error message includes: requested variable, available variables
   - Raised at: Query execution start

### Runtime Errors

5. **LLM Response Validation Error**: Raised when LLM returns invalid value
   - Error message includes: LLM response, variable domain, prompt sent
   - Raised at: LLM strategy elimination step

6. **Cardinality Inconsistency**: Raised when same variable has different cardinalities
   - Error message includes: variable name, conflicting cardinalities, factor sources
   - Raised at: Network validation

7. **Empty Factor Scope**: Handled gracefully by treating as scalar
   - No error raised, logged in verbose mode
   - Occurs at: Factor reduction or elimination

### Error Handling Strategy

- **Fail Fast**: Validate all inputs before starting inference
- **Descriptive Messages**: Include context about what went wrong and valid options
- **Preserve State**: Errors should not leave network in inconsistent state
- **Logging**: All errors logged with full context in verbose mode

## Testing Strategy

### Focused Testing Approach

Given the emphasis on correctness of core numerical operations, testing will focus on:

1. **Critical NumPy Operations**: Unit tests for factor multiplication, marginalization, and maximization
2. **Basic Functionality**: Simple tests to verify each component works
3. **Known Examples**: Rain-Sprinkler-GrassWet network with hand-calculated results

### Unit Test Coverage

**Priority 1: NumPy Operations (CRITICAL)**
These operations must be mathematically correct:

1. **Factor Multiplication**:
   - Test with known values and verify pointwise product
   - Test scope union is correct
   - Test with different scope orderings
   - Verify einsum subscript generation is correct

2. **Factor Marginalization**:
   - Test summing out variables produces correct marginals
   - Verify scope reduction
   - Test with different axis orderings

3. **Factor Maximization**:
   - Test max operation produces correct values
   - Verify argmax tracking is correct
   - Test backtracking reconstruction

4. **Evidence Reduction**:
   - Test slicing produces correct reduced factors
   - Verify scope updates correctly

**Priority 2: Basic Functionality**

1. **Variable and Network**:
   - Test variable creation and validation
   - Test network construction and validation
   - Test Pydantic validation catches errors

2. **Strategies**:
   - Test each strategy with simple 2-variable example
   - Test LLM prompt construction
   - Test strategy swapping

3. **End-to-End**:
   - Rain-Sprinkler-GrassWet with all three strategies
   - Verify Sum-Product produces correct posterior
   - Verify Max-Product produces correct MAP assignment

### Test Organization

```
tests/
├── test_factor_operations.py      # CRITICAL: NumPy operations
├── test_variable_network.py       # Basic Pydantic models
├── test_strategies.py             # Strategy implementations
└── test_integration.py            # End-to-end example
```

**Testing Framework**: `pytest` with minimal dependencies

## Implementation Notes

### NumPy Best Practices for Factor Operations

#### Factor Multiplication with `np.einsum`

**Challenge**: Efficiently multiply factors with different scopes while handling arbitrary variable orderings

**Solution**: Use `np.einsum` with dynamically generated subscripts
```python
def multiply(self, other: Factor) -> Factor:
    """
    Multiply two factors using Einstein summation for efficiency.
    
    Example: Factor(A,B) × Factor(B,C) → Factor(A,B,C)
    Einsum: "ab,bc->abc"
    """
    # Build output scope (union, preserving order)
    result_scope = self.scope.copy()
    for var in other.scope:
        if var not in result_scope:
            result_scope.append(var)
    
    # Generate einsum subscript string
    # Use letters a-z for variable indices
    var_to_char = {var.name: chr(97 + i) for i, var in enumerate(result_scope)}
    
    self_subscript = ''.join(var_to_char[v.name] for v in self.scope)
    other_subscript = ''.join(var_to_char[v.name] for v in other.scope)
    result_subscript = ''.join(var_to_char[v.name] for v in result_scope)
    
    einsum_str = f"{self_subscript},{other_subscript}->{result_subscript}"
    
    # Perform multiplication
    result_values = np.einsum(einsum_str, self.values, other.values)
    
    return Factor(scope=result_scope, values=result_values)
```

**Why einsum?**
- Handles arbitrary dimensional arrays
- Automatically broadcasts and aligns dimensions
- Optimized C implementation
- More efficient than manual reshaping and broadcasting

#### Factor Marginalization (Sum-Product)

**Challenge**: Sum out a variable efficiently

**Solution**: Use `np.sum` with correct axis
```python
def marginalize(self, variable: Variable) -> Factor:
    """
    Sum out a variable from the factor.
    
    Example: Factor(A,B,C) marginalize B → Factor(A,C)
    """
    if variable not in self.scope:
        return self  # Variable not in scope, return unchanged
    
    # Find axis corresponding to variable
    axis = self.scope.index(variable)
    
    # Sum along that axis
    result_values = np.sum(self.values, axis=axis)
    
    # Remove variable from scope
    result_scope = [v for v in self.scope if v != variable]
    
    return Factor(scope=result_scope, values=result_values)
```

#### Factor Maximization (Max-Product)

**Challenge**: Maximize over a variable and track argmax for reconstruction

**Solution**: Use `np.max` and `np.argmax` together
```python
def maximize(self, variable: Variable) -> tuple[Factor, dict[tuple, str]]:
    """
    Maximize over a variable, returning reduced factor and argmax mapping.
    
    Returns:
        - Reduced factor with variable eliminated
        - Dict mapping remaining variable assignments to optimal value
    """
    if variable not in self.scope:
        return self, {}
    
    axis = self.scope.index(variable)
    
    # Get maximum values
    max_values = np.max(self.values, axis=axis)
    
    # Get argmax indices
    argmax_indices = np.argmax(self.values, axis=axis)
    
    # Build argmax mapping: (other_var_values) -> optimal_value
    result_scope = [v for v in self.scope if v != variable]
    argmax_map = {}
    
    # Iterate over all configurations of remaining variables
    for idx in np.ndindex(argmax_indices.shape):
        optimal_idx = argmax_indices[idx]
        optimal_value = variable.domain[optimal_idx]
        argmax_map[idx] = optimal_value
    
    return Factor(scope=result_scope, values=max_values), argmax_map
```

### Elimination Ordering Heuristics

**Min-Degree Heuristic**:
- Choose variable that appears in fewest factors
- Minimizes size of intermediate factors
- Simple to implement, reasonably effective

**Implementation**:
```python
def min_degree_ordering(
    factors: list[Factor], 
    nuisance_vars: set[str]
) -> list[str]:
    """
    Compute elimination ordering using min-degree heuristic.
    Greedy algorithm: always eliminate variable in fewest factors.
    """
    ordering = []
    remaining_factors = factors.copy()
    remaining_vars = nuisance_vars.copy()
    
    while remaining_vars:
        # Count appearances of each variable
        var_counts = {
            var: sum(1 for f in remaining_factors 
                    if any(v.name == var for v in f.scope))
            for var in remaining_vars
        }
        
        # Pick variable with minimum count
        next_var = min(var_counts, key=var_counts.get)
        ordering.append(next_var)
        remaining_vars.remove(next_var)
        
        # Simulate elimination: remove factors with next_var
        remaining_factors = [
            f for f in remaining_factors 
            if not any(v.name == next_var for v in f.scope)
        ]
    
    return ordering
```

### LLM Interface Design

**Abstraction**: Use callable interface for flexibility
```python
class LLMSemanticStrategy(EliminationStrategy):
    def __init__(self, llm_query_fn: Callable[[str], str]):
        self.llm_query_fn = llm_query_fn
```

**Mock for Testing**:
```python
def mock_llm(prompt: str) -> str:
    # Simple heuristic: return first domain value
    # Or parse prompt and use rules
    if "Rain" in prompt and "yes" in prompt:
        return "on"  # Sprinkler likely on if raining
    return "off"
```

**Real LLM Integration** (future):
```python
def openai_llm(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0  # Deterministic
    )
    return response.choices[0].message.content.strip()
```

### Performance Considerations and Best Practices

**Optimized For**:
- **Correctness First**: All NumPy operations are mathematically verified
- **Modularity**: Easy to swap strategies and extend functionality
- **Reusability**: Clean interfaces using Pydantic models
- **Extensibility**: Strategy pattern allows adding new inference methods

**NumPy Optimizations Applied**:
- `np.einsum` for efficient multi-dimensional multiplication
- Vectorized operations (no Python loops over array elements)
- In-place operations where safe (e.g., normalization)
- Avoid unnecessary array copies (use views when possible)

**Not Optimized For** (acceptable trade-offs for research code):
- Very large networks (>50 variables)
- Real-time inference
- Production deployment at scale

**Future Optimizations** (if needed):
- Log-space computation for numerical stability with very small probabilities
- Sparse factor representation for large domains
- Caching of intermediate factors
- JIT compilation with Numba for hot paths

## Project Structure and Dependencies

### UV Project Management

The project will be managed using `uv` for fast, reliable dependency management:

```toml
# pyproject.toml
[project]
name = "pgm-llm-inference"
version = "0.1.0"
description = "Bayesian Network inference with pluggable strategies including LLM-driven MAP"
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.26.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",  # Fast linting and formatting
]
llm = [
    "openai>=1.0.0",  # Optional: for real LLM integration
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### Project Structure

```
pgm-llm-inference/
├── pyproject.toml
├── README.md
├── .env.example
├── src/
│   └── pgm_llm_inference/
│       ├── __init__.py
│       ├── models.py          # Variable, Factor, BayesianNetwork
│       ├── config.py          # InferenceConfig
│       ├── factor_ops.py      # NumPy operations (multiply, marginalize, etc.)
│       ├── strategies.py      # EliminationStrategy and implementations
│       ├── inference.py       # InferenceEngine
│       └── utils.py           # Helper functions
├── tests/
│   ├── test_factor_operations.py
│   ├── test_variable_network.py
│   ├── test_strategies.py
│   └── test_integration.py
└── examples/
    └── rain_sprinkler_grass.py
```

**Core Dependencies**:
- `numpy>=1.26.0`: Array operations and numerical computation (Python 3.12 compatible)
- `pydantic>=2.5.0`: Data validation and settings management
- `pydantic-settings>=2.1.0`: Environment-based configuration
- `python>=3.12`: Modern type hints (native dict, list, etc.)

**Development Dependencies**:
- `pytest>=7.4.0`: Test framework
- `pytest-cov>=4.1.0`: Coverage reporting
- `ruff>=0.1.0`: Fast linting and formatting (replaces black, flake8, isort)

**Optional Dependencies**:
- `openai>=1.0.0`: For real LLM integration (not required for core library)

## Modern Python Best Practices

### Type Hints (Python 3.12+)

**Use native types instead of typing module**:
```python
# ✅ Modern (Python 3.12+)
def process_factors(factors: list[Factor]) -> dict[str, float]:
    results: dict[str, float] = {}
    return results

# ❌ Old style (avoid)
from typing import List, Dict
def process_factors(factors: List[Factor]) -> Dict[str, float]:
    ...
```

**Use type aliases for clarity**:
```python
Assignment = dict[str, str]
VariableName = str
Probability = float

def query(evidence: Assignment) -> dict[VariableName, Probability]:
    ...
```

### Pydantic Best Practices

**Use validators for complex validation**:
```python
class Factor(BaseModel):
    scope: list[Variable]
    values: NDArray[np.float64]
    
    @field_validator('values')
    @classmethod
    def validate_probabilities(cls, v: NDArray) -> NDArray:
        if np.any(v < 0) or np.any(v > 1):
            raise ValueError("Probabilities must be in [0, 1]")
        return v
```

**Use model_validator for cross-field validation**:
```python
class BayesianNetwork(BaseModel):
    variables: dict[str, Variable]
    factors: list[Factor]
    
    @model_validator(mode='after')
    def validate_consistency(self) -> 'BayesianNetwork':
        # Check all factor variables are defined
        for factor in self.factors:
            for var in factor.scope:
                if var.name not in self.variables:
                    raise ValueError(f"Undefined variable: {var.name}")
        return self
```

**Use Pydantic Settings for configuration**:
```python
from pydantic_settings import BaseSettings

class InferenceConfig(BaseSettings):
    verbose: bool = False
    epsilon: float = 1e-10
    
    model_config = {
        "env_prefix": "PGM_",
        "env_file": ".env"
    }
```

### Code Organization

**Separate concerns into modules**:
- `models.py`: Pydantic models (Variable, Factor, BayesianNetwork)
- `config.py`: Configuration management
- `factor_ops.py`: Pure NumPy operations (no business logic)
- `strategies.py`: Strategy implementations
- `inference.py`: High-level inference orchestration

**Benefits**:
- Easy to test (pure functions in factor_ops.py)
- Easy to extend (add new strategies)
- Easy to maintain (clear separation of concerns)

## Future Extensions

1. **Loopy Belief Propagation**: Approximate inference for networks with cycles
2. **Sampling Methods**: MCMC, Gibbs sampling for large networks
3. **Learning**: Parameter learning from data (EM algorithm)
4. **Structure Learning**: Discover network structure from data
5. **Continuous Variables**: Extend beyond discrete domains
6. **GPU Acceleration**: Use PyTorch/JAX for large-scale inference
7. **LLM Prompt Optimization**: Experiment with different prompt templates
8. **Hybrid Inference**: Combine numerical and LLM strategies adaptively
