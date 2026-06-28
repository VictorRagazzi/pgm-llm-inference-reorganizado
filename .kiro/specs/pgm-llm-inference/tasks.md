# Implementation Plan

- [x] 1. Set up project structure with UV
  - Initialize UV project with pyproject.toml
  - Create src/pgm_llm_inference package structure
  - Set up basic directory layout (models, config, strategies, etc.)
  - Create .env.example for configuration
  - _Requirements: Project Structure_

- [ ] 2. Implement core data models with Pydantic
  - [x] 2.1 Create Variable model with validation
    - Implement Variable class with name and domain fields
    - Add property for cardinality computation
    - Add validator for unique domain values
    - _Requirements: 1.1_

  - [x] 2.2 Create Factor model with NumPy integration
    - Implement Factor class with scope and values
    - Add Pydantic validators for shape and probability bounds
    - Configure arbitrary_types_allowed for NumPy arrays
    - _Requirements: 2.1_

  - [x] 2.3 Create BayesianNetwork model with validation
    - Implement BayesianNetwork class with variables and factors
    - Add model_validator for factor scope validation
    - Add model_validator for cardinality consistency
    - Add helper methods (add_variable, add_factor, get_variable)
    - _Requirements: 1.2, 1.3, 1.4, 9.4_

  - [ ]* 2.4 Write unit tests for data models
    - Test Variable creation and validation
    - Test Factor creation with valid/invalid shapes
    - Test BayesianNetwork validation catches errors
    - _Requirements: 1.1, 1.2, 1.3_

- [ ] 3. Implement configuration management
  - [x] 3.1 Create InferenceConfig with Pydantic Settings
    - Implement InferenceConfig class with all settings
    - Configure environment variable prefix (PGM_)
    - Set up .env file support
    - Add settings for logging, numerical stability, LLM
    - _Requirements: 10.5_

- [ ] 4. Implement critical NumPy factor operations
  - [x] 4.1 Implement factor multiplication with einsum
    - Write multiply method using np.einsum
    - Generate dynamic einsum subscripts based on scope
    - Handle scope union correctly
    - _Requirements: 2.2_

  - [x] 4.2 Write unit tests for factor multiplication
    - Test with known values (2x2 × 2x2 = 2x2x2)
    - Test commutativity (F1 × F2 = F2 × F1)
    - Test scope union is correct
    - Test with different variable orderings
    - _Requirements: 2.2_

  - [x] 4.3 Implement factor marginalization
    - Write marginalize method using np.sum
    - Find correct axis for variable to eliminate
    - Update scope correctly
    - _Requirements: 2.5, 4.1_

  - [x] 4.4 Write unit tests for marginalization
    - Test with known values
    - Test scope reduction
    - Test summing over different axes
    - Verify probabilities sum correctly
    - _Requirements: 4.1_

  - [x] 4.5 Implement factor maximization with argmax tracking
    - Write maximize method using np.max and np.argmax
    - Build argmax mapping for reconstruction
    - Return both reduced factor and argmax dict
    - _Requirements: 5.1, 5.3_

  - [x] 4.6 Write unit tests for maximization
    - Test with known values
    - Test argmax tracking is correct
    - Test scope reduction
    - Verify max values are correct
    - _Requirements: 5.1, 5.3_

  - [x] 4.7 Implement evidence reduction
    - Write reduce method to fix evidence variables
    - Use advanced indexing to slice factor
    - Update scope to remove evidence variables
    - _Requirements: 2.5_

  - [x] 4.8 Write unit tests for evidence reduction
    - Test with known values
    - Test scope updates correctly
    - Test multiple evidence variables
    - Verify reduced values match manual slicing
    - _Requirements: 2.5_

  - [x] 4.9 Implement factor normalization
    - Write normalize method to scale values to sum to 1
    - Handle edge case of zero sum
    - _Requirements: 4.5_

  - [x] 4.10 Implement factor value lookup
    - Write get_value method for assignment lookup
    - Convert string assignment to array indices
    - _Requirements: 2.3_

- [x] 5. Checkpoint - Verify factor operations are correct
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement elimination strategy interface
  - [x] 6.1 Create abstract EliminationStrategy base class
    - Define abstract eliminate method signature
    - Add type hints for factor, variable, context
    - _Requirements: 7.1, 7.3_

  - [x] 6.2 Implement SumProductStrategy
    - Create SumProductStrategy class inheriting from base
    - Implement eliminate method using factor.marginalize
    - _Requirements: 4.1, 7.2_

  - [x] 6.3 Implement MaxProductStrategy
    - Create MaxProductStrategy class inheriting from base
    - Implement eliminate method using factor.maximize
    - Store argmax values for later reconstruction
    - _Requirements: 5.1, 5.3, 7.2_

  - [x] 6.4 Implement LLMSemanticStrategy
    - Create LLMSemanticStrategy class with injected LLM function
    - Implement prompt construction with variable, domain, evidence
    - Call LLM function and validate response
    - Reduce factor by fixing variable to LLM prediction
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 6.5 Write unit tests for strategies
    - Test SumProductStrategy with simple factor
    - Test MaxProductStrategy with simple factor
    - Test LLMSemanticStrategy with mock LLM
    - Verify prompt excludes probabilities
    - Test LLM response validation
    - _Requirements: 4.1, 5.1, 6.3, 6.4_

- [ ] 7. Implement elimination ordering heuristics
  - [x] 7.1 Implement min-degree ordering
    - Write function to compute min-degree ordering
    - Greedily select variable in fewest factors
    - Simulate elimination to update factor list
    - _Requirements: 3.4_

  - [ ]* 7.2 Write unit tests for ordering
    - Test ordering contains all nuisance variables
    - Test ordering is valid (no duplicates)
    - Test with simple network structure
    - _Requirements: 3.4_

- [ ] 8. Implement InferenceEngine
  - [x] 8.1 Create InferenceEngine class with Pydantic
    - Define InferenceEngine with network, strategy, config
    - Add model_config for arbitrary types
    - _Requirements: 3.1_

  - [x] 8.2 Implement query method
    - Validate query variables and evidence
    - Apply evidence to all factors
    - Identify nuisance variables
    - Get elimination ordering
    - Call _eliminate_variables
    - Post-process based on strategy type
    - _Requirements: 3.2, 4.3, 9.1, 9.5_

  - [x] 8.3 Implement _eliminate_variables helper
    - Loop through elimination ordering
    - Collect factors containing current variable
    - Multiply factors together
    - Call strategy.eliminate
    - Add result back to factor list
    - _Requirements: 3.2, 3.3_

  - [x] 8.4 Implement _multiply_factors helper
    - Multiply list of factors pairwise
    - Use factor.multiply method
    - _Requirements: 3.2_

  - [x] 8.5 Implement MAP assignment reconstruction
    - For MaxProductStrategy, backtrack through argmax values
    - Build complete assignment from stored argmax dicts
    - _Requirements: 5.4_

  - [x] 8.6 Add verbose logging throughout
    - Log elimination steps when config.verbose is True
    - Log factor operations
    - Log strategy-specific actions
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 9. Checkpoint - Verify inference engine works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Create comprehensive integration test
  - [x] 10.1 Implement Rain-Sprinkler-GrassWet example
    - Define variables (Rain, Sprinkler, GrassWet)
    - Create factors with known CPT values
    - Build BayesianNetwork
    - _Requirements: 8.4_

  - [x] 10.2 Test Sum-Product inference
    - Run posterior query P(Rain | GrassWet=yes)
    - Verify result matches hand-calculated value
    - Verify normalization (sums to 1)
    - _Requirements: 4.2, 4.5_

  - [x] 10.3 Test Max-Product inference
    - Run MAP query for most likely configuration
    - Verify assignment is correct
    - Verify probability value is correct
    - _Requirements: 5.2_

  - [x] 10.4 Test LLM Semantic inference
    - Create mock LLM function with reasonable heuristics
    - Run MAP query using LLM strategy
    - Verify prompt construction
    - Verify assignment is returned
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 10.5 Test strategy swapping
    - Run same query with all three strategies
    - Verify no errors occur
    - Verify each returns appropriate result format
    - _Requirements: 3.5, 7.5_

- [ ] 11. Create example script and documentation
  - [x] 11.1 Create examples/rain_sprinkler_grass.py
    - Demonstrate network construction
    - Show all three inference strategies
    - Print results with explanations
    - _Requirements: 8.3, 8.4_

  - [x] 11.2 Write README.md
    - Project overview and motivation
    - Installation instructions with UV
    - Quick start example
    - Configuration options
    - API documentation
    - _Requirements: 8.5_

  - [x] 11.3 Add docstrings to all public APIs
    - Document all classes with purpose and attributes
    - Document all public methods with parameters and returns
    - Include mathematical explanations where relevant
    - _Requirements: 8.1_

- [x] 12. Final checkpoint - Complete system test
  - Run all tests and verify they pass
  - Run example script and verify output
  - Ensure all tests pass, ask the user if questions arise.
