# Requirements Document

## Introduction

This document specifies the requirements for an experimental Python library that implements Variable Elimination (VE) for Bayesian Networks with pluggable inference strategies. The library supports traditional numerical inference (Sum-Product, Max-Product) and a novel LLM-driven semantic inference approach for MAP queries. The system is designed to facilitate research comparing numerical probability-based inference against LLM-based semantic reasoning.

## Glossary

- **Bayesian Network (BN)**: A directed acyclic graph representing probabilistic dependencies between random variables
- **Factor**: A multi-dimensional table representing a conditional probability distribution or potential function over a subset of variables
- **Variable Elimination (VE)**: An exact inference algorithm that systematically eliminates variables by multiplying and marginalizing factors
- **Scope**: The set of variables that a factor is defined over
- **Cardinality**: The number of possible values a discrete random variable can take
- **Evidence**: Observed values for a subset of variables in the network
- **Query Variables**: Variables for which we want to compute posterior probabilities or MAP assignments
- **Nuisance Variables**: Variables that are neither query variables nor evidence variables and must be eliminated
- **Sum-Product VE**: Variable Elimination using summation as the elimination operation for computing posterior marginals
- **Max-Product VE**: Variable Elimination using maximization as the elimination operation for MAP inference
- **MAP (Maximum A Posteriori)**: The most probable assignment to query variables given evidence
- **MPE (Most Probable Explanation)**: The most probable complete assignment to all non-evidence variables
- **Elimination Ordering**: The sequence in which nuisance variables are eliminated during VE
- **Inference Engine**: The core component that orchestrates the Variable Elimination algorithm
- **Elimination Strategy**: A pluggable component that defines how variables are eliminated from factors (sum, max, or LLM query)
- **LLM Semantic Strategy**: An elimination strategy that uses a Large Language Model to predict the most plausible variable value based on context

## Requirements

### Requirement 1

**User Story:** As a researcher, I want to define Bayesian Networks with discrete random variables, so that I can represent probabilistic models for inference experiments.

#### Acceptance Criteria

1. WHEN a user creates a variable definition THEN the system SHALL store the variable name, cardinality, and domain values
2. WHEN a user adds a factor to the network THEN the system SHALL validate that all variables in the factor scope are defined in the network
3. WHEN a user specifies a CPT for a factor THEN the system SHALL verify that the table dimensions match the cardinalities of variables in the scope
4. WHEN a user queries the network structure THEN the system SHALL return the list of variables, their domains, and the factors
5. THE system SHALL support factors with arbitrary scope sizes (unary, binary, n-ary)

### Requirement 2

**User Story:** As a researcher, I want to represent factors as multi-dimensional probability tables, so that I can store and manipulate conditional probability distributions efficiently.

#### Acceptance Criteria

1. WHEN a factor is created THEN the system SHALL store the scope (variable list) and the probability values
2. WHEN two factors are multiplied THEN the system SHALL compute the pointwise product over the union of their scopes
3. WHEN a factor is queried for a specific variable assignment THEN the system SHALL return the corresponding probability value
4. THE system SHALL support both dense array representations and sparse dictionary representations for factor storage
5. WHEN evidence is applied to a factor THEN the system SHALL reduce the factor by fixing the evidence variables to their observed values

### Requirement 3

**User Story:** As a researcher, I want to perform Variable Elimination with different elimination strategies, so that I can compare numerical and LLM-based inference approaches.

#### Acceptance Criteria

1. WHEN the inference engine is initialized with an elimination strategy THEN the system SHALL use that strategy for all elimination operations
2. WHEN a variable is eliminated from a set of factors THEN the system SHALL multiply all factors containing that variable and apply the elimination strategy
3. WHEN the elimination ordering is provided THEN the system SHALL process variables in that specified order
4. WHEN no elimination ordering is provided THEN the system SHALL compute a heuristic ordering (e.g., min-fill or min-degree)
5. THE system SHALL support swapping elimination strategies without modifying the core VE algorithm implementation

### Requirement 4

**User Story:** As a researcher, I want to compute posterior marginals using Sum-Product Variable Elimination, so that I can perform standard probabilistic inference queries.

#### Acceptance Criteria

1. WHEN Sum-Product strategy eliminates a variable THEN the system SHALL sum out that variable from the factor product
2. WHEN a posterior query is executed THEN the system SHALL return a normalized probability distribution over the query variables
3. WHEN evidence is provided THEN the system SHALL condition all factors on the evidence before elimination
4. WHEN multiple query variables are specified THEN the system SHALL return the joint posterior distribution
5. THE system SHALL normalize the final factor to ensure probabilities sum to one

### Requirement 5

**User Story:** As a researcher, I want to compute MAP assignments using Max-Product Variable Elimination, so that I can find the most probable configuration using numerical optimization.

#### Acceptance Criteria

1. WHEN Max-Product strategy eliminates a variable THEN the system SHALL maximize over that variable in the factor product
2. WHEN a MAP query is executed THEN the system SHALL return both the maximum probability and the optimal variable assignment
3. WHEN maximizing over a variable THEN the system SHALL track the argmax values for later reconstruction
4. WHEN the elimination is complete THEN the system SHALL backtrack through stored argmax values to construct the full MAP assignment
5. THE system SHALL perform maximization locally during each elimination step rather than constructing the full joint distribution

### Requirement 6

**User Story:** As a researcher, I want to compute MAP assignments using LLM Semantic Strategy, so that I can compare semantic reasoning against numerical optimization.

#### Acceptance Criteria

1. WHEN LLM Semantic Strategy eliminates a variable THEN the system SHALL construct a textual context describing the variable, its domain, and current evidence
2. WHEN the LLM is queried THEN the system SHALL receive a predicted value from the variable's domain
3. WHEN constructing the LLM prompt THEN the system SHALL include variable names, domain values, and evidence assignments but SHALL NOT include numerical probabilities
4. WHEN the LLM returns a prediction THEN the system SHALL validate that the returned value is in the variable's domain
5. THE system SHALL support injecting custom LLM query functions without modifying the strategy implementation

### Requirement 7

**User Story:** As a researcher, I want the elimination strategies to follow a common interface, so that I can easily swap between Sum-Product, Max-Product, and LLM strategies.

#### Acceptance Criteria

1. WHEN an elimination strategy is implemented THEN the system SHALL provide an `eliminate` method that takes a factor and variable name
2. WHEN the `eliminate` method is called THEN the system SHALL return a new factor with the specified variable removed from the scope
3. THE system SHALL define a base strategy interface that all concrete strategies implement
4. WHEN a strategy requires additional context (e.g., evidence for LLM) THEN the system SHALL pass this context through the strategy interface
5. WHEN switching strategies THEN the system SHALL require no changes to the Variable Elimination algorithm implementation

### Requirement 8

**User Story:** As a researcher, I want comprehensive documentation and examples, so that I can understand how to use the library for my experiments.

#### Acceptance Criteria

1. WHEN a user reads the code THEN the system SHALL provide docstrings for all public classes and methods
2. WHEN a user examines a class THEN the system SHALL include docstrings explaining the mathematical operations and algorithmic purpose
3. THE system SHALL include a complete usage example demonstrating all three inference strategies on a simple Bayesian Network
4. WHEN a user runs the example THEN the system SHALL demonstrate inference on a "Rain-Sprinkler-GrassWet" type network
5. THE system SHALL document the expected input formats for factors, evidence, and query specifications

### Requirement 9

**User Story:** As a researcher, I want the library to handle edge cases gracefully, so that I can trust the results of my experiments.

#### Acceptance Criteria

1. WHEN invalid evidence is provided (variable not in network or value not in domain) THEN the system SHALL raise a descriptive error
2. WHEN a factor operation results in an empty scope THEN the system SHALL handle it as a scalar factor
3. WHEN the LLM returns an invalid value THEN the system SHALL raise an error with the invalid response and valid domain
4. WHEN factors have inconsistent cardinalities for the same variable THEN the system SHALL detect and report the inconsistency
5. WHEN a query requests a variable that doesn't exist THEN the system SHALL raise a descriptive error

### Requirement 10

**User Story:** As a researcher, I want to inspect intermediate steps of the Variable Elimination process, so that I can debug and understand the algorithm's behavior.

#### Acceptance Criteria

1. WHEN verbose mode is enabled THEN the system SHALL log each elimination step with the variable being eliminated
2. WHEN a factor is created or modified THEN the system SHALL optionally log the factor's scope and dimensions
3. WHEN the LLM strategy makes a prediction THEN the system SHALL optionally log the prompt sent and response received
4. WHEN Max-Product stores argmax values THEN the system SHALL optionally log the backtracking reconstruction process
5. THE system SHALL provide a way to enable or disable verbose logging without code modification
