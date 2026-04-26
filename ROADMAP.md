# Roadmap

Incremental delivery of a minimal neural network library in Go.
Each phase produces a shippable milestone with user-facing value.

## Phase 1: Math Core ✅
Foundation. Matrix operations and activation primitives.

- Matrix: new, transpose, multiply, element-wise ops
- Calc: sigmoid, sigmoid_derivative
- Validate: dimension checks, ragged row guards
- Full unit test coverage

**Value:** Developers trust the math before building on it.

## Phase 2: Layer Abstraction
A composable neural network layer. Decoupled from any specific task.

- Activation function as a first-class concept: `Activation` type with Sigmoid, ReLU, None
- Calc: ReLU, ReLU derivative (alongside existing Sigmoid)
- Layer struct: weights Matrix + Activation + output cache
- `New(weights Matrix, activation)` constructor (takes a pre-built weight matrix)
- `Forward(input Matrix) Matrix`: matmul then conditionally apply activation
- `Weights()`, `Output()` accessors
- Tests with known inputs, multiple activations, panic paths

**Value:** Building block for any feedforward architecture. Not locked to XOR.

## Phase 3: Network Engine
Feedforward MLP with backpropagation training.

- `Network`: ordered list of Layers, predict, train
- Backpropagation with configurable learning rate
- Solver-agnostic: train any dataset that fits feedforward shape
- Integration test: train XOR to convergence as a canonical example

**Value:** The library works end-to-end for any feedforward problem.

## Phase 4: Training Scenarios
Demonstrate the library on diverse problems beyond logic gates.

- Logic gates: AND, OR, XOR with truth-table verification
- Regression: learn simple continuous functions
- Classification: multi-class problems with softmax output
- Each scenario is a self-contained Go program + integration test

**Value:** Proof that the library generalizes. Not just a XOR toy.

## Phase 5: CLI Tooling
A human interface for training and inference.

- `dipirona train --config file` with epochs, learning rate, architecture
- `dipirona predict --model file --input ...`
- `dipirona demo` to run built-in scenarios

**Value:** Use the library without writing Go code.

## Phase 6: Model Persistence
Save and load trained weights.

- `Save(path)` and `Load(path)` on `Network`
- JSON format
- Roundtrip tests

**Value:** Reuse trained models. No re-training on every run.

## Phase 7: Advanced Architectures
Beyond vanilla MLP.

- Configurable loss functions (MSE, cross-entropy)
- Softmax output layer
- Regularization (L2)
- Optional bias per layer

**Value:** The library handles real-world problems, not just toy datasets.