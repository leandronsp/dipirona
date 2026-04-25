# Roadmap

Incremental delivery of a minimal neural network library in Go.
Each phase produces a shippable milestone with user-facing value.

## Phase 1: Math Core
Foundation. Matrix operations and sigmoid activation.

- Matrix: new, transpose, multiply, element-wise ops
- Calc: sigmoid, sigmoid_derivative
- Full unit test coverage

**Value:** Developers trust the math before building on it.

## Phase 2: Layer Abstraction
A single neural network layer with forward pass and state caching.

- Struct `Layer` with weights and forwarded output cache
- `Forward(input) -> output` API
- Tests with fixed weights and known inputs

**Value:** Isolated component you can inspect and test.

## Phase 3: Neural Network Engine
Feedforward network with backpropagation training.

- `NeuralNetwork`: chain layers, predict, train
- Gradient descent via backprop
- Integration test: train XOR to convergence

**Value:** The library works end-to-end.

## Phase 4: Logic Gate Scenarios
Train boolean gates and print truth tables.

- Training programs for AND, OR, XOR
- 10k epochs, threshold 0.5, verified output
- Integration tests on truth tables

**Value:** First "wow" moment. Concrete, verifiable learning.

## Phase 5: CLI Tooling
A human interface for training and inference.

- `dipirona train <gate>` with `--epochs`, `--lr`
- `dipirona predict <gate> <input>`
- `dipirona demo` to run all gates

**Value:** Use the library without writing Go code.

## Phase 6: Model Persistence
Save and load trained weights.

- `Save(path)` and `Load(path)` on `NeuralNetwork`
- JSON format
- Roundtrip tests

**Value:** Reuse trained models. No re-training on every run.

## Phase 7: Neural Computer
Build computer components from trained gates.

- Half Adder, Full Adder, 4-bit ALU
- End-to-end tests on arithmetic operations

**Value:** The aspirina showcase. Neural networks doing deterministic computation.
