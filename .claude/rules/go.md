# Dipirona Go Rules

## Functional Style
- Structures are immutable. Methods return new instances, never mutate `receiver`.
- Receiver by **value**, not pointer, unless justified exception.
- Pure functions when possible. Avoid global state.
- Higher-order functions (`Map`, `Zip`) take `func(float64) float64`, not interfaces.

## TDD Discipline
- No production code without a failing test first (RED).
- One behavior per test. One assertion per test when possible.
- Table-driven tests with descriptive `name` field.
- 100% coverage for `model` and `calc` packages.
- Incremental commits after each RED-GREEN cycle.

## Matrix Conventions
- Panic with clear message for dimension mismatch (programming error, not runtime).
- `String()` for readable debugging in test failures.
- Benchmarks for O(n³) operations (multiplication).

## Naming Conventions
- Packages named after **domain concepts**, not data structures.
  - `pkg/model` — neural network parameters, activations, gradients (not `mat` or `matrix`)
  - `pkg/calc` — mathematical primitives (not `math` or `functions`)
- Types named after what they are: `Matrix`, `Layer`, `Network`.
- Functions rely on package context: `model.New()` not `model.NewMatrixFromSlice()`.

## Project Layout
- `pkg/model`: neural network numerical model (Matrix, weights, activations)
- `pkg/calc`: pure math functions (sigmoid, etc.)
- `main.go`: manual demonstration of the public API
