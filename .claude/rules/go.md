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
- 100% coverage for `mat` and `calc` packages.
- Incremental commits after each RED-GREEN cycle.

## Matrix Conventions
- Panic with clear message for dimension mismatch (programming error, not runtime).
- `String()` for readable debugging in test failures.
- Benchmarks for O(n³) operations (multiplication).

## Project Layout
- `pkg/mat`: Matrix type and operations
- `pkg/calc`: pure math functions (sigmoid, etc.)
- `main.go`: manual demonstration of the public API
