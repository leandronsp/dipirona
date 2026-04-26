# Dipirona

Minimal neural network library in Go. Inspired by [aspirina](https://github.com/leandronsp/aspirina).

## Stack
- Go 1.26.2 (mise-managed)
- Zero external dependencies for math core
- Standard `testing` package

## Domain
- `pkg/model`: numerical structures (Matrix) that represent weights, activations, and gradients
- `pkg/calc`: pure math functions (sigmoid, sigmoid_derivative)
- Functional style: operations return new values, never mutate receivers

## Layout
```
pkg/model/     Neural network numerical model (weights, activations, gradients)
pkg/calc/      Math primitives
main.go        API smoke test
```

## Conventions
- TDD: RED → GREEN → commit
- Table-driven tests, 100% coverage on math packages
- Panic only for programmer errors (dimension mismatch)

## Roadmap
See [ROADMAP.md](ROADMAP.md)
