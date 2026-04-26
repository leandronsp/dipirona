# Dipirona

Minimal neural network library in Go. Inspired by [aspirina](https://github.com/leandronsp/aspirina).

## Stack
- Go 1.26.2 (mise-managed)
- Zero external dependencies for math core
- Standard `testing` package

## Domain
- `pkg/model`: numerical structures (Matrix) that represent weights, activations, and gradients
- `pkg/calc`: pure math functions (sigmoid, sigmoid_derivative)
- `pkg/validate`: input validation (dimension checks, ragged rows)
- Functional style: operations return new values, never mutate receivers

## Layout
```
pkg/model/     Neural network numerical model (weights, activations, gradients)
pkg/calc/      Math primitives
pkg/validate/  Input validation (primitive types only, no model dependency)
main.go        API smoke test
```

## Commands
Always use `make`, never raw `go` commands directly.

| Command | Purpose |
|---------|---------|
| `make test` | Run all tests |
| `make fmt` | Format all Go files |
| `make vet` | Run static analysis |
| `make build` | Build the binary (`./dipirona`) |
| `make bench` | Run benchmarks for `pkg/model` |

## Conventions
- TDD: RED → GREEN → commit
- Table-driven tests, 100% coverage on math packages
- Panic only for programmer errors (dimension mismatch)
- Always use `make` targets, never raw `go` commands

## Roadmap
See [ROADMAP.md](ROADMAP.md)
