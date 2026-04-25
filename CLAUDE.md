# Dipirona

Minimal neural network library in Go. Inspired by [aspirina](https://github.com/leandronsp/aspirina).

## Stack
- Go 1.26.2 (mise-managed)
- Zero external dependencies for math core
- Standard `testing` package

## Domain
- `pkg/mat`: immutable Matrix with transpose, multiply, Map, Zip
- `pkg/calc`: pure math functions (sigmoid, sigmoid_derivative)
- Functional style: operations return new values, never mutate receivers

## Layout
```
pkg/mat/       Matrix type and ops
pkg/calc/      Math primitives
main.go        API smoke test
```

## Conventions
- TDD: RED → GREEN → commit
- Table-driven tests, 100% coverage on math packages
- Panic only for programmer errors (dimension mismatch)

## Roadmap
See [ROADMAP.md](ROADMAP.md)
