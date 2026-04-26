# dipirona

Minimal neural network library in Go, inspired by [aspirina](https://github.com/leandronsp/aspirina).

Composable math core and layer abstraction. Build any feedforward architecture — not just XOR.

## Quick Start

```go
package main

import (
    "fmt"

    "dipirona/pkg/calc"
    "dipirona/pkg/model"
    "dipirona/pkg/model/layer"
)

func main() {
    // Create a layer with known weights and Sigmoid activation
    weights := model.New([][]float64{
        {0.5, -0.2},
        {0.1, 0.8},
    })
    input := model.New([][]float64{
        {1.0, 1.0},
    })

    l := layer.New(weights, model.ActivationSigmoid)
    output := l.Forward(input)
    fmt.Println("Output:", output) // Matrix(1x2)[[0.6456... 0.6456...]]
}
```

## Packages

| Package | Purpose |
|---------|---------|
| `pkg/model` | Matrix operations, Activation type (None, Sigmoid, ReLU) |
| `pkg/model/layer` | Layer struct with Forward pass, Weights, and Output |
| `pkg/calc` | Pure math functions — Sigmoid, SigmoidDerivative, Relu, ReluDerivative |
| `pkg/validate` | Input validation (dimension checks, ragged row guards) |

## Commands

All commands use `make`. Never run raw `go` commands.

| Command | Purpose | Example |
|---------|---------|---------|
| `make test` | Run all tests | `make test` |
| `make test PKG=./pkg/calc` | Run tests for a specific package | `make test PKG=./pkg/calc` |
| `make build` | Build the binary (`./dipirona`) | `make build` |
| `make bench` | Run benchmarks for `pkg/model` | `make bench` |
| `make fmt` | Format all Go files | `make fmt` |
| `make vet` | Run static analysis | `make vet` |

## Architecture

- **Functional style**: operations return new values, never mutate receivers
- **Activation as enum**: `ActivationNone`, `ActivationSigmoid`, `ActivationReLU` — pluggable per layer
- **Layer.Forward**: computes `input × weights` (matmul), then applies activation element-wise
- **Panic for programmer errors**: dimension mismatch, calling Output before Forward
- **Zero external dependencies** for the math core

## Roadmap

See [ROADMAP.md](ROADMAP.md) for planned phases.