// Package random provides random matrix initialization for weight seeding.
package random

import (
	"dipirona/pkg/model"
	"math/rand"
)

// New creates a Matrix of the given dimensions with values sampled uniformly from (-0.5, 0.5).
// Uses a package-level default seed for reproducibility within a single run.
func New(rows, cols int) model.Matrix {
	return NewWithSeed(rows, cols, 42)
}

// NewWithSeed creates a Matrix of the given dimensions with values sampled uniformly from (-0.5, 0.5),
// using the provided seed for deterministic results.
func NewWithSeed(rows, cols int, seed int64) model.Matrix {
	r := rand.New(rand.NewSource(seed))
	data := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		data[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			data[i][j] = r.Float64() - 0.5
		}
	}
	return model.New(data)
}