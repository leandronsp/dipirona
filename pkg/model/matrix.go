// Package model provides the numerical data structures and operations
// that constitute a neural network: weights, activations, and gradients.
package model

import (
	"fmt"
	"strings"

	"dipirona/pkg/validate"
)

// Matrix represents an immutable 2D matrix of float64 values.
type Matrix struct {
	Rows int
	Cols int
	data []float64
}

// offset returns the flat index for element (row, col) in row-major layout.
func offset(cols, row, col int) int {
	return row*cols + col
}

// New creates a new Matrix from a 2D slice. The slice is deep-copied.
func New(values [][]float64) Matrix {
	rows := len(values)
	cols := 0
	if rows > 0 {
		cols = len(values[0])
	}

	validate.RowsUniform(values, cols)

	data := make([]float64, rows*cols)
	for r := range values {
		start := offset(cols, r, 0)
		end := start + cols
		copy(data[start:end], values[r])
	}

	return Matrix{
		Rows: rows,
		Cols: cols,
		data: data,
	}
}

// At returns the element at the given row and column.
func (m Matrix) At(row, col int) float64 {
	return m.data[offset(m.Cols, row, col)]
}

// Transpose returns a new Matrix with rows and columns swapped.
func (m Matrix) Transpose() Matrix {
	data := make([]float64, m.Rows*m.Cols)

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			src := offset(m.Cols, r, c)
			dst := offset(m.Rows, c, r)
			data[dst] = m.data[src]
		}
	}

	return Matrix{
		Rows: m.Cols,
		Cols: m.Rows,
		data: data,
	}
}

// Multiply returns the matrix product of m and other.
func (m Matrix) Multiply(other Matrix) Matrix {
	validate.MultiplyCompatible(m.Rows, m.Cols, other.Rows, other.Cols)

	data := make([]float64, m.Rows*other.Cols)

	for r := 0; r < m.Rows; r++ {
		for k := 0; k < m.Cols; k++ {
			ark := m.data[offset(m.Cols, r, k)]
			for c := 0; c < other.Cols; c++ {
				dst := offset(other.Cols, r, c)
				src := offset(other.Cols, k, c)
				data[dst] += ark * other.data[src]
			}
		}
	}

	return Matrix{
		Rows: m.Rows,
		Cols: other.Cols,
		data: data,
	}
}

// Zip returns a new Matrix where each element is the result of applying fn
// to the corresponding elements of m and other.
func (m Matrix) Zip(other Matrix, fn func(float64, float64) float64) Matrix {
	validate.SameDimensions(m.Rows, m.Cols, other.Rows, other.Cols)

	data := make([]float64, m.Rows*m.Cols)
	for i := range data {
		data[i] = fn(m.data[i], other.data[i])
	}

	return Matrix{
		Rows: m.Rows,
		Cols: m.Cols,
		data: data,
	}
}

// Map returns a new Matrix where each element is the result of applying fn
// to the corresponding element of m.
func (m Matrix) Map(fn func(float64) float64) Matrix {
	data := make([]float64, len(m.data))
	for i := range data {
		data[i] = fn(m.data[i])
	}

	return Matrix{
		Rows: m.Rows,
		Cols: m.Cols,
		data: data,
	}
}

// String returns a human-readable representation of the matrix.
func (m Matrix) String() string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("Matrix(%dx%d)[", m.Rows, m.Cols))
	for r := 0; r < m.Rows; r++ {
		if r > 0 {
			b.WriteString(" ")
		}
		b.WriteString("[")
		for c := 0; c < m.Cols; c++ {
			if c > 0 {
				b.WriteString(" ")
			}
			b.WriteString(fmt.Sprintf("%v", m.data[offset(m.Cols, r, c)]))
		}
		b.WriteString("]")
	}
	b.WriteString("]")
	return b.String()
}
