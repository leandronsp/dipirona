// Package model provides the numerical data structures and operations
// that constitute a neural network: weights, activations, and gradients.
package model

import (
	"fmt"
	"strings"
)

// Matrix represents an immutable 2D matrix of float64 values.
type Matrix struct {
	Rows int
	Cols int
	data []float64
}

// idx returns the flat index for (row, col).
func (m Matrix) idx(row, col int) int {
	return row*m.Cols + col
}

// New creates a new Matrix from a 2D slice. The slice is deep-copied.
func New(values [][]float64) Matrix {
	rows := len(values)
	cols := 0
	if rows > 0 {
		cols = len(values[0])
	}

	for i := range values {
		if len(values[i]) != cols {
			panic(fmt.Sprintf("ragged input: row %d has %d elements, expected %d", i, len(values[i]), cols))
		}
	}

	data := make([]float64, rows*cols)
	for r := range values {
		copy(data[r*cols:(r+1)*cols], values[r])
	}

	return Matrix{
		Rows: rows,
		Cols: cols,
		data: data,
	}
}

// At returns the element at the given row and column.
func (m Matrix) At(row, col int) float64 {
	return m.data[m.idx(row, col)]
}

// Transpose returns a new Matrix with rows and columns swapped.
func (m Matrix) Transpose() Matrix {
	data := make([]float64, m.Rows*m.Cols)

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			data[c*m.Rows+r] = m.data[m.idx(r, c)]
		}
	}

	return Matrix{
		Rows: m.Cols,
		Cols: m.Rows,
		data: data,
	}
}

// Multiply returns the matrix product of m and other.
// Panics if the number of columns in m does not equal the number of rows in other.
func (m Matrix) Multiply(other Matrix) Matrix {
	if m.Cols != other.Rows {
		panic(fmt.Sprintf("dimension mismatch: cannot multiply %dx%d by %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}

	data := make([]float64, m.Rows*other.Cols)

	for r := 0; r < m.Rows; r++ {
		for k := 0; k < m.Cols; k++ {
			ark := m.data[m.idx(r, k)]
			for c := 0; c < other.Cols; c++ {
				data[r*other.Cols+c] += ark * other.data[other.idx(k, c)]
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
// Panics if the matrices have different dimensions.
func (m Matrix) Zip(other Matrix, fn func(float64, float64) float64) Matrix {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		panic(fmt.Sprintf("dimension mismatch: cannot zip %dx%d with %dx%d", m.Rows, m.Cols, other.Rows, other.Cols))
	}

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
			b.WriteString(fmt.Sprintf("%v", m.data[m.idx(r, c)]))
		}
		b.WriteString("]")
	}
	b.WriteString("]")
	return b.String()
}
