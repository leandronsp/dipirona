// Package mat provides immutable matrix operations for neural networks.
package mat

import (
	"fmt"
	"strings"
)

// Matrix represents an immutable 2D matrix of float64 values.
type Matrix struct {
	Rows int
	Cols int
	data [][]float64
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

	data := make([][]float64, rows)
	for i := range values {
		data[i] = make([]float64, cols)
		copy(data[i], values[i])
	}

	return Matrix{
		Rows: rows,
		Cols: cols,
		data: data,
	}
}

// At returns the element at the given row and column.
func (m Matrix) At(row, col int) float64 {
	return m.data[row][col]
}

// Transpose returns a new Matrix with rows and columns swapped.
func (m Matrix) Transpose() Matrix {
	data := make([][]float64, m.Cols)
	for i := range data {
		data[i] = make([]float64, m.Rows)
	}

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			data[c][r] = m.data[r][c]
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

	data := make([][]float64, m.Rows)
	for i := range data {
		data[i] = make([]float64, other.Cols)
	}

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < other.Cols; c++ {
			var sum float64
			for k := 0; k < m.Cols; k++ {
				sum += m.data[r][k] * other.data[k][c]
			}
			data[r][c] = sum
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

	data := make([][]float64, m.Rows)
	for i := range data {
		data[i] = make([]float64, m.Cols)
	}

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			data[r][c] = fn(m.data[r][c], other.data[r][c])
		}
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
	data := make([][]float64, m.Rows)
	for i := range data {
		data[i] = make([]float64, m.Cols)
	}

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			data[r][c] = fn(m.data[r][c])
		}
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
			b.WriteString(fmt.Sprintf("%v", m.data[r][c]))
		}
		b.WriteString("]")
	}
	b.WriteString("]")
	return b.String()
}
