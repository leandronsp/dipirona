// Package mat provides immutable matrix operations for neural networks.
package mat

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
