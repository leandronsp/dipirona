// Package validate provides input validation for matrix operations.
// All functions panic on invalid input with clear messages.
package validate

import "fmt"

// RowsUniform panics if any row length differs from expected.
func RowsUniform(values [][]float64, expected int) {
	for i := range values {
		if len(values[i]) != expected {
			panic(fmt.Sprintf("ragged input: row %d has %d elements, expected %d", i, len(values[i]), expected))
		}
	}
}

// MultiplyCompatible panics if inner dimensions don't match for multiplication.
func MultiplyCompatible(rowsA, colsA, rowsB, colsB int) {
	if colsA != rowsB {
		panic(fmt.Sprintf("dimension mismatch: cannot multiply %dx%d by %dx%d", rowsA, colsA, rowsB, colsB))
	}
}

// SameDimensions panics if two matrices have different dimensions.
func SameDimensions(rowsA, colsA, rowsB, colsB int) {
	if rowsA != rowsB || colsA != colsB {
		panic(fmt.Sprintf("dimension mismatch: cannot zip %dx%d with %dx%d", rowsA, colsA, rowsB, colsB))
	}
}
