package main

import (
	"fmt"

	"dipirona/pkg/calc"
	"dipirona/pkg/mat"
)

func main() {
	// Create matrices
	a := mat.New([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	b := mat.New([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})

	fmt.Println("Matrix A:", a)
	fmt.Println("Matrix B:", b)

	// Transpose
	fmt.Println("A^T:", a.Transpose())

	// Multiply
	product := a.Multiply(b)
	fmt.Println("A * B:", product)

	// Element-wise operations
	doubled := a.Map(func(x float64) float64 { return x * 2 })
	fmt.Println("A * 2:", doubled)

	// Sigmoid
	fmt.Printf("Sigmoid(0) = %v\n", calc.Sigmoid(0))
	fmt.Printf("Sigmoid(2) = %v\n", calc.Sigmoid(2))
	fmt.Printf("SigmoidDerivative(0) = %v\n", calc.SigmoidDerivative(0))
}
