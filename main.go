package main

import (
	"fmt"

	"dipirona/pkg/calc"
	"dipirona/pkg/model"
	"dipirona/pkg/model/layer"
)

func main() {
	// Matrix operations
	a := model.New([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	b := model.New([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})

	fmt.Println("Matrix A:", a)
	fmt.Println("Matrix B:", b)
	fmt.Println("A^T:", a.Transpose())
	fmt.Println("A * B:", a.Multiply(b))
	fmt.Println("A * 2:", a.Map(func(x float64) float64 { return x * 2 }))

	// Activation functions
	fmt.Printf("Sigmoid(0) = %v\n", calc.Sigmoid(0))
	fmt.Printf("Sigmoid(2) = %v\n", calc.Sigmoid(2))
	fmt.Printf("Relu(2.5) = %v\n", calc.Relu(2.5))
	fmt.Printf("Relu(-1.0) = %v\n", calc.Relu(-1.0))

	// Layer — forward pass with different activations
	weights := model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	})
	input := model.New([][]float64{
		{1.0, 1.0},
	})

	lNone := layer.New(weights, model.ActivationNone)
	outputNone := lNone.Forward(input)
	fmt.Println("Layer (None):", outputNone)

	lSigmoid := layer.New(weights, model.ActivationSigmoid)
	outputSigmoid := lSigmoid.Forward(input)
	fmt.Println("Layer (Sigmoid):", outputSigmoid)

	lReLU := layer.New(weights, model.ActivationReLU)
	outputReLU := lReLU.Forward(input)
	fmt.Println("Layer (ReLU):", outputReLU)
}
