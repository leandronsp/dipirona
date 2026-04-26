package main

import (
	"fmt"
	"math"

	"dipirona/pkg/calc"
	"dipirona/pkg/model"
	"dipirona/pkg/model/layer"
	"dipirona/pkg/network"
	"dipirona/pkg/random"
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
	fmt.Println("A - A/2:", a.Subtract(a.Map(func(x float64) float64 { return x / 2 })))
	fmt.Println("A * 0.5:", a.Scale(0.5))

	// Activation functions
	fmt.Printf("Sigmoid(0) = %v\n", calc.Sigmoid(0))
	fmt.Printf("Sigmoid(2) = %v\n", calc.Sigmoid(2))
	fmt.Printf("Relu(2.5) = %v\n", calc.Relu(2.5))
	fmt.Printf("Relu(-1.0) = %v\n", calc.Relu(-1.0))

	// MSE
	predicted := []float64{0.8, 0.2, 0.6}
	target := []float64{1.0, 0.0, 0.5}
	fmt.Printf("MSE = %v\n", calc.MSE(predicted, target))
	fmt.Printf("MSEDerivative = %v\n", calc.MSEDerivative(predicted, target))

	// Layer — forward pass with different activations
	weights := model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	})
	input := model.New([][]float64{{1.0, 1.0}})

	lNone := layer.New(weights, model.ActivationNone)
	outputNone := lNone.Forward(input)
	fmt.Println("Layer (None):", outputNone)

	lSigmoid := layer.New(weights, model.ActivationSigmoid)
	outputSigmoid := lSigmoid.Forward(input)
	fmt.Println("Layer (Sigmoid):", outputSigmoid)

	lReLU := layer.New(weights, model.ActivationReLU)
	outputReLU := lReLU.Forward(input)
	fmt.Println("Layer (ReLU):", outputReLU)

	// Activation derivative
	z := model.New([][]float64{{0.0, 1.0, -1.0}})
	fmt.Println("Sigmoid derivative:", model.ActivationSigmoid.ApplyDerivative(z))
	fmt.Println("ReLU derivative:", model.ActivationReLU.ApplyDerivative(z))
	fmt.Println("None derivative:", model.ActivationNone.ApplyDerivative(z))

	// Layer backward pass
	delta := model.New([][]float64{{0.1, -0.3}})
	grad := lNone.Backward(delta, 0.5)
	fmt.Println("Gradient for previous layer:", grad)
	fmt.Println("Updated weights:", lNone.Weights())

	// Random weight initialization
	randWeights := random.New(2, 3)
	fmt.Println("Random weights (2x3):", randWeights)
	randSeeded := random.NewWithSeed(2, 3, 42)
	fmt.Println("Seeded random (2x3, seed=42):", randSeeded)

	// XOR training with MLP
	fmt.Println("\n--- XOR Training ---")

	w1 := random.NewWithSeed(2, 2, 1)
	w2 := random.NewWithSeed(2, 1, 101)
	hidden := layer.New(w1, model.ActivationSigmoid)
	output := layer.New(w2, model.ActivationSigmoid)
	mlp := network.New([]layer.Layer{hidden, output}, network.WithLearningRate(5.0))

	batchInput := model.New([][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
	batchTarget := model.New([][]float64{{0}, {1}, {1}, {0}})

	xorInputs := [][][]float64{{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}
	xorTargets := []float64{0, 1, 1, 0}

	// Train
	for epoch := 0; epoch < 10000; epoch++ {
		mlp.Train(batchInput, batchTarget)
	}

	// Verify
	fmt.Println("Results after 10000 epochs:")
	for i := 0; i < 4; i++ {
		pred := mlp.Predict(model.New(xorInputs[i]))
		fmt.Printf("  XOR %v = %.4f (expected %.1f, diff %.4f)\n",
			xorInputs[i][0], pred.At(0, 0), xorTargets[i], math.Abs(pred.At(0, 0)-xorTargets[i]))
	}
}