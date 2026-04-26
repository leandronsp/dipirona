// Package layer provides a composable neural network layer with pluggable activation.
package layer

import (
	"dipirona/pkg/model"
)

// Layer represents a single neural network layer with weights, activation, and cached state.
type Layer struct {
	weights    model.Matrix
	activation model.Activation
	input      *model.Matrix
	z          *model.Matrix // pre-activation: input * weights
	output     *model.Matrix // post-activation: activation(z)
}

// New creates a Layer with the given weights and activation function.
func New(weights model.Matrix, activation model.Activation) Layer {
	return Layer{
		weights:    weights,
		activation: activation,
	}
}

// Weights returns the layer's weight matrix.
func (l Layer) Weights() model.Matrix {
	return l.weights
}

// Forward computes the output of the layer for the given input.
// It multiplies input by weights, caches input and pre-activation for backpropagation,
// and applies the activation function.
func (l *Layer) Forward(input model.Matrix) model.Matrix {
	l.input = &input
	z := input.Multiply(l.weights)
	l.z = &z
	applied := l.activation.Apply(z)
	l.output = &applied
	return applied
}

// PreActivation returns the cached pre-activation (z) from the last Forward call.
// Panics if Forward has not been called yet.
func (l Layer) PreActivation() model.Matrix {
	if l.z == nil {
		panic("pre-activation not available: call Forward first")
	}
	return *l.z
}

// Input returns the cached input from the last Forward call.
// Panics if Forward has not been called yet.
func (l Layer) Input() model.Matrix {
	if l.input == nil {
		panic("input not available: call Forward first")
	}
	return *l.input
}

// Output returns the cached output from the last Forward call.
// Panics if Forward has not been called yet.
func (l Layer) Output() model.Matrix {
	if l.output == nil {
		panic("output not available: call Forward first")
	}
	return *l.output
}

// Backward propagates the gradient backward through the layer, updates weights,
// and returns the gradient for the previous layer.
// delta is the gradient from the next layer (or loss derivative for the output layer).
// lr is the learning rate for weight updates.
// Panics if Forward has not been called yet.
func (l *Layer) Backward(delta model.Matrix, lr float64) model.Matrix {
	if l.z == nil || l.input == nil {
		panic("backward not available: call Forward first")
	}

	// Apply activation derivative to get local delta
	localDelta := l.activation.ApplyDerivative(*l.z).Zip(delta, func(a, b float64) float64 {
		return a * b
	})

	// dW = input.T * localDelta
	dW := l.input.Transpose().Multiply(localDelta)

	// Gradient for previous layer: localDelta * weights.T (before update)
	grad := localDelta.Multiply(l.weights.Transpose())

	// Update weights: W = W - lr * dW
	l.weights = l.weights.Subtract(dW.Scale(lr))

	return grad
}
