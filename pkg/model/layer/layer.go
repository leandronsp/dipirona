// Package layer provides a composable neural network layer with pluggable activation.
package layer

import (
	"dipirona/pkg/model"
)

// Layer represents a single neural network layer with weights, activation, and cached output.
type Layer struct {
	weights       model.Matrix
	activation    model.Activation
	output        model.Matrix
	forwardCalled bool
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
// It multiplies input by weights and applies the activation function.
func (l *Layer) Forward(input model.Matrix) model.Matrix {
	result := input.Multiply(l.weights)
	l.output = l.activation.Apply(result)
	l.forwardCalled = true
	return l.output
}

// Output returns the cached output from the last Forward call.
// Panics if Forward has not been called yet.
func (l Layer) Output() model.Matrix {
	if !l.forwardCalled {
		panic("output not available: call Forward first")
	}
	return l.output
}
