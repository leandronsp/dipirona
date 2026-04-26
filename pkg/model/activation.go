package model

import "dipirona/pkg/calc"

// Activation represents a pluggable activation function for a neural network layer.
type Activation int

const (
	ActivationNone    Activation = iota // Identity — returns input unchanged
	ActivationSigmoid                   // Logistic sigmoid
	ActivationReLU                      // Rectified linear unit
)

// Apply returns a new Matrix with the activation function applied element-wise.
func (a Activation) Apply(m Matrix) Matrix {
	switch a {
	case ActivationNone:
		return m
	case ActivationSigmoid:
		return m.Map(calc.Sigmoid)
	case ActivationReLU:
		return m.Map(calc.Relu)
	default:
		panic("unknown activation function")
	}
}
