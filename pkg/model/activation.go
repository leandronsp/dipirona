package model

import "dipirona/pkg/calc"

// Activation represents a pluggable activation function for a neural network layer.
type Activation int

const (
	ActivationNone    Activation = iota // Identity — returns input unchanged
	ActivationSigmoid                   // Logistic sigmoid
	ActivationReLU                      // Rectified linear unit
)

// Apply returns the result of applying the activation function element-wise.
// For ActivationNone, returns the input unchanged.
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
