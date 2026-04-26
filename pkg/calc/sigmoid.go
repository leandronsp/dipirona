// Package calc provides pure mathematical functions for neural networks.
package calc

import "math"

// Sigmoid returns the sigmoid of x: 1 / (1 + e^(-x)).
func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// SigmoidDerivative returns the derivative of sigmoid at x.
// This is equal to sigmoid(x) * (1 - sigmoid(x)).
func SigmoidDerivative(x float64) float64 {
	s := Sigmoid(x)
	return s * (1.0 - s)
}
