package calc

// Relu returns x if x is positive, 0 otherwise.
func Relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReluDerivative returns 1 if x is positive, 0 otherwise.
func ReluDerivative(x float64) float64 {
	if x > 0 {
		return 1.0
	}
	return 0
}
