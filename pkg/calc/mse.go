package calc

// MSE returns the mean squared error between predicted and target values.
func MSE(predicted, target []float64) float64 {
	n := len(predicted)
	var sum float64
	for i := 0; i < n; i++ {
		diff := predicted[i] - target[i]
		sum += diff * diff
	}
	return sum / float64(n)
}

// MSEDerivative returns the per-element derivative of MSE with respect to predicted values.
// derivative_i = 2 * (predicted_i - target_i) / n
func MSEDerivative(predicted, target []float64) []float64 {
	n := len(predicted)
	n64 := float64(n)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		result[i] = 2 * (predicted[i] - target[i]) / n64
	}
	return result
}