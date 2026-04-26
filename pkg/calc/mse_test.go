package calc

import (
	"math"
	"testing"
)

func TestMSE_ReturnsMeanSquaredError(t *testing.T) {
	predicted := []float64{0.8, 0.2, 0.6}
	target := []float64{1.0, 0.0, 0.5}

	// MSE = ((0.8-1.0)² + (0.2-0.0)² + (0.6-0.5)²) / 3
	//     = (0.04 + 0.04 + 0.01) / 3
	//     = 0.09 / 3
	//     = 0.03
	got := MSE(predicted, target)
	want := 0.03
	if math.Abs(got-want) > 1e-10 {
		t.Errorf("expected MSE = %v, got %v", want, got)
	}
}

func TestMSE_PerfectPredictionReturnsZero(t *testing.T) {
	predicted := []float64{1.0, 0.0, 0.5}
	target := []float64{1.0, 0.0, 0.5}

	got := MSE(predicted, target)
	if got != 0.0 {
		t.Errorf("expected MSE = 0.0 for perfect prediction, got %v", got)
	}
}

func TestMSEDerivative_ReturnsPerElementDerivative(t *testing.T) {
	predicted := []float64{0.8, 0.2}
	target := []float64{1.0, 0.0}
	// derivative = 2*(predicted - target) / n
	// [2*(0.8-1.0)/2, 2*(0.2-0.0)/2] = [-0.2, 0.2]
	got := MSEDerivative(predicted, target)
	want := []float64{-0.2, 0.2}
	for i := range got {
		if math.Abs(got[i]-want[i]) > 1e-10 {
			t.Errorf("expected MSEDerivative[%d] = %v, got %v", i, want[i], got[i])
		}
	}
}

func TestMSEDerivative_ZeroDifferenceReturnsZero(t *testing.T) {
	predicted := []float64{0.5, 0.5}
	target := []float64{0.5, 0.5}

	got := MSEDerivative(predicted, target)
	for i, v := range got {
		if v != 0.0 {
			t.Errorf("expected MSEDerivative[%d] = 0.0 for zero difference, got %v", i, v)
		}
	}
}