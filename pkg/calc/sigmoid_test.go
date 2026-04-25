package calc

import "testing"

func TestSigmoid_ZeroReturnsHalf(t *testing.T) {
	got := Sigmoid(0.0)
	if got != 0.5 {
		t.Errorf("expected Sigmoid(0.0) = 0.5, got %v", got)
	}
}
