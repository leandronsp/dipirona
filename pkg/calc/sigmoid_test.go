package calc

import "testing"

func TestSigmoid_ZeroReturnsHalf(t *testing.T) {
	got := Sigmoid(0.0)
	if got != 0.5 {
		t.Errorf("expected Sigmoid(0.0) = 0.5, got %v", got)
	}
}

func TestSigmoid_LargePositiveApproachesOne(t *testing.T) {
	got := Sigmoid(10.0)
	if got <= 0.999 {
		t.Errorf("expected Sigmoid(10.0) > 0.999, got %v", got)
	}
}

func TestSigmoid_LargeNegativeApproachesZero(t *testing.T) {
	got := Sigmoid(-10.0)
	if got >= 0.001 {
		t.Errorf("expected Sigmoid(-10.0) < 0.001, got %v", got)
	}
}
