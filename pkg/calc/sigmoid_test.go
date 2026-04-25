package calc

import (
	"math"
	"testing"
)

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

func TestSigmoidDerivative_EqualsSigmoidTimesOneMinusSigmoid(t *testing.T) {
	cases := []float64{0.0, 1.0, -1.0, 2.5, -2.5}
	for _, x := range cases {
		got := SigmoidDerivative(x)
		s := Sigmoid(x)
		want := s * (1 - s)
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("SigmoidDerivative(%v) = %v, want %v", x, got, want)
		}
	}
}
