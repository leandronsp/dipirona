package calc

import "testing"

func TestRelu_PositiveReturnsInput(t *testing.T) {
	got := Relu(2.5)
	if got != 2.5 {
		t.Errorf("expected Relu(2.5) = 2.5, got %v", got)
	}
}

func TestRelu_ZeroReturnsZero(t *testing.T) {
	got := Relu(0)
	if got != 0 {
		t.Errorf("expected Relu(0) = 0, got %v", got)
	}
}

func TestRelu_NegativeReturnsZero(t *testing.T) {
	got := Relu(-3.0)
	if got != 0 {
		t.Errorf("expected Relu(-3.0) = 0, got %v", got)
	}
}

func TestReluDerivative_PositiveReturnsOne(t *testing.T) {
	got := ReluDerivative(2.5)
	if got != 1.0 {
		t.Errorf("expected ReluDerivative(2.5) = 1.0, got %v", got)
	}
}

func TestReluDerivative_ZeroReturnsZero(t *testing.T) {
	got := ReluDerivative(0)
	if got != 0 {
		t.Errorf("expected ReluDerivative(0) = 0, got %v", got)
	}
}

func TestReluDerivative_NegativeReturnsZero(t *testing.T) {
	got := ReluDerivative(-1.0)
	if got != 0 {
		t.Errorf("expected ReluDerivative(-1.0) = 0,0, got %v", got)
	}
}
