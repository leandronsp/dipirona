package model

import (
	"dipirona/pkg/calc"
	"math"
	"testing"
)

func TestActivationNone_ApplyReturnsUnchanged(t *testing.T) {
	m := New([][]float64{
		{1.0, -2.0},
		{3.0, -4.0},
	})

	result := ActivationNone.Apply(m)

	if result.At(0, 0) != 1.0 || result.At(0, 1) != -2.0 || result.At(1, 0) != 3.0 || result.At(1, 1) != -4.0 {
		t.Errorf("expected identity, got %v", result)
	}
}

func TestActivationSigmoid_ApplyAppliesSigmoid(t *testing.T) {
	m := New([][]float64{
		{0.0, 1.0},
	})

	result := ActivationSigmoid.Apply(m)

	if math.Abs(result.At(0, 0)-calc.Sigmoid(0.0)) > 1e-10 {
		t.Errorf("expected Sigmoid(0.0) = 0.5, got %v", result.At(0, 0))
	}
	if math.Abs(result.At(0, 1)-calc.Sigmoid(1.0)) > 1e-10 {
		t.Errorf("expected Sigmoid(1.0), got %v", result.At(0, 1))
	}
}

func TestActivationNone_ApplyDerivativeReturnsUnchanged(t *testing.T) {
	m := New([][]float64{
		{1.0, -2.0},
		{3.0, -4.0},
	})

	result := ActivationNone.ApplyDerivative(m)

	if result.At(0, 0) != 1.0 || result.At(0, 1) != 1.0 || result.At(1, 0) != 1.0 || result.At(1, 1) != 1.0 {
		t.Errorf("expected all ones for None derivative, got %v", result)
	}
}

func TestActivationSigmoid_ApplyDerivativeAppliesSigmoidDerivative(t *testing.T) {
	m := New([][]float64{
		{0.0, 1.0},
	})

	result := ActivationSigmoid.ApplyDerivative(m)

	if math.Abs(result.At(0, 0)-calc.SigmoidDerivative(0.0)) > 1e-10 {
		t.Errorf("expected SigmoidDerivative(0.0) = %v, got %v", calc.SigmoidDerivative(0.0), result.At(0, 0))
	}
	if math.Abs(result.At(0, 1)-calc.SigmoidDerivative(1.0)) > 1e-10 {
		t.Errorf("expected SigmoidDerivative(1.0) = %v, got %v", calc.SigmoidDerivative(1.0), result.At(0, 1))
	}
}

func TestActivationReLU_ApplyDerivativeAppliesReluDerivative(t *testing.T) {
	m := New([][]float64{
		{2.5, -1.0},
		{0.0, 3.0},
	})

	result := ActivationReLU.ApplyDerivative(m)

	expected := [][]float64{
		{1.0, 0.0},
		{0.0, 1.0},
	}
	for r := 0; r < 2; r++ {
		for c := 0; c < 2; c++ {
			if result.At(r, c) != expected[r][c] {
				t.Errorf("expected At(%d,%d) = %v, got %v", r, c, expected[r][c], result.At(r, c))
			}
		}
	}
}

func TestApplyDerivative_UnknownActivationPanics(t *testing.T) {
	m := New([][]float64{{1.0}})

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic for unknown activation")
		}
		msg, ok := r.(string)
		if !ok || msg != "unknown activation function" {
			t.Errorf("expected 'unknown activation function', got %v", r)
		}
	}()

	Activation(99).ApplyDerivative(m)
}

func TestApply_UnknownActivationPanics(t *testing.T) {
	m := New([][]float64{{1.0}})

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic for unknown activation")
		}
		msg, ok := r.(string)
		if !ok || msg != "unknown activation function" {
			t.Errorf("expected 'unknown activation function', got %v", r)
		}
	}()

	Activation(99).Apply(m)
}

func TestActivationReLU_ApplyClampsNegatives(t *testing.T) {
	m := New([][]float64{
		{2.5, -1.0},
		{0.0, 3.0},
	})

	result := ActivationReLU.Apply(m)

	expected := [][]float64{
		{2.5, 0.0},
		{0.0, 3.0},
	}
	for r := 0; r < 2; r++ {
		for c := 0; c < 2; c++ {
			if result.At(r, c) != expected[r][c] {
				t.Errorf("expected At(%d,%d) = %v, got %v", r, c, expected[r][c], result.At(r, c))
			}
		}
	}
}
