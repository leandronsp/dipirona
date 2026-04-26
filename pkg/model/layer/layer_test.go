package layer

import (
	"dipirona/pkg/model"
	"math"
	"testing"
)

func TestNew_CreatesLayerWithWeightsAndActivation(t *testing.T) {
	w := model.New([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	l := New(w, model.ActivationSigmoid)

	if l.Weights().At(0, 0) != 1.0 {
		t.Errorf("expected Weights().At(0,0) = 1.0, got %v", l.Weights().At(0, 0))
	}
}

func TestForward_WithNoneActivation_ReturnsRawMatmul(t *testing.T) {
	// input: [[1.0, 1.0]]  (1x2)
	// weights: [[0.5, -0.2], [0.1, 0.8]]  (2x2)
	// matmul: [1*0.5+1*0.1, 1*(-0.2)+1*0.8] = [0.6, 0.6]
	w := model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	})
	l := New(w, model.ActivationNone)

	input := model.New([][]float64{
		{1.0, 1.0},
	})

	result := l.Forward(input)

	if result.Rows != 1 || result.Cols != 2 {
		t.Fatalf("expected 1x2, got %dx%d", result.Rows, result.Cols)
	}
	if math.Abs(result.At(0, 0)-0.6) > 1e-10 {
		t.Errorf("expected At(0,0) = 0.6, got %v", result.At(0, 0))
	}
	if math.Abs(result.At(0, 1)-0.6) > 1e-10 {
		t.Errorf("expected At(0,1) = 0.6, got %v", result.At(0, 1))
	}
}

func TestForward_WithSigmoidActivation_AppliesSigmoid(t *testing.T) {
	// Same as None test: matmul = [0.6, 0.6], then sigmoid applied
	w := model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	})
	l := New(w, model.ActivationSigmoid)

	input := model.New([][]float64{
		{1.0, 1.0},
	})

	result := l.Forward(input)

	expected := 1.0 / (1.0 + math.Exp(-0.6))

	if math.Abs(result.At(0, 0)-expected) > 1e-10 {
		t.Errorf("expected sigmoid(0.6), got %v", result.At(0, 0))
	}
	if math.Abs(result.At(0, 1)-expected) > 1e-10 {
		t.Errorf("expected sigmoid(0.6), got %v", result.At(0, 1))
	}
}

func TestForward_WithReLUActivation_ClampsNegatives(t *testing.T) {
	// input: [[2.0, 0.5]]  (1x2)
	// weights: [[1.0, -1.0], [0.5, 0.5]]  (2x2)
	// matmul: [2*1+0.5*0.5, 2*(-1)+0.5*0.5] = [2.25, -1.75]
	// ReLU: [2.25, 0.0]
	w := model.New([][]float64{
		{1.0, -1.0},
		{0.5, 0.5},
	})
	l := New(w, model.ActivationReLU)

	input := model.New([][]float64{
		{2.0, 0.5},
	})

	result := l.Forward(input)

	if result.At(0, 0) != 2.25 {
		t.Errorf("expected At(0,0) = 2.25, got %v", result.At(0, 0))
	}
	if result.At(0, 1) != 0.0 {
		t.Errorf("expected At(0,1) = 0.0, got %v", result.At(0, 1))
	}
}

func TestOutput_ReturnsCachedForwardResult(t *testing.T) {
	w := model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	})
	l := New(w, model.ActivationNone)

	input := model.New([][]float64{
		{1.0, 1.0},
	})

	forwardResult := l.Forward(input)
	cachedResult := l.Output()

	if cachedResult.Rows != forwardResult.Rows || cachedResult.Cols != forwardResult.Cols {
		t.Fatalf("expected Output dimensions %dx%d, got %dx%d", forwardResult.Rows, forwardResult.Cols, cachedResult.Rows, cachedResult.Cols)
	}
	if cachedResult.At(0, 0) != forwardResult.At(0, 0) {
		t.Errorf("expected Output().At(0,0) = %v, got %v", forwardResult.At(0, 0), cachedResult.At(0, 0))
	}
	if cachedResult.At(0, 1) != forwardResult.At(0, 1) {
		t.Errorf("expected Output().At(0,1) = %v, got %v", forwardResult.At(0, 1), cachedResult.At(0, 1))
	}
}

func TestOutput_BeforeForwardPanics(t *testing.T) {
	w := model.New([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	l := New(w, model.ActivationNone)

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic when calling Output before Forward")
		}
		msg, ok := r.(string)
		if !ok || msg != "output not available: call Forward first" {
			t.Errorf("expected panic message 'output not available: call Forward first', got %v", r)
		}
	}()

	l.Output()
}

func TestForward_DimensionMismatchPanics(t *testing.T) {
	// weights: 3 rows, 2 cols
	// input: 1 row, 1 col — input.Cols=1 != weights.Rows=3 → mismatch
	w := model.New([][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
		{5.0, 6.0},
	})
	l := New(w, model.ActivationNone)

	input := model.New([][]float64{
		{1.0},
	})

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for dimension mismatch")
		}
	}()

	l.Forward(input)
}
