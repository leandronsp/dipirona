package layer

import (
	"dipirona/pkg/calc"
	"dipirona/pkg/model"
	"math"
	"testing"
)

func TestForward_CachesInputAndPreActivation(t *testing.T) {
	w := model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	})
	l := New(w, model.ActivationNone)

	input := model.New([][]float64{
		{1.0, 1.0},
	})

	l.Forward(input)

	// Pre-activation (z) = input * weights = [0.6, 0.6]
	z := l.PreActivation()
	if z.Rows != 1 || z.Cols != 2 {
		t.Fatalf("expected z dimensions 1x2, got %dx%d", z.Rows, z.Cols)
	}
	if math.Abs(z.At(0, 0)-0.6) > 1e-10 {
		t.Errorf("expected z.At(0,0) = 0.6, got %v", z.At(0, 0))
	}

	// Cached input
	cachedInput := l.Input()
	if cachedInput.Rows != 1 || cachedInput.Cols != 2 {
		t.Fatalf("expected input dimensions 1x2, got %dx%d", cachedInput.Rows, cachedInput.Cols)
	}
	if cachedInput.At(0, 0) != 1.0 || cachedInput.At(0, 1) != 1.0 {
		t.Errorf("expected cached input [1.0, 1.0], got %v", cachedInput)
	}
}

func TestPreActivation_BeforeForwardPanics(t *testing.T) {
	w := model.New([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	l := New(w, model.ActivationNone)

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic when calling PreActivation before Forward")
		}
		msg, ok := r.(string)
		if !ok || msg != "pre-activation not available: call Forward first" {
			t.Errorf("expected panic message 'pre-activation not available: call Forward first', got %v", r)
		}
	}()

	l.PreActivation()
}

func TestInput_BeforeForwardPanics(t *testing.T) {
	w := model.New([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	l := New(w, model.ActivationNone)

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic when calling Input before Forward")
		}
		msg, ok := r.(string)
		if !ok || msg != "input not available: call Forward first" {
			t.Errorf("expected panic message 'input not available: call Forward first', got %v", r)
		}
	}()

	l.Input()
}

func TestBackward_UpdatesWeightsAndReturnsGradient(t *testing.T) {
	// Layer with None activation so we can compute exact values
	// weights: [[0.5, -0.2], [0.1, 0.8]] (2x2)
	// input: [[1.0, 1.0]] (1x2)
	// z = input * weights = [0.6, 0.6] (1x2)  — pre-activation
	// output = z since activation is None
	w := model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	})
	l := New(w, model.ActivationNone)

	input := model.New([][]float64{
		{1.0, 1.0},
	})
	l.Forward(input)

	// output delta: [[0.1, -0.3]] (1x2)
	delta := model.New([][]float64{
		{0.1, -0.3},
	})

	lr := 0.5
	grad := l.Backward(delta, lr)

	// For None activation: derivative = 1, so delta passes through
	// dW = input.T * delta = [[1], [1]] * [[0.1, -0.3]] = [[0.1, -0.3], [0.1, -0.3]] (2x2)
	// new weights = W - lr * dW = [[0.5-0.05, -0.2+0.15], [0.1-0.05, 0.8+0.15]]
	//             = [[0.45, -0.05], [0.05, 0.95]]

	newWeights := l.Weights()
	if math.Abs(newWeights.At(0, 0)-0.45) > 1e-10 {
		t.Errorf("expected new W(0,0) = 0.45, got %v", newWeights.At(0, 0))
	}
	if math.Abs(newWeights.At(0, 1)-(-0.05)) > 1e-10 {
		t.Errorf("expected new W(0,1) = -0.05, got %v", newWeights.At(0, 1))
	}
	if math.Abs(newWeights.At(1, 0)-0.05) > 1e-10 {
		t.Errorf("expected new W(1,0) = 0.05, got %v", newWeights.At(1, 0))
	}
	if math.Abs(newWeights.At(1, 1)-0.95) > 1e-10 {
		t.Errorf("expected new W(1,1) = 0.95, got %v", newWeights.At(1, 1))
	}

	// grad for previous layer = delta * weights.T
	// delta (1x2) * weights.T (2x2) = [0.1*0.5+(-0.3)*(-0.2), 0.1*0.1+(-0.3)*0.8]
	//                               = [0.05+0.06, 0.01-0.24] = [0.11, -0.23]
	if math.Abs(grad.At(0, 0)-0.11) > 1e-10 {
		t.Errorf("expected grad.At(0,0) = 0.11, got %v", grad.At(0, 0))
	}
	if math.Abs(grad.At(0, 1)-(-0.23)) > 1e-10 {
		t.Errorf("expected grad.At(0,1) = -0.23, got %v", grad.At(0, 1))
	}
}

func TestBackward_WithSigmoidActivation(t *testing.T) {
	// weights: [[0.5]] (1x1), activation: Sigmoid
	// input: [[0.5]] (1x1)
	// z = 0.5 * 0.5 = 0.25, output = sigmoid(0.25) ≈ 0.56218
	w := model.New([][]float64{{0.5}})
	l := New(w, model.ActivationSigmoid)

	input := model.New([][]float64{{0.5}})
	l.Forward(input)

	// delta from next layer: [[1.0]]
	delta := model.New([][]float64{{1.0}})

	lr := 1.0
	grad := l.Backward(delta, lr)

	z := l.PreActivation()
	sigDeriv := calc.SigmoidDerivative(z.At(0, 0))

	// local_delta = delta * sigmoid'(z) = 1.0 * sigDeriv
	// dW = input.T * local_delta = [[0.5]] * [[sigDeriv]] = [[0.5*sigDeriv]]
	// new weight = 0.5 - 1.0 * 0.5*sigDeriv
	expectedNewW := 0.5 - 1.0*0.5*sigDeriv
	newW := l.Weights()
	if math.Abs(newW.At(0, 0)-expectedNewW) > 1e-10 {
		t.Errorf("expected new W(0,0) = %v, got %v", expectedNewW, newW.At(0, 0))
	}

	// grad = local_delta * weights.T = sigDeriv * 0.5
	expectedGrad := sigDeriv * 0.5
	if math.Abs(grad.At(0, 0)-expectedGrad) > 1e-10 {
		t.Errorf("expected grad.At(0,0) = %v, got %v", expectedGrad, grad.At(0, 0))
	}
}

func TestBackward_BeforeForwardPanics(t *testing.T) {
	w := model.New([][]float64{{1.0, 2.0}, {3.0, 4.0}})
	l := New(w, model.ActivationNone)

	delta := model.New([][]float64{{1.0, 1.0}})

	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic when calling Backward before Forward")
		}
		msg, ok := r.(string)
		if !ok || msg != "backward not available: call Forward first" {
			t.Errorf("expected panic message 'backward not available: call Forward first', got %v", r)
		}
	}()

	l.Backward(delta, 1.0)
}