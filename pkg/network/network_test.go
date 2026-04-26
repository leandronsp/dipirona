package network

import (
	"dipirona/pkg/model"
	"dipirona/pkg/model/layer"
	"math"
	"testing"
)

// --- network.New tests ---

func TestNew_CreatesMLPWithLayersAndDefaultLearningRate(t *testing.T) {
	l1 := layer.New(model.New([][]float64{{0.5, -0.2}, {0.1, 0.8}}), model.ActivationSigmoid)
	l2 := layer.New(model.New([][]float64{{0.4}, {0.3}}), model.ActivationSigmoid)

	mlp := New([]layer.Layer{l1, l2})

	if mlp.LearningRate() != 1.0 {
		t.Errorf("expected default learning rate 1.0, got %v", mlp.LearningRate())
	}
	if len(mlp.Layers()) != 2 {
		t.Errorf("expected 2 layers, got %d", len(mlp.Layers()))
	}
}

func TestNew_WithLearningRateOption(t *testing.T) {
	l1 := layer.New(model.New([][]float64{{0.5, -0.2}, {0.1, 0.8}}), model.ActivationSigmoid)

	mlp := New([]layer.Layer{l1}, WithLearningRate(0.5))

	if mlp.LearningRate() != 0.5 {
		t.Errorf("expected learning rate 0.5, got %v", mlp.LearningRate())
	}
}

func TestNew_EmptyLayerListPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic for empty layer list")
		}
		msg, ok := r.(string)
		if !ok || msg == "" {
			t.Errorf("expected non-empty panic message, got %v", r)
		}
	}()

	New([]layer.Layer{})
}

// --- Predict tests ---

func TestPredict_ChainsForwardAcrossLayers(t *testing.T) {
	// 2 layers: input(1x2) -> hidden(2x2, Sigmoid) -> output(2x1, Sigmoid)
	l1 := layer.New(model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	}), model.ActivationSigmoid)
	l2 := layer.New(model.New([][]float64{
		{0.4},
		{0.3},
	}), model.ActivationSigmoid)

	mlp := New([]layer.Layer{l1, l2})

	input := model.New([][]float64{{1.0, 1.0}})
	output := mlp.Predict(input)

	if output.Rows != 1 || output.Cols != 1 {
		t.Fatalf("expected 1x1 output, got %dx%d", output.Rows, output.Cols)
	}

	// Manual calculation:
	// hidden = sigmoid([1.0, 1.0] * [[0.5, -0.2], [0.1, 0.8]])
	//       = sigmoid([0.6, 0.6])
	//       = [sigmoid(0.6), sigmoid(0.6)]
	sig06 := 1.0 / (1.0 + math.Exp(-0.6))
	// output = sigmoid([sig06, sig06] * [[0.4], [0.3]])
	//        = sigmoid(sig06*0.4 + sig06*0.3)
	//        = sigmoid(sig06 * 0.7)
	preOut := sig06 * 0.7
	expected := 1.0 / (1.0+math.Exp(-preOut))

	if math.Abs(output.At(0, 0)-expected) > 1e-10 {
		t.Errorf("expected output %v, got %v", expected, output.At(0, 0))
	}
}

func TestPredict_WrongDimensionInputPanics(t *testing.T) {
	// Hidden layer expects 2 inputs (2 rows in weights)
	l1 := layer.New(model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	}), model.ActivationSigmoid)

	mlp := New([]layer.Layer{l1})

	// Input has 3 columns but layer expects 2
	input := model.New([][]float64{{1.0, 1.0, 1.0}})

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for dimension mismatch")
		}
	}()

	mlp.Predict(input)
}