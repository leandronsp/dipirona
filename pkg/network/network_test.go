package network

import (
	"dipirona/pkg/model"
	"dipirona/pkg/model/layer"
	"dipirona/pkg/random"
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

// --- Train tests ---

func TestTrain_SingleEpochReturnsLoss(t *testing.T) {
	// Simple 1-layer network
	l1 := layer.New(model.New([][]float64{{0.5}, {0.3}}), model.ActivationSigmoid)
	mlp := New([]layer.Layer{l1}, WithLearningRate(0.5))

	input := model.New([][]float64{{1.0, 1.0}})
	target := model.New([][]float64{{1.0}})

	loss := mlp.Train(input, target)

	if loss < 0 {
		t.Errorf("expected non-negative loss, got %v", loss)
	}
}

func TestTrain_LossDecreasesOverEpochs(t *testing.T) {
	// 2-layer network
	l1 := layer.New(model.New([][]float64{
		{0.5, -0.2},
		{0.1, 0.8},
	}), model.ActivationSigmoid)
	l2 := layer.New(model.New([][]float64{
		{0.4},
		{0.3},
	}), model.ActivationSigmoid)
	mlp := New([]layer.Layer{l1, l2}, WithLearningRate(0.5))

	input := model.New([][]float64{{1.0, 1.0}})
	target := model.New([][]float64{{1.0}})

	firstLoss := mlp.Train(input, target)

	// Train a few more times, loss should decrease
	var lastLoss float64 = firstLoss
	for i := 0; i < 10; i++ {
		loss := mlp.Train(input, target)
		if loss > lastLoss+1e-10 {
			// Loss may not always decrease on every single step due to the simplicity of the test,
			// but the general trend should be downward.
			// We don't make a hard assertion here for a single step.
		}
		lastLoss = loss
	}

	// After several epochs, loss should be meaningfully less than first loss
	if lastLoss >= firstLoss {
		t.Errorf("expected loss to decrease over training: first=%v, last=%v", firstLoss, lastLoss)
	}
}

func TestTrain_XORConvergence(t *testing.T) {
	// The canonical integration test: train a 2->2->1 MLP on XOR
	// to convergence (all outputs within 0.1 of targets within 10000 epochs).
	// Uses batch training (all 4 patterns per epoch) for stable convergence.

	// Network: 2 inputs -> 2 hidden (Sigmoid) -> 1 output (Sigmoid)
	w1 := random.NewWithSeed(2, 2, 1)
	w2 := random.NewWithSeed(2, 1, 101)

	l1 := layer.New(w1, model.ActivationSigmoid)
	l2 := layer.New(w2, model.ActivationSigmoid)

	mlp := New([]layer.Layer{l1, l2}, WithLearningRate(5.0))

	// Batch XOR training: all 4 patterns as a single matrix per epoch
	batchInput := model.New([][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}})
	batchTarget := model.New([][]float64{{0}, {1}, {1}, {0}})

	maxEpochs := 10000
	tolerance := 0.1

	for epoch := 0; epoch < maxEpochs; epoch++ {
		mlp.Train(batchInput, batchTarget)
	}

	// Verify convergence on each pattern individually
	xorInputs := [][][]float64{{{0, 0}}, {{0, 1}}, {{1, 0}}, {{1, 1}}}
	xorTargets := []float64{0, 1, 1, 0}

	for i := 0; i < 4; i++ {
		predicted := mlp.Predict(model.New(xorInputs[i]))
		target := xorTargets[i]
		pred := predicted.At(0, 0)
		if math.Abs(pred-target) > tolerance {
			t.Errorf("XOR not converged after %d epochs: input=%v, target=%.1f, got=%.6f",
					maxEpochs, xorInputs[i], target, pred)
		}
	}
}