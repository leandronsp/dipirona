// Package network provides a feedforward MLP with backpropagation training.
package network

import (
	"dipirona/pkg/calc"
	"dipirona/pkg/model"
	"dipirona/pkg/model/layer"
)

// MLP represents a multi-layer perceptron.
type MLP struct {
	layers       []layer.Layer
	learningRate float64
}

// Option configures an MLP during construction.
type Option func(*MLP)

// WithLearningRate sets the learning rate for training. Default is 1.0.
func WithLearningRate(lr float64) Option {
	return func(m *MLP) {
		m.learningRate = lr
	}
}

// New creates an MLP with the given layers and options.
// Panics if the layer list is empty.
func New(layers []layer.Layer, opts ...Option) MLP {
	if len(layers) == 0 {
		panic("network: at least one layer is required")
	}

	m := MLP{
		layers:       layers,
		learningRate: 1.0,
	}

	for _, opt := range opts {
		opt(&m)
	}

	return m
}

// Layers returns the MLP's layers.
func (m MLP) Layers() []layer.Layer {
	return m.layers
}

// LearningRate returns the MLP's learning rate.
func (m MLP) LearningRate() float64 {
	return m.learningRate
}

// Predict chains Forward across all layers and returns the final output.
func (m MLP) Predict(input model.Matrix) model.Matrix {
	output := input
	for i := range m.layers {
		output = m.layers[i].Forward(output)
	}
	return output
}

// Train runs one forward pass, computes MSE loss, backpropagates gradients,
// updates weights, and returns the loss value.
func (m *MLP) Train(input, target model.Matrix) float64 {
	// Forward pass
	output := m.Predict(input)

	// Compute MSE loss
	predicted := flatten(output)
	targets := flatten(target)
	loss := calc.MSE(predicted, targets)

	// Compute output delta: MSE derivative element-wise * activation derivative
	lossDeriv := calc.MSEDerivative(predicted, targets)
	delta := model.New(unflatten(lossDeriv, output.Rows, output.Cols))

	// Backward pass (reverse order)
	for i := len(m.layers) - 1; i >= 0; i-- {
		delta = m.layers[i].Backward(delta, m.learningRate)
	}

	return loss
}

// flatten converts a Matrix to a []float64 in row-major order.
func flatten(m model.Matrix) []float64 {
	result := make([]float64, m.Rows*m.Cols)
	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			result[r*m.Cols+c] = m.At(r, c)
		}
	}
	return result
}

// unflatten converts a []float64 to a 2D slice with the given dimensions.
func unflatten(data []float64, rows, cols int) [][]float64 {
	result := make([][]float64, rows)
	for r := 0; r < rows; r++ {
		result[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			result[r][c] = data[r*cols+c]
		}
	}
	return result
}