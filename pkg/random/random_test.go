package random

import (
	"testing"
)

func TestNew_CorrectDimensions(t *testing.T) {
	m := New(3, 4)

	if m.Rows != 3 {
		t.Errorf("expected Rows = 3, got %d", m.Rows)
	}
	if m.Cols != 4 {
		t.Errorf("expected Cols = 4, got %d", m.Cols)
	}
}

func TestNew_ValuesInRange(t *testing.T) {
	m := New(100, 100)

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			v := m.At(r, c)
			if v <= -0.5 || v >= 0.5 {
				t.Errorf("value at (%d,%d) = %v, expected in (-0.5, 0.5)", r, c, v)
			}
		}
	}
}

func TestNewWithSeed_DeterministicWithSameSeed(t *testing.T) {
	m1 := NewWithSeed(4, 5, 42)
	m2 := NewWithSeed(4, 5, 42)

	for r := 0; r < 4; r++ {
		for c := 0; c < 5; c++ {
			if m1.At(r, c) != m2.At(r, c) {
				t.Errorf("deterministic mismatch at (%d,%d): %v vs %v", r, c, m1.At(r, c), m2.At(r, c))
			}
		}
	}
}

func TestNewWithSeed_DifferentSeedsProduceDifferentMatrices(t *testing.T) {
	m1 := NewWithSeed(4, 5, 42)
	m2 := NewWithSeed(4, 5, 99)

	different := false
	for r := 0; r < 4; r++ {
		for c := 0; c < 5; c++ {
			if m1.At(r, c) != m2.At(r, c) {
				different = true
				break
			}
		}
		if different {
			break
		}
	}

	if !different {
		t.Errorf("expected different seeds to produce different matrices")
	}
}

func TestNewWithSeed_ValuesInRange(t *testing.T) {
	m := NewWithSeed(50, 50, 1)

	for r := 0; r < m.Rows; r++ {
		for c := 0; c < m.Cols; c++ {
			v := m.At(r, c)
			if v <= -0.5 || v >= 0.5 {
				t.Errorf("value at (%d,%d) = %v, expected in (-0.5, 0.5)", r, c, v)
			}
		}
	}
}