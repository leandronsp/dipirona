package mat

import "testing"

func TestNew_CreatesMatrixWithCorrectDimensions(t *testing.T) {
	input := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}

	m := New(input)

	if m.Rows != 2 {
		t.Errorf("expected Rows = 2, got %d", m.Rows)
	}
	if m.Cols != 3 {
		t.Errorf("expected Cols = 3, got %d", m.Cols)
	}
}

func TestNew_DeepCopiesValues(t *testing.T) {
	input := [][]float64{
		{1, 2},
		{3, 4},
	}

	m := New(input)
	input[0][0] = 99

	if got := m.At(0, 0); got != 1.0 {
		t.Errorf("expected At(0,0) = 1.0 after mutation, got %v", got)
	}
}
