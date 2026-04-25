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
