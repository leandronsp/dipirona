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

func TestTranspose_SwapsRowsAndCols(t *testing.T) {
	m := New([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})

	tr := m.Transpose()

	if tr.Rows != 3 {
		t.Errorf("expected Rows = 3, got %d", tr.Rows)
	}
	if tr.Cols != 2 {
		t.Errorf("expected Cols = 2, got %d", tr.Cols)
	}
	if got := tr.At(0, 1); got != 4 {
		t.Errorf("expected At(0,1) = 4, got %v", got)
	}
	if got := tr.At(2, 0); got != 3 {
		t.Errorf("expected At(2,0) = 3, got %v", got)
	}
}

func TestMultiply_CompatibleDimensions(t *testing.T) {
	a := New([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	b := New([][]float64{
		{7, 8},
		{9, 10},
		{11, 12},
	})

	result := a.Multiply(b)

	if result.Rows != 2 {
		t.Errorf("expected Rows = 2, got %d", result.Rows)
	}
	if result.Cols != 2 {
		t.Errorf("expected Cols = 2, got %d", result.Cols)
	}
	if got := result.At(0, 0); got != 58 {
		t.Errorf("expected At(0,0) = 58, got %v", got)
	}
	if got := result.At(1, 1); got != 154 {
		t.Errorf("expected At(1,1) = 154, got %v", got)
	}
}
