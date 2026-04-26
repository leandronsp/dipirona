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

func TestNew_RaggedRowsPanics(t *testing.T) {
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("expected panic for ragged rows")
		}
		msg, ok := r.(string)
		if !ok || msg == "" {
			t.Errorf("expected non-empty panic message, got %v", r)
		}
	}()

	New([][]float64{
		{1, 2, 3},
		{4, 5},
	})
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

func TestMultiply_IncompatibleDimensionsPanics(t *testing.T) {
	a := New([][]float64{
		{1, 2, 3},
		{4, 5, 6},
	})
	b := New([][]float64{
		{1, 2},
		{3, 4},
	})

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for incompatible dimensions")
		}
	}()

	a.Multiply(b)
}

func TestZip_AddsCorrespondingElements(t *testing.T) {
	a := New([][]float64{
		{1, 2},
		{3, 4},
	})
	b := New([][]float64{
		{10, 20},
		{30, 40},
	})

	result := a.Zip(b, func(x, y float64) float64 { return x + y })

	if got := result.At(0, 0); got != 11 {
		t.Errorf("expected At(0,0) = 11, got %v", got)
	}
	if got := result.At(1, 1); got != 44 {
		t.Errorf("expected At(1,1) = 44, got %v", got)
	}
}

func TestZip_IncompatibleDimensionsPanics(t *testing.T) {
	a := New([][]float64{
		{1, 2},
		{3, 4},
	})
	b := New([][]float64{
		{1, 2, 3},
	})

	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for incompatible dimensions")
		}
	}()

	a.Zip(b, func(x, y float64) float64 { return x + y })
}

func TestMap_AppliesScalarFunction(t *testing.T) {
	m := New([][]float64{
		{1, 2},
		{3, 4},
	})

	result := m.Map(func(x float64) float64 { return x * 2 })

	if got := result.At(0, 0); got != 2 {
		t.Errorf("expected At(0,0) = 2, got %v", got)
	}
	if got := result.At(1, 1); got != 8 {
		t.Errorf("expected At(1,1) = 8, got %v", got)
	}
}

func TestString_ReturnsReadableFormat(t *testing.T) {
	m := New([][]float64{
		{1, 2},
		{3, 4},
	})

	s := m.String()
	if s == "" {
		t.Errorf("expected non-empty string")
	}
}

func BenchmarkMultiply_10x10(b *testing.B) {
	a := New(makeMatrix(10, 10))
	c := New(makeMatrix(10, 10))
	for i := 0; i < b.N; i++ {
		a.Multiply(c)
	}
}

func BenchmarkMultiply_50x50(b *testing.B) {
	a := New(makeMatrix(50, 50))
	c := New(makeMatrix(50, 50))
	for i := 0; i < b.N; i++ {
		a.Multiply(c)
	}
}

func makeMatrix(rows, cols int) [][]float64 {
	m := make([][]float64, rows)
	for i := range m {
		m[i] = make([]float64, cols)
		for j := range m[i] {
			m[i][j] = float64(i*cols + j)
		}
	}
	return m
}
