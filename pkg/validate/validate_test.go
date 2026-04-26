package validate

import "testing"

func TestRowsUniform_RaggedPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for ragged rows")
		}
	}()
	RowsUniform([][]float64{{1, 2, 3}, {4, 5}}, 3)
}

func TestRowsUniform_UniformPasses(t *testing.T) {
	RowsUniform([][]float64{{1, 2}, {3, 4}}, 2)
}

func TestMultiplyCompatible_MismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for incompatible dimensions")
		}
	}()
	MultiplyCompatible(2, 3, 2, 4)
}

func TestMultiplyCompatible_MatchPasses(t *testing.T) {
	MultiplyCompatible(2, 3, 3, 4)
}

func TestSameDimensions_MismatchPanics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic for different dimensions")
		}
	}()
	SameDimensions(2, 3, 2, 4)
}

func TestSameDimensions_MatchPasses(t *testing.T) {
	SameDimensions(2, 3, 2, 3)
}
