package utils

import (
	"testing"

	"github.com/vorduin/nune"
)

func TestAdd(t *testing.T) {
	a := nune.Range[int](0, 10, 1)
	b := nune.Range[int](0, 10, 1)
	expected := nune.Range[int](0, 20, 2)
	c := Add(a, b)
	if !Equal1D(c, expected, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected, c)
	}
}

func TestDot(t *testing.T) {
	testCases := []struct {
		a        nune.Tensor[int]
		b        nune.Tensor[int]
		expected nune.Tensor[int]
	}{
		{a: nune.Range[int](0, 9, 1).Reshape(3, 3),
			b: nune.Range[int](0, 9, 1).Reshape(3, 3),
			expected: nune.FromBuffer([]int{
				15, 18, 21,
				42, 54, 66,
				69, 90, 111,
			}).Reshape(3, 3),
		},
		{a: nune.Range[int](0, 12, 1).Reshape(3, 4),
			b: nune.Range[int](0, 12, 1).Reshape(4, 3),
			expected: nune.FromBuffer([]int{
				42, 48, 54,
				114, 136, 158,
				186, 224, 262,
			}).Reshape(3, 3),
		},
	}
	for _, tt := range testCases {
		c := Dot(tt.a, tt.b)
		if !Equal2D(c, tt.expected, func(a, b int) bool { return a == b }) {
			t.Errorf("Expected %v, got %v", tt.expected, c)
		}
	}
}

func TestCrossEntropyErrorBatch(t *testing.T) {
	testCases := []struct {
		yBatch   nune.Tensor[float64]
		tBatch   nune.Tensor[int]
		expected float64
	}{
		{
			yBatch:   nune.FromBuffer([]float64{0.7, 0.2, 0.1}).Reshape(1, 3),
			tBatch:   nune.FromBuffer([]int{0, 0, 1}).Reshape(1, 3),
			expected: 2.3025,
		},
		{
			yBatch:   nune.FromBuffer([]float64{0.0, 0.0, 0.8, 0.2}).Reshape(1, 4),
			tBatch:   nune.FromBuffer([]int{0, 0, 1, 0}).Reshape(1, 4),
			expected: 0.2231,
		},
		{
			yBatch:   nune.FromBuffer([]float64{0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.8, 0.2}).Reshape(3, 4),
			tBatch:   nune.FromBuffer([]int{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0}).Reshape(3, 4),
			expected: 0.2231,
		},
	}
	for _, tt := range testCases {
		if !EqualFloat(CrossEntropyErrorBatch(tt.yBatch, tt.tBatch, 1e-7), tt.expected, 0.0001) {
			t.Errorf("Expected %v, got %v", tt.expected, CrossEntropyErrorBatch(tt.yBatch, tt.tBatch, 1e-7))
		}
	}
}

func TestSoftmaxBatch(t *testing.T) {
	testCases := []struct {
		xBatch   nune.Tensor[float64]
		expected nune.Tensor[float64]
	}{
		{
			xBatch:   nune.FromBuffer([]float64{1., 2., 3., 4., 5.}).Reshape(1, 5),
			expected: nune.FromBuffer([]float64{0.0117, 0.0317, 0.0861, 0.2341, 0.6364}).Reshape(1, 5),
		},
		{
			xBatch:   nune.FromBuffer([]float64{1., 2., 3., 4., 5., 1., 2., 3., 2., 1.}).Reshape(2, 5),
			expected: nune.FromBuffer([]float64{0.0117, 0.0317, 0.0861, 0.2341, 0.6364, 0.0675, 0.1834, 0.4984, 0.1834, 0.0675}).Reshape(2, 5),
		},
	}
	for _, tt := range testCases {
		output := SoftmaxBatch(tt.xBatch)
		if !Equal2D(output, tt.expected, func(a, b float64) bool { return EqualFloat(a, b, 0.0001) }) {
			t.Errorf("Expected %v, got %v", output, tt.expected)
		}
	}
}
