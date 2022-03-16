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
	if !Equal1D(c, expected) {
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
		if !Equal2D(c, tt.expected) {
			t.Errorf("Expected %v, got %v", tt.expected, c)
		}
	}
}
