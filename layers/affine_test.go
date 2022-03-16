package layers

import (
	"testing"

	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

func TestAffineForward(t *testing.T) {
	W := nune.FromBuffer([]int{1, 2, 3, 4, 5, 6}).Reshape(3, 2)
	B := nune.FromBuffer([]int{1, 1})
	X := nune.FromBuffer([]int{1, 2, 3, 1, 2, 3, 1, 2, 3}).Reshape(3, 3)
	affine := NewAffine(&W, &B)
	expected := nune.FromBuffer([]int{23, 29, 23, 29, 23, 29}).Reshape(3, 2)
	Y := affine.Forward(X)
	if !utils.Equal2D(Y, expected, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected, Y)
	}
}

func TestAffineBackward(t *testing.T) {
	W := nune.FromBuffer([]int{1, 2, 3, 4, 5, 6}).Reshape(3, 2)
	B := nune.FromBuffer([]int{1, 1})
	X := nune.FromBuffer([]int{1, 2, 3, 1, 2, 3, 1, 2, 3}).Reshape(3, 3)
	affine := NewAffine(&W, &B)
	expected := nune.FromBuffer([]int{23, 29, 23, 29, 23, 29}).Reshape(3, 2)
	Y := affine.Forward(X)
	if !utils.Equal2D(Y, expected, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected, Y)
	}
}
