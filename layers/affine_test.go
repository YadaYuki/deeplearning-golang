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
	affine.Forward(X)
	dx, dW, dB := affine.Backward(nune.FromBuffer([]int{1, 2, 3, 1, 2, 3}).Reshape(3, 2))

	expected_dx := nune.FromBuffer([]int{5, 11, 17, 5, 13, 21, 8, 18, 28}).Reshape(3, 3)
	expected_dW := nune.FromBuffer([]int{6, 6, 12, 12, 18, 18}).Reshape(3, 2)
	expected_dB := nune.FromBuffer([]int{6, 6})
	expected_W := nune.FromBuffer([]int{-5, -4, -9, -8, -13, -12}).Reshape(3, 2)
	expected_B := nune.FromBuffer([]int{-5, -5})

	if !utils.Equal2D(dx, expected_dx, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected_dx, dx)
	}
	if !utils.Equal2D(dW, expected_dW, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected_dW, dW)
	}
	if !utils.Equal1D(dB, expected_dB, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected_dB, dB)
	}
	if !utils.Equal2D(*affine.W, expected_W, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected_W, affine.W)
	}
	if !utils.Equal1D(*affine.B, expected_B, func(a, b int) bool { return a == b }) {
		t.Errorf("Expected %v, got %v", expected_B, affine.B)
	}
}
