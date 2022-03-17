package layers

import (
	"testing"

	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

func TestReluForward(t *testing.T) {
	a := nune.Range[float64](-4, 5, 1).Reshape(3, 3)
	expected := nune.FromBuffer([]float64{0, 0, 0, 0, 0, 1, 2, 3, 4}).Reshape(3, 3)
	relu := NewRelu[float64]()
	out := relu.Forward(a)
	if !utils.Equal2D(out, expected, func(a, b float64) bool { return utils.EqualFloat(a, b, 0.0001) }) {
		t.Errorf("Expected %v, got %v", expected, out)
	}
}

func TestReluBackward(t *testing.T) {
	type DataType float64
	a := nune.Range[DataType](-4, 5, 1).Reshape(3, 3)
	relu := NewRelu[DataType]()
	relu.Forward(a)
	dy := nune.Ones[DataType](3, 3)
	expected := nune.FromBuffer([]DataType{0, 0, 0, 0, 0, 1, 1, 1, 1}).Reshape(3, 3)
	dx := relu.Backward(dy)
	if !utils.Equal2D(dx, expected, func(a, b DataType) bool { return utils.EqualFloat(float64(a), float64(b), 0.0001) }) {
		t.Errorf("Expected %v, got %v", expected, dx)
	}
}
