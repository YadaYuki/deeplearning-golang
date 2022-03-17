package layers

import (
	"testing"

	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

func TestSoftmaxWithLossForward(t *testing.T) {
	testCases := []struct {
		x            nune.Tensor[float64]
		t            nune.Tensor[float64]
		expectedLoss float64
	}{
		{
			x:            nune.FromBuffer([]float64{0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1}).Reshape(3, 3),
			t:            nune.FromBuffer([]float64{0, 1, 0, 0, 1, 0, 0, 1, 0}).Reshape(3, 3),
			expectedLoss: 1.10194,
		},
		{
			x:            nune.FromBuffer([]float64{0.3, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1}).Reshape(3, 4),
			t:            nune.FromBuffer([]float64{0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}).Reshape(3, 4),
			expectedLoss: 1.36477,
		},
	}
	for _, tt := range testCases {
		softmaxWithLoss := NewSoftmaxWithLoss[float64]()
		loss := softmaxWithLoss.Forward(tt.x, tt.t)
		if !utils.EqualFloat(loss, tt.expectedLoss, 0.0001) {
			t.Errorf("Expected %v, got %v", tt.expectedLoss, loss)
		}
	}
}

func TestSoftmaxWithLossBackward(t *testing.T) {
	testCases := []struct {
		x          nune.Tensor[float64]
		t          nune.Tensor[float64]
		expectedDx nune.Tensor[float64]
		equal      func(a, b nune.Tensor[float64]) bool
	}{
		{
			x:          nune.FromBuffer([]float64{0.3, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3, 0.2, 0.1}).Reshape(3, 3),
			t:          nune.FromBuffer([]float64{0, 1, 0, 0, 1, 0, 0, 1, 0}).Reshape(3, 3),
			expectedDx: nune.FromBuffer([]float64{0.12238, -0.22259, 0.10020, 0.12238, -0.22259, 0.10020, 0.12238, -0.22259, 0.10020}).Reshape(3, 3),
			equal: func(a, b nune.Tensor[float64]) bool {
				return utils.Equal2D(a, b, func(a, b float64) bool { return utils.EqualFloat(a, b, 1e-5) })
			},
		},
		{
			x:          nune.FromBuffer([]float64{0.3, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1, 0.3, 0.2, 0.1, 0.1}).Reshape(3, 4),
			t:          nune.FromBuffer([]float64{0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0}).Reshape(3, 4),
			expectedDx: nune.FromBuffer([]float64{0.09410, -0.24818, 0.07704, 0.07704, 0.09410, -0.24818, 0.0770, 0.0770, 0.09410, -0.24818, 0.07704, 0.07704}).Reshape(3, 4),
			equal: func(a, b nune.Tensor[float64]) bool {
				return utils.Equal2D(a, b, func(a, b float64) bool { return utils.EqualFloat(a, b, 1e-4) })
			},
		},
	}
	for _, tt := range testCases {
		softmaxWithLoss := NewSoftmaxWithLoss[float64]()
		softmaxWithLoss.Forward(tt.x, tt.t)
		dx := softmaxWithLoss.Backward(nune.Zeros[float64](1))
		if !tt.equal(tt.expectedDx, dx) {
			t.Errorf("Expected %v, got %v", tt.expectedDx, dx)
		}
	}
}
