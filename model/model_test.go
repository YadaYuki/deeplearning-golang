package model

import (
	"math/rand"
	"testing"

	"github.com/vorduin/nune"
)

func TestPredict(t *testing.T) {
	weightInitilizer := func() float64 {
		return rand.Float64()
	}
	batchSize := 10
	inputDim := 15
	hiddenDim := 5
	outDim := 10
	net := NewTwoLayerNet(inputDim, hiddenDim, outDim, weightInitilizer, weightInitilizer)
	out := net.Predict(nune.Ones[float64](batchSize, inputDim))
	if out.Shape()[0] != batchSize || out.Shape()[1] != outDim {
		t.Errorf("out.Shape() != []int{batchSize, outDim}")
	}
	for i := 0; i < batchSize; i++ {
		for j := 0; j < outDim; j++ {
			if out.Index(i, j).Scalar() > 1.0 {
				t.Errorf("out.Index(%d, %d).Scalar() > 1.0", i, j)
			}
		}
	}
}

func TestTrainStep(t *testing.T) {
	weightInitilizer := func() float64 {
		return rand.Float64()
	}
	batchSize := 10
	inputDim := 15
	hiddenDim := 5
	outDim := 10
	net := NewTwoLayerNet(inputDim, hiddenDim, outDim, weightInitilizer, weightInitilizer)
	x := nune.Ones[float64](batchSize, inputDim)
	tt := nune.FromBuffer([]float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}).Reshape(batchSize, outDim)
	loss := net.TrainStep(x, tt)
	if loss == 0.0 {
		t.Errorf("loss = 0.0")
	}
}
