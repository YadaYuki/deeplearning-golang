package main

import (
	"fmt"
	"math/rand"
	"os"
	"path"

	"github.com/YadaYuki/deeplearning-golang/mnist"
	"github.com/YadaYuki/deeplearning-golang/model"
	"github.com/vorduin/nune"
)

func shuffle(a []int) []int {
	rand.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
	return a
}

func getRange(from int, to int) []int {
	a := make([]int, to-from)
	for i := from; i < to; i++ {
		a[i-from] = i
	}
	return a
}

func getBatchData[T nune.Number](idxes []int, data nune.Tensor[T]) nune.Tensor[T] {
	batchSize := len(idxes)
	dataDim := data.Shape()[1]
	batchX := nune.Zeros[T](batchSize, dataDim)
	for i, idx := range idxes {
		for j := 0; j < dataDim; j++ {
			batchX.Index(i, j).Ravel()[0] = data.Index(idx, j).Scalar()
		}
	}
	return batchX
}

func splitSlice[T any](slices []T, n int) [][]T {
	result := [][]T{}
	for i := 0; i < len(slices); i += n {
		if i+n > len(slices) {
			break
		}
		result = append(result, slices[i:i+n])

	}
	if len(slices)%n != 0 {
		result = append(result, slices[len(slices)-len(slices)%n:])
	}
	return result
}

func train[T nune.Number](net *model.TwoLayerNet[T], xTrain nune.Tensor[T], tTrain nune.Tensor[T], batchSize int) {
	dataSize := xTrain.Shape()[0]
	batchIdxes := splitSlice(shuffle(getRange(0, dataSize)), batchSize)
	for epoch := 0; epoch < 10; epoch++ {
		for i := 0; i < len(batchIdxes); i++ {
			xBatch := getBatchData(batchIdxes[i], xTrain)
			tBatch := getBatchData(batchIdxes[i], tTrain)
			loss := net.TrainStep(xBatch, tBatch)
			fmt.Println("train loss: \n", loss)
		}
	}
}

func main() {
	wd, _ := os.Getwd()
	pathToMnistDir := path.Join(wd, "mnist/data")
	xTrain, tTrain, _, _, _ := mnist.LoadMnist[float64](pathToMnistDir, true)
	weightInitilizer := func() float64 {
		return rand.Float64() * 0.01
	}
	biasInitilizer := func() float64 {
		return rand.Float64()
	}
	net := model.NewTwoLayerNet(784, 50, 10, weightInitilizer, biasInitilizer)
	train(net, xTrain, tTrain, 100)
}
