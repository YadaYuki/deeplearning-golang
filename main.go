package main

import (
	"fmt"
	"math/rand"
	"os"
	"path"

	"github.com/YadaYuki/deeplearning-golang/mnist"
	"github.com/YadaYuki/deeplearning-golang/model"
	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

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

func where[T nune.Number](data nune.Tensor[T], target T) (int, error) {
	size := data.Size(0)
	for i := 0; i < size; i++ {
		if data.Index(i).Scalar() == target {
			return i, nil
		}
	}
	return -1, fmt.Errorf("target not found")
}

func getCorrectNum[T nune.Number](yBatch nune.Tensor[T], tBatch nune.Tensor[T]) int {
	if yBatch.Shape()[0] != tBatch.Shape()[0] || yBatch.Shape()[1] != tBatch.Shape()[1] {
		panic("Dimension mismatch")
	}
	batchSize := yBatch.Shape()[0]
	correct := 0
	for i := 0; i < batchSize; i++ {
		y_pred, _ := where(yBatch.Index(i), yBatch.Index(i).Max().Scalar())
		t_true, _ := where(tBatch.Index(i), tBatch.Index(i).Max().Scalar())
		if y_pred == t_true {
			correct++
		}
	}
	return correct
}

func train[T nune.Number](net *model.TwoLayerNet[T], xTrain nune.Tensor[T], tTrain nune.Tensor[T], batchSize int) {
	dataSize := xTrain.Shape()[0]
	batchIdxes := utils.SplitSlice(utils.Shuffle(utils.Range(0, dataSize)), batchSize)
	fmt.Println("Train")
	for epoch := 0; epoch < 10; epoch++ {
		fmt.Println("epoch:", epoch+1)
		for i := 0; i < len(batchIdxes); i++ {
			xBatch := getBatchData(batchIdxes[i], xTrain)
			tBatch := getBatchData(batchIdxes[i], tTrain)
			loss := net.TrainStep(xBatch, tBatch)
			if (i+1)%20 == 0 {
				fmt.Printf("iter:%d/%d train loss: %v \n", (i + 1), len(batchIdxes), loss)
			}
		}
	}
	fmt.Println("Train finished")
}

func test[T nune.Number](net *model.TwoLayerNet[T], xVal nune.Tensor[T], tVal nune.Tensor[T], batchSize int) (accuracy float64, loss float64) {
	dataSize := xVal.Shape()[0]
	batchIdxes := utils.SplitSlice(utils.Range(0, dataSize), batchSize)
	loss = 0.0
	correct := 0
	fmt.Println("Test")
	for i := 0; i < len(batchIdxes); i++ {
		xBatch := getBatchData(batchIdxes[i], xVal)
		tBatch := getBatchData(batchIdxes[i], tVal)
		yBatch := net.Predict(xBatch)
		correct += getCorrectNum(yBatch, tBatch)
		fmt.Println(correct, getCorrectNum(yBatch, tBatch))
		loss += float64(net.ForwardAndLoss(xBatch, tBatch))
	}
	fmt.Println("Test finished")
	loss /= float64(len(batchIdxes))
	return float64(correct) / float64(dataSize), loss
}

func main() {
	// load mnist data
	wd, _ := os.Getwd()
	pathToMnistDir := path.Join(wd, "mnist/data")
	xTrain, tTrain, xTest, tTest, _ := mnist.LoadMnist[float64](pathToMnistDir, true)

	// initialize network
	weightInitilizer := func() float64 {
		return rand.Float64() * 0.01
	}
	biasInitilizer := func() float64 {
		return rand.Float64()
	}
	net := model.NewTwoLayerNet(784, 50, 10, weightInitilizer, biasInitilizer)
	// train & test network
	batchSize := 100
	train(net, xTrain, tTrain, batchSize)
	accuracy, loss := test(net, xTest, tTest, batchSize)
	fmt.Printf("accuracy: %v, loss: %v\n", accuracy, loss)
}
