package main

import (
	"fmt"
	"os"
	"path"

	"github.com/YadaYuki/deeplearning-golang/mnist"
)

func main() {
	wd, _ := os.Getwd()
	pathToMnistDir := path.Join(wd, "mnist/data")
	xTrain, tTrain, xTest, tTest, _ := mnist.LoadMnist[float64](pathToMnistDir, true)
	fmt.Println(xTrain.Shape())
	fmt.Println(tTrain.Shape())
	fmt.Println(xTest.Shape())
	fmt.Println(tTest.Shape())
}
