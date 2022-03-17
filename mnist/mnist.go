package mnist

import (
	"io/ioutil"
	"path"

	"github.com/vorduin/nune"
)

func LoadMnist[T nune.Number](pathToMnistDir string, normalize bool) (xTrain, tTrain, xTest, tTest nune.Tensor[T], err error) {
	xTrain, err = loadImage[T](path.Join(pathToMnistDir, "train-images-idx3-ubyte"), normalize)
	if err != nil {
		var empty nune.Tensor[T]
		return empty, empty, empty, empty, err
	}
	xTest, err = loadImage[T](path.Join(pathToMnistDir, "t10k-images-idx3-ubyte"), normalize)
	if err != nil {
		var empty nune.Tensor[T]
		return empty, empty, empty, empty, err
	}
	tTrain, err = loadLabel[T](path.Join(pathToMnistDir, "train-labels-idx1-ubyte"))
	if err != nil {
		var empty nune.Tensor[T]
		return empty, empty, empty, empty, err
	}
	tTest, err = loadLabel[T](path.Join(pathToMnistDir, "t10k-labels-idx1-ubyte"))
	if err != nil {
		var empty nune.Tensor[T]
		return empty, empty, empty, empty, err
	}
	return
}

func loadImage[T nune.Number](pathToImage string, normalize bool) (imgsData nune.Tensor[T], err error) {
	_data, err := ioutil.ReadFile(pathToImage)
	if err != nil {
		var empty nune.Tensor[T]
		return empty, err
	}
	offset := 16
	_data = _data[offset:]
	data := make([]T, len(_data))
	for i := 0; i < len(data); i++ {
		data[i] = T(_data[i])
	}
	w, h := 28, 28
	imgSize := w * h
	if len(data)%imgSize != 0 {
		panic("Invalid image size")
	}
	imgNum := int(len(data) / imgSize)
	imgsData = nune.FromBuffer(data).Reshape(imgNum, imgSize)
	if normalize {
		imgsData.Div(255.0)
	}
	return imgsData, nil
}

func loadLabel[T nune.Number](pathToLabel string) (labelData nune.Tensor[T], err error) {
	_data, err := ioutil.ReadFile(pathToLabel)
	if err != nil {
		var empty nune.Tensor[T]
		return empty, err
	}
	offset := 8
	_data = _data[offset:]
	labelNum := 10
	dataSize := len(_data)
	labelData = nune.Zeros[T](dataSize, labelNum)
	for i := 0; i < dataSize; i++ {
		labelData.Index(i, int(_data[i])).Ravel()[0] = 1
	}
	return labelData, nil
}
