package model

import (
	"github.com/YadaYuki/deeplearning-golang/layers"
	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

type Model[T nune.Number] interface {
	Predict(x nune.Tensor[T]) nune.Tensor[T]
	Backward()
}

type TwoLayerNet[T nune.Number] struct {
	Model[T]
	AffineLayer1         *layers.Affine[T]
	ReluLayer            *layers.Relu[T]
	AffineLayer2         *layers.Affine[T]
	SoftmaxWithLossLayer *layers.SoftmaxWithLoss[T]
}

func NewTwoLayerNet[T nune.Number](inputDim int, hiddenDim int, outDim int, weightInitilizer func() T, biasInitializer func() T) *TwoLayerNet[T] {
	f := func(x T) T {
		return weightInitilizer()
	}
	f2 := func(x T) T {
		return biasInitializer()
	}
	W1 := nune.Zeros[T](inputDim, hiddenDim).Map(f)
	B1 := nune.Zeros[T](hiddenDim).Map(f2)
	W2 := nune.Zeros[T](hiddenDim, outDim).Map(f)
	B2 := nune.Zeros[T](outDim).Map(f2)
	return &TwoLayerNet[T]{
		AffineLayer1:         layers.NewAffine(&W1, &B1),
		ReluLayer:            layers.NewRelu[T](),
		AffineLayer2:         layers.NewAffine(&W2, &B2),
		SoftmaxWithLossLayer: layers.NewSoftmaxWithLoss[T](),
	}
}

func (model *TwoLayerNet[T]) Predict(x nune.Tensor[T]) (y nune.Tensor[T]) {
	y = model.AffineLayer1.Forward(x)
	y = model.ReluLayer.Forward(y)
	y = model.AffineLayer2.Forward(y)
	return utils.SoftmaxBatch(y)
}

func (model *TwoLayerNet[T]) ForwardAndLoss(x nune.Tensor[T], t nune.Tensor[T]) (loss T) {
	y := model.AffineLayer1.Forward(x)
	y = model.ReluLayer.Forward(y)
	y = model.AffineLayer2.Forward(y)
	return model.SoftmaxWithLossLayer.Forward(y, t)
}

func (model *TwoLayerNet[T]) Backward() {
	dy := model.SoftmaxWithLossLayer.Backward()
	dy, _, _ = model.AffineLayer2.Backward(dy)
	dy = model.ReluLayer.Backward(dy)
	model.AffineLayer1.Backward(dy)
}

func (model *TwoLayerNet[T]) TrainStep(x nune.Tensor[T], t nune.Tensor[T]) (loss T) {
	loss = model.ForwardAndLoss(x, t)
	model.Backward()
	return loss
}
