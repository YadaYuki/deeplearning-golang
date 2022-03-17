package layers

import (
	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

type Affine[T nune.Number] struct {
	Layer[T]
	W  *nune.Tensor[T]
	B  *nune.Tensor[T]
	xM nune.Tensor[T]
}

func NewAffine[T nune.Number](W *nune.Tensor[T], B *nune.Tensor[T]) *Affine[T] {
	return &Affine[T]{W: W, B: B}
}

func (affine *Affine[T]) Forward(X nune.Tensor[T]) nune.Tensor[T] {
	affine.xM = X

	Y := utils.Dot(X, *affine.W)
	batchSize := X.Shape()[0]
	// repeat B for each sample
	Y = utils.Add(Y, (*affine.B).Repeat(batchSize))
	return Y
}

func (affine *Affine[T]) Backward(dy nune.Tensor[T]) (dx nune.Tensor[T], dW nune.Tensor[T], dB nune.Tensor[T]) {
	dx = utils.Dot(dy, utils.Transpose(*affine.W))
	dW = utils.Dot(utils.Transpose(affine.xM), dy)
	dB = utils.Sum(dy)
	*affine.W = utils.Add(*affine.W, dW.Clone().Mul(-1.0)) // dW自体も書き換えられてしまうため、Cloneしておく
	*affine.B = utils.Add(*affine.B, dB.Clone().Mul(-1.0)) // dB自体も書き換えられてしまうため、Cloneしておく
	return dx, dW, dB
}
