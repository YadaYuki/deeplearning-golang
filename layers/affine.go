package layers

import (
	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

type Affine[T nune.Number] struct {
	Layer[T]
	W *nune.Tensor[T]
	B *nune.Tensor[T]
}

func NewAffine[T nune.Number](W *nune.Tensor[T], B *nune.Tensor[T]) *Affine[T] {
	return &Affine[T]{W: W, B: B}
}

func (affine *Affine[T]) Forward(X nune.Tensor[T]) nune.Tensor[T] {
	Y := utils.Dot(X, *affine.W)
	batchSize := X.Shape()[0]
	// repeat B for each sample
	Y = utils.Add(Y, (*affine.B).Repeat(batchSize))
	return Y
}

func (affine *Affine[T]) Backward(dy nune.Tensor[T]) nune.Tensor[T] {
	return dy
}
