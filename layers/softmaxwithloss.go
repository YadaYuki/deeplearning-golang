package layers

import (
	"github.com/YadaYuki/deeplearning-golang/utils"
	"github.com/vorduin/nune"
)

type SoftmaxWithLoss[T nune.Number] struct {
	Layer[T]
	yM nune.Tensor[T]
	tM nune.Tensor[T]
}

func NewSoftmaxWithLoss[T nune.Number]() *SoftmaxWithLoss[T] {
	return &SoftmaxWithLoss[T]{}
}

func (softmaxWithLoss *SoftmaxWithLoss[T]) Forward(x nune.Tensor[T], t nune.Tensor[T]) T {
	softmaxWithLoss.yM = utils.SoftmaxBatch(x)
	softmaxWithLoss.tM = t
	loss := utils.CrossEntropyErrorBatch(softmaxWithLoss.yM, softmaxWithLoss.tM, 1e-7)
	return loss
}

func (softmaxWithLoss *SoftmaxWithLoss[T]) Backward() nune.Tensor[T] {
	batchSize := softmaxWithLoss.yM.Shape()[0]
	dx := utils.Add(softmaxWithLoss.yM, softmaxWithLoss.tM.Clone().Mul(-1))
	dx = dx.Div(batchSize)
	return dx
}
