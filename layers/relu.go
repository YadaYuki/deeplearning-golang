package layers

import "github.com/vorduin/nune"

type Relu[T nune.Number] struct {
	Layer[T]
	inputM nune.Tensor[T]
}

func NewRelu[T nune.Number]() *Relu[T] {
	return &Relu[T]{}
}

func (relu *Relu[T]) Forward(x nune.Tensor[T]) nune.Tensor[T] {
	relu.inputM = x
	y := x.Map(func(x T) T {
		if x < 0 {
			return 0
		}
		return x
	})
	return y
}

func (relu *Relu[T]) Backward(dy nune.Tensor[T]) nune.Tensor[T] {
	dx := nune.ZerosLike[T](relu.inputM)
	for i := 0; i < dy.Size(0); i++ {
		for j := 0; j < dy.Size(1); j++ {
			if relu.inputM.Index(i, j).Scalar() > 0 {
				dx.Index(i, j).Ravel()[0] = dy.Index(i, j).Scalar()
			}
		}
	}
	return dx
}
