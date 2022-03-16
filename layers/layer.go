package layers

import "github.com/vorduin/nune"

type Layer[T nune.Number] interface {
	Forward(x nune.Tensor[T]) nune.Tensor[T]
	Backward(dy nune.Tensor[T]) nune.Tensor[T]
}
