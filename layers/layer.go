package layers

import "github.com/vorduin/nune"

type Layer[T nune.Number] interface {
	Forward(input nune.Tensor[T]) nune.Tensor[T]  // TODO: remove any
	Backward(input nune.Tensor[T]) nune.Tensor[T] // TODO: remove any
}
