package utils

import (
	"github.com/vorduin/nune"
)

// TODO: correspond to 2 dimensional tensor
func Equal[T nune.Number](a, b nune.Tensor[T]) bool {
	if a.Size(0) != b.Size(0) {
		return false
	}
	for i := 0; i < a.Size(0); i++ {
		if a.Index(i).Scalar() != b.Index(i).Scalar() {
			return false
		}
	}
	return true
}

func Add[T nune.Number](a, b nune.Tensor[T]) nune.Tensor[T] {
	if a.Size(0) != b.Size(0) {
		panic("Tensors must have the same size")
	}
	add := make([]T, a.Size(0))
	for i := 0; i < a.Size(0); i++ {
		add[i] = a.Index(i).Scalar() + b.Index(i).Scalar()
	}
	return nune.From[T](add)
}
