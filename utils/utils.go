package utils

import (
	"fmt"

	"github.com/vorduin/nune"
)

func Equal1D[T nune.Number](a, b nune.Tensor[T]) bool {
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

func Equal2D[T nune.Number](a, b nune.Tensor[T]) bool {
	if a.Shape()[0] != b.Shape()[0] || a.Shape()[1] != b.Shape()[1] {
		return false
	}
	I := a.Shape()[0]
	J := a.Shape()[1]
	for i := 0; i < I; i++ {
		for j := 0; j < J; j++ {
			if a.Index(i, j).Scalar() != b.Index(i, j).Scalar() {
				return false
			}
		}
	}
	return true
}

func Add[T nune.Number](a, b nune.Tensor[T]) nune.Tensor[T] {
	if a.Size(0) != b.Size(0) {
		panic("Tensors must have the same size")
	}
	result := make([]T, a.Size(0))
	for i := 0; i < a.Size(0); i++ {
		result[i] = a.Index(i).Scalar() + b.Index(i).Scalar()
	}
	return nune.FromBuffer(result)
}

// dot 2D tensor & 2D tensor a * b
func Dot[T nune.Number](a, b nune.Tensor[T]) nune.Tensor[T] {
	Ia := a.Size(0)
	Ja := a.Size(1)
	Ib := b.Size(0)
	Jb := b.Size(1)
	if Ja != Ib {
		panic(fmt.Sprint("a.Size(1),b.Size(0) must have the same size", a.Shape(), b.Shape()))
	}
	result := nune.Zeros[T](Ia, Jb)
	for ia := 0; ia < Ia; ia++ {
		for jb := 0; jb < Jb; jb++ {
			sum := T(0)
			for j := 0; j < Ja; j++ {
				ib := j
				ja := j
				sum += a.Index(ia, ja).Scalar() * b.Index(ib, jb).Scalar()
			}
			result.Index(ia, jb).Ravel()[0] = sum
		}
	}
	return result
}
