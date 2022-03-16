package utils

import (
	"fmt"
	"math"

	"github.com/vorduin/nune"
)

func Equal1D[T nune.Number](a, b nune.Tensor[T], eq func(a, b T) bool) bool {
	if a.Size(0) != b.Size(0) {
		return false
	}
	for i := 0; i < a.Size(0); i++ {
		if !eq(a.Index(i).Scalar(), b.Index(i).Scalar()) {
			return false
		}
	}
	return true
}

func Equal2D[T nune.Number](a, b nune.Tensor[T], eq func(a, b T) bool) bool {
	if a.Shape()[0] != b.Shape()[0] || a.Shape()[1] != b.Shape()[1] {
		return false
	}
	I := a.Shape()[0]
	J := a.Shape()[1]
	for i := 0; i < I; i++ {
		for j := 0; j < J; j++ {
			if !eq(a.Index(i, j).Scalar(), b.Index(i, j).Scalar()) {
				return false
			}
		}
	}
	return true
}

func EqualFloat(a, b, tolerance float64) bool {
	if diff := math.Abs(a - b); diff < tolerance {
		return true
	} else {
		return false
	}
}

func Add[T nune.Number](a, b nune.Tensor[T]) nune.Tensor[T] {
	var result nune.Tensor[T]
	if len(a.Shape()) != len(b.Shape()) {
		panic(fmt.Sprintf("Dimension mismatch: %d != %d", len(a.Shape()), len(b.Shape())))
	}
	if len(a.Shape()) == 1 {
		if a.Size(0) != b.Size(0) {
			panic("Tensors must have the same size")
		}
		result = nune.ZerosLike[T](a)
		for i := 0; i < a.Size(0); i++ {
			result.Index(i).Ravel()[0] = a.Index(i).Scalar() + b.Index(i).Scalar()
		}
	}
	if len(a.Shape()) == 2 {
		if a.Size(0) != b.Size(0) || a.Size(1) != b.Size(1) {
			panic("Tensors must have the same size")
		}
		result = nune.ZerosLike[T](a)
		for i := 0; i < a.Size(0); i++ {
			for j := 0; j < a.Size(1); j++ {
				result.Index(i, j).Ravel()[0] = a.Index(i, j).Scalar() + b.Index(i, j).Scalar()
			}
		}
	}

	return result
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

func CrossEntropyError[T nune.Number, S nune.Number](y nune.Tensor[T], t nune.Tensor[S], eps T) T {
	labelNum := y.Size(0)
	v := nune.Zeros[T](labelNum)
	for i := 0; i < labelNum; i++ {
		if t.Index(i).Scalar() == 1 {
			v.Index(i).Ravel()[0] = y.Index(i).Scalar() + eps
		} else {
			v.Index(i).Ravel()[0] = 1 - y.Index(i).Scalar() + eps
		}
	}
	v = v.Log()
	for i := 0; i < labelNum; i++ {
		v.Index(i).Ravel()[0] = v.Index(i).Ravel()[0] * T(t.Index(i).Scalar())
	}
	return -v.Sum().Scalar()
}

func CrossEntropyErrorBatch[T ~float64, S nune.Number](yBatch nune.Tensor[T], tBatch nune.Tensor[S], eps T) T {
	if (yBatch.Size(0) != tBatch.Size(0)) || (yBatch.Size(1) != tBatch.Size(1)) {
		panic(fmt.Sprintf("yBatch and tBatch must have the same size yBatch.Shape:%v,tBatch.Sahpe:%v", yBatch.Shape(), tBatch.Shape()))
	}
	batchSize := yBatch.Size(0)
	result := T(0)
	for i := 0; i < batchSize; i++ {
		result += CrossEntropyError(yBatch.Index(i), tBatch.Index(i), eps)
	}
	return result / T(batchSize)
}

func Softmax[T ~float64](x nune.Tensor[T]) nune.Tensor[T] {
	max_x := x.Max().Scalar()
	x = x.Sub(max_x)
	exp_x := x.Exp()
	exp_x_sum := exp_x.Sum().Scalar()
	return exp_x.Div(exp_x_sum)
}

func SoftmaxBatch[T ~float64](xBatch nune.Tensor[T]) nune.Tensor[T] {
	result := nune.Zeros[T](xBatch.Size(0), xBatch.Size(1))
	for i := 0; i < xBatch.Size(0); i++ {
		out := Softmax(xBatch.Index(i))
		for j := 0; j < xBatch.Size(1); j++ {
			result.Index(i, j).Ravel()[0] = out.Index(j).Scalar()
		}
	}
	return result
}
