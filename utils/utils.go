package utils

import "math/rand"

func Shuffle(a []int) []int {
	rand.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
	return a
}

func Range(from int, to int) []int {
	a := make([]int, to-from)
	for i := from; i < to; i++ {
		a[i-from] = i
	}
	return a
}

func SplitSlice[T any](slices []T, n int) [][]T {
	result := [][]T{}
	for i := 0; i < len(slices); i += n {
		if i+n > len(slices) {
			break
		}
		result = append(result, slices[i:i+n])

	}
	if len(slices)%n != 0 {
		result = append(result, slices[len(slices)-len(slices)%n:])
	}
	return result
}
