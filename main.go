package main

import (
	"fmt"
	"math/rand"

	"github.com/vorduin/nune"
)

func main() {
	t := nune.Zeros[float64](5, 5)
	weightInitilizer := func() float64 {
		return rand.Float64()
	}
	t.Map(func(x float64) float64 {
		return weightInitilizer()
	})
	fmt.Println(t)
}
