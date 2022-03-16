package main

import (
	"fmt"
	"math"

	"github.com/vorduin/nune"
)

func main() {
	t := nune.Range[float64](0, 5, 1)
	fmt.Println(t.Log())
	fmt.Println(math.Log(10))
}
