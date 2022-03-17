package main

import (
	"fmt"

	"github.com/vorduin/nune"
)

func main() {
	t := nune.Range[float64](0, 5, 1)
	// fmt.Println(t.Log())
	fmt.Println(t.Repeat(5))
}
