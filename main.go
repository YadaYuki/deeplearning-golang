package main

import (
	"fmt"

	"github.com/vorduin/nune"
)

func main() {
	t := nune.Range[float64](0, 10, 1)
	fmt.Println(t)
}
