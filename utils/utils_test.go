package utils

import (
	"testing"

	"github.com/vorduin/nune"
)

func TestAdd(t *testing.T) {
	a := nune.Range[int](0, 10, 1)
	b := nune.Range[int](0, 10, 1)
	expected := nune.Range[int](0, 20, 2)
	c := Add(a, b)
	if !Equal(c, expected) {
		t.Errorf("Expected %v, got %v", expected, c)
	}
}
