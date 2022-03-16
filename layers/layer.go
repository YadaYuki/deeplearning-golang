package layers

type Layer[T any] interface {
	Forward(input T) any  // TODO: remove any
	Backward(input T) any // TODO: remove any
}
