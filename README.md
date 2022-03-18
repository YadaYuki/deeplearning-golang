# Deep Leaning by Golang 👻

- Deep Leaning implementation by Golang
- Dataset: [MNIST](http://yann.lecun.com/exdb/mnist/)

## Requirements
- [Go](https://go.dev/dl/) 1.18+

## Setup
### Download MNIST

you can download MNIST Dataset by running:
```
$ ./mnist/mnist.sh 
```
<!-- 実行権限を与える、という旨を書く。 -->

## Train & Test Model

you can train **[TwoLayerNet](https://github.com/YadaYuki/deeplearning-golang/blob/main/model/model.go#L14)** by running:

```
$ go run main.go
Train
epoch: 1
iter:20/600 train loss: 2.2526680169043094 
iter:40/600 train loss: 2.2060880533621425 
iter:60/600 train loss: 2.004868903448843 
iter:80/600 train loss: 1.6958167317582715 
iter:100/600 train loss: 1.4482442877179182 
iter:120/600 train loss: 0.9726532993333357 
iter:140/600 train loss: 0.9454905601894334 
iter:160/600 train loss: 0.8181915867804608 
...
iter:520/600 train loss: 0.15054788493244708 
iter:540/600 train loss: 0.08847004249275113 
iter:560/600 train loss: 0.03189571300337565 
iter:580/600 train loss: 0.11025929565750328 
iter:600/600 train loss: 0.1796958582400985 
Train finished
Test
Test finished
accuracy: 0.9634, loss: 0.1273054300107751 
```
