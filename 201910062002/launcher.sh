#!/bin/bash

SAVEDIR=experiments/$(date +%Y%m%d%H%M)
mkdir -p $SAVEDIR

for i in $(seq 1 10); do
	python examples/odenet_mnist.py --network=odenet --adjoint=False --dataset=fasion-mnist --save=${SAVEDIR}/try${i}
done
