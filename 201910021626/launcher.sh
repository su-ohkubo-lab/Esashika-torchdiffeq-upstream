#!/bin/bash

SAVEDIR=experiments/$(date +%Y%m%d%H%M)
mkdir -p SAVEDIR

for ldep in $(seq 0 10); do
for i in $(seq 1 10); do
	python examples/odenet_mnist.py --network=resnet --adjoint=True --uniform=True --dataset=fasion-mnist --layer_depth=${ldep} --save=${SAVEDIR}/layer${ldep}-try${i}
done
done
