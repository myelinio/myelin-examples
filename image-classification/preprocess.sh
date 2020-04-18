#!/usr/bin/env bash

DATA=${DATA_PATH:-/tmp/data/}

mkdir -p ${DATA}
cd ${DATA}

wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar zxvf cifar-10-python.tar.gz

mkdir cifar
mv cifar-10-batches-py/data_batch_1 cifar/
rm -r cifar-10-batches-py/
rm cifar-10-python.tar.gz