#!/usr/bin/env bash

CUDA_REPO_PKG="cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb"
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/${CUDA_REPO_PKG}
dpkg -i ${CUDA_REPO_PKG}
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x$
apt-get update
CUDA_REPO_PKG="cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb"
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/${CUDA_REPO_PKG}
dpkg -i ${CUDA_REPO_PKG}
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x$
apt-get update
wget https://github.com/ashokpant/cudnn_archive/raw/master/v7.0/${CUDNN_PKG}
apt-get install -y cuda-command-line-tools-9-0