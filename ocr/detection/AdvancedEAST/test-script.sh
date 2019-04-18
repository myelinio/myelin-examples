#!/usr/bin/env bash
set -o errexit
set -o pipefail
set -o nounset

apt-get update && apt-get install -y --no-install-recommends \
		wget \
		gnupg \
		dirmngr \
	&& rm -rf /var/lib/apt/lists/*



# NVIDIA Drivers
echo "deb http://httpredir.debian.org/debian/ stretch main contrib non-free" >> /etc/apt/sources.list
DEBIAN_FRONTEND=noninteractive  apt install nvidia-kernel-common-410 \
                                            libnvidia-gl-410 \
                                            xserver-xorg-video-nvidia-410 \
                                            libnvidia-ifr1-410 \
                                            nvidia-settings \
                                            libnvidia-compute-410 \
                                            libnvidia-decode-410 \
                                            libnvidia-encode-410 \
                                            libnvidia-ifr1-410 \
                                            libnvidia-fbc1-410 \
                                            libnvidia-gl-410 \
                                            xserver-xorg-core \
                                            nvidia-driver-410  nvidia-dkms-410 nvidia-kernel-dkms

DEBIAN_FRONTEND=noninteractive  apt install nvidia-driver-410  nvidia-dkms-410 nvidia-kernel-dkms

# FIXME: apt purge --autoremove '*nvidia*'
# CUDA
CUDA_REPO_PKG="cuda-repo-ubuntu1804-10-0-local-10.0.130-410.48_1.0-1_amd64"
wget https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/${CUDA_REPO_PKG}
dpkg -i ${CUDA_REPO_PKG}
apt-key adv --fetch-keys  http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt-get update
apt-get install  cuda-drivers cuda-runtime-10-0  cuda-10-0


# CUDNN
wget https://developer.download.nvidia.com/compute/redist/cudnn/v7.4.2/cudnn-10.0-linux-x64-v7.4.2.24.tgz
tar -xvf cudnn-10.0-linux-x64-v7.4.2.24.tgz
cp include/cudnn.h /usr/local/cuda-10.0/include/cudnn.h
cp lib64/libcudnn.so.7.4.2 /usr/local/cuda-10.0/lib64/libcudnn.so.7.4.2
ln -s /usr/local/cuda-10.0/lib64/libcudnn.so.7.4.2 /usr/local/cuda-10.0/lib64/libcudnn.so.7
ln -s /usr/local/cuda-10.0/lib64/libcudnn.so.7 /usr/local/cuda-10.0/lib64/libcudnn.so


export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export CUDA_HOME=/usr/local/cuda-10.0


apt-get install -y cuda-command-line-tools-10-0

rm -rf /var/lib/apt/lists/*
rm ${CUDA_REPO_PKG}
rm  cudnn-10.0-linux-x64-v7.4.2.24.tgz include lib64
