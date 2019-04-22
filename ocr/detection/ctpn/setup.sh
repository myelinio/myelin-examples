#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset


#echo "deb http://ftp.debian.org/debian jessie main non-free contrib" >> /etc/apt/sources.list
#echo "deb-src http://ftp.debian.org/debian jessie main non-free contrib" >> /etc/apt/sources.list
#echo "deb http://ftp.debian.org/debian jessie-updates main contrib non-free" >> /etc/apt/sources.list
#echo "deb-src http://ftp.debian.org/debian jessie-updates main contrib non-free" >> /etc/apt/sources.list

apt update
apt-get install -y gcc libglib2.0-0 libsm6 libxext6 libxrender1

#pip install numpy scipy matplotlib pillow
#pip install easydict opencv-python keras h5py PyYAML
pip install cython==0.24

# for gpu
#pip install tensorflow-gpu==1.3.0

chmod +x ./lib/utils/make.sh
cd ./lib/utils/ && ./make.sh

# for cpu
# pip install tensorflow==1.3.0
# chmod +x ./ctpn/lib/utils/make_cpu.sh
# cd ./ctpn/lib/utils/ && ./make_cpu.sh