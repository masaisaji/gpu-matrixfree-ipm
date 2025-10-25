# Prerequisites

## Dependencies

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

2. Install other dependencies:

- Ubuntu / Debian

```sh
sudo apt update
sudo apt install -y build-essential make gcc \
     liblapacke-dev liblapack-dev libblas-dev
```

- Fedora / RHEL

```sh
sudo dnf install -y gcc gcc-c++ make \
     lapack-devel lapacke-devel blas-devel
```

- Arch Linux

```sh
sudo pacman -S base-devel lapack blas lapacke cuda
```

## CUDA Environment Setup

This project assumes CUDA is installed in:

```sh
/usr/local/cuda
```

If your CUDA installation is in a different location, set:

```sh
export CUDA_HOME=/path/to/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Implementing **GPU Acceleration of the Matrix-Free Interior Point Method**

- [Original paper](https://link.springer.com/chapter/10.1007/978-3-642-31464-3_69)
- [Overleaf for project report](https://www.overleaf.com/project/67a0d44096c84f5425307c91)
