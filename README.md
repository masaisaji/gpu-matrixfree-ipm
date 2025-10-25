Implementing **a GPU accelerated matrix-free interior point method for LP**, originally proposed by E. Smith et al in
> Smith, E., Gondzio, J., Hall, J. (2012). GPU Acceleration of the Matrix-Free Interior Point Method. In: Wyrzykowski, R., Dongarra, J., Karczewski, K., Waśniewski, J. (eds) Parallel Processing and Applied Mathematics. PPAM 2011. Lecture Notes in Computer Science, vol 7203. Springer, Berlin, Heidelberg. <https://doi.org/10.1007/978-3-642-31464-3_69>

# Installation

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
