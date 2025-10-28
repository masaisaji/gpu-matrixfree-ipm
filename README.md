Implementing **a GPU accelerated matrix-free interior point method for LP**, originally proposed by E. Smith et al. in
> Smith, E., Gondzio, J., Hall, J. (2012). "GPU Acceleration of the Matrix-Free Interior Point Method". In: Wyrzykowski, R., Dongarra, J., Karczewski, K., Waśniewski, J. (eds) Parallel Processing and Applied Mathematics. PPAM 2011. Lecture Notes in Computer Science, vol 7203. Springer, Berlin, Heidelberg. <https://doi.org/10.1007/978-3-642-31464-3_69>

Other techiniuqes such as preconditioner based on partial cholesky decomposition, Mehretora preconditioner corrector method, and strictly feasible initial guess generation, are also applied.

The implementation assumes LPs of the form:

$$
\begin{align}
\min_x & c^\top x \\
\text{s.t.} & Ax=b\\
& x\geq 0
\end{align}
$$

# Installation
1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). This project assumes CUDA is installed in `/usr/local/cuda`, but a different path can be passed to `make` as an option like: `make CUDA_HOME=/path/to/cuda`.

2. Install other dependencies:

- Ubuntu / Debian

```sh
sudo apt install -y build-essential make gcc liblapacke-dev liblapack-dev libblas-dev
```

- Fedora / RHEL

```sh
sudo dnf install -y gcc gcc-c++ make lapack-devel lapacke-devel blas-devel
```

- Arch Linux

```sh
sudo pacman -S base-devel lapack blas lapacke cuda
```

# Example Usage
## GPU-Accelerated Implementation
To build the executable, run
```sh
make
```
in the root directory. This will place the executable in `src/run`, which can be run like `src/run <lp_instance_name> <gpu_indicator>`. 
For example, to solve `nug07` LP instance with CPU + GPU,
```sh
src/run nug07 1
```
`src/run nug07 0` will solve the problem using CPU only.
### Built-in Test Problems
Currently, the supported built-in problem instances are `nug05, nug06, nug07, nug08, nug12, nug15, nug20, nug30`.
See [Suite Sparse website](https://www.cise.ufl.edu/research/sparse/matrices/Qaplib/) for more details on the instances.
You may also define your own LP with $A$ matrix and $b, c$ vectors.

## Comparison/Validation with Gurobi
To validate solutions and/or compare solve time, you can also solve problems with Gurobi (CPU only).
*Note that Gurobi needs to be installed and configured (e.g., PATH) to use this feature*.
To compile, run
```sh
make USE_GUROBI=1
```
in the root directory. 
If needed, Gurobi path can be passed explicitly here like: `make USE_GUROBI=1 GUROBI_HOME=/path/to/gurobi`.

The executable can be found in `src/run_gurobi_test`. Use it as `src/run_gurobi_test <lp_instance_name>`.
For example, to solve `nug07` with Gurobi, run

```sh
src/run_gurobi_test nug07
```

## Additional references:
> Gondzio, J. "Matrix-free interior point method". Comput Optim Appl 51, 457–480 (2012). https://doi.org/10.1007/s10589-010-9361-3

> Bellavia, S., Gondzio, J., and Morini, B. “A Matrix-Free Preconditioner for Sparse
Symmetric Positive Definite Systems and Least-Squares Problems”. In: SIAM Journal on Scientific
Computing 35.1 (Jan. 2013), A192–A211. https://doi.org/10.1137/110840819.

## Credits
Other contributor(s): [ecrwrig4](https://github.com/ecwrig4)
