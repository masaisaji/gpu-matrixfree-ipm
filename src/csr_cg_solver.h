#include "csr_utils.h"
#include <stdbool.h>

#ifndef CUDA_CG_SOLVER_H
#define CUDA_CG_SOLVER_H

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Standard CG solver (no preconditioning).
     */
    void cg_solve_csr(const CSRMatrix *A, double *x, const double *b, double tol, int max_iter,
                      bool use_gpu, bool verbose);
    void pcg_solve_csr(const CSRMatrix *A, double *x, const double *b, int n, double tol,
                       int max_iter, int max_col, bool verbose, bool use_ldl_solve,
                       bool use_gpu); //, const double *A_dense

#ifdef __cplusplus
}
#endif

#endif
