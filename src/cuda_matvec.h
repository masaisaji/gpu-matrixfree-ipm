#pragma once
#ifdef __cplusplus
extern "C"
{
#endif

#include "csr_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
    void cuda_alloc_csr_matrix(int n, int nnz, const double *csr_values, const int *csr_row_ptr,
                               const int *csr_col_idx);

    void cuda_matvec_csr(const CSRMatrix *A, const double *x, double *Ax);
    // void cuda_free_csr_matrix();
    //
    void alloc_and_upload_dense_matvec(const double *A_host, const double *x_host, int A_rows,
                                       int A_cols, double **d_A, double **d_x, double **d_y);
    void cuda_matvec_dense(const double *d_A, const double *d_x, double *d_y, double *h_y,
                           int A_rows, int A_cols, bool transpose);
    void cuda_matvec_dense_handle(cublasHandle_t handle, const double *d_A, const double *d_x,
                                  double *d_y, double *h_y, int A_rows, int A_cols, bool transpose);
    void free_cuda_matvec_memory(double *d_A, double *d_x, double *d_y);

    // "warm up" cuda, use before the first cuda call
    void init_cuda();

#ifdef __cplusplus
}
#endif
