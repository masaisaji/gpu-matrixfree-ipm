#include "csr_utils.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>

extern "C" void init_cuda()
{
    // Optional: reset device (only use for full reset, not usually necessary)
    // cudaDeviceReset();

    // Force context initialization
    cudaError_t cerr = cudaFree(0);
    if (cerr != cudaSuccess)
    {
        fprintf(stderr, "cudaFree(0) failed: %s\n", cudaGetErrorString(cerr));
        return;
    }

    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "cuSPARSE initialization failed! Status = %d\n", status);
        return;
    }

    cusparseDestroy(handle);
}

extern "C" void cuda_matvec_csr(const CSRMatrix *A, const double *h_x, double *Ax)
{

    //--------------------------------------------------------------------------
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Unpack CSRMatrix A
    int m = A->rows;
    int n = A->cols;
    int nnz = A->nnz;
    const double *csr_values = A->val;
    const int *csr_row_ptr = A->row_ptr;
    const int *csr_col_idx = A->col_idx;

    // Allocate device memory
    double *d_vals, *d_x, *d_y;
    int *d_row_ptr, *d_col_idx;
    cudaMalloc(&d_vals, nnz * sizeof(double));
    cudaMalloc(&d_row_ptr, (m + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_y, m * sizeof(double));

    // Copy data to device
    cudaMemcpy(d_vals, csr_values, nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptr, csr_row_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_idx, csr_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, n * sizeof(double), cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    double alpha = 1.0, beta = 0.0;

    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        fprintf(stderr, "cusparseCreate failed: %d\n", status);
        return;
    }

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;

    // Create matrix A in CSR format
    cusparseCreateCsr(&matA, m, n, nnz, d_row_ptr, d_col_idx, d_vals, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, m, d_y, CUDA_R_64F);

    //--------------------------------------------------------------------------
    cudaEventRecord(start, 0);
    //--------------------------------------------------------------------------

    // allocate buffter
    size_t bufferSize = 0;
    void *dBuffer = NULL;

    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
                            vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute preprocess (optional)
    cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
                            vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    // excecute SpMV
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);

    //--------------------------------------------------------------------------
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    // printf("Time elapsed: %.6f s\n", elapsedTime / 1000.0f);
    //--------------------------------------------------------------------------

    // Cleanup
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);

    // copy result back to host
    cudaMemcpy(Ax, d_y, m * sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_vals);
    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(dBuffer);
    //--------------------------------------------------------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    //--------------------------------------------------------------------------
}

// Allocates and uploads A (dense matrix), x (vector), and allocates y (vector)
extern "C" void alloc_and_upload_dense_matvec(const double *A_host, const double *x_host,
                                              int A_rows, int A_cols, double **d_A, double **d_x,
                                              double **d_y)
{
    {
        size_t A_size = A_rows * A_cols * sizeof(double);
        size_t x_size = A_cols * sizeof(double); // assuming A*x, so x has dim A_cols
        size_t y_size = A_rows * sizeof(double); // y has dim A_rows

        cudaMalloc((void **)d_A, A_size);
        cudaMalloc((void **)d_x, x_size);
        cudaMalloc((void **)d_y, y_size);

        cudaMemcpy(*d_A, A_host, A_size, cudaMemcpyHostToDevice);
        cudaMemcpy(*d_x, x_host, x_size, cudaMemcpyHostToDevice);
        cudaMemset(*d_y, 0, y_size); // optionally zero y
    }
}

// Dense matrix-vector multiplication using cuBLAS
extern "C" void cuda_matvec_dense(const double *d_A, const double *d_x, double *d_y, double *h_y,
                                  int A_rows, int A_cols, bool transpose)
{
    {
        cublasHandle_t handle;
        cublasCreate(&handle);

        const double alpha = 1.0;
        const double beta = 0.0;
        cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

        cublasDgemv(handle, op, A_rows, A_cols, &alpha, d_A, A_rows, d_x, 1, &beta, d_y, 1);

        cublasDestroy(handle);
        if (h_y)
        {
            size_t y_size = (transpose ? A_cols : A_rows) * sizeof(double);
            cudaMemcpy(h_y, d_y, y_size, cudaMemcpyDeviceToHost);
        }
    }
}

extern "C" void free_cuda_matvec_memory(double *d_A, double *d_x, double *d_y)
{
    {
        if (d_A)
            cudaFree(d_A);
        if (d_x)
            cudaFree(d_x);
        if (d_y)
            cudaFree(d_y);
    }
}

extern "C" void cuda_matvec_dense_handle(cublasHandle_t handle, const double *d_A,
                                         const double *d_x, double *d_y, double *h_y, int A_rows,
                                         int A_cols, bool transpose)
{
    {
        const double alpha = 1.0;
        const double beta = 0.0;
        cublasOperation_t op = transpose ? CUBLAS_OP_T : CUBLAS_OP_N;

        cublasDgemv(handle, op, A_rows, A_cols, &alpha, d_A, A_rows, d_x, 1, &beta, d_y, 1);

        if (h_y)
        {
            size_t y_size = (transpose ? A_cols : A_rows) * sizeof(double);
            cudaMemcpy(h_y, d_y, y_size, cudaMemcpyDeviceToHost);
        }
    }
}
