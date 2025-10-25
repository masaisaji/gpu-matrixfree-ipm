#include "csr_utils.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(func)                                                                           \
    {                                                                                              \
        cudaError_t status = (func);                                                               \
        if (status != cudaSuccess)                                                                 \
        {                                                                                          \
            printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,                   \
                   cudaGetErrorString(status), status);                                            \
            goto cleanup;                                                                          \
        }                                                                                          \
    }

#define CHECK_CUSPARSE(func)                                                                       \
    {                                                                                              \
        cusparseStatus_t status = (func);                                                          \
        if (status != CUSPARSE_STATUS_SUCCESS)                                                     \
        {                                                                                          \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__,               \
                   cusparseGetErrorString(status), status);                                        \
            goto cleanup;                                                                          \
        }                                                                                          \
    }

extern "C"
{
    CSRMatrix cusparse_compute_AAT(int m, int n, int nnz, const int *csr_row_ptr,
                                   const int *csr_col_idx, const double *csr_val)
    {
        double alpha = 1.0;
        double beta = 0.0;
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cudaDataType computeType = CUDA_R_64F;
        cudaDataType valueType = CUDA_R_64F;

        // Allocate and copy input matrix A to device
        int *d_row_ptr_A, *d_col_idx_A;
        double *d_val_A;
        int *d_row_ptr_AT, *d_col_idx_AT;
        double *d_val_AT;
        int *d_row_ptr_AAT, *d_col_idx_AAT;
        double *d_val_AAT;
        // CHECK_CUDA(cudaMalloc((void **)&d_row_ptr_A, (m + 1) * sizeof(int)))
        // CHECK_CUDA(cudaMalloc((void **)&d_col_idx_A, nnz * sizeof(int)))
        // CHECK_CUDA(cudaMalloc((void **)&d_val_A, nnz * sizeof(double)))
        // CHECK_CUDA(cudaMalloc((void **)&d_row_ptr_AT, (m + 1) * sizeof(int)))
        // CHECK_CUDA(cudaMalloc((void **)&d_col_idx_AT, nnz * sizeof(int)))
        // CHECK_CUDA(cudaMalloc((void **)&d_val_AT, nnz * sizeof(double)))
        cudaMalloc((void **)&d_row_ptr_A, (m + 1) * sizeof(int));
        cudaMalloc((void **)&d_col_idx_A, nnz * sizeof(int));
        cudaMalloc((void **)&d_val_A, nnz * sizeof(double));
        cudaMalloc((void **)&d_row_ptr_AT, (m + 1) * sizeof(int));
        cudaMalloc((void **)&d_col_idx_AT, nnz * sizeof(int));
        cudaMalloc((void **)&d_val_AT, nnz * sizeof(double));
        // column indices and values of AA^T are allocated later
        cudaMalloc((void **)&d_row_ptr_AAT, (m + 1) * sizeof(int));
        cudaMemcpy(d_row_ptr_A, csr_row_ptr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idx_A, csr_col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_A, csr_val, nnz * sizeof(double), cudaMemcpyHostToDevice);
        printf("d_val_AT: %p, d_col_idx_AT: %p, d_row_ptr_AT: %p\n", d_val_AT, d_col_idx_AT,
               d_row_ptr_AT);

        // -------------------------------------------------------------------------------------//

        // Create matrix descriptors
        cusparseHandle_t handle = NULL;
        cusparseSpMatDescr_t matA, matAT, matAAT;
        void *dBuffer1 = NULL, *dBuffer2 = NULL, *dBufferAT = NULL;
        size_t bufferSize1 = 0, bufferSize2 = 0, bufferSizeAT = 0;
        cusparseCreate(&handle);

        cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, d_val_A, d_row_ptr_A, d_col_idx_A,
                                      d_val_AT, d_col_idx_AT, d_row_ptr_AT, CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1, &bufferSizeAT);
        printf("bufferSizeAT = %zu\n", bufferSizeAT);

        cudaMalloc(&dBufferAT, bufferSizeAT);
        cusparseCsr2cscEx2(handle, m, n, nnz, d_val_A, d_row_ptr_A, d_col_idx_A, d_val_AT,
                           d_col_idx_AT, d_row_ptr_AT, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC,
                           CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, dBufferAT);

        cusparseCreateCsr(&matA, m, n, nnz, d_row_ptr_A, d_col_idx_A, d_val_A, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, valueType);
        cusparseCreateCsr(&matAT, n, m, nnz, d_row_ptr_AT, d_col_idx_AT, d_val_AT,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          valueType);
        cusparseCreateCsr(&matAAT, m, m, 0, d_row_ptr_AAT, NULL, NULL, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, valueType);

        // -------------------------------------------------------------------------------------//
        // Create SpGEMM descriptor
        cusparseSpGEMMDescr_t spgemmDesc;
        cusparseSpGEMM_createDescr(&spgemmDesc);

        // ask bufferSize1 bytes for external memory
        cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matAT, &beta, matAAT,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                      &bufferSize1, NULL);
        cudaMalloc(&dBuffer1, bufferSize1);

        // Inspect matrix A and A^T to understand the memory requirements
        cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matAT, &beta, matAAT,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                      &bufferSize1, dBuffer1);

        // ask bufferSize2 bytes for external memory
        cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matAT, &beta, matAAT,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                      &bufferSize2, NULL);
        cudaMalloc(&dBuffer2, bufferSize2);

        // compute the intermediate product of A * A^T
        cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matAT, &beta, matAAT, computeType,
                               CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2);

        // get matrix AAT non-zero entries AAT_nnz1
        int64_t C_num_rows1, C_num_cols1, AAT_nnz1;
        cusparseSpMatGetSize(matAAT, &C_num_rows1, &C_num_cols1, &AAT_nnz1);

        // allocate matrix AAT
        cudaMalloc((void **)&d_col_idx_AAT, AAT_nnz1 * sizeof(int));
        cudaMalloc((void **)&d_val_AAT, AAT_nnz1 * sizeof(double));

        // Update AAT with the new poitners
        cusparseCsrSetPointers(matAAT, d_row_ptr_AAT, d_col_idx_AAT, d_val_AAT);

        // -------------------------------------------------------------------------------------//
        cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matAT, &beta, matAAT, computeType,
                            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
        // -------------------------------------------------------------------------------------//
        cusparseDestroySpMat(matA);
        cusparseDestroySpMat(matAT);
        cusparseDestroySpMat(matAAT);
        cusparseSpGEMM_destroyDescr(spgemmDesc);
        cusparseDestroy(handle);
        // -------------------------------------------------------------------------------------//
        int *h_row_ptr_AAT = (int *)malloc((m + 1) * sizeof(int));
        int *h_col_idx_AAT = (int *)malloc(AAT_nnz1 * sizeof(int));
        double *h_val_AAT = (double *)malloc(AAT_nnz1 * sizeof(double));
        cudaMemcpy(h_row_ptr_AAT, d_row_ptr_AAT, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_col_idx_AAT, d_col_idx_AAT, AAT_nnz1 * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val_AAT, d_val_AAT, AAT_nnz1 * sizeof(double), cudaMemcpyDeviceToHost);
        // -------------------------------------------------------------------------------------//
        cudaFree(dBuffer1);
        cudaFree(dBuffer2);
        cudaFree(dBufferAT);
        cudaFree(d_row_ptr_A);
        cudaFree(d_col_idx_A);
        cudaFree(d_val_A);
        cudaFree(d_row_ptr_AT);
        cudaFree(d_col_idx_AT);
        cudaFree(d_val_AT);
        cudaFree(d_row_ptr_AAT);
        cudaFree(d_col_idx_AAT);
        cudaFree(d_val_AAT);
        CSRMatrix result = {h_row_ptr_AAT, h_col_idx_AAT, h_val_AAT, (int)AAT_nnz1};
        //     goto done;

        // cleanup:
        //     if (dBuffer1)
        //         cudaFree(dBuffer1);
        //     if (dBuffer2)
        //         cudaFree(dBuffer2);
        //     if (dBufferAT)
        //         cudaFree(dBufferAT);
        //     if (d_row_ptr_A)
        //         cudaFree(d_row_ptr_A);
        //     if (d_col_idx_A)
        //         cudaFree(d_col_idx_A);
        //     if (d_val_A)
        //         cudaFree(d_val_A);
        //     if (d_row_ptr_AT)
        //         cudaFree(d_row_ptr_AT);
        //     if (d_col_idx_AT)
        //         cudaFree(d_col_idx_AT);
        //     if (d_val_AT)
        //         cudaFree(d_val_AT);
        //     if (d_row_ptr_AAT)
        //         cudaFree(d_row_ptr_AAT);
        //     if (d_col_idx_AAT)
        //         cudaFree(d_col_idx_AAT);
        //     if (d_val_AAT)
        //         cudaFree(d_val_AAT);
        //     if (matA)
        //         cusparseDestroySpMat(matA);
        //     if (matAT)
        //         cusparseDestroySpMat(matAT);
        //     if (matAAT)
        //         cusparseDestroySpMat(matAAT);
        //     if (spgemmDesc)
        //         cusparseSpGEMM_destroyDescr(spgemmDesc);
        //     if (handle)
        //         cusparseDestroy(handle);
        //     CSRMatrix empty = {NULL, NULL, NULL, 0};
        //     return empty;
        //     //
        //     -------------------------------------------------------------------------------------//

        // done:
        return result;
    }
}
