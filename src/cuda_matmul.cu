#include "csr_utils.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

extern "C"
{
    CSRMatrix cuda_matmul_csr(const CSRMatrix *A, const CSRMatrix *B)
    {
        // unpack CSRMatrix A and B
        int A_rows = A->rows;
        int A_cols = A->cols;
        int A_nnz = A->nnz;
        const double *h_A_val = A->val;
        const int *h_A_row_ptr = A->row_ptr;
        const int *h_A_col_idx = A->col_idx;

        int B_rows = B->rows;
        int B_cols = B->cols;
        int B_nnz = B->nnz;
        const double *h_B_val = B->val;
        const int *h_B_row_ptr = B->row_ptr;
        const int *h_B_col_idx = B->col_idx;

        if (A_cols != B_rows)
        {
            printf("ERROR: Dimension mismatch A: %d x %d, B: %d x %d\n", A_rows, A_cols, B_rows,
                   B_cols);
            return (CSRMatrix){0};
        }

        // Allocate device memory for A
        double *d_A_val;
        int *d_A_row_ptr, *d_A_col_idx;
        cudaMalloc((void **)&d_A_val, A_nnz * sizeof(double));
        cudaMalloc((void **)&d_A_row_ptr, (A_rows + 1) * sizeof(int));
        cudaMalloc((void **)&d_A_col_idx, A_nnz * sizeof(int));

        // Allocate device memory for B
        double *d_B_val;
        int *d_B_row_ptr, *d_B_col_idx;
        cudaMalloc((void **)&d_B_val, B_nnz * sizeof(double));
        cudaMalloc((void **)&d_B_row_ptr, (B_rows + 1) * sizeof(int));
        cudaMalloc((void **)&d_B_col_idx, B_nnz * sizeof(int));

        // Allocate device memory for C, values and column indices are allocated later
        double *d_C_val;
        int *d_C_row_ptr, *d_C_col_idx;
        cudaMalloc((void **)&d_C_row_ptr, (A_rows + 1) * sizeof(int));

        // Copy data from host to device
        cudaMemcpy(d_A_val, h_A_val, A_nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_row_ptr, h_A_row_ptr, (A_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_col_idx, h_A_col_idx, A_nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_val, h_B_val, B_nnz * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_row_ptr, h_B_row_ptr, (B_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B_col_idx, h_B_col_idx, B_nnz * sizeof(int), cudaMemcpyHostToDevice);

        // Create matrix descriptors
        double alpha = 1.0;
        double beta = 0.0;
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cudaDataType computeType = CUDA_R_64F;
        cudaDataType valueType = CUDA_R_64F;
        cusparseHandle_t handle = NULL;
        cusparseSpMatDescr_t matA, matB, matC;
        void *dBuffer1 = NULL, *dBuffer2 = NULL;
        size_t bufferSize1 = 0, bufferSize2 = 0;
        cusparseCreate(&handle);

        cusparseCreateCsr(&matA, A_rows, A_cols, A_nnz, d_A_row_ptr, d_A_col_idx, d_A_val,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          valueType);
        cusparseCreateCsr(&matB, B_rows, B_cols, B_nnz, d_B_row_ptr, d_B_col_idx, d_B_val,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          valueType);
        cusparseCreateCsr(&matC, A_rows, B_cols, 0, d_C_row_ptr, NULL, NULL, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, valueType);
        // -------------------------------------------------------------------------------------//
        // Create SpGEMM descriptor
        cusparseSpGEMMDescr_t spgemmDesc;
        cusparseSpGEMM_createDescr(&spgemmDesc);
        cusparseStatus_t status;

        // ask bufferSize1 bytes for external memory
        cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                      &bufferSize1, NULL);
        cudaMalloc(&dBuffer1, bufferSize1);

        // Inspect matrix A and B to understand the memory requirements
        cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc,
                                      &bufferSize1, dBuffer1);

        // ask bufferSize2 bytes for external memory
        cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                               CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, NULL);
        cudaMalloc(&dBuffer2, bufferSize2);

        // compute the intermediate product of A * B
        status =
            cusparseSpGEMM_compute(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                                   CUSPARSE_SPGEMM_DEFAULT, spgemmDesc, &bufferSize2, dBuffer2);

        if (status != CUSPARSE_STATUS_SUCCESS)
        {
            fprintf(stderr, "SpGEMM compute failed with code %d\n", status);
            exit(EXIT_FAILURE);
        }

        // get matrix C non-zero entries
        int64_t C_num_rows1, C_num_cols1, C_nnz;
        cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz);

        // allocate matrix C
        cudaMalloc((void **)&d_C_col_idx, C_nnz * sizeof(int));
        cudaMalloc((void **)&d_C_val, C_nnz * sizeof(double));

        // Update C with the new poitners
        cusparseCsrSetPointers(matC, d_C_row_ptr, d_C_col_idx, d_C_val);
        // -------------------------------------------------------------------------------------//
        cusparseSpGEMM_copy(handle, opA, opB, &alpha, matA, matB, &beta, matC, computeType,
                            CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);
        cudaDeviceSynchronize(); // <-- right after cusparseSpGEMM_copy
        // -------------------------------------------------------------------------------------//
        size_t nnz = (size_t)C_nnz; // safeguard for huge nnz
        int *h_C_row_ptr = (int *)malloc((A_rows + 1) * sizeof(int));
        int *h_C_col_idx = (int *)malloc(nnz * sizeof(int));
        double *h_C_val = (double *)malloc(nnz * sizeof(double));
        cudaMemcpy(h_C_row_ptr, d_C_row_ptr, (A_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_col_idx, d_C_col_idx, nnz * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C_val, d_C_val, nnz * sizeof(double), cudaMemcpyDeviceToHost);
        // -------------------------------------------------------------------------------------//
        // destroy matrix/vector descriptors
        cusparseDestroySpMat(matA);
        cusparseDestroySpMat(matB);
        cusparseDestroySpMat(matC);
        cusparseSpGEMM_destroyDescr(spgemmDesc);
        cusparseDestroy(handle);
        // -------------------------------------------------------------------------------------//
        cudaFree(dBuffer1);
        cudaFree(dBuffer2);
        cudaFree(d_A_row_ptr);
        cudaFree(d_A_col_idx);
        cudaFree(d_A_val);
        cudaFree(d_B_row_ptr);
        cudaFree(d_B_col_idx);
        cudaFree(d_B_val);
        cudaFree(d_C_row_ptr);
        cudaFree(d_C_col_idx);
        cudaFree(d_C_val);

        CSRMatrix C = {
            .row_ptr = h_C_row_ptr,
            .col_idx = h_C_col_idx,
            .val = h_C_val,
            .nnz = (int)nnz,
            .rows = A_rows,
            .cols = B_cols,
        };
        return C;
    }

    void cuda_matmul_free(double *d_A_val, int *d_A_row_ptr, int *d_A_col_idx, double *d_B_val,
                          int *d_B_row_ptr, int *d_B_col_idx)
    {
        // Free device memory
        cudaFree(d_A_val);
        cudaFree(d_A_row_ptr);
        cudaFree(d_A_col_idx);
        cudaFree(d_B_val);
        cudaFree(d_B_row_ptr);
        cudaFree(d_B_col_idx);
    }
}
