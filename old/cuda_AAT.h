#ifndef CUDA_AAT_H
#define CUDA_AAT_H

#include "csr_utils.h"
#ifdef __cplusplus
extern "C"
{
#endif

    CSRMatrix cusparse_compute_AAT(int m, int n, int nnz, const int *csr_row_ptr,
                                   const int *csr_col_idx, const double *csr_val);

#ifdef __cplusplus
}
#endif

#endif // CUDA_AAT_H
