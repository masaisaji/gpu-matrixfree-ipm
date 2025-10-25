#pragma once
#include "csr_utils.h"

#ifdef __cplusplus
extern "C"
{
#endif

    CSRMatrix cuda_matmul_csr(const CSRMatrix *A, const CSRMatrix *B);

#ifdef __cplusplus
}
#endif
