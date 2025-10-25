#include "csr_utils.h"
#include <stdbool.h>

#ifndef CSR_IPM_H
#define CSR_IPM_H

void IPM_with_preconditioner_csr(const CSRMatrix *A, const double *A_dense, const double *b,
                                 const double *c, double *x0, double *y0, double *s0, int m, int n,
                                 int max_col, bool use_mehrotra, bool use_ldl_solve, bool use_gpu,
                                 bool verbose);

#endif
