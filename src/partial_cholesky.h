#ifndef PARTIAL_CHOLESKY_H
#define PARTIAL_CHOLESKY_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* Cholesky with or without pivoting, partial factor up to max_col. */
    void cholesky_w_pivoting(const double *A, /* n x n input matrix */
                             double *L,       /* n x n output, lower-tri factor */
                             int *perm,       /* length n, output permutation */
                             bool allow_pivot, int n, int max_col);

    /* LDL from pivoted Cholesky. L_out, D_out, perm_out are outputs. */
    void LDL_w_pivoting(const double *A, double *L_out, double *D_out, int *perm_out,
                        bool allow_pivot, int n, int max_col);

#ifdef __cplusplus
}
#endif

#endif
