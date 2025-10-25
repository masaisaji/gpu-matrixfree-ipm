#include "partial_cholesky.h"
#include "lin_alg_helper.h"
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*******************************************************************************
 * Cholesky factorization WITHOUT pivoting, partial if max_col < n
 * L is stored in an n x n buffer (row-major), with zeros outside the factor.
 ******************************************************************************/
void cholesky_no_pivoting(const double *A, double *L, int n, int max_col)
{
    if (max_col <= 0 || max_col > n)
    {
        max_col = n;
    }

    // Zero L
    for (int i = 0; i < max_col * max_col; i++)
    {
        L[i] = 0.0;
    }
    // "Classic" algorithm
    for (int col = 0; col < max_col; col++)
    {
        // L[col,col] = sqrt(A[col,col] - sum_{k=0..col-1} L[col,k]^2)
        double sum_sq = 0.0;
        for (int k = 0; k < col; k++)
        {
            double val = MAT(L, col, k, max_col);
            sum_sq += val * val;
        }
        double diag = MAT(A, col, col, n) - sum_sq;
        MAT(L, col, col, max_col) = sqrt(diag);

        // L[row,col] = (A[row,col] - sum_{k=0..col-1} L[row,k]*L[col,k]) /
        // L[col,col]
        for (int row = col + 1; row < max_col; row++)
        {
            double sum_l = 0.0;
            for (int k = 0; k < col; k++)
            {
                sum_l += MAT(L, row, k, max_col) * MAT(L, col, k, max_col);
            }
            double val = (MAT(A, row, col, n) - sum_l) / MAT(L, col, col, max_col);
            MAT(L, row, col, max_col) = val;
        }
    }
}

/*******************************************************************************
 * Pivoted Cholesky for SPD matrix: at each step 'col', pick the largest diag
 * in [col..n-1], swap row col with row pivot, etc.
 * We do partial factor if max_col < n. Store factor in L, and record perm in
 *'perm'.
 ******************************************************************************/
void cholesky_w_pivoting(const double *A, double *L, int *perm, bool allow_pivot, int n,
                         int max_col)
{
    if (max_col <= 0 || max_col > n)
    {
        max_col = n;
    }

    // If pivot is not allowed, just do no-pivot version
    if (!allow_pivot)
    {
        for (int i = 0; i < n * max_col; i++)
        {
            L[i] = 0.0;
        }
        for (int i = 0; i < n; i++)
        {
            perm[i] = i;
        }

        for (int col = 0; col < max_col; col++)
        {
            double sum_sq = 0.0;
            for (int k = 0; k < col; k++)
            {
                double val = MAT(L, col, k, max_col);
                sum_sq += val * val;
            }
            double diag = MAT(A, col, col, n) - sum_sq;
            MAT(L, col, col, max_col) = sqrt(diag);

            for (int row = col + 1; row < n; row++)
            {
                double sum_l = 0.0;
                for (int k = 0; k < col; k++)
                {
                    sum_l += MAT(L, row, k, max_col) * MAT(L, col, k, max_col);
                }
                double val = (MAT(A, row, col, n) - sum_l) / MAT(L, col, col, max_col);
                MAT(L, row, col, max_col) = val;
            }
        }

        return;
    }

    // Zero L (n x max_col)
    for (int i = 0; i < n * max_col; i++)
    {
        L[i] = 0.0;
    }

    // Initialize perm = identity
    for (int i = 0; i < n; i++)
    {
        perm[i] = i;
    }

    // D array holds the diagonal entries of A[perm[i], perm[i]]
    double *D = (double *)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
    {
        D[i] = MAT(A, i, i, n);
    }

    // Pivoted Cholesky
    for (int col = 0; col < max_col; col++)
    {
        // find pivot
        int piv_idx = col;
        double max_val = fabs(D[col]);
        for (int r = col + 1; r < n; r++)
        {
            double v = fabs(D[r]);
            if (v > max_val)
            {
                max_val = v;
                piv_idx = r;
            }
        }

        // swap perm
        int tmpi = perm[col];
        perm[col] = perm[piv_idx];
        perm[piv_idx] = tmpi;

        // swap D
        double tmpd = D[col];
        D[col] = D[piv_idx];
        D[piv_idx] = tmpd;

        // Also swap rows in L (entire row, up to col, because L is lower triangular)
        for (int k = 0; k < col; k++)
        {
            double t = MAT(L, col, k, max_col);
            MAT(L, col, k, max_col) = MAT(L, piv_idx, k, max_col);
            MAT(L, piv_idx, k, max_col) = t;
        }

        // L[col, col]
        MAT(L, col, col, max_col) = sqrt(D[col]);

        for (int row = col + 1; row < n; row++)
        {
            double sum_l = 0.0;
            for (int k = 0; k < col; k++)
            {
                sum_l += MAT(L, row, k, max_col) * MAT(L, col, k, max_col);
            }
            int rperm = perm[row];
            int cperm = perm[col];
            double val = (MAT(A, rperm, cperm, n) - sum_l) / MAT(L, col, col, max_col);
            MAT(L, row, col, max_col) = val;
        }

        for (int row = col + 1; row < n; row++)
        {
            double vv = MAT(L, row, col, max_col);
            D[row] -= vv * vv;
        }
    }

    free(D);
}

/*******************************************************************************
 * LDL decomposition from pivoted Cholesky: (Π^T)A(Π) = L_c L_c^T.
 * Then L = L_c diag(L_c)^{-1}, D = diag(L_c)^2.
 * We only do partial columns up to max_col.
 ******************************************************************************/
void LDL_w_pivoting(const double *A,
                    double *L_out, // n x n, partial factor in top-left
                    double *D_out, // length n, holds diag
                    int *perm_out, // length n
                    bool allow_pivot, int n, int max_col)
{
    if (max_col <= 0 || max_col > n)
    {
        max_col = n;
    }

    // pivoted Cholesky
    double *L_chol = (double *)malloc(n * max_col * sizeof(double));
    int *perm = (int *)malloc(n * sizeof(int));
    cholesky_w_pivoting(A, L_chol, perm, allow_pivot, n, max_col);
    copy_mat(L_out, L_chol, n, max_col);

    for (int col = 0; col < max_col; col++)
    {
        double diag = MAT(L_chol, col, col, max_col);
        D_out[col] = diag * diag;

        for (int row = col; row < n; row++)
        {
            MAT(L_out, row, col, max_col) /= diag;
        }
    }

    // for (int col = 0; col < max_col; col++)
    // {
    //     D_out[col] = MAT(L_chol, col, col, max_col) * MAT(L_chol, col, col, max_col);
    //     for (int row = col; row < max_col; row++)
    //     {
    //         MAT(L_out, row, col, max_col) /= MAT(L_chol, col, col, max_col);
    //     }
    // }
    // copy permutation
    for (int i = 0; i < n; i++)
    {
        perm_out[i] = perm[i];
    }

    free(L_chol);
    free(perm);
}

/*******************************************************************************
 * Quick matrix transpose in-place. (n x n)
 ******************************************************************************/
void transpose_inplace(double *A, int n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            double tmp = MAT(A, i, j, n);
            MAT(A, i, j, n) = MAT(A, j, i, n);
            MAT(A, j, i, n) = tmp;
        }
    }
}

/*******************************************************************************
 * Test routine
 ******************************************************************************/
void test_partial_cholesky(int n, int max_col)
{

    // Generate random SPD matrix A
    double *A = alloc_mat(n, n);
    srand((unsigned)time(NULL));
    make_random_spd(A, n);

    // 2) Reference full Cholesky with LAPACK's dpotrf (no pivot, entire matrix)
    double *A_ref = alloc_mat(n, n);
    copy_mat(A_ref, A, n, n);
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A_ref, n);
    if (info != 0)
    {
        printf("LAPACKE_dpotrf failed, info=%d\n", info);
    }
    // A_ref now has the factor in the lower triangle.

    // Convert that to a full L matrix for comparison: L_chol_ref
    double *L_chol_ref = alloc_mat(n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                MAT(L_chol_ref, i, j, n) = MAT(A_ref, i, j, n);
            }
            else if (i > j)
            {
                MAT(L_chol_ref, i, j, n) = MAT(A_ref, i, j, n);
            }
            else
            {
                MAT(L_chol_ref, i, j, n) = 0.0;
            }
        }
    }

    // Full Cholesky (no pivot) via our code
    double *L_chol = alloc_mat(n, n);
    cholesky_no_pivoting(A, L_chol, n, n);
    // Compare L_chol with L_chol_ref
    double diff = matrix_diff_norm(L_chol, L_chol_ref, n);
    printf("\nFull Cholesky (no pivot) vs. LAPACK dpotrf:\n%g\n", diff);

    // Partial Cholesky (no pivot) up to max_col
    double *L_chol_part = alloc_mat(n, n);
    cholesky_no_pivoting(A, L_chol_part, n, max_col);
    // Compare top-left block with L_chol_ref
    double sumsq_p = 0.0;
    for (int i = 0; i < max_col; i++)
    {
        for (int j = 0; j < max_col; j++)
        {
            double d = MAT(L_chol_part, i, j, n) - MAT(L_chol_ref, i, j, n);
            sumsq_p += d * d;
        }
    }
    printf("\nPartial Cholesky (no pivot), error in top-left:\n%g\n", sqrt(sumsq_p));

    // Full & partial pivoted Cholesky using LAPACK dpstrf
    double *L_pivot = alloc_mat(n, n);
    int *perm = (int *)malloc(n * sizeof(int));
    cholesky_w_pivoting(A, L_pivot, perm, true, n, n);

    double *A_perm2 = alloc_mat(n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            MAT(A_perm2, i, j, n) = MAT(A, perm[i], perm[j], n);
        }
    }
    double *A_recovered = alloc_mat(n, n);
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans, // treat the first matrix as-is
                CblasTrans,   // treat the second matrix as transpose
                n, n, n, 1.0, L_pivot, n, L_pivot, n, 0.0, A_recovered, n);

    printf("\nFull Cholesky (no pivot) vs. LAPACK dpotrf 2:\n%g\n",
           matrix_diff_norm(A_recovered, A_perm2, n));
    free_mat(A_recovered);
    free_mat(A_perm2);

    double *L_pivot_part = alloc_mat(n, n);
    int *perm_part = (int *)malloc(n * sizeof(int));
    cholesky_w_pivoting(A, L_pivot_part, perm_part, true, n, max_col);

    double *A_pstrf = alloc_mat(n, n);
    copy_mat(A_pstrf, A, n, n);
    lapack_int *piv = (lapack_int *)malloc(n * sizeof(lapack_int));
    lapack_int info_pstrf = 0;
    int rank = 0;
    double tol = 0.0; // you can adjust tolerance if matrix is near-singular

    // If you have the older 8-argument dpstrf signature:
    info_pstrf = LAPACKE_dpstrf(LAPACK_ROW_MAJOR, // matrix layout
                                'L',              // store factor in lower triangle
                                n,                // size of matrix
                                A_pstrf,          // the matrix data, in-place
                                n,                // leading dimension
                                piv,              // pivot info (output)
                                &rank,            // numeric rank
                                tol               // tolerance
    );
    if (info_pstrf != 0)
    {
        printf("dpstrf failed, info=%d\n", (int)info_pstrf);
    }
    else
    {
        double *L_chol_ref_piv = alloc_mat(n, n);
        for (int i = 0; i < n * n; i++)
        {
            L_chol_ref_piv[i] = 0.0;
        }
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                L_chol_ref_piv[i * n + j] = A_pstrf[i * n + j];
            }
        }

        int *perm_dpstrf = (int *)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            perm_dpstrf[i] = (int)(piv[i] - 1);
        }

        double diff = matrix_diff_norm(L_pivot, L_chol_ref_piv, n);
        printf("\nFull pivoted Cholesky (dpstrf):\n%g\n", diff);

        // Partial pivoted Cholesky
        // Compare top-left block with L_pivot
        double sumsq_pp = 0.0;
        for (int i = 0; i < max_col; i++)
        {
            for (int j = 0; j < max_col; j++)
            {
                double d = MAT(L_pivot_part, i, j, n) - MAT(L_chol_ref_piv, i, j, n);
                sumsq_pp += d * d;
            }
        }
        printf("\nPartial pivoted Cholesky, error in top-left:\n%g\n", sqrt(sumsq_pp));

        free(L_chol_ref_piv);
        free(perm_dpstrf);
    }

    // Full LDL from pivoted Cholesky
    double *L_ldl = alloc_mat(n, n);
    double *D_ldl = (double *)malloc(n * sizeof(double));
    int *perm_ldl = (int *)malloc(n * sizeof(int));
    double *LDLt = alloc_mat(n, n);

    LDL_w_pivoting(A, L_ldl, D_ldl, perm_ldl, false, n, n);
    // Compute L*D*L (this is slow, but we're just testing)
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += (L_ldl[i * n + k] * D_ldl[k]) * L_ldl[j * n + k];
            }
            LDLt[i * n + j] = sum;
        }
    }

    double diff_ldl = matrix_diff_norm(A, LDLt, n);
    printf("\nFull LDL (no pivot):\n%g\n", diff_ldl);

    // Partial LDL with no pivoting
    // Overwrite L_ldl, D_ldl, perm_ldl
    LDL_w_pivoting(A, L_ldl, D_ldl, perm_ldl, false, n, max_col);
    for (int i = 0; i < max_col; i++)
    {
        for (int j = 0; j < max_col; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += (L_ldl[i * n + k] * D_ldl[k]) * L_ldl[j * n + k];
            }
            LDLt[i * n + j] = sum;
        }
    }
    double err_ldl_partial = 0.0;
    for (int i = 0; i < max_col; i++)
    {
        for (int j = 0; j < max_col; j++)
        {
            double d = MAT(A, i, j, n) - MAT(LDLt, i, j, n);
            err_ldl_partial += d * d;
        }
    }
    printf("\nPartial LDL (no pivot), error in top left:\n%g\n", err_ldl_partial);

    // LDL with pivoting
    LDL_w_pivoting(A, L_ldl, D_ldl, perm_ldl, true, n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += (L_ldl[i * n + k] * D_ldl[k]) * L_ldl[j * n + k];
            }
            LDLt[i * n + j] = sum;
        }
    }

    // Build A[perm,perm]
    double *A_perm = alloc_mat(n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            MAT(A_perm, i, j, n) = MAT(A, perm_ldl[i], perm_ldl[j], n);
        }
    }

    double diff_ldl_piv = matrix_diff_norm(A_perm, LDLt, n);
    printf("\nFull pivoted LDL:\n%g\n", diff_ldl_piv);

    // Partial LDL
    LDL_w_pivoting(A, L_ldl, D_ldl, perm_ldl, true, n, max_col);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < n; k++)
            {
                sum += (L_ldl[i * n + k] * D_ldl[k]) * L_ldl[j * n + k];
            }
            LDLt[i * n + j] = sum;
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            MAT(A_perm, i, j, n) = MAT(A, perm_ldl[i], perm_ldl[j], n);
        }
    }
    // We'll just check top-left block or measure partial error
    double sumsq_ldl_part = 0.0;
    for (int i = 0; i < max_col; i++)
    {
        for (int j = 0; j < max_col; j++)
        {
            double d = MAT(LDLt, i, j, n) - MAT(A_perm, i, j, n);
            sumsq_ldl_part += d * d;
        }
    }
    printf("\nPartial pivoted LDL vs full pivoted LDL, error in top-left:\n%g\n",
           sqrt(sumsq_ldl_part));

    /* Cleanup */
    free_mat(A);
    free_mat(A_ref);
    free_mat(L_chol_ref);
    free_mat(L_chol);
    free_mat(L_chol_part);
    free_mat(L_pivot);
    free(perm);
    free_mat(L_pivot_part);
    free(perm_part);
    free_mat(L_ldl);
    free(D_ldl);
    free(perm_ldl);
    free_mat(A_perm);
    free_mat(LDLt);
    free(A_pstrf);
    free(piv);
}

/*******************************************************************************
 * main
 ******************************************************************************/
int main_partial_chol()
{
    int n = 500;
    int max_col = 30;
    printf("=== Test Partial Cholesky with n=%d, max_col=%d ===\n", n, max_col);

    /* We do one test. You might loop or do multiple tests, etc. */
    test_partial_cholesky(n, max_col);

    return 0;
}
