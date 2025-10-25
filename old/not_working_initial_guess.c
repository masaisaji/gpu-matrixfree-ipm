#include "cs.h"
#include "lin_alg_helper.h"
#include "mm_loader.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Convert COO to CSparse CSC format */
cs *coo_to_CS_csc(int m, int n, int nnz, const int *row_indices, const int *col_indices,
                  const double *values)
{
    // Step 1: Allocate triplet matrix
    cs *T = cs_spalloc(m, n, nnz, 1, 1); // triplet format

    // Step 2: Fill triplet entries
    for (int k = 0; k < nnz; ++k)
    {
        cs_entry(T, row_indices[k], col_indices[k], values[k]);
    }

    // Step 3: Compress to CSC format
    cs *A = cs_compress(T);

    // Optional: remove duplicates (if any)
    cs_dupl(A);

    // Cleanup
    cs_spfree(T);
    return A; // Now A is ready for SpMV, multiplication, etc.
}

/**
 * Solves the linear system A x = b using sparse Cholesky factorization,
 * where A is a symmetric positive definite (SPD) matrix stored in CSC format.
 *
 * This function uses CSparse's symbolic and numeric Cholesky decomposition
 * routines: `cs_schol` and `cs_chol`, and performs the solve in three steps:
 *   1. Symbolic analysis (with AMD or other fill-reducing ordering)
 *   2. Numeric Cholesky factorization (A = L * Lᵗ)
 *   3. Forward and backward substitution (L y = P * b, then Lᵗ x = y)
 *
 * A permutation is always applied to reduce fill-in.
 * Caller must ensure A is symmetric positive definite.
 *
 * @param[in]  A   Pointer to input SPD matrix in CSC format (size n x n)
 * @param[in]  b   Right-hand side vector (length n)
 * @param[out] x   Solution vector (length n), overwritten with result
 *
 * @return 1 if successful, 0 if any step fails (invalid input or numerical failure)
 */
int cs_solve_SPD_w_chol(const cs *A, const double *b, double *x)
{
    if (!A || !b || !x || A->m != A->n)
    {
        fprintf(stderr, "Invalid input to cs_solve_SPD_w_chol\n");
        return 0;
    }

    int n = A->n;

    css *S = cs_schol(1, A); // symbolic analysis with AMD ordering
    if (!S)
    {
        fprintf(stderr, "Symbolic Cholesky failed\n");
        return 0;
    }

    csn *N = cs_chol(A, S);
    if (!N)
    {
        fprintf(stderr, "Numeric Cholesky failed\n");
        cs_sfree(S);
        return 0;
    }

    double *tmp = (double *)calloc(n, sizeof(double));
    if (!tmp)
    {
        fprintf(stderr, "Memory allocation failed\n");
        cs_nfree(N);
        cs_sfree(S);
        return 0;
    }

    // Apply permutation: tmp = P * b
    cs_ipvec(S->pinv, b, tmp, n);

    // Solve L y = P b
    cs_lsolve(N->L, tmp);

    // Solve Lᵗ x = y
    cs_ltsolve(N->L, tmp);

    // Inverse permutation: x = Pᵗ * y
    cs_pvec(S->pinv, tmp, x, n);

    free(tmp);
    cs_nfree(N);
    cs_sfree(S);
    return 1;
}

// Matrix-vector multiplication: y = A x (A is m x n)
void spmv_coo_Ax(int m, int n, int nz, const int *row_i, const int *col_j, const double *val,
                 const double *x, double *y)
{
    for (int i = 0; i < m; ++i)
        y[i] = 0.0;
    for (int k = 0; k < nz; ++k)
    {
        y[row_i[k]] += val[k] * x[col_j[k]];
    }
}

// Matrix-vector multiplication: y = A^T x (A is m x n)
void spmv_coo_ATx(int m, int n, int nz, const int *row_i, const int *col_j, const double *val,
                  const double *x, double *y)
{
    for (int j = 0; j < n; ++j)
        y[j] = 0.0;
    for (int k = 0; k < nz; ++k)
    {
        y[col_j[k]] += val[k] * x[row_i[k]];
    }
}

void generate_feasible_start(int m, int n, int nz, const int *row_i, const int *col_j,
                             const double *val, const double *b, const double *c, double **x0_out,
                             double **y0_out, double **s0_out)
{
    cs *A = coo_to_CS_csc(m, n, nz, row_i, col_j, val);
    cs *AT = cs_transpose(A, 1);
    cs *AAT = cs_multiply(A, AT);
    double *x_tilde = calloc(n, sizeof(double));
    double *y_tilde = calloc(m, sizeof(double));
    double *s_tilde = calloc(n, sizeof(double));

    // Solve x_tilde = A^T (A A^T)^-1 b in two steps:
    // First solve A A^T x_slack = b, then x_tilde = A^T x_slack
    double *x_slack = calloc(n, sizeof(double));
    cs_solve_SPD_w_chol(AAT, b, x_slack);
    cs_gaxpy(A, x_slack, x_tilde);
    free(x_slack);

    // Solve y_tilde = (A A^T)^-1 Ac
    double *Ac = calloc(m, sizeof(double));
    cs_gaxpy(A, c, Ac);
    cs_solve_SPD_w_chol(AAT, Ac, y_tilde);
    free(Ac);

    // Solve s_tilde = c - A^T y_tilde
    double *ATy = calloc(n, sizeof(double));
    cs_gaxpy(AT, y_tilde, ATy);
    vecsub(n, c, ATy, s_tilde, 1);
    free(ATy);

    cs_spfree(A);
    cs_spfree(AT);
    cs_spfree(AAT);

    // Step 4: positivity adjustments
    double min_xt = x_tilde[0], min_st = s_tilde[0];
    for (int i = 1; i < n; ++i)
    {
        if (x_tilde[i] < min_xt)
            min_xt = x_tilde[i];
        if (s_tilde[i] < min_st)
            min_st = s_tilde[i];
    }
    double delta_x = fmax(0.0, 1.5 * -min_xt);
    double delta_s = fmax(0.0, 1.5 * -min_st);

    double *x_hat = malloc(n * sizeof(double));
    double *s_hat = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
    {
        x_hat[i] = x_tilde[i] + delta_x;
        s_hat[i] = s_tilde[i] + delta_s;
    }

    double xTs = 0.0, eTs = 0.0, eTx = 0.0;
    for (int i = 0; i < n; ++i)
    {
        xTs += x_hat[i] * s_hat[i];
        eTs += s_hat[i];
        eTx += x_hat[i];
    }
    double delta_hat_x = 0.5 * xTs / eTs;
    double delta_hat_s = 0.5 * xTs / eTx;

    double *x0 = malloc(n * sizeof(double));
    double *s0 = malloc(n * sizeof(double));
    for (int i = 0; i < n; ++i)
    {
        x0[i] = x_hat[i] + delta_hat_x;
        s0[i] = s_hat[i] + delta_hat_s;
    }

    *x0_out = x0;
    *s0_out = s0;
    *y0_out = y_tilde;

    free(x_tilde);
    free(y_tilde);
    free(s_tilde);
    free(x_hat);
    free(s_hat);
    free(Ac);
}

int main(void)
{
    const char *lp_name = "nug30";
    int m, n, nz;
    int *row_i, *col_j;
    double *val, *b, *c;

    LPPaths paths = get_lp_filepaths(lp_name);
    if (load_mtx_lp(&paths, &m, &n, &nz, &row_i, &col_j, &val, &b, &c) != 0)
    {
        fprintf(stderr, "Failed to load LP.\n");
        return 1;
    }

    double *x0, *y0, *s0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    generate_feasible_start(m, n, nz, row_i, col_j, val, b, c, &x0, &y0, &s0);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Feasible start generated in %.6g.\nFirst 5 entries of x0 and s0:\n", time);
    for (int i = 0; i < 5 && i < n; ++i)
        printf("x0[%d] = %.6f, s0[%d] = %.6f\n", i, x0[i], i, s0[i]);

    // Free
    free_lp_paths(&paths);
    free(row_i);
    free(col_j);
    free(val);
    free(b);
    free(c);
    free(x0);
    free(y0);
    free(s0);
    return 0;
}
