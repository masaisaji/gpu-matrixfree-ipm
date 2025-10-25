#include "gen_initial_guess.h"
#include "cg_solver.h"
#include "csr_cg_solver.h"
#include "csr_utils.h"
#include "cuda_matmul.h"
#include "cuda_matvec.h"
#include "lin_alg_helper.h"
#include "mm_loader.h"
#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

void form_dense_AAT_from_csr(int m, int n, const int *csr_row_ptr, const int *csr_col_idx,
                             const double *csr_val, double *AAT_dense)
{
    for (int i = 0; i < m; ++i)
    {
        for (int k = i; k < m; ++k)
        {
            double sum = 0.0;

            int pa = csr_row_ptr[i], qa = csr_row_ptr[i + 1];
            int pb = csr_row_ptr[k], qb = csr_row_ptr[k + 1];

            while (pa < qa && pb < qb)
            {
                int col_a = csr_col_idx[pa];
                int col_b = csr_col_idx[pb];

                if (col_a == col_b)
                {
                    sum += csr_val[pa] * csr_val[pb];
                    pa++;
                    pb++;
                }
                else if (col_a < col_b)
                    pa++;
                else
                    pb++;
            }

            AAT_dense[i * m + k] = sum;
            if (i != k)
                AAT_dense[k * m + i] = sum;
        }
    }
}

void generate_feasible_start(const CSRMatrix *A, const double *b, const double *c, double **x0_out,
                             double **y0_out, double **s0_out, bool use_gpu)
{
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int m = A->rows;
    int n = A->cols;

    double *x0 = calloc(n, sizeof(double));
    double *y0 = calloc(m, sizeof(double));
    double *s0 = calloc(n, sizeof(double));
    double *x_slack = calloc(m, sizeof(double));
    double *Ac = calloc(m, sizeof(double));
    double *ATy = calloc(n, sizeof(double));

    CSRMatrix AT = csr_transpose(A);
    CSRMatrix AAT;
    if (use_gpu)
    {
        AAT = cuda_matmul_csr(A, &AT);
    }
    else
    {
        AAT = get_AAT_csr(A);
    }

    // Solve x0 = A^T (A A^T)^-1 b in two steps:
    // First solve A A^T x_slack = b
    cg_solve_csr(&AAT, x_slack, b, 1e-8, 100, use_gpu, false);
    //  then x0 = A^T x_slack
    if (use_gpu)
    {
        cuda_matvec_csr(&AT, x_slack, x0);
    }
    else
    {
        matvec_csr(&AT, x_slack, x0, false);
    }

    free(x_slack);

    // Solve (A A^T) y0 = Ac
    if (use_gpu)
    {
        cuda_matvec_csr(A, c, Ac);
    }
    else
    {
        matvec_csr(A, c, Ac, false);
    }
    cg_solve_csr(&AAT, y0, Ac, 1e-8, 100, use_gpu, false);
    free(Ac);

    // s0 = c - Aᵗ * y0
    if (use_gpu)
    {
        cuda_matvec_csr(&AT, y0, ATy);
    }
    else
    {
        matvec_csr(A, y0, ATy, true);
    }
    vecsub(n, c, ATy, s0, 1);

    free(ATy);

    // positivity adjustments
    double min_xt = x0[0], min_st = s0[0];
    for (int i = 1; i < n; ++i)
    {
        if (x0[i] < min_xt)
            min_xt = x0[i];
        if (s0[i] < min_st)
            min_st = s0[i];
    }
    double delta_x = fmax(0.0, 1.5 * -min_xt);
    double delta_s = fmax(0.0, 1.5 * -min_st);

    for (int i = 0; i < n; ++i)
    {
        x0[i] += delta_x;
        s0[i] += delta_s;
    }

    double min_x0 = min_xt + delta_x;
    double min_s0 = min_st + delta_s;
    if (min_x0 < 1e-10 || min_s0 < 1e-10)
    {
        double xTs = dot(x0, s0, n);
        double eTs = 0.0, eTx = 0.0;

        for (int i = 0; i < n; ++i)
        {
            eTs += s0[i];
            eTx += x0[i];
        }
        double delta_hat_x = 0.5 * xTs / eTs;
        double delta_hat_s = 0.5 * xTs / eTx;

        for (int i = 0; i < n; ++i)
        {
            x0[i] += delta_hat_x;
            s0[i] += delta_hat_s;
        }
    }

    // re-correct y s.t. AA^T y = A(c - s)
    // if s0 is shifted
    if (delta_s > 1e-8)
    {
        double *c_min_s = malloc(n * sizeof(double));
        double *Ac_min_s = malloc(m * sizeof(double));
        vecsub(n, c, s0, c_min_s, 1);
        if (use_gpu)
        {
            cuda_matvec_csr(A, c_min_s, Ac_min_s);
        }
        else
        {
            matvec_csr(A, c_min_s, Ac_min_s, false);
        }
        cg_solve_csr(&AAT, y0, Ac_min_s, 1e-8, 100, use_gpu, false);
    }

    *x0_out = x0;
    *y0_out = y0;
    *s0_out = s0;

    freeCSRmat(&AT);
    freeCSRmat(&AAT);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Feasible start generated in %.6g s.\n", time);
}

void test_cuda_AAT_csr(CSRMatrix *A)
{
    printf("\nTesting AA^T matrix generation with cuda...\n");
    // Dense AAT from trusted code
    double *AAT_dense = calloc(A->rows * A->rows, sizeof(double));
    form_dense_AAT_from_csr(A->rows, A->cols, A->row_ptr, A->col_idx, A->val, AAT_dense);

    // CSR AAT to compare
    CSRMatrix AT = csr_transpose(A);
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix AAT = cuda_matmul_csr(A, &AT);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("AA^T calculated with cusparse in %g\n", time);

    printf("nnz of A: %d\n", A->nnz);
    printf("nnz of AAT_csr: %d\n", AAT.nnz);
    printf("Sparsity of AAT_csr: %.12f%%\n", 1.0 * AAT.nnz / (A->rows * A->rows));
    double *AAT_csr_dense = csr_to_dense_row_major(&AAT);
    double err = matrix_diff_norm(AAT_dense, AAT_csr_dense, AAT.rows);
    printf("‖AAT_dense - AAT_sparse‖_F = %.12e\n", err);

    free(AAT_dense);
    freeCSRmat(&AT);
    freeCSRmat(&AAT);
    free(AAT_csr_dense);
}

void test_cpu_AAT_csr(const CSRMatrix *A)
{
    printf("\nTesting AA^T matrix generation with CPU...\n");

    // Trusted dense AAᵗ
    double *AAT_dense = calloc(A->rows * A->rows, sizeof(double));
    form_dense_AAT_from_csr(A->rows, A->cols, A->row_ptr, A->col_idx, A->val, AAT_dense);

    // CPU-computed AAᵗ in CSR
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    CSRMatrix AAT = get_AAT_csr(A);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("AA^T calculated with CPU in %g seconds\n", elapsed);

    printf("nnz of A: %d\n", A->nnz);
    printf("nnz of AAT_csr: %d\n", AAT.nnz);
    printf("Sparsity of AAT_csr: %.12f%%\n", 100.0 * AAT.nnz / (A->rows * A->rows));

    double *AAT_csr_dense = csr_to_dense_row_major(&AAT);
    double err = matrix_diff_norm(AAT_dense, AAT_csr_dense, A->rows);
    printf("‖AAT_dense - AAT_csr‖_F = %.12e\n", err);

    free(AAT_dense);
    free(AAT_csr_dense);
    freeCSRmat(&AAT);
}

void test_matvec_csr(CSRMatrix *A)
{
    printf("\nTesting matvec with cuda...\n");
    int n = A->cols;
    double *v = malloc(n * sizeof(double));
    generate_rhs_vector(v, n);
    double *y_cpu = malloc(A->rows * sizeof(double));
    double *y_gpu = malloc(A->rows * sizeof(double));
    struct timespec startcpu, endcpu;
    clock_gettime(CLOCK_MONOTONIC, &startcpu);
    matvec_csr(A, v, y_cpu, false);
    clock_gettime(CLOCK_MONOTONIC, &endcpu);
    double timecpu = (endcpu.tv_sec - startcpu.tv_sec) + (endcpu.tv_nsec - startcpu.tv_nsec) / 1e9;
    printf("Matvec calculated with CPU %g\n", timecpu);

    struct timespec startgpu, endgpu;
    clock_gettime(CLOCK_MONOTONIC, &startgpu);
    cuda_matvec_csr(A, v, y_gpu);
    clock_gettime(CLOCK_MONOTONIC, &endgpu);
    double timegpu = (endgpu.tv_sec - startgpu.tv_sec) + (endgpu.tv_nsec - startgpu.tv_nsec) / 1e9;
    printf("Matvec calculated with GPU in %g\n", timegpu);

    double *diff = malloc(A->rows * sizeof(double));
    vecsub(A->rows, y_cpu, y_gpu, diff, 1);
    double err = vec_L2_norm(diff, A->rows);
    printf("‖y_cpu - y_gpu‖_2 = %.12e\n", err);
    free(v);
    free(y_cpu);
    free(y_gpu);
    free(diff);
}

void test_initial_guess_gen(CSRMatrix *A, const double *b, const double *c, bool use_gpu)
{
    printf("\nTesting initial guess generation...\n");
    int m = A->rows;
    int n = A->cols;
    double *x0, *y0, *s0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    generate_feasible_start(A, b, c, &x0, &y0, &s0, use_gpu);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Feasible start generated in %.6g s.\n", time);

    double *Ax0 = malloc(m * sizeof(double));
    double *prim_res = malloc(m * sizeof(double));
    cuda_matvec_csr(A, x0, Ax0);
    vecsub(m, Ax0, b, prim_res, 1);

    double *ATy0 = malloc(n * sizeof(double));
    double *dual_res = malloc(n * sizeof(double));
    matvec_csr(A, y0, ATy0, true);
    for (int i = 0; i < n; ++i)
        dual_res[i] = c[i] - ATy0[i] - s0[i];

    printf("Primal residual norm with x0: %.6g\n", vec_L2_norm(prim_res, m));
    printf("Dual residual norm with y0, s0: %.6g\n", vec_L2_norm(dual_res, n));
    free(x0);
    free(y0);
    free(s0);
    free(Ax0);
    free(prim_res);
    free(ATy0);
    free(dual_res);
}

int main_int_guess(void)
{
    const char *lp_name = "nug15";
    bool use_gpu = false;
    int m, n, nnz;
    int *csr_row_ptr, *csr_col_idx;
    double *csr_val, *b, *c;

    LPPaths paths = get_lp_filepaths(lp_name);
    if (load_mtx_lp(&paths, &m, &n, &nnz, &csr_row_ptr, &csr_col_idx, &csr_val, &b, &c) != 0)
    {
        fprintf(stderr, "Failed to load LP.\n");
        return 1;
    }
    CSRMatrix A = {csr_row_ptr, csr_col_idx, csr_val, nnz, m, n};

    init_cuda();
    test_cuda_AAT_csr(&A);
    test_cpu_AAT_csr(&A);
    test_matvec_csr(&A);
    test_initial_guess_gen(&A, b, c, use_gpu);

    // Free
    free_lp_paths(&paths);
    freeCSRmat(&A);
    free(b);
    free(c);
    return 0;
}
