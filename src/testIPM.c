#include "IPM.h"
#include "cg_solver.h"
#include "csr_IPM.h"
#include "csr_utils.h"
#include "cuda_matvec.h"
#include "gen_initial_guess.h"
#include "lin_alg_helper.h"
#include "mm_loader.h"
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TOL 1e-6
#define EPS 1e-8

void generate_test_lp(double *A, double *b, double *c, double *x_feas, int m, int n,
                      double sparsity)
{
    for (int i = 0; i < n; i++)
    {
        x_feas[i] = ((double)rand() / RAND_MAX) + 0.5; // strictly positive
        c[i] = ((double)rand() / RAND_MAX) + 0.5;
    }
    gen_full_row_rank_sparse_mat(A, m, n, 10, sparsity);
    matvec(A, x_feas, b, m, n, false);
}

int solve_random_lp(void)
{
    int m = 50;
    int n = 500;
    double *A = (double *)calloc(m * n, sizeof(double));
    double *b = (double *)malloc(m * sizeof(double));
    double *c = (double *)malloc(n * sizeof(double));
    double *x0 = malloc(n * sizeof(double));
    double *y0 = malloc(m * sizeof(double));
    double *s0 = malloc(n * sizeof(double));

    generate_test_lp(A, b, c, x0, m, n, 0.2);
    printf("Starting IPM test with known LP solution verification...\n");

    int max_col = m / 2; // rounded down

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < n; i++)
    {
        if (c[i] > 0)
        {
            s0[i] = c[i];
        }
        else
        {
            s0[i] = 1.0;
        }
    }
    for (int i = 0; i < m; i++)
    {
        y0[i] = 0.0;
    }

    IPM_with_preconditioner(A, b, c, x0, y0, s0, m, n, max_col, true, false, true);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("IPM test completed in %.6f seconds.\n", elapsed_time);

    free(A);
    free(b);
    free(c);
    free(x0);
    free(y0);
    free(s0);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Missing argument(s). Expected usage:\n%s <lp_name> <use_gpu: 0 or 1>\n", argv[0]);
        return 1;
    }

    const char *lp_name = argv[1];     // e.g., "nug05"
    bool use_gpu = atoi(argv[2]) != 0; // 1 for true, 0 for false
    // Name of LP problem
    // bool use_gpu = true;
    bool use_ldl_solve = false;
    bool use_mehrotra = true;
    int m, n, nnz;
    int *csr_row_ptr, *csr_col_idx;
    double *csr_val, *b, *c;
    LPPaths paths = get_lp_filepaths(lp_name);

    // Load the LP problem
    if (load_mtx_lp(&paths, &m, &n, &nnz, &csr_row_ptr, &csr_col_idx, &csr_val, &b, &c) != 0)
    {
        free_lp_paths(&paths);
        return 1;
    }
    CSRMatrix A = {
        .row_ptr = csr_row_ptr,
        .col_idx = csr_col_idx,
        .val = csr_val,
        .nnz = (int)nnz,
        .rows = m,
        .cols = n,
    };

    printf("\nLP problem ID: %s\n", lp_name);
    printf("GPU option: %s\n", use_gpu ? "ON" : "OFF");
    printf("Loaded matrix A: %d x %d with %d nonzeros\n", m, n, nnz);
    printf("Loaded vectors b and c\n");
    printf("Starting IPM test with known LP solution verification...\n");
    init_cuda();

    double *x0, *y0, *s0;
    generate_feasible_start(&A, b, c, &x0, &y0, &s0, use_gpu);

    int max_col = m / 4; // rounded down

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    double *A_dense = csr_to_dense_row_major(&A);
    IPM_with_preconditioner_csr(&A, A_dense, b, c, x0, y0, s0, m, n, max_col, use_mehrotra,
                                use_ldl_solve, use_gpu, true);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("IPM test completed in %.6f seconds.\n\n", elapsed_time);

    free_lp_paths(&paths);
    freeCSRmat(&A);
    free(A_dense);
    free(b);
    free(c);
    free(x0);
    free(y0);
    free(s0);
    return 0;
}
