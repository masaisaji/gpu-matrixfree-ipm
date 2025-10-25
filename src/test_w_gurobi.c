#include "csr_utils.h"
#include "cuda_matvec.h"
#include "mm_loader.h"
#include "solve_w_gurobi.h"
#include <float.h>
#include <glpk.h>
#include <gurobi_c.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        printf("Missing argument(s). Expected usage:\n%s <lp_name>\n", argv[0]);
        return 1;
    }

    const char *lp_name = argv[1]; // e.g., "nug05"
    bool use_gpu = false;
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
    printf("Solving LP with Gurobi...\n");
    init_cuda();

    double *x0 = malloc(n * sizeof(double));
    double *A_dense = csr_to_dense_row_major(&A);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    solve_lp_with_gurobi(A_dense, b, c, m, n, x0);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Gurobi solved the test problem in %.6f seconds.\n\n", elapsed_time);

    free_lp_paths(&paths);
    freeCSRmat(&A);
    free(A_dense);
    free(b);
    free(c);
    free(x0);
    return 0;
}
