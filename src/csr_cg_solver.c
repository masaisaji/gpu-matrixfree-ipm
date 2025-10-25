#include "csr_cg_solver.h"
#include "cg_solver.h"
#include "csr_utils.h"
#include "cuda_matvec.h"
#include "lin_alg_helper.h"
#include "preconditioner.h"
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

void cg_solve_csr(const CSRMatrix *A, double *x, const double *b, double tol, int max_iter,
                  bool use_gpu, bool verbose)
{
    if (use_gpu)
    {
        init_cuda();
    }
    // Check if the matrix is square
    {
        int m = A->rows;
        int n = A->cols;
        if (m != n)
        {
            fprintf(stderr, "Matrix is not square: %d x %d\n", m, n);
            return;
        }
        double *r = (double *)malloc(n * sizeof(double));
        double *p = (double *)malloc(n * sizeof(double));
        double *Ap = (double *)malloc(n * sizeof(double));

        // r = b - A*x_ref
        if (use_gpu)
        {
            cuda_matvec_csr(A, x, Ap);
        }
        else
        {
            matvec_csr(A, x, Ap, false);
        }
        for (int i = 0; i < n; i++)
        {
            r[i] = b[i] - Ap[i];
            p[i] = r[i];
        }

        double r_norm_sq = dot(r, r, n);
        double b_norm_sq = dot(b, b, n);
        if (b_norm_sq < 1e-14)
        {
            b_norm_sq = 1.0;
        }

        int k;
        for (k = 0; k < max_iter; k++)
        {
            if (use_gpu)
            {
                cuda_matvec_csr(A, p, Ap);
            }
            else
            {
                matvec_csr(A, p, Ap, false);
            }
            double pAp = dot(p, Ap, n);
            double alpha = r_norm_sq / pAp;
            if (fabs(pAp) < 1e-14)
            {
                fprintf(stderr, "CG breakdown: p^T A p ≈ 0\n");
                break;
            }

            // Update x, r
            for (int i = 0; i < n; i++)
            {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            double new_r_norm_sq = dot(r, r, n);
            if (sqrt(new_r_norm_sq / b_norm_sq) < tol)
            {
                if (verbose)
                    printf("CG converged in %d iterations\n", k + 1);
                break;
            }

            double beta = new_r_norm_sq / r_norm_sq;
            r_norm_sq = new_r_norm_sq;

            for (int i = 0; i < n; i++)
            {
                p[i] = r[i] + beta * p[i];
            }
        }

        free(r);
        free(p);
        free(Ap);
    }
}

void pcg_solve_csr(const CSRMatrix *A, double *x, const double *b, int n, double tol, int max_iter,
                   int max_col, bool verbose, bool use_ldl_solve, bool use_gpu)
{
    double *r = (double *)malloc(n * sizeof(double));
    double *z = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));
    int *perm = NULL;
    double *inv_P_dense = alloc_mat(n, n);

    double *A_dense = csr_to_dense_row_major(A);
    // Get P first, name it inv_P then invert in-place
    if (use_ldl_solve)
    {
        double *L, *D;
        get_preconditioner_LD(A_dense, n, max_col, &L, &D, &perm);
        invert_ldlt(L, D, inv_P_dense, n);
        free_mat(L);
        free(D);
    }
    else
    {
        get_preconditioner(A_dense, n, max_col, &inv_P_dense, &perm, false);
        invert_spd_matrix(inv_P_dense, n);
    }
    CSRMatrix inv_P = dense_to_csr(inv_P_dense, n, n);
    free_mat(inv_P_dense);
    // printf("sparsity of P^-1: %g\n", 100.0 * inv_P.nnz / (inv_P.rows * inv_P.cols));

    double *A_perm_dense = alloc_mat(n, n);
    double *b_perm = malloc(n * sizeof(double));
    double *x_perm = calloc(n, sizeof(double)); // x_perm stores y during CG

    int *perm_inv = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        b_perm[i] = b[perm[i]];
        for (int j = 0; j < n; j++)
        {
            MAT(A_perm_dense, i, j, n) = MAT(A_dense, perm[i], perm[j], n);
        }
        perm_inv[perm[i]] = i; // for x = Πᵀ y
    }
    CSRMatrix A_perm = dense_to_csr(A_perm_dense, n, n);
    free_mat(A_dense);
    free_mat(A_perm_dense);

    // Initial residual r = b - A*x
    if (use_gpu)
    {
        cuda_matvec_csr(&A_perm, x_perm, Ap);
    }
    else
    {
        matvec_csr(&A_perm, x_perm, Ap, false);
    }
    for (int i = 0; i < n; i++)
    {
        r[i] = b_perm[i] - Ap[i];
    }

    // z = P^{-1}(r)
    if (use_gpu)
    {
        cuda_matvec_csr(&inv_P, r, z);
    }
    else
    {
        matvec_csr(&inv_P, r, z, false);
    }
    // p = z
    for (int i = 0; i < n; i++)
    {
        p[i] = z[i];
    }

    double rz_old = dot(r, z, n);
    double b_norm_sq = dot(b_perm, b_perm, n);
    if (b_norm_sq < 1e-14)
    {
        b_norm_sq = 1.0;
    }

    int k;
    for (k = 0; k < max_iter; k++)
    {
        if (use_gpu)
        {
            cuda_matvec_csr(&A_perm, p, Ap);
        }
        else
        {
            matvec_csr(&A_perm, p, Ap, false);
        }
        double pAp = dot(p, Ap, n);
        double alpha = rz_old / pAp;

        // Update x_perm, r
        for (int i = 0; i < n; i++)
        {
            x_perm[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double new_r_norm_sq = dot(r, r, n);
        if (sqrt(new_r_norm_sq / b_norm_sq) < tol)
        {
            if (verbose)
            {
                printf("PCG converged in %d iterations\n", k + 1);
            }
            break;
        }

        // z = P^{-1}(r)
        if (use_gpu)
        {

            cuda_matvec_csr(&inv_P, r, z);
        }
        else
        {
            matvec_csr(&inv_P, r, z, false);
        }
        double rz_new = dot(r, z, n);
        double beta = rz_new / rz_old;
        rz_old = rz_new;

        for (int i = 0; i < n; i++)
        {
            p[i] = z[i] + beta * p[i];
        }
    }

    for (int i = 0; i < n; i++)
    {
        x[i] = x_perm[perm_inv[i]]; // x = Πᵀ x_perm
    }

    free(r);
    free(z);
    free(p);
    free(Ap);
    freeCSRmat(&A_perm);
    freeCSRmat(&inv_P);
    free(perm);
    free(b_perm);
    free(x_perm);
    free(perm_inv);
}

int main_cg_csr(void)
{
    int n = 500;
    double cond_num = 1e6;

    double *A_dense = malloc(n * n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *x_ref = calloc(n, sizeof(double));
    double *x_cpu = calloc(n, sizeof(double));
    double *x_cpu_csr = calloc(n, sizeof(double));
    double *x_gpu = calloc(n, sizeof(double));
    double *x_pcg_csr = calloc(n, sizeof(double));
    double *x_pcg_gpu_csr = calloc(n, sizeof(double));

    generate_ill_conditioned_spd_2(A_dense, n, cond_num);
    generate_rhs_vector(b, n);

    // Reference solution
    double *inv_A = alloc_mat(n, n);
    copy_mat(inv_A, A_dense, n, n);
    invert_spd_matrix(inv_A, n);
    matvec(inv_A, b, x_ref, n, n, false);
    free_mat(inv_A);

    // Solve with CPU CG
    cg_solve(A_dense, x_cpu, b, n, 1e-8, 10000);

    // Convert to CSR
    CSRMatrix A_csr = dense_to_csr(A_dense, n, n);
    // Solve CG for CSR with GPU
    cg_solve_csr(&A_csr, x_gpu, b, 1e-8, 10000, true, true);
    // Solve CG for CSR Matrix with CPU
    cg_solve_csr(&A_csr, x_cpu_csr, b, 1e-8, 10000, true, false);
    // Solve PCG for CSR Matrix with CPU
    pcg_solve_csr(&A_csr, x_pcg_csr, b, n, 1e-8, 10000, 50, true, false, false);
    // Solve PCG for CSR Matrix with GPU
    pcg_solve_csr(&A_csr, x_pcg_gpu_csr, b, n, 1e-8, 10000, 50, true, false, true);

    // Compare CPU CG results, dense vs CSR
    double diff_csr = 0.0;
    for (int i = 0; i < n; i++)
        diff_csr += (x_cpu_csr[i] - x_ref[i]) * (x_cpu_csr[i] - x_ref[i]);
    printf("L2 diff (CSR CG vs Dense CG): %.12e\n", sqrt(diff_csr));

    // Compare CPU vs GPU results
    double diff = 0.0;
    for (int i = 0; i < n; i++)
        diff += (x_cpu[i] - x_gpu[i]) * (x_cpu[i] - x_gpu[i]);
    printf("L2 diff (GPU CG vs CPU CG): %.12e\n", sqrt(diff));

    // Compare PCG vs CG results
    double diff_pcg = 0.0;
    for (int i = 0; i < n; i++)
        diff_pcg += (x_pcg_csr[i] - x_ref[i]) * (x_pcg_csr[i] - x_ref[i]);
    printf("L2 diff (PCG CSR vs Dense CG): %.12e\n", sqrt(diff_pcg));

    // Compare GPU PCG vs CG results
    double diff_pcg_gpu = 0.0;
    for (int i = 0; i < n; i++)
        diff_pcg_gpu += (x_pcg_gpu_csr[i] - x_ref[i]) * (x_pcg_gpu_csr[i] - x_ref[i]);
    printf("L2 diff (GPU PCG CSR vs Dense CG): %.12e\n", sqrt(diff_pcg_gpu));

    // Clean up
    free(A_dense);
    free(b);
    free(x_ref);
    free(x_cpu);
    free(x_cpu_csr);
    free(x_gpu);
    free(x_pcg_csr);
    free(x_pcg_gpu_csr);
    freeCSRmat(&A_csr);

    return 0;
}
