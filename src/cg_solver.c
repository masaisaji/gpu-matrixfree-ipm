#include "cg_solver.h"
#include "cuda_matvec.h"
#include "lin_alg_helper.h"
#include "preconditioner.h"
#include <cblas.h>
#include <cuda_runtime.h>
#include <lapacke.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*----------------------------------------------------------
 * 1) Unpreconditioned Conjugate Gradient
 *---------------------------------------------------------*/
void cg_solve(const double *A, double *x, const double *b, int n, double tol, int max_iter)
{
    double *r = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));

    // r = b - A*x
    matvec(A, x, Ap, n, n, false);
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
        matvec(A, p, Ap, n, n, false);
        double pAp = dot(p, Ap, n);
        double alpha = r_norm_sq / pAp;

        // Update x, r
        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        double new_r_norm_sq = dot(r, r, n);
        if (sqrt(new_r_norm_sq / b_norm_sq) < tol)
        {
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

/* solving for z in L D Lᵗ z = r
 * where L is lower triangular, D is diagonal
 * and r is the right-hand side vector
 */
void ldlt_solve(const double *L, const double *D, const double *r, double *z, int n)
{
    double *tmp = malloc(n * sizeof(double));

    // 1. Forward solve: L y = r
    for (int i = 0; i < n; i++)
    {
        tmp[i] = r[i];
        for (int j = 0; j < i; j++)
        {
            tmp[i] -= L[i * n + j] * tmp[j]; // row-major
        }
    }

    // 2. Diagonal solve: y = y ./ D
    for (int i = 0; i < n; i++)
    {
        tmp[i] /= D[i];
    }

    // 3. Backward solve: Lᵗ z = y
    for (int i = n - 1; i >= 0; i--)
    {
        z[i] = tmp[i];
        for (int j = i + 1; j < n; j++)
        {
            z[i] -= L[j * n + i] * z[j]; // row-major L → Lᵗ access
        }
    }

    free(tmp);
}

// Given: L (lower-triangular n x n), D (diag n)
// Output: invP = P^{-1} (n x n)
void invert_ldlt(const double *L, const double *D, double *invP, int n)
{
    // Step 1: Compute L_inv (L^{-1})
    double *L_inv = alloc_mat(n, n);
    memcpy(L_inv, L, n * n * sizeof(double)); // copy L since dtrtri is in-place

    int info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N', n, L_inv, n);
    if (info != 0)
    {
        fprintf(stderr, "LAPACKE_dtrtri failed with info = %d\n", info);
        exit(EXIT_FAILURE);
    }

    // Step 2: Apply D^{-1}
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            L_inv[i * n + j] /= D[j]; // post-multiply by D^{-1}

    // Step 3: invP = L_invᵀ * L_inv
    matmul(L_inv, L_inv, invP, n, n, n, n, true, false);

    free_mat(L_inv);
}

/*----------------------------------------------------------
 * 2) Preconditioned CG using partial pivoted Cholesky (LDL)
 *---------------------------------------------------------*/
void pcg_solve(const double *A, double *x, const double *b, int n, double tol, int max_iter,
               int max_col, bool verbose, bool use_ldl_solve, bool use_gpu)
{
    bool use_gpu_local = false; // use_gpu;
    double *r = (double *)malloc(n * sizeof(double));
    double *z = (double *)malloc(n * sizeof(double));
    double *p = (double *)malloc(n * sizeof(double));
    double *Ap = (double *)malloc(n * sizeof(double));
    int *perm = NULL;
    double *inv_P = alloc_mat(n, n);

    // Get P first, name it inv_P then invert in-place
    struct timespec p_start, p_end;
    clock_gettime(CLOCK_MONOTONIC, &p_start);
    if (use_ldl_solve)
    {
        double *L, *D;
        get_preconditioner_LD(A, n, max_col, &L, &D, &perm);
        invert_ldlt(L, D, inv_P, n);
        free_mat(L);
        free(D);
    }
    else
    {
        get_preconditioner(A, n, max_col, &inv_P, &perm, false);
    }
    clock_gettime(CLOCK_MONOTONIC, &p_end);
    double p_time_spent =
        (p_end.tv_sec - p_start.tv_sec) + 1e-9 * (p_end.tv_nsec - p_start.tv_nsec);
    if (verbose)
    {
        printf("Time spent calculating preconditioner: %g s\n", p_time_spent);
    };

    struct timespec inv_start, inv_end;
    clock_gettime(CLOCK_MONOTONIC, &inv_start);

    if (verbose)
    {
        printf("inverting preconditioner P\n");
    }
    if (!use_ldl_solve)
    {
        invert_spd_matrix(inv_P, n);
    }
    clock_gettime(CLOCK_MONOTONIC, &inv_end);
    double inv_time_spent =
        (inv_end.tv_sec - inv_start.tv_sec) + 1e-9 * (inv_end.tv_nsec - inv_start.tv_nsec);
    if (verbose)
    {
        printf("Time spent inverting preconditioner matrix: %g s\n", inv_time_spent);
    };

    double *A_perm = alloc_mat(n, n);
    double *b_perm = malloc(n * sizeof(double));
    double *x_perm = calloc(n, sizeof(double)); // x_perm stores y during CG

    int *perm_inv = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        b_perm[i] = b[perm[i]];
        for (int j = 0; j < n; j++)
        {
            MAT(A_perm, i, j, n) = MAT(A, perm[i], perm[j], n);
        }
        perm_inv[perm[i]] = i; // for x = Πᵀ y
    }

    // Initial residual r = b - A*x
    if (use_gpu_local)
    {
        double *d_A = NULL, *d_x = NULL, *d_y = NULL;
        alloc_and_upload_dense_matvec(A_perm, x_perm, n, n, &d_A, &d_x, &d_y);
        cuda_matvec_dense(d_A, d_x, d_y, Ap, n, n, false); // Ap = A_perm * x_perm
        free_cuda_matvec_memory(d_A, d_x, d_y);
    }
    else
    {
        matvec(A_perm, x_perm, Ap, n, n, false);
    }
    for (int i = 0; i < n; i++)
    {
        r[i] = b_perm[i] - Ap[i];
    }

    // z = P^{-1}(r)
    if (use_gpu_local)
    {
        double *d_P = NULL, *d_r = NULL, *d_z = NULL;
        alloc_and_upload_dense_matvec(inv_P, r, n, n, &d_P, &d_r, &d_z);
        cuda_matvec_dense(d_P, d_r, d_z, z, n, n, false); // z = inv_P * r
        free_cuda_matvec_memory(d_P, d_r, d_z);
    }
    else
    {
        matvec(inv_P, r, z, n, n, false);
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
        if (use_gpu_local)
        {
            double *d_A = NULL, *d_x = NULL, *d_y = NULL;
            alloc_and_upload_dense_matvec(A_perm, p, n, n, &d_A, &d_x, &d_y);
            cuda_matvec_dense(d_A, d_x, d_y, Ap, n, n, false); // Ap = A_perm * x_perm
            free_cuda_matvec_memory(d_A, d_x, d_y);
        }
        else
        {
            matvec(A_perm, p, Ap, n, n, false);
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
        if (use_gpu_local)
        {
            double *d_invP = NULL, *d_r = NULL, *d_z = NULL;
            alloc_and_upload_dense_matvec(inv_P, r, n, n, &d_invP, &d_r, &d_z);
            cuda_matvec_dense(d_invP, d_r, d_z, z, n, n, false); // z = inv_P * r
            free_cuda_matvec_memory(d_invP, d_r, d_z);
        }
        else
        {
            matvec(inv_P, r, z, n, n, false);
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
    free_mat(inv_P);
    free(perm);
    free_mat(A_perm);
    free(b_perm);
    free(x_perm);
    free(perm_inv);
}

int main_cg(void)
{
    // Example: n=500, cond_num=1e6 for a bigger ill-conditioned SPD
    {
        int n = 500;
        double cond_num = 1e6; // or 1e8 if you want it even worse

        // allocate
        double *A = (double *)malloc(n * n * sizeof(double));
        double *b = (double *)malloc(n * sizeof(double));
        double *x_cg = (double *)calloc(n, sizeof(double));
        double *x_pcg = (double *)calloc(n, sizeof(double));
        double *x_ref = (double *)calloc(n, sizeof(double));

        // Generate ill-conditioned SPD and b
        generate_ill_conditioned_spd_2(A, n, cond_num);
        generate_rhs_vector(b, n);

        // generate reference solution with x = A^{-1} b
        double *inv_A = alloc_mat(n, n);
        copy_mat(inv_A, A, n, n);

        invert_spd_matrix(inv_A, n);
        matvec(inv_A, b, x_ref, n, n, false);

        printf("Solving ill-conditioned SPD, n=%d, cond_num=%.2e, with CG...\n", n, cond_num);
        struct timespec cg_start, cg_end;
        clock_gettime(CLOCK_MONOTONIC, &cg_start);
        cg_solve(A, x_cg, b, n, 1e-6, 10000);
        clock_gettime(CLOCK_MONOTONIC, &cg_end);
        double cg_time_spent =
            (cg_end.tv_sec - cg_start.tv_sec) + 1e-9 * (cg_end.tv_nsec - cg_start.tv_nsec);
        double cg_sol_error = vec_diff_norm(x_ref, x_cg, n);
        printf("Solution error: %g\n", cg_sol_error);
        printf("Total computataional time: %g s\n", cg_time_spent);

        int max_col = 20;
        printf("\nSolving ill-conditioned SPD with PCG (partial pivoted Cholesky, "
               "max_col=%d)...\n",
               max_col);
        struct timespec pcg_start, pcg_end;
        clock_gettime(CLOCK_MONOTONIC, &pcg_start);
        pcg_solve(A, x_pcg, b, n, 1e-6, 10000, max_col, true, true, false);
        clock_gettime(CLOCK_MONOTONIC, &pcg_end);
        double pcg_time_spent =
            (pcg_end.tv_sec - pcg_start.tv_sec) + 1e-9 * (pcg_end.tv_nsec - pcg_start.tv_nsec);
        double pcg_sol_error = vec_diff_norm(x_ref, x_pcg, n);
        printf("Solution error: %g\n", pcg_sol_error);
        printf("Total computataional time: %g s\n", pcg_time_spent);

        // free everything
        free(A);
        free(b);
        free(x_cg);
        free(x_pcg);
        free(x_ref);
        free_mat(inv_A);
    }

    return 0;
}
