#include "lin_alg_helper.h"   // matrix operations
#include "partial_cholesky.h" // pivoted partial LDL and Cholesky
#include <assert.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Compute D_2 = diag(bottom (n-max_col) x (n-max_col) submatrix of A_perm)
 *  - diag(L_21 D_1 L_21^T)
 *  where A_perm is n x n, L_21 is (n - max_col) x max_col,
 *  D_1 is a vector of size max_col, D_2 is a vector of size n-max_col
 *  */
void compute_D2(const double *A_perm, const double *L_21, const double *D_1, double *D_2, int n,
                int max_col)
{
    for (int row = 0; row < n - max_col; ++row)
    {
        double sum = 0.0;
        for (int col = 0; col < max_col; ++col)
        {
            sum += D_1[col] * MAT(L_21, row, col, max_col) *
                   MAT(L_21, row, col, max_col); // avoiding pow for speedup
        }
        D_2[row] = MAT(A_perm, max_col + row, max_col + row, n) - sum;
    };
}

double compute_mat_density(const double *P, int n, double tol)
{
    int nnz = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (fabs(MAT(P, i, j, n)) > tol)
            {
                nnz++;
            }
        }
    }

    double total = (double)(n * n);
    return (double)nnz / total;
}

void get_preconditioner_LD(const double *A_in, int n, int max_col, double **L_out, double **D_out,
                           int **perm_out)
{
    if (max_col > n)
    {
        fprintf(stderr, "Error: max_col (%d) > n (%d)\n", max_col, n);
        return;
    }

    // Allocate intermediate blocks
    double *L_rec = alloc_mat(n, max_col); // n x max_col
    double *D_1_vec = (double *)malloc(max_col * sizeof(double));
    double *D_2_vec = (double *)malloc((n - max_col) * sizeof(double));
    int *perm = (int *)malloc(n * sizeof(int));
    double *L_11 = alloc_mat(max_col, max_col);
    double *L_21 = alloc_mat(n - max_col, max_col);

    // Factor
    LDL_w_pivoting(A_in, L_rec, D_1_vec, perm, true, n, max_col);

    // Extract L_11 and L_21 from L_rec
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < max_col; j++)
        {
            if (i < max_col)
                MAT(L_11, i, j, max_col) = MAT(L_rec, i, j, max_col);
            else
                MAT(L_21, i - max_col, j, max_col) = MAT(L_rec, i, j, max_col);
        }
    }

    // Compute A_perm for D2 computation
    double *A_perm = alloc_mat(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            MAT(A_perm, i, j, n) = MAT(A_in, perm[i], perm[j], n);

    // Compute D2 via Schur complement approx
    compute_D2(A_perm, L_21, D_1_vec, D_2_vec, n, max_col);

    // Allocate full L and D
    double *L = alloc_mat(n, n);
    double *D = malloc(n * sizeof(double));
    for (int i = 0; i < n * n; i++)
        L[i] = 0.0;

    // Fill L (row-major)
    for (int i = 0; i < max_col; i++)
    {
        for (int j = 0; j <= i; j++)
            L[i * n + j] = L_11[i * max_col + j];
    }
    for (int i = 0; i < n - max_col; i++)
    {
        for (int j = 0; j < max_col; j++)
            L[(i + max_col) * n + j] = L_21[i * max_col + j];
    }
    for (int i = max_col; i < n; i++)
    {
        L[i * n + i] = 1.0;
    }

    // Fill D
    for (int i = 0; i < max_col; i++)
        D[i] = D_1_vec[i];
    for (int i = 0; i < n - max_col; i++)
        D[max_col + i] = D_2_vec[i];

    // Output
    *L_out = L;
    *D_out = D;
    *perm_out = perm;

    // Clean up intermediates
    free_mat(L_rec);
    free_mat(A_perm);
    free_mat(L_11);
    free_mat(L_21);
    free(D_1_vec);
    free(D_2_vec);
}

/* Compute Preconditioning Matrix P for n x n matrix A_in
 * The preconditioner "normalizes" max_col-largest eigenvalues of A_in,
 * effectively lowering the condition number. The preconditioner must be
 * used for the matrix after permutation; the permutation is returned in
 * perm_out. i.e., κ(P⁻¹ΠᵀAΠ) is much lower than κ(A) where Π is the
 * permutation matrix.
 */
void get_preconditioner(const double *A_in, int n, int max_col,
                        double **P_out, /* returns allocated P matrix */
                        int **perm_out, /* returns allocated perm array */
                        bool is_test)
{
    if (max_col > n)
    {
        fprintf(stderr,
                "Error: argument max_col must be smaller than matrix size n. "
                "max_col=%d > n=%d\n",
                max_col, n);
        return;
    }

    if (is_test)
    {
        printf("matrix size: %d x %d, max_col: %d\n", n, n, max_col);
    }
    /* Partial pivoted LDL => L_11, D_1, perm */
    double *L_rec = alloc_mat(n, max_col);
    double *D_1_vec = (double *)malloc(max_col * sizeof(double));
    int *perm = (int *)malloc(n * sizeof(int));
    double *L_11 = alloc_mat(max_col, max_col);
    double *L_21 = alloc_mat(n - max_col, max_col);
    double *D_2_vec = (double *)malloc((n - max_col) * sizeof(double));

    LDL_w_pivoting(A_in, L_rec, D_1_vec, perm, true, n, max_col);

    /*  Build A_perm = A_in[perm,perm] and extract its submatrices */
    double *A_perm = alloc_mat(n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            MAT(A_perm, i, j, n) = MAT(A_in, perm[i], perm[j], n);
        }
    }

    // Extract L_rec of size (n x max_col), row-major
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < max_col; col++)
        {
            if (row < max_col)
            {
                // Fill L_11
                MAT(L_11, row, col, max_col) = MAT(L_rec, row, col, max_col);
            }
            else
            {
                // Fill L_21
                MAT(L_21, row - max_col, col, max_col) = MAT(L_rec, row, col, max_col);
            }
        }
    }
    free_mat(L_rec);
    compute_D2(A_perm, L_21, D_1_vec, D_2_vec, n, max_col);

    // Step 1: Create block_mat_1 = [L_11, 0; L_21, I]
    double *block_mat_1 = alloc_mat(n, n);
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (row < max_col && col < max_col)
            {
                MAT(block_mat_1, row, col, n) = MAT(L_11, row, col, max_col);
            }
            else if (row >= max_col && col < max_col)
            {
                MAT(block_mat_1, row, col, n) = MAT(L_21, row - max_col, col, max_col);
            }
            else if (row == col)
            {
                MAT(block_mat_1, row, col, n) = 1.0; // identity
            }
            else
            {
                MAT(block_mat_1, row, col, n) = 0.0;
            }
        }
    }

    // Step 2: Multiply block_mat_1 * diag(D1, D2) to get block_mat_2
    double *block_mat_2 = alloc_mat(n, n);
    for (int row = 0; row < n; row++)
    {
        for (int col = 0; col < n; col++)
        {
            if (col < max_col)
            {
                MAT(block_mat_2, row, col, n) = MAT(block_mat_1, row, col, n) * D_1_vec[col];
            }
            else
            {
                MAT(block_mat_2, row, col, n) =
                    MAT(block_mat_1, row, col, n) * D_2_vec[col - max_col];
            }
        }
    }

    // Step 3: Final multiplication: P = block_mat_2 * block_mat_1ᵀ
    double *P = alloc_mat(n, n);
    matmul(block_mat_2, block_mat_1, P, n, n, n, n, false, true);

    // Now P is your preconditioner: return or use it for testing

    // Cleanup
    free_mat(block_mat_1);
    free_mat(block_mat_2);

    /* Test routine to make sure the original matrix can be recovered,
     * and Schur complement diagonal elements are correctly calculated*/
    if (is_test)
    {
        double *inv_L_11 = alloc_mat(max_col, max_col);
        copy_mat(inv_L_11, L_11, max_col, max_col);
        invert_matrix_inplace(inv_L_11, max_col);

        // A_11 (max_col x max_col)
        double *A_perm_11 = alloc_mat(max_col, max_col);
        for (int row = 0; row < max_col; row++)
        {
            for (int col = 0; col < max_col; col++)
            {
                MAT(A_perm_11, row, col, max_col) = MAT(A_perm, row, col, n);
            }
        }

        // inv(A_11) (max_col x max_col)
        double *inv_A_perm_11 = alloc_mat(max_col, max_col);
        // copy A perm first
        copy_mat(inv_A_perm_11, A_perm_11, max_col, max_col);
        // A_11 is SPD bc it is a principal submatrix of A, which is SPD
        printf("inverting submatrix of GR\n");
        invert_spd_matrix(inv_A_perm_11, max_col);

        // A_21 (n-max_col x max_col)
        double *A_perm_21 = alloc_mat(n - max_col, max_col);
        for (int row = 0; row < n - max_col; row++)
        {
            for (int col = 0; col < max_col; col++)
            {
                MAT(A_perm_21, row, col, max_col) = MAT(A_perm, row + max_col, col, n);
            }
        }

        // L_21 (n-max_col x max_col)= A_perm_21 (n -max_col x max_col) * inv(L_11')
        // (max_col x max_col)* inv(D_1) (max_col x max_col)
        double *A_21_inv_L_11_T = alloc_mat(n - max_col, max_col);
        matmul(A_perm_21, inv_L_11, A_21_inv_L_11_T, n - max_col, max_col, max_col, max_col, false,
               true);
        for (int col = 0; col < max_col; col++)
        {
            for (int row = 0; row < n - max_col; row++)
            {
                MAT(L_21, row, col, max_col) =
                    MAT(A_21_inv_L_11_T, row, col, max_col) / D_1_vec[col];
            }
        }
        free_mat(inv_L_11);
        free_mat(A_21_inv_L_11_T);

        double *D_2_vec_test = (double *)malloc((n - max_col) * sizeof(double));
        compute_D2(A_perm, L_21, D_1_vec, D_2_vec_test, n, max_col);

        // block_mat_1 = [L_11, 0; L_21 , I]
        double *block_mat_1 = alloc_mat(n, n);
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < n; col++)
            {
                if (row < max_col)
                {
                    if (col < max_col)
                    {
                        MAT(block_mat_1, row, col, n) = MAT(L_11, row, col, max_col);
                    }
                    else
                    {
                        MAT(block_mat_1, row, col, n) = 0.0;
                    }
                }
                else
                {
                    if (col < max_col)
                    {
                        MAT(block_mat_1, row, col, n) = MAT(L_21, row - max_col, col, max_col);
                    }
                    else
                    {
                        MAT(block_mat_1, row, col, n) = (row == col ? 1.0 : 0.0);
                    }
                }
            }
        }

        // block_mat_2 = block_mat_1 * [D_1, 0; 0, D_2]
        double *block_mat_2 = alloc_mat(n, n);
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < n; col++)
            {
                if (col < max_col)
                {
                    MAT(block_mat_2, row, col, n) = MAT(block_mat_1, row, col, n) * D_1_vec[col];
                }
                else
                {
                    MAT(block_mat_2, row, col, n) =
                        MAT(block_mat_1, row, col, n) * D_2_vec_test[col - max_col];
                }
            }
        }

        // P = block_mat_2 * block_mat_1'
        matmul(block_mat_2, block_mat_1, P, n, n, n, n, false, true);
        free_mat(block_mat_2);

        double *A_perm_22 = alloc_mat(n - max_col, n - max_col);
        for (int row = 0; row < n - max_col; row++)
        {
            for (int col = 0; col < n - max_col; col++)
            {
                MAT(A_perm_22, row, col, n - max_col) =
                    MAT(A_perm, row + max_col, col + max_col, n);
            }
        }

        // schur_comp (n-max_col x n-max_col) = A_22(n-max_col x n-max_col) - A_21
        // (n-max_col x max_col)* inv(A_11) (max_col x max_col)* A_21' (max_col x
        // n-max_col)
        double *A_21_inv_A_11 = alloc_mat(n - max_col, max_col);
        matmul(A_perm_21, inv_A_perm_11, A_21_inv_A_11, n - max_col, max_col, max_col, max_col,
               false, false);
        double *A_21_inv_A_11_A_21_T = alloc_mat(n - max_col, n - max_col);
        matmul(A_21_inv_A_11, A_perm_21, A_21_inv_A_11_A_21_T, n - max_col, max_col, n - max_col,
               max_col, false, true);

        double *schur_comp = alloc_mat(n - max_col, n - max_col);
        matsub(A_perm_22, A_21_inv_A_11_A_21_T, schur_comp, n - max_col, n - max_col);

        free_mat(A_perm_22);
        free_mat(A_21_inv_A_11);
        free_mat(A_21_inv_A_11_A_21_T);

        // D2_ref = diag(schur_comp)
        double *D_2_vec_ref = (double *)malloc((n - max_col) * sizeof(double));
        for (int i = 0; i < n - max_col; i++)
        {
            D_2_vec_ref[i] = MAT(schur_comp, i, i, n - max_col);
        }
        printf("comparing D2 accuracy: %g\n",
               vec_diff_norm(D_2_vec_test, D_2_vec_ref, n - max_col));

        // block_mat_test = [D_1, 0; 0, schur_comp]
        double *block_mat_test = alloc_mat(n, n);
        for (int row = 0; row < n; row++)
        {
            for (int col = 0; col < n; col++)
            {
                if (row < max_col && col < max_col)
                {
                    if (row == col)
                    {
                        MAT(block_mat_test, row, col, n) = D_1_vec[col];
                    }
                    else
                    {
                        MAT(block_mat_test, row, col, n) = 0.0;
                    }
                }
                else if (row >= max_col && col >= max_col)
                {
                    MAT(block_mat_test, row, col, n) =
                        MAT(schur_comp, row - max_col, col - max_col, n - max_col);
                }
                else
                {
                    MAT(block_mat_test, row, col, n) = 0.0;
                }
            }
        }

        free_mat(schur_comp);
        free(D_2_vec_ref);
        free(D_2_vec_test);

        // block_mat_test_2 = block_mat_1 * block_mat_test
        double *block_mat_test_2 = alloc_mat(n, n);
        matmul(block_mat_1, block_mat_test, block_mat_test_2, n, n, n, n, false, false);
        // recovered_A_perm = block_mat_test_2 * block_mat_1'
        double *recovered_A_perm = alloc_mat(n, n);
        matmul(block_mat_test_2, block_mat_1, recovered_A_perm, n, n, n, n, false, true);
        free_mat(block_mat_test);
        free_mat(block_mat_test_2);
        printf("\nRecovered Matrix Error:\n%g\n", matrix_diff_norm(recovered_A_perm, A_perm, n));
        free_mat(recovered_A_perm);
        free_mat(A_perm_11);
        free_mat(inv_A_perm_11);
        free_mat(block_mat_1);
        free_mat(A_perm_21);
    }

    free_mat(L_11);
    free_mat(L_21);
    free_mat(A_perm);
    free(D_1_vec);
    free(D_2_vec);
    if (is_test)
    {
        double sparsity = compute_mat_density(P, n, 1e-12);
        printf("Preconditioner density: %g\n", sparsity);
    }
    /* Write P into *P_out, perm into *perm_out */
    *P_out = P;
    *perm_out = perm;
}

/*==============================================================================
  main()
==============================================================================*/
int main_precond(void)
{
    int n = 100; // size of the original matrix A (n x n)
    int max_col = 30;

    // Generate ill-conditioned SPD
    double cond_num = 1e5;
    double *A = generate_ill_conditioned_spd(n, cond_num);

    // Build preconditioner from partial LDL approach
    double *P = NULL;
    int *perm = NULL;
    get_preconditioner(A, n, max_col, &P, &perm, false);

    // Compare condition number
    double *A_perm = alloc_mat(n, n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            MAT(A_perm, i, j, n) = MAT(A, perm[i], perm[j], n);
        }
    }
    invert_spd_matrix(P, n);
    double *inv_P_A_perm = alloc_mat(n, n);
    matmul(P, A_perm, inv_P_A_perm, n, n, n, n, false, false);
    printf("Condition number of A: %g\n", estimate_condition_number(A, n));
    printf("Condition number of P^-1 A_perm: %g\n", estimate_condition_number(inv_P_A_perm, n));

    free_mat(A_perm);
    free_mat(inv_P_A_perm);
    free_mat(A);
    if (P)
    {
        free_mat(P);
    }
    if (perm)
    {
        free(perm);
    }
    return 0;
}
