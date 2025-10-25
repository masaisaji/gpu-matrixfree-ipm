#include "csr_utils.h"
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void freeCSRmat(CSRMatrix *mat)
{
    if (mat->row_ptr)
        free(mat->row_ptr);
    if (mat->col_idx)
        free(mat->col_idx);
    if (mat->val)
        free(mat->val);

    // Optional: zero fields to avoid dangling pointers
    mat->row_ptr = NULL;
    mat->col_idx = NULL;
    mat->val = NULL;
    mat->nnz = 0;
    mat->rows = 0;
    mat->cols = 0;
}

CSRMatrix csr_transpose(const CSRMatrix *A)
{
    int m = A->rows;
    int n = A->cols;
    int nnz = A->nnz;

    // Allocate transpose CSR structure
    int *row_ptr_T = (int *)calloc(n + 1, sizeof(int));
    int *col_idx_T = (int *)malloc(nnz * sizeof(int));
    double *val_T = (double *)malloc(nnz * sizeof(double));

    // Step 1: Count nonzeros in each column of A
    for (int i = 0; i < nnz; i++)
    {
        int col = A->col_idx[i];
        row_ptr_T[col + 1]++;
    }

    // Step 2: Cumulative sum to get row_ptr of A^T
    for (int i = 0; i < n; i++)
    {
        row_ptr_T[i + 1] += row_ptr_T[i];
    }

    // Step 3: Fill col_idx and val of A^T
    int *counter = (int *)calloc(n, sizeof(int));
    for (int row = 0; row < m; row++)
    {
        for (int jj = A->row_ptr[row]; jj < A->row_ptr[row + 1]; jj++)
        {
            int col = A->col_idx[jj];
            int dst = row_ptr_T[col] + counter[col]++;
            col_idx_T[dst] = row;
            val_T[dst] = A->val[jj];
        }
    }

    free(counter);

    CSRMatrix AT = {
        .row_ptr = row_ptr_T, .col_idx = col_idx_T, .val = val_T, .nnz = nnz, .rows = n, .cols = m};

    return AT;
}

double *csr_to_dense_row_major(const CSRMatrix *A)
{
    int m = A->rows;
    int n = A->cols;
    int *csr_row_ptr = A->row_ptr;
    int *csr_col_idx = A->col_idx;
    double *csr_values = A->val;

    double *A_dense = (double *)calloc(m * n, sizeof(double));
    if (!A_dense)
    {
        perror("Failed to allocate dense matrix");
        return NULL;
    }

    for (int i = 0; i < m; ++i)
    {
        for (int idx = csr_row_ptr[i]; idx < csr_row_ptr[i + 1]; ++idx)
        {
            int j = csr_col_idx[idx];
            if (j < 0 || j >= n)
            {
                fprintf(stderr, "Invalid column index %d at row %d\n", j, i);
                continue;
            }
            A_dense[i * n + j] = csr_values[idx]; // row-major
        }
    }

    return A_dense;
}

CSRMatrix dense_to_csr(const double *A, int rows, int cols)
{
    int nnz = 0;
    for (int i = 0; i < rows * cols; i++)
        if (fabs(A[i]) > 1e-14)
            nnz++;

    int *row_ptr = (int *)malloc((rows + 1) * sizeof(int));
    int *col_idx = (int *)malloc(nnz * sizeof(int));
    double *vals = (double *)malloc(nnz * sizeof(double));

    int idx = 0;
    row_ptr[0] = 0;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            double v = A[i * cols + j]; // row-major layout
            if (fabs(v) > 1e-14)
            {
                col_idx[idx] = j;
                vals[idx] = v;
                idx++;
            }
        }
        row_ptr[i + 1] = idx;
    }

    CSRMatrix result = {.row_ptr = row_ptr,
                        .col_idx = col_idx,
                        .val = vals,
                        .nnz = nnz,
                        .rows = rows,
                        .cols = cols};
    return result;
}

/* * Matrix-vector multiplication for CSR matrix.
 * If transpose is true, computes y = A^T * x, otherwise computes y = A * x.
 */
void matvec_csr(const CSRMatrix *A, const double *x, double *y, bool transpose)
{
    int m = A->rows;
    int n = A->cols;
    if (!transpose)
    {
        for (int i = 0; i < m; i++)
        {
            y[i] = 0.0;
            for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
            {
                y[i] += A->val[j] * x[A->col_idx[j]];
            }
        }
    }
    else
    {
        for (int i = 0; i < n; i++)
            y[i] = 0.0;
        for (int i = 0; i < m; i++)
        {
            for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
            {
                int col = A->col_idx[j];
                y[col] += A->val[j] * x[i];
            }
        }
    }
}

CSRMatrix get_ADAT_csr(const CSRMatrix *A, const double *D)
{
    int m = A->rows;
    const int *A_row_ptr = A->row_ptr;
    const int *A_col_idx = A->col_idx;
    const double *A_val = A->val;

    int *row_nnz = calloc(m, sizeof(int));
    int **temp_col = malloc(m * sizeof(int *));
    double **temp_val = malloc(m * sizeof(double *));
    int alloc_per_row = 64;

    for (int i = 0; i < m; ++i)
    {
        temp_col[i] = malloc(alloc_per_row * sizeof(int));
        temp_val[i] = malloc(alloc_per_row * sizeof(double));
    }

    for (int i = 0; i < m; ++i)
    {
        for (int j = i; j < m; ++j)
        {
            double dot = 0.0;
            int ai = A_row_ptr[i], aj = A_row_ptr[j];
            int ai_end = A_row_ptr[i + 1], aj_end = A_row_ptr[j + 1];

            while (ai < ai_end && aj < aj_end)
            {
                int ci = A_col_idx[ai];
                int cj = A_col_idx[aj];

                if (ci == cj)
                {
                    // Insert D into dot product
                    dot += A_val[ai] * D[ci] * A_val[aj];
                    ai++;
                    aj++;
                }
                else if (ci < cj)
                {
                    ai++;
                }
                else
                {
                    aj++;
                }
            }

            if (fabs(dot) > 1e-14)
            {
                if (row_nnz[i] % alloc_per_row == 0 && row_nnz[i] > 0)
                {
                    temp_col[i] = realloc(temp_col[i], 2 * row_nnz[i] * sizeof(int));
                    temp_val[i] = realloc(temp_val[i], 2 * row_nnz[i] * sizeof(double));
                }
                temp_col[i][row_nnz[i]] = j;
                temp_val[i][row_nnz[i]] = dot;
                row_nnz[i]++;

                if (i != j)
                {
                    if (row_nnz[j] % alloc_per_row == 0 && row_nnz[j] > 0)
                    {
                        temp_col[j] = realloc(temp_col[j], 2 * row_nnz[j] * sizeof(int));
                        temp_val[j] = realloc(temp_val[j], 2 * row_nnz[j] * sizeof(double));
                    }
                    temp_col[j][row_nnz[j]] = i;
                    temp_val[j][row_nnz[j]] = dot;
                    row_nnz[j]++;
                }
            }
        }
    }

    int total_nnz = 0;
    for (int i = 0; i < m; ++i)
        total_nnz += row_nnz[i];

    int *row_ptr = malloc((m + 1) * sizeof(int));
    int *col_idx = malloc(total_nnz * sizeof(int));
    double *val = malloc(total_nnz * sizeof(double));

    row_ptr[0] = 0;
    int idx = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < row_nnz[i]; ++j)
        {
            col_idx[idx] = temp_col[i][j];
            val[idx] = temp_val[i][j];
            idx++;
        }
        row_ptr[i + 1] = idx;
        free(temp_col[i]);
        free(temp_val[i]);
    }

    free(temp_col);
    free(temp_val);
    free(row_nnz);

    CSRMatrix ADAT = {
        .row_ptr = row_ptr,
        .col_idx = col_idx,
        .val = val,
        .nnz = total_nnz,
        .rows = m,
        .cols = m,
    };

    return ADAT;
}

static double sparse_dot_product(const int *A_col, const double *A_val, int start1, int end1,
                                 const int *B_col, const double *B_val, int start2, int end2)
{
    double sum = 0.0;
    int i = start1, j = start2;
    while (i < end1 && j < end2)
    {
        if (A_col[i] == B_col[j])
        {
            sum += A_val[i] * B_val[j];
            i++;
            j++;
        }
        else if (A_col[i] < B_col[j])
        {
            i++;
        }
        else
        {
            j++;
        }
    }
    return sum;
}

CSRMatrix get_AAT_csr(const CSRMatrix *A)
{
    int m = A->rows;
    // int n = A->cols;
    const int *A_row_ptr = A->row_ptr;
    const int *A_col_idx = A->col_idx;
    const double *A_val = A->val;

    int *row_nnz = calloc(m, sizeof(int));
    int **temp_col = malloc(m * sizeof(int *));
    double **temp_val = malloc(m * sizeof(double *));
    int alloc_per_row = 64;

    for (int i = 0; i < m; ++i)
    {
        temp_col[i] = malloc(alloc_per_row * sizeof(int));
        temp_val[i] = malloc(alloc_per_row * sizeof(double));
    }

    for (int i = 0; i < m; ++i)
    {
        for (int j = i; j < m; ++j)
        {
            double dot = sparse_dot_product(A_col_idx, A_val, A_row_ptr[i], A_row_ptr[i + 1],
                                            A_col_idx, A_val, A_row_ptr[j], A_row_ptr[j + 1]);
            if (fabs(dot) > 1e-14)
            {
                if (row_nnz[i] % alloc_per_row == 0 && row_nnz[i] > 0)
                {
                    temp_col[i] = realloc(temp_col[i], 2 * row_nnz[i] * sizeof(int));
                    temp_val[i] = realloc(temp_val[i], 2 * row_nnz[i] * sizeof(double));
                }
                temp_col[i][row_nnz[i]] = j;
                temp_val[i][row_nnz[i]] = dot;
                row_nnz[i]++;

                if (i != j)
                {
                    if (row_nnz[j] % alloc_per_row == 0 && row_nnz[j] > 0)
                    {
                        temp_col[j] = realloc(temp_col[j], 2 * row_nnz[j] * sizeof(int));
                        temp_val[j] = realloc(temp_val[j], 2 * row_nnz[j] * sizeof(double));
                    }
                    temp_col[j][row_nnz[j]] = i;
                    temp_val[j][row_nnz[j]] = dot;
                    row_nnz[j]++;
                }
            }
        }
    }

    int total_nnz = 0;
    for (int i = 0; i < m; ++i)
        total_nnz += row_nnz[i];

    int *row_ptr = malloc((m + 1) * sizeof(int));
    int *col_idx = malloc(total_nnz * sizeof(int));
    double *val = malloc(total_nnz * sizeof(double));

    row_ptr[0] = 0;
    int idx = 0;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < row_nnz[i]; ++j)
        {
            col_idx[idx] = temp_col[i][j];
            val[idx] = temp_val[i][j];
            idx++;
        }
        row_ptr[i + 1] = idx;
        free(temp_col[i]);
        free(temp_val[i]);
    }

    free(temp_col);
    free(temp_val);
    free(row_nnz);

    CSRMatrix AAT = {
        .row_ptr = row_ptr,
        .col_idx = col_idx,
        .val = val,
        .nnz = total_nnz,
        .rows = m,
        .cols = m,
    };

    return AAT;
}

CSRMatrix shift_csr_diagonal_safe(const CSRMatrix *A, double shift)
{
    int n = A->rows;
    int nnz = A->nnz;

    // Worst case: we may add one diagonal element per row
    int max_nnz = nnz + n;

    int *new_row_ptr = malloc((n + 1) * sizeof(int));
    int *new_col_idx = malloc(max_nnz * sizeof(int));
    double *new_val = malloc(max_nnz * sizeof(double));

    int pos = 0;
    new_row_ptr[0] = 0;

    for (int i = 0; i < n; i++)
    {
        int inserted_diag = 0;

        for (int jj = A->row_ptr[i]; jj < A->row_ptr[i + 1]; jj++)
        {
            int col = A->col_idx[jj];

            if (!inserted_diag && col > i)
            {
                // Insert diagonal before current entry
                new_col_idx[pos] = i;
                new_val[pos] = shift;
                pos++;
                inserted_diag = 1;
            }

            if (col == i)
            {
                // Found diagonal — shift it
                new_col_idx[pos] = col;
                new_val[pos] = A->val[jj] + shift;
                pos++;
                inserted_diag = 1;
            }
            else
            {
                // Copy non-diagonal entry
                new_col_idx[pos] = col;
                new_val[pos] = A->val[jj];
                pos++;
            }
        }

        if (!inserted_diag)
        {
            // No diagonal in this row — add it at the end
            new_col_idx[pos] = i;
            new_val[pos] = shift;
            pos++;
        }

        new_row_ptr[i + 1] = pos;
    }

    CSRMatrix shifted = {
        .row_ptr = new_row_ptr,
        .col_idx = new_col_idx,
        .val = new_val,
        .nnz = pos,
        .rows = n,
        .cols = A->cols,
    };

    return shifted;
}

void copy_csr_matrix(const CSRMatrix *src, CSRMatrix *dest)
{
    dest->rows = src->rows;
    dest->cols = src->cols;
    dest->nnz = src->nnz;

    dest->row_ptr = (int *)malloc((src->rows + 1) * sizeof(int));
    dest->col_idx = (int *)malloc(src->nnz * sizeof(int));
    dest->val = (double *)malloc(src->nnz * sizeof(double));

    memcpy(dest->row_ptr, src->row_ptr, (src->rows + 1) * sizeof(int));
    memcpy(dest->col_idx, src->col_idx, src->nnz * sizeof(int));
    memcpy(dest->val, src->val, src->nnz * sizeof(double));
}
