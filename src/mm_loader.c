#include "mm_loader.h"
#include "csr_utils.h"
#include "mmio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Allocates and fills the filepaths for a given LP name (e.g., "nug05")
// Caller must free the returned strings.
// This assumes the format of:
//  - data/lp_<name>/lp_<name>.mtx for A matrix
//  - data/lp_<name>/lp_<name>_b.mtx for b vector
//  - data/lp_<name>/lp_<name>_c.mtx for c vector
LPPaths get_lp_filepaths(const char *name)
{
    LPPaths paths;

    // Compute full strings with format: data/lp_<name>/lp_<name>*.mtx
    size_t base_len = strlen(name);
    size_t prefix_len = strlen("data/lp_/lp__c.mtx");

    // Allocate +1 for null-terminator
    paths.A_path = (char *)malloc(prefix_len + 2 * base_len + 1);
    paths.b_path = (char *)malloc(prefix_len + 2 * base_len + 3); // _b
    paths.c_path = (char *)malloc(prefix_len + 2 * base_len + 3); // _c

    if (!paths.A_path || !paths.b_path || !paths.c_path)
    {
        perror("Failed to allocate file path strings");
        exit(1);
    }

    sprintf(paths.A_path, "data/lp_%s/lp_%s.mtx", name, name);
    sprintf(paths.b_path, "data/lp_%s/lp_%s_b.mtx", name, name);
    sprintf(paths.c_path, "data/lp_%s/lp_%s_c.mtx", name, name);

    return paths;
}

void free_lp_paths(LPPaths *paths)
{
    if (paths->A_path)
        free(paths->A_path);
    if (paths->b_path)
        free(paths->b_path);
    if (paths->c_path)
        free(paths->c_path);
}

int load_mtx_matrix(const char *filename, int *m, int *n, int *nnz, int **row_indices,
                    int **col_indices, double **values)
{
    FILE *f;
    MM_typecode matcode;

    if ((f = fopen(filename, "r")) == NULL)
    {
        perror("Error opening file");
        return -1;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        fclose(f);
        return -1;
    }

    if (!mm_is_coordinate(matcode) || (!mm_is_real(matcode) && !mm_is_integer(matcode)))
    {
        fprintf(stderr, "Only real or integer coordinate matrices are supported.\n");
        fclose(f);
        return -1;
    }

    if (mm_read_mtx_crd_size(f, m, n, nnz) != 0)
    {
        fprintf(stderr, "Failed to read matrix size.\n");
        fclose(f);
        return -1;
    }

    // Allocate memory
    *row_indices = (int *)malloc(*nnz * sizeof(int));
    *col_indices = (int *)malloc(*nnz * sizeof(int));
    *values = (double *)malloc(*nnz * sizeof(double));

    if (!*row_indices || !*col_indices || !*values)
    {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(f);
        return -1;
    }

    // Read entries
    char line[128];
    for (int i = 0; i < *nnz; i++)
    {
        if (!fgets(line, sizeof(line), f))
        {
            fprintf(stderr, "Unexpected EOF at entry %d\n", i);
            free(*row_indices);
            free(*col_indices);
            free(*values);
            fclose(f);
            return -1;
        }

        int row, col;
        double value;
        int count = sscanf(line, "%d %d %lf", &row, &col, &value);

        if (count != 3)
        {
            fprintf(stderr, "Error: Failed to parse entry %d\n", i + 1);
            for (int j = 0; line[j]; j++)
            {
                printf("byte[%d] = 0x%02x ", j, (unsigned char)line[j]);
                if (line[j] == '\n')
                    printf("\\n");
                if (line[j] == '\r')
                    printf("\\r");
                printf("\n");
            }
            free(*row_indices);
            free(*col_indices);
            free(*values);
            fclose(f);
            return -1;
        }

        (*row_indices)[i] = row - 1;
        (*col_indices)[i] = col - 1;
        (*values)[i] = value;
    }

    fclose(f);
    return 0;
}

int load_mtx_vector(const char *filename, int *length, double **values)
{
    FILE *f;
    MM_typecode matcode;

    if ((f = fopen(filename, "r")) == NULL)
    {
        perror("Error opening file");
        return -1;
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        fclose(f);
        return -1;
    }

    if (!mm_is_array(matcode) || (!mm_is_real(matcode) && !mm_is_integer(matcode)))
    {
        fprintf(stderr, "Only real or integer array matrices are supported.\n");
        fclose(f);
        return -1;
    }

    int m, n;
    if (mm_read_mtx_array_size(f, &m, &n) != 0)
    {
        fprintf(stderr, "Failed to read matrix size.\n");
        fclose(f);
        return -1;
    }

    if (n != 1 && m != 1)
    {
        fprintf(stderr, "Not a vector (expected 1 column or 1 row).\n");
        fclose(f);
        return -1;
    }

    int len = (m > n) ? m : n; // if m x 1, len = m, if 1 x n, len = n
    *length = len;
    *values = (double *)malloc(len * sizeof(double));

    for (int i = 0; i < len; i++)
    {
        if (fscanf(f, "%lf", &((*values)[i])) != 1)
        {
            fprintf(stderr, "Error reading value %d from file: %s\n", i, filename);
            free(*values);
            fclose(f);
            return -1;
        }
    }

    fclose(f);
    return 0;
}

double *coo_to_dense_row_major(int m, int n, int nz, const int *row_indices, const int *col_indices,
                               const double *A_vals)
{
    // Allocate zero-initialized row-major matrix
    double *A_dense = (double *)calloc(m * n, sizeof(double));
    if (!A_dense)
    {
        perror("Failed to allocate dense matrix");
        return NULL;
    }

    for (int k = 0; k < nz; k++)
    {
        int i = row_indices[k];
        int j = col_indices[k];
        A_dense[i * n + j] = A_vals[k]; // row-major
    }

    return A_dense;
}

typedef struct
{
    int row, col;
    double val;
} CooEntry;

int compare_coo(const void *a, const void *b)
{
    const CooEntry *ea = (const CooEntry *)a;
    const CooEntry *eb = (const CooEntry *)b;
    if (ea->row != eb->row)
        return ea->row - eb->row;
    return ea->col - eb->col;
}

/**
 * @brief Convert a sparse matrix from unsorted COO format to CSR format.
 *
 * This function takes input COO arrays (row, col, val) and produces sorted CSR arrays
 * (csr_row_ptr, csr_col_idx, csr_values). The COO entries are first packed into triplets,
 * sorted by (row, col), and then converted to CSR format.
 *
 * This is an in-place converter in the sense that the output CSR arrays must be preallocated
 * by the caller. The original COO arrays are not modified.
 *
 * @param[in]  n_rows        Number of rows in the matrix
 * @param[in]  nnz           Number of nonzero entries
 * @param[in]  coo_row       Array of COO row indices (length nnz)
 * @param[in]  coo_col       Array of COO column indices (length nnz)
 * @param[in]  coo_val       Array of COO nonzero values (length nnz)
 * @param[out] csr_row_ptr   CSR row pointer array (length n_rows + 1)
 * @param[out] csr_col_idx   CSR column indices array (length nnz)
 * @param[out] csr_values    CSR nonzero values array (length nnz)
 *
 * @note The output CSR format will have rows sorted in ascending order, and columns
 *       sorted within each row. Duplicate (row, col) entries are not merged.
 *       This function allocates a temporary buffer internally and frees it before returning.
 */
void coo_to_csr_inplace(int n_rows, int nnz, const int *coo_row, const int *coo_col,
                        const double *coo_val, int *csr_row_ptr, int *csr_col_idx,
                        double *csr_values)
{
    // Pack COO entries into struct for sorting
    CooEntry *entries = (CooEntry *)malloc(nnz * sizeof(CooEntry));
    for (int i = 0; i < nnz; ++i)
    {
        entries[i].row = coo_row[i];
        entries[i].col = coo_col[i];
        entries[i].val = coo_val[i];
    }

    // Sort by (row, col)
    qsort(entries, nnz, sizeof(CooEntry), compare_coo);

    // Fill csr_row_ptr histogram
    for (int i = 0; i < nnz; ++i)
        csr_row_ptr[entries[i].row + 1]++;

    for (int i = 0; i < n_rows; ++i)
        csr_row_ptr[i + 1] += csr_row_ptr[i];

    // Fill col_idx and val arrays
    for (int i = 0; i < nnz; ++i)
    {
        csr_col_idx[i] = entries[i].col;
        csr_values[i] = entries[i].val;
    }

    free(entries);
}

/**
 * @brief Load a linear program (LP) from Matrix Market files in COO format,
 *        and convert the constraint matrix to CSR format.
 *
 * This function loads the LP problem:
 *     minimize    cᵀx
 *     subject to  Ax = b
 *                 x ≥ 0
 *
 * It expects three Matrix Market files:
 *   - A matrix in COO format (can be unsorted): <matrix_path>
 *   - Right-hand side vector b: <b_path>
 *   - Cost vector c: <c_path>
 *
 * After loading, the function:
 *   - Converts the matrix A from COO to CSR format (sorted by row and column)
 *   - Returns the matrix in CSR format through `row_indices`, `col_indices`, and `val`
 *   - Returns vectors b and c
 *
 * @param[in]  matrix_path    Path to Matrix Market file containing matrix A in COO format
 * @param[in]  b_path         Path to Matrix Market file containing vector b
 * @param[in]  c_path         Path to Matrix Market file containing vector c
 * @param[out] m              Number of rows of A
 * @param[out] n              Number of columns of A
 * @param[out] nnz            Number of nonzeros in A
 * @param[out] row_indices    CSR row pointer array of size (m + 1)
 * @param[out] col_indices    CSR column index array of size nnz
 * @param[out] val            CSR nonzero values array of size nnz
 * @param[out] b              Right-hand side vector of size m
 * @param[out] c              Cost vector of size n
 *
 * @return 0 on success, nonzero on failure
 *
 * @note The row_indices output corresponds to the CSR row pointer array (not COO row indices).
 *       Internally, the input COO triplets are sorted before CSR conversion.
 */
int load_mtx_lp_w_path(const char *matrix_path, const char *b_path, const char *c_path, int *m,
                       int *n, int *nnz, int **row_indices, int **col_indices, double **val,
                       double **b, double **c)
{

    int b_len, c_len;

    // Load matrix A (COO)
    if (load_mtx_matrix(matrix_path, m, n, nnz, row_indices, col_indices, val) != 0)
    {
        fprintf(stderr, "Failed to load matrix A from %s\n", matrix_path);
        return 1;
    }

    // Load RHS vector b
    if (load_mtx_vector(b_path, &b_len, b) != 0)
    {
        fprintf(stderr, "Failed to load b vector from %s\n", b_path);
        return 1;
    }

    if (b_len != *m)
    {
        fprintf(stderr, "Mismatch: b has length %d but A has %d rows\n", b_len, *m);
        return 1;
    }

    // Load cost vector c
    if (load_mtx_vector(c_path, &c_len, c) != 0)
    {
        fprintf(stderr, "Failed to load c vector from %s\n", c_path);
        return 1;
    }

    if (c_len != *n)
    {
        fprintf(stderr, "Mismatch: c has length %d but A has %d columns\n", c_len, *n);
        return 1;
    }

    // Convert COO → CSR in place
    int *csr_row_ptr = (int *)calloc(*m + 1, sizeof(int));
    int *csr_col_idx = (int *)malloc(*nnz * sizeof(int));
    double *csr_val = (double *)malloc(*nnz * sizeof(double));

    coo_to_csr_inplace(*m, *nnz, *row_indices, *col_indices, *val, csr_row_ptr, csr_col_idx,
                       csr_val);

    // Free COO arrays and replace them with CSR
    free(*row_indices);
    free(*col_indices);
    free(*val);
    *row_indices = csr_row_ptr; // row_indices now holds csr_row_ptr
    *col_indices = csr_col_idx;
    *val = csr_val;

    return 0;
}

// This func assumes the path structure of:
//  - data/lp_<name>/lp_<name>.mtx for A matrix
//  - data/lp_<name>/lp_<name>_b.mtx for b vector
//  - data/lp_<name>/lp_<name>_c.mtx for c vector
// To specify paths individually, use load_mtx_lp_w_path
int load_mtx_lp(const LPPaths *paths, int *m, int *n, int *nz, int **csr_row_ptr, int **csr_col_idx,
                double **val, double **b, double **c)
{
    return load_mtx_lp_w_path(paths->A_path, paths->b_path, paths->c_path, m, n, nz, csr_row_ptr,
                              csr_col_idx, val, b, c);
}

int main_mm_loader()
{
    const char *lp_name = "nug05"; // Name of LP problem
    int m, n, nnz;
    int *csr_row_ptr, *csr_col_idx;
    double *csr_val, *b, *c;
    LPPaths paths = get_lp_filepaths(lp_name);

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

    double *A_dense = csr_to_dense_row_major(&A);
    freeCSRmat(&A);

    printf("LP problem ID: %s\n", lp_name);
    printf("Loaded matrix A: %d x %d with %d nonzeros\n", m, n, nnz);
    printf("Loaded vectors b and c\n");

    // (Optional) Print a few entries
    for (int i = 0; i < 5 && i < m; i++)
    {
        printf("b[%d] = %f\n", i, b[i]);
    }
    for (int j = 0; j < 5 && j < n; j++)
    {
        printf("c[%d] = %f\n", j, c[j]);
    }

    free_lp_paths(&paths);
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
    free(A_dense);
    free(b);
    free(c);
    free_lp_paths(&paths);

    return 0;
}
