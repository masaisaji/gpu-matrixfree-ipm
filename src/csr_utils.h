#ifndef CSR_UTILS_H
#define CSR_UTILS_H
#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct
    {
        int *row_ptr;
        int *col_idx;
        double *val;
        int nnz;
        int rows;
        int cols;
    } CSRMatrix;

    // Computes the transpose of a CSR matrix on CPU.
    // The returned CSRMatrix contains newly allocated memory.
    // The caller is responsible for freeing row_ptr, col_idx, and val.
    CSRMatrix csr_transpose(const CSRMatrix *A);
    void freeCSRmat(CSRMatrix *mat);
    double *csr_to_dense_row_major(const CSRMatrix *A);
    CSRMatrix dense_to_csr(const double *A, int rows, int cols);
    void matvec_csr(const CSRMatrix *A, const double *x, double *y, bool transpose);

    CSRMatrix get_ADAT_csr(const CSRMatrix *A, const double *D);
    CSRMatrix get_AAT_csr(const CSRMatrix *A);
    CSRMatrix shift_csr_diagonal_safe(const CSRMatrix *A, double shift);
    void copy_csr_matrix(const CSRMatrix *src, CSRMatrix *dest);

#ifdef __cplusplus
}
#endif

#endif // CSR_UTILS_H
