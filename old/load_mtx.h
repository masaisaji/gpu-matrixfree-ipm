#ifndef LOAD_MTX_H
#define LOAD_MTX_H

int load_mtx_to_csr(const char *filename,
                    double **values,
                    int **col_indices,
                    int **row_ptr,
                    int *m, int *n, int *nnz);

int load_mtx_vector(const char *filename,
                    double **vec,
                    int *length);

#endif
