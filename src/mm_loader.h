#ifndef MM_LOADER_H
#define MM_LOADER_H

typedef struct
{
    char *A_path;
    char *b_path;
    char *c_path;
} LPPaths;

int load_mtx_lp(const LPPaths *paths, int *m, int *n, int *nz, int **csr_row_ptr, int **csr_col_idx,
                double **val, double **b, double **c);
int load_mtx_lp_w_path(const char *matrix_path, const char *b_path, const char *c_path, int *m,
                       int *n, int *nnz, int **row_indices, int **col_indices, double **val,
                       double **b, double **c);
void free_lp_paths(LPPaths *paths);
LPPaths get_lp_filepaths(const char *name);
#endif
