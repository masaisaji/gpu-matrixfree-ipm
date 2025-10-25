#ifndef LIN_ALG_HELPER_H
#define LIN_ALG_HELPER_H

#include <glpk.h>
#include <stdbool.h>

#define MAT(A, i, j, n) A[(i) * (n) + (j)]

double *alloc_mat(int rows, int cols);
void free_mat(double *M);
void eye_mat(double *M, int n);
void copy_mat(double *A_dst, const double *A_src, int m, int n);
void matadd(const double *A, const double *B, double *C, int rows, int cols);
void matsub(const double *A, const double *B, double *C, int rows, int cols);
void matmul(const double *A, const double *B, double *C, int A_rows, int A_cols,
            int B_rows, int B_cols, bool A_trans, bool B_trans);
int invert_matrix_inplace(double *A, int n);
int invert_spd_matrix(double *A, int n);
double matrix_diff_norm(const double *A, const double *B, int n);
double vec_diff_norm(const double *a, const double *b, int vec_dim);
void make_random_spd(double *A, int n);
double *generate_ill_conditioned_spd(int n, double cond_num);
void generate_ill_conditioned_spd_2(double *A, int n, double cond_num);
void generate_rhs_vector(double *b, int n);
double dot(const double *a, const double *b, int n);
void matvec(const double *A, const double *x, double *y, int A_rows, int A_cols,
            bool transpose);
void ADAT_prod(const double *A, const double *D, double *ADAT, int m, int n);
bool is_spd(double *GR, int m);
void gen_random_dense_mat(double *A, int rows, int cols, double scaler);
void gen_random_sparse_mat(double *A, int rows, int cols, double scaler,
                           double sparsity);
void gen_full_row_rank_sparse_mat(double *A, int m, int n, double scale,
                                  double sparsity);

void vecadd_inplace(int size, const double *x, double *y, double alpha);
void vecadd(int size, const double *x, const double *y, double *z,
            double alpha);
void vecsub_inplace(int size, const double *x, double *y, double alpha);
void vecsub(int size, const double *x, const double *y, double *z,
            double alpha);
double vec_L2_norm(const double *x, int size);
double max_val(const double *x, int n);
int solve_spd_system(const double *A, double *x, const double *b, int n);
double estimate_condition_number(const double *A, int n);
void transpose_row_to_col_major(const double *A_row, double *A_col, int m, int n);
#endif
