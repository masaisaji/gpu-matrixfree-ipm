#include "lin_alg_helper.h"
#include <assert.h>
#include <cblas.h>
#include <lapacke.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#define MAT(A, i, j, n) A[(i) * (n) + (j)]

#ifdef USE_CUDA
#include <cusparse.h>
#include "cuda_matvec.h"

int csr_n;
int csr_nnz;
double *csr_values = NULL;
int *csr_row_ptr = NULL;
int *csr_col_idx = NULL;
#endif

void cuda_sparse_matvec(const double *A, const double *x, double *y, int A_rows, int A_cols, bool transpose) {
#ifdef USE_CUDA
  // CUDA path assumes CSR format already loaded
  // cuSPARSE-based CSR matrix-vector multiplication
  cusparseHandle_t handle;
  cusparseMatDescr_t descr;
  double alpha = 1.0;
  double beta = 0.0;

  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  cusparseDcsrmv(handle,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 csr_n, csr_n, csr_nnz,
                 &alpha,
                 descr,
                 csr_values, csr_row_ptr, csr_col_idx,
                 x,
                 &beta,
                 y);

  cusparseDestroyMatDescr(descr);
  cusparseDestroy(handle);
#else
  if (!transpose) {
    for (int i = 0; i < A_rows; i++) {
      y[i] = 0.0;
      for (int j = 0; j < A_cols; j++) {
        y[i] += A[i * A_cols + j] * x[j];
      }
    }
  } else {
    for (int i = 0; i < A_cols; i++) {
      y[i] = 0.0;
      for (int j = 0; j < A_rows; j++) {
        y[i] += A[j * A_cols + i] * x[j];
      }
    }
  }
#endif
}

double *alloc_mat(int rows, int cols) {
  return (double *)malloc(rows * cols * sizeof(double));
}
void free_mat(double *M) { free(M); }

void eye_mat(double *M, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      MAT(M, i, j, n) = (i == j ? 1.0 : 0.0);
    }
  }
}

/* Copy (m x n) source matrix A_src to desitination matrix A_dst */
void copy_mat(double *A_dst, const double *A_src, int m, int n) {
  for (int i = 0; i < m * n; i++) {
    A_dst[i] = A_src[i];
  }
}

/* Matrix Addition C = A + B*/
void matadd(const double *A, const double *B, double *C, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = i * rows + j; /* row-major index */
      C[idx] = A[idx] + B[idx];
    }
  }
}

/* Matrix Subtraction C = A - B*/
void matsub(const double *A, const double *B, double *C, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      int idx = i * rows + j; /* row-major index */
      C[idx] = A[idx] - B[idx];
    }
  }
}

/* Multiply C = A (A_rows x A_cols) * B (B_rows x B_cols)
 * A_rows, A_cols, B_rows, B_cols should be defined as dimensions before any
 * transposition. Reference for row-major BLAS dgemm call:
 * https://stackoverflow.com/questions/71095291/cblas-on-entry-to-sgemm-dgemm-parameter-number-x-had-an-illegal-value?rq=2
 */
void matmul(const double *A, const double *B, double *C, int A_rows, int A_cols,
            int B_rows, int B_cols, bool A_trans, bool B_trans) {

  // transposition logic
  int A_rows_eff = (A_trans ? A_cols : A_rows);
  int A_cols_eff = (A_trans ? A_rows : A_cols);
  int B_rows_eff = (B_trans ? B_cols : B_rows);
  int B_cols_eff = (B_trans ? B_rows : B_cols);

  // Debugging statements before calling matmul
  // printf("\nmatmul called\n");
  // printf("A: %d x %d, B: %d x %d, C: %d x %d\n", A_rows_eff, A_cols_eff,
  //        B_rows_eff, B_cols_eff, A_rows_eff, B_cols_eff);

  // dimension check
  if (A_cols_eff != B_rows_eff) {
    fprintf(stderr,
            "Error: dimension mismatch in matmul. A_cols=%d != B_rows=%d\n",
            A_cols_eff, B_rows_eff);
    return;
  }

  // These are the only settings working for row-major CBLAS
  int lda = (A_trans ? A_rows_eff : A_cols_eff); // leading dimension of A
  int ldb = (B_trans ? B_rows_eff : B_cols_eff); // leading dimension of B
  int ldc = B_cols_eff;                          // leading dimension of C

  // set up the CBLAS_TRANSPOSE enum
  CBLAS_TRANSPOSE transA = A_trans ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = B_trans ? CblasTrans : CblasNoTrans;

  cblas_dgemm(CblasRowMajor,
              transA,     // A transpose indicator
              transB,     // B transpose indicator
              A_rows_eff, // number of rows of A, and C
              B_cols_eff, // number of columns of B, and C
              A_cols_eff, // shared dimension
              1.0,        // alpha
              A, lda, B, ldb,
              0.0, // beta
              C, ldc);
}

/* In-place matrix inversion using LAPACK's LU decomposition */
int invert_matrix_inplace(double *A, int n) {
  // Allocate pivot array (just for factorization)
  int *ipiv = (int *)malloc(n * sizeof(int));
  if (!ipiv) {
    // optional: handle malloc error
    return -1;
  }

  // Factor A with LU decomposition
  lapack_int info =
      LAPACKE_dgetrf(LAPACK_ROW_MAJOR, /* data is in row-major format */
                     n, n,             /* dimensions of the matrix */
                     A, n,             /* pointer to A, leading dimension n */
                     ipiv);
  if (info != 0) {
    // matrix is singular or factorization failed
    free(ipiv);
    return info;
  }

  // Invert using the LU factor
  info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, A, n, ipiv);

  // We no longer need ipiv, so just free it
  free(ipiv);

  return info;
}

/**
 * Invert an SPD matrix A in-place using Cholesky decomposition.
 *
 * A will be overwritten with its inverse.
 *
 * @param A   Pointer to the matrix A (size n x n), row-major.
 * @param n   Dimension of the matrix.
 * @return    0 if successful, nonzero if an error occurred.
 */
int invert_spd_matrix(double *A, int n) {
  // Cholesky decomposition: A = L * Lᵀ
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A, n);
  if (info != 0) {
    fprintf(stderr, "dpotrf failed (info=%d). Matrix may not be SPD.\n", info);
    return info;
  }

  // Compute inverse from Cholesky factor
  info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', n, A, n);
  if (info != 0) {
    fprintf(stderr, "dpotri failed (info=%d)\n", info);
    return info;
  }

  // Fill upper triangle to make A fully symmetric
  for (int i = 0; i < n; i++) {
    for (int j = i + 1; j < n; j++) {
      A[i * n + j] = A[j * n + i];
    }
  }

  return 0;
}

/* A naive 2-norm (Frobenius) difference: ||A - B||_F. */
// TODO: rectanguar matrix support
double matrix_diff_norm(const double *A, const double *B, int n) {
  double sumsq = 0.0;
  for (int i = 0; i < n * n; i++) {
    double diff = A[i] - B[i];
    sumsq += diff * diff;
  }
  return sqrt(sumsq);
}

double vec_diff_norm(const double *a, const double *b, int vec_dim) {
  double sum = 0.0;
  for (int i = 0; i < vec_dim; i++) {
    sum += fabs(a[i] - b[i]);
  }
  return sum;
}

/* Make a random SPD matrix: A <- rand(n,n); A <- A' * A + I */
void make_random_spd(double *A, int n) {
  for (int i = 0; i < n * n; i++) {
    A[i] = (double)rand() / (double)RAND_MAX; // random in [0,1)
  }
  /* We want A <- A^T * A + I. We'll do this with a temporary. */
  double *temp = alloc_mat(n, n);

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0, A, n, A, n,
              0.0, temp, n);

  // Now temp = A^T * A. Overwrite A with temp
  copy_mat(A, temp, n, n);

  // Add identity
  for (int i = 0; i < n; i++) {
    MAT(A, i, i, n) += 1.0;
  }

  free_mat(temp);
}

/* A quick function to generate an ill-conditioned SPD matrix */
double *generate_ill_conditioned_spd(int n, double cond_num) {
  /* Basic approach: random Q * diag(...) * Q^T. For brevity, just do a simple
   * approach. */

  double *A = alloc_mat(n, n);
  srand((unsigned)time(NULL));

  // Fill A with random. Then A <- A^T * A for SPD
  for (int i = 0; i < n * n; i++) {
    A[i] = ((double)rand()) / RAND_MAX - 0.5;
  }

  // We want to embed a wide range of eigenvalues.
  // Let's do: scale each row i by something between 1..cond_num
  // This is a hack, not a stable approach:
  for (int i = 0; i < n; i++) {
    double scale = 1.0 + (cond_num - 1.0) * (double)i / (double)(n - 1);
    for (int j = 0; j < n; j++) {
      A[i * n + j] *= scale;
    }
  }

  // A <- A^T * A
  double *temp = alloc_mat(n, n);
  // compute temp = A^T * A
  for (int i = 0; i < n * n; i++) {
    temp[i] = 0.0;
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++) {
        sum += A[k * n + i] *
               A[k * n + j]; // note "A^T" => swap i/k for that index
      }
      temp[i * n + j] = sum;
    }
  }
  // now copy temp->A
  for (int i = 0; i < n * n; i++) {
    A[i] = temp[i];
  }

  // add identity
  for (int i = 0; i < n; i++) {
    A[i * n + i] += 1.0;
  }

  free_mat(temp);
  return A;
}

/**
 * Generates a random orthonormal matrix Q via QR factorization of a random NxN,
 * then multiplies Q * diag(...) * Q^T to form an SPD matrix with a desired
 * condition number.
 *
 * @param A         Output NxN SPD matrix
 * @param n         Matrix dimension
 * @param cond_num  Approx. condition number to aim for (e.g. 1e6)
 */
void generate_ill_conditioned_spd_2(double *A, int n, double cond_num) {
  // 1) Create random NxN matrix "Rand"
  double *Rand = (double *)malloc(n * n * sizeof(double));
  srand((unsigned int)time(NULL));
  for (int i = 0; i < n * n; i++) {
    Rand[i] = ((double)rand() / RAND_MAX) - 0.5; // random in [-0.5..0.5]
  }

  // step a) do QR factorization
  double *tau = (double *)malloc(n * sizeof(double));
  int info = LAPACKE_dgeqrf(101 /* row-major */, n, n, Rand, n, tau);
  if (info != 0) {
    printf("LAPACKE_dgeqrf failed, info=%d\n", info);
  }

  // step b) generate Q from R. (Rand now has partial R in upper tri)
  info = LAPACKE_dorgqr(101 /* row-major */, n, n, n, Rand, n, tau);
  if (info != 0) {
    printf("LAPACKE_dorgqr failed, info=%d\n", info);
  }
  // Now "Rand" is actually Q

  // 2) Create diag with eigenvalues from "cond_num" down to 1.0
  double *Lambda = (double *)calloc(n * n, sizeof(double));
  for (int i = 0; i < n; i++) {
    double lambda_i = 1.0 + (cond_num - 1.0) * (double)(n - 1 - i) / (n - 1);
    // that spreads them from cond_num (for i=0) down to 1.0 (for i=n-1)
    Lambda[i * n + i] = lambda_i;
  }

  // 3) A = Q * Lambda * Q^T
  // We'll do a temporary M = Q * Lambda, then A = M * Q^T
  double *M = (double *)calloc(n * n, sizeof(double));

  // cblas_dgemm( order=101 row-major,
  //              transA=NoTrans, transB=NoTrans,
  //              M=n, N=n, K=n, alpha=1.0,
  //              Rand, lda=n,
  //              Lambda, ldb=n,
  //              beta=0.0,
  //              M, ldc=n )
  cblas_dgemm(101, 111, 111, n, n, n, 1.0, Rand, n, Lambda, n, 0.0, M, n);

  // A = M * Q^T
  // Q^T is just Rand^T, so we do cblas_dgemm with transB=Trans
  cblas_dgemm(101, 111, 112, // 112 => transB=Yes
              n, n, n, 1.0, M, n, Rand, n, 0.0, A, n);

  free(M);
  free(Lambda);
  free(tau);
  free(Rand);
}

/**
 * A simple random right-hand-side vector
 */
void generate_rhs_vector(double *b, int n) {
  // Also random in [1..10]
  for (int i = 0; i < n; i++) {
    b[i] = (rand() % 10) + 1.0;
  }
}

// simple dot product of two vectors a * b
double dot(const double *a, const double *b, int n) {
  return cblas_ddot(n, a, 1, b, 1);
}

/**
 * @brief Performs matrix-vector multiplication: y = A * x or y = Aᵗ * x
 *
 * Computes:
 *   - y = A * x         if transpose == false
 *   - y = Aᵗ * x        if transpose == true
 *
 * @param A        Pointer to matrix A (row-major, size A_rows × A_cols)
 * @param x        Pointer to vector x
 * @param y        Pointer to result vector y (must be preallocated)
 * @param A_rows   Number of rows in matrix A (before transposition)
 * @param A_cols   Number of columns in matrix A (before transposition)
 * @param transpose If true, compute Aᵗ * x; else compute A * x
 */ 
void matvec(const double *A, const double *x, double *y, int A_rows, int A_cols,
            bool transpose) {
  CBLAS_TRANSPOSE trans = transpose ? CblasTrans : CblasNoTrans;
  cblas_dgemv(CblasRowMajor, trans, A_rows, A_cols, 1.0, A, A_cols, x, 1, 0.0,
              y, 1);
}

/**
 * @brief Compute ADAᵗ where A is (m × n), D is a diagonal matrx (n x n)
 * where elements are stored as a vector of length n,
 * and the result is an m × m symmetric matrix.
 *
 * Uses BLAS dsyr for efficient symmetric rank-1 updates.
 *
 * @param A       Pointer to matrix A (row-major, size m × n)
 * @param D       Pointer to diagonal entries of D (size n)
 * @param ADAT    Pointer to result matrix (row-major, size m × m), must be
 * zero-initialized
 * @param m       Number of rows of A
 * @param n       Number of columns of A
 */
void ADAT_prod(const double *A, const double *D, double *ADAT, int m, int n) {
  double *a_k = malloc(m * sizeof(double));
  if (!a_k) {
    perror("Failed to allocate a_k");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < m * m; ++i)
    ADAT[i] = 0.0;

  for (int k = 0; k < n; ++k) {
    double d_k = D[k];

    // Extract column k of A into a_k
    for (int i = 0; i < m; ++i) {
      a_k[i] = A[i * n + k]; // A[i][k] in row-major
    }

    // ADAT += d_k * a_k * a_kᵗ (rank-1 update)
    cblas_dsyr(CblasRowMajor, CblasLower, m, d_k, a_k, 1, ADAT, m);
  }

  // Fill in upper triangle to make ADAT symmetric
  for (int i = 0; i < m; ++i) {
    for (int j = i + 1; j < m; ++j) {
      ADAT[i * m + j] = ADAT[j * m + i];
    }
  }
  // printf("Printing G_R matrix...\n");
  // for (int i = 0; i < fmin(10, m); i++) {
  //   for (int j = 0; j < fmin(10, m); j++) {
  //     printf("%g ", ADAT[i * m + j]);
  //   }
  //   printf("\n");
  // }

  free(a_k);
}

/**
 * @brief Compute y = y + alpha * x in-place (y will be overwritten)
 *
 * @param size  size of vectors
 * @param x  pointer to first vector
 * @param y  pointer to second vector, will be overwritten
 */
void vecadd_inplace(int size, const double *x, double *y, double alpha) {
  cblas_daxpy(size, alpha, x, 1, y, 1);
}

bool is_spd(double *GR, int m) {
  // Copy GR to avoid modifying the original
  double *GR_copy = (double *)malloc(m * m * sizeof(double));
  for (int i = 0; i < m * m; ++i) {
    GR_copy[i] = GR[i];
  }

  // Try Cholesky (lower triangle, row-major)
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', m, GR_copy, m);

  free(GR_copy);
  return (info == 0);
}

/**
 * @brief Compute z = x + alpha *y
 *
 * @param size  size of vectors
 * @param x  pointer to first vector
 * @param y  pointer to second vector
 * @param z  pointer to result vector
 */
void vecadd(int size, const double *x, const double *y, double *z,
            double alpha) {
  for (int i = 0; i < size; ++i)
    z[i] = x[i] + y[i];
}

/**
 * @brief Compute y = y - x in-place (y will be overwritten)
 *
 * @param size  size of vectors
 * @param x  pointer to first vector
 * @param y  pointer to second vector, will be overwritten
 */
void vecsub_inplace(int size, const double *x, double *y, double alpha) {
  cblas_daxpy(size, -1.0 * alpha, x, 1, y, 1);
}

/**
 * @brief Compute z = x - alpha * y
 *
 * @param size  size of vectors
 * @param x  pointer to first vector
 * @param y  pointer to second vector
 * @param z  pointer to result vector
 */
void vecsub(int size, const double *x, const double *y, double *z,
            double alpha) {
  for (int i = 0; i < size; ++i)
    z[i] = x[i] - alpha * y[i];
}

double vec_L2_norm(const double *x, int size) {
  return cblas_dnrm2(size, x, 1);
}

double max_val(const double *x, int n) {
  double max = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] > max) {
      max = x[i];
    }
  }
  return max;
}

/* Solves Ax = b where A is SPD matrix */
int solve_spd_system(const double *A, double *x, const double *b, int n) {
  double *A_copy = alloc_mat(n, n);
  double *b_copy = (double *)malloc(n * sizeof(double));
  if (!A_copy || !b_copy) {
    fprintf(stderr, "Memory allocation failed\n");
    return -1;
  }

  // Copy inputs
  copy_mat(A_copy, A, n, n);
  for (int i = 0; i < n; i++) {
    b_copy[i] = b[i];
  }

  // Cholesky decomposition (lower triangular)
  int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n, A_copy, n);
  if (info != 0) {
    fprintf(stderr, "Cholesky decomposition failed, info = %d\n", info);
    free(A_copy);
    free(b_copy);
    return info;
  }

  // Solve system
  info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n, 1, A_copy, n, b_copy, 1);
  if (info != 0) {
    fprintf(stderr, "Cholesky solve failed, info = %d\n", info);
    free(A_copy);
    free(b_copy);
    return info;
  }

  // Copy solution to x
  for (int i = 0; i < n; i++) {
    x[i] = b_copy[i];
  }

  free(A_copy);
  free(b_copy);
  return 0;
}

double estimate_condition_number(const double *A, int n) {
  // Make a copy since LAPACKE_dgecon requires factored matrix
  double *A_copy = alloc_mat(n, n);
  copy_mat(A_copy, A, n, n);

  // Allocate pivot array
  int *ipiv = malloc(n * sizeof(int));
  if (!ipiv) {
    fprintf(stderr, "malloc failed for ipiv\n");
    exit(EXIT_FAILURE);
  }

  // Compute LU factorization (even for SPD matrix)
  lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, A_copy, n, ipiv);
  if (info != 0) {
    fprintf(stderr, "LU factorization failed (info=%d)\n", info);
    free_mat(A_copy);
    free(ipiv);
    return -1.0;
  }

  // Estimate 1-norm of A
  double norm = LAPACKE_dlange(LAPACK_ROW_MAJOR, '1', n, n, A, n);

  // Estimate reciprocal of condition number
  double rcond;
  info = LAPACKE_dgecon(LAPACK_ROW_MAJOR, '1', n, A_copy, n, norm, &rcond);
  if (info != 0) {
    fprintf(stderr, "Condition number estimate failed (info=%d)\n", info);
    free_mat(A_copy);
    free(ipiv);
    return -1.0;
  }

  free_mat(A_copy);
  free(ipiv);

  return (rcond > 0.0) ? (1.0 / rcond) : INFINITY;
}


void gen_random_dense_mat(double *A, int rows, int cols, double scaler) {
  for (int i = 0; i < rows * cols; ++i) {
    A[i] = scaler * (((double)rand() / RAND_MAX) * 2.0 -
                     1.0); // Uniform in [-scaler, scaler]
  }
}

void gen_random_sparse_mat(double *A, int rows, int cols, double scaler,
                           double sparsity) {
  for (int i = 0; i < rows * cols; ++i) {
    double r = (double)rand() / RAND_MAX;
    if (r < sparsity) {
      A[i] = scaler * (((double)rand() / RAND_MAX) * 2.0 - 1.0); // nonzero
    } else {
      A[i] = 0.0;
    }
  }
}

void gen_full_row_rank_sparse_mat(double *A, int m, int n, double scale,
                                  double sparsity) {
  if (n < m) {
    fprintf(stderr,
            "Error: Cannot construct full row rank matrix because n < m\n");
    exit(EXIT_FAILURE);
  }

  // Zero out A
  for (int i = 0; i < m * n; ++i)
    A[i] = 0.0;

  // Fill identity part: A[i][i] = 1 for first m columns
  for (int i = 0; i < m; ++i) {
    A[i * n + i] = 1.0;
  }

  // Fill random sparse part: columns m to n-1
  for (int i = 0; i < m; ++i) {
    for (int j = m; j < n; ++j) {
      double r = (double)rand() / RAND_MAX;
      if (r < sparsity) {
        A[i * n + j] = scale * (((double)rand() / RAND_MAX) * 2.0 - 1.0);
      }
    }
  }
}

void transpose_row_to_col_major(const double *A_row, double *A_col, int m, int n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      A_col[j * m + i] = A_row[i * n + j];  // convert to column-major
    }
  }
}