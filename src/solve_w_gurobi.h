#ifndef SOLVE_W_GUROBI_H
#define SOLVE_W_GUROBI_H

/**
 * Solve LP: min cᵀx s.t. Ax = b, x ≥ 0
 * @param A        Row-major constraint matrix A (size m x n)
 * @param b        RHS vector (length m)
 * @param c        Cost vector (length n)
 * @param m        Number of constraints
 * @param n        Number of variables
 * @param x_out    Output array of length n (will be filled with optimal x)
 * @return         0 on success, nonzero on error
 */
int solve_lp_with_gurobi(const double *A, const double *b, const double *c, int m, int n,
                         double *x_out);
#endif
