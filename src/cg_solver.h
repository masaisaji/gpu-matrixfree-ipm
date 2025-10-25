//
//  cg_solver.h
//  Project
//
//  Created by Mitchell Hebner on 3/18/25.
//
#include <stdbool.h>

#ifndef CG_SOLVER_H
#define CG_SOLVER_H

#ifdef __cplusplus
extern "C"
{
#endif

    /**
     * Standard CG solver (no preconditioning).
     */
    void cg_solve(const double *A, double *x, const double *b, int n, double tol, int max_iter);

    /**
     * Preconditioned CG solver using partial pivoted Cholesky from
     * partial_cholesky.c.
     *
     * @param A         The system matrix (n x n).
     * @param x         The solution vector (size n). On entry, the initial guess;
     * on exit, the solution.
     * @param b         The right-hand side vector (size n).
     * @param n         The matrix dimension.
     * @param tol       Convergence tolerance.
     * @param max_iter  Maximum iterations.
     * @param max_col   How many columns of A are factorized (partial or full).
     */
    void pcg_solve(const double *A, double *x, const double *b, int n, double tol, int max_iter,
                   int max_col, bool verbose, bool use_ldl_solve, bool use_gpu);

    void invert_ldlt(const double *L, const double *D, double *invP, int n);

#ifdef __cplusplus
}
#endif

#endif // CG_SOLVER_H
