#ifdef USE_GUROBI

#include "solve_w_gurobi.h"
#include <float.h>
#include <glpk.h>
#include <gurobi_c.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
                         double *x_out)
{
    GRBenv *env = NULL;
    GRBmodel *model = NULL;
    int error = 0;

    // 1. Create environment and model
    error = GRBloadenv(&env, "gurobi.log");
    if (error)
        goto QUIT;

    error = GRBnewmodel(env, &model, "lp_test", n, (double *)c, NULL, NULL, NULL, NULL);
    if (error)
        goto QUIT;

    // 2. Set variable lower bounds (x ≥ 0)
    double *lb = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++)
        lb[i] = 0.0;
    error = GRBsetdblattrarray(model, GRB_DBL_ATTR_LB, 0, n, lb);
    free(lb);
    if (error)
        goto QUIT;

    // 3. Add constraints (dense row-wise)
    for (int i = 0; i < m; i++)
    {
        int *ind = malloc(n * sizeof(int));
        double *val = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++)
        {
            ind[j] = j;
            val[j] = A[i * n + j]; // row-major
        }
        error = GRBaddconstr(model, n, ind, val, GRB_EQUAL, b[i], NULL);
        free(ind);
        free(val);
        if (error)
            goto QUIT;
    }

    // 4. Optimize
    error = GRBoptimize(model);
    if (error)
        goto QUIT;

    // 5. Retrieve solution
    error = GRBgetdblattrarray(model, GRB_DBL_ATTR_X, 0, n, x_out);

QUIT:
    if (error)
    {
        fprintf(stderr, "Gurobi error: %d\n", error);
    }

    if (model)
        GRBfreemodel(model);
    if (env)
        GRBfreeenv(env);

    return error;
}

#endif // USE_GUROBI
