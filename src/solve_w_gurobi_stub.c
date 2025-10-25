#ifndef USE_GUROBI

#include "solve_w_gurobi.h"
#include <stdio.h>

// fallback stub if Gurobi support is not built
int solve_lp_with_gurobi(const double *A, const double *b, const double *c, int m, int n,
                         double *x_out)
{
    (void)A;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)x_out;
    fprintf(stderr, "[WARN] Gurobi support not built. Rebuild with USE_GUROBI=1.\n");
    return -1;
}

#endif /* !USE_GUROBI */
