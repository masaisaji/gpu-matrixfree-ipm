#include <stdbool.h>

#ifndef IPM_H
#define IPM_H

void IPM_with_preconditioner(double *A, double *b, double *c, double *x0, double *y0, double *s0,
                             int m, int n, int max_col, bool use_mehrotra, bool use_ldl_solve,
                             bool verbose);
void compute_step_sizes(const double *x, const double *dx, const double *s, const double *ds, int n,
                        double *alpha_p_out, double *alpha_d_out, bool for_aff_dir);
#endif // IPM_H
