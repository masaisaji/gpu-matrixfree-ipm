#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void get_preconditioner(const double *A_in, int n, int max_col,
                            double **P_out, /* returns allocated P matrix */
                            int **perm_out, /* returns allocated perm array */
                            bool is_test);

    void get_preconditioner_LD(const double *A_in, int n, int max_col, double **L_out,
                               double **D_out, int **perm_out);

#ifdef __cplusplus
}
#endif

#endif
