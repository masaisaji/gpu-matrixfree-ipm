#ifndef GEN_INITIAL_GUESS_H
#define GEN_INITIAL_GUESS_H
#include "csr_utils.h"
#include <stdbool.h>

#ifdef __cplusplus
extrn "C"
{
#endif

    void generate_feasible_start(const CSRMatrix *A, const double *b, const double *c,
                                 double **x0_out, double **y0_out, double **s0_out, bool use_gpu);
#ifdef __cplusplus
}
#endif
#endif
