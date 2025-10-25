#include "solver_interface.h"
#include <float.h>
#include <glpk.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

LinearProgram read_lp_from_mps(const char *filename)
{
    glp_prob *lp = glp_create_prob();
    glp_read_mps(lp, GLP_MPS_FILE, NULL, filename);

    int rows = glp_get_num_rows(lp);
    int cols = glp_get_num_cols(lp);
    double opt_obj = glp_get_obj_val(lp); // returns 0.0 if not known

    double *A = malloc(rows * cols * sizeof(double));
    double *b = malloc(rows * sizeof(double));
    double *c = malloc(cols * sizeof(double));
    double *lb = malloc(cols * sizeof(double));
    double *ub = malloc(cols * sizeof(double));

    // Read c vector
    for (int j = 0; j < cols; j++)
    {
        c[j] = glp_get_obj_coef(lp, j + 1);
    }

    // x lower and upper bounds
    for (int j = 0; j < cols; j++)
    {
        lb[j] = glp_get_col_lb(lp, j + 1);
        ub[j] = glp_get_col_ub(lp, j + 1);
    }

    int *ind = malloc((cols + 1) * sizeof(int));
    double *val = malloc((cols + 1) * sizeof(double));

    for (int i = 0; i < rows; i++)
    {
        int type = glp_get_row_type(lp, i + 1);
        if (type == GLP_FX || type == GLP_DB || type == GLP_UP || type == GLP_LO)
            b[i] = glp_get_row_lb(lp, i + 1);
        else
            b[i] = 0.0;

        int len = glp_get_mat_row(lp, i + 1, ind, val);
        for (int k = 1; k <= len; k++)
        {
            int j = ind[k] - 1;
            A[i * cols + j] = val[k];
        }
    }

    free(ind);
    free(val);

    glp_delete_prob(lp);

    return (LinearProgram){
        .rows = rows, .cols = cols, .opt_obj = opt_obj, A = A, .b = b, .c = c, .lb = lb, .ub = ub};
}

void free_lp(LinearProgram *lp)
{
    free(lp->A);
    free(lp->b);
    free(lp->c);
    free(lp->lb);
    free(lp->ub);
}
