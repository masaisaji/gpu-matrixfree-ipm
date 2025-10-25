#ifndef LP_READER_H
#define LP_READER_H

typedef struct
{
    int rows;
    int cols;
    double opt_obj;
    double *A; // Row-major (rows x cols)
    double *b; // size = rows
    double *c; // size = cols
    double *lb;
    double *ub;
} LinearProgram;

LinearProgram read_lp_from_mps(const char *filename);
void free_lp(LinearProgram *lp);

#endif
