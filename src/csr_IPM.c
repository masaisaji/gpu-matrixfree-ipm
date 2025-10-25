#include "csr_IPM.h"
#include "IPM.h"
#include "cg_solver.h"
#include "csr_cg_solver.h"
#include "csr_utils.h"
#include "cuda_matmul.h"
#include "cuda_matvec.h"
#include "lin_alg_helper.h"
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_ITER 100
#define TOL 1e-6
#define ALPHA 0.99
#define EPS 1e-8

void update_residuals_and_mu_csr(const CSRMatrix *A, const CSRMatrix *AT, double *Ax, double *ATy,
                                 const double *b, const double *c, const double *x, const double *y,
                                 const double *s, double *rp, double *rp_norm, double *rd,
                                 double *rd_norm, double *normalized_comp, const int m, const int n,
                                 const double b_norm, const double c_norm, double *x_s_dot,
                                 double *mu, double mu_frac, bool use_gpu)
{
    if (use_gpu)
    {
        cuda_matvec_csr(A, x, Ax);
        cuda_matvec_csr(AT, y, ATy);
    }
    else
    {
        matvec_csr(A, x, Ax, false);
        matvec_csr(A, y, ATy, true);
    }
    *x_s_dot = dot(x, s, n);
    *mu = *x_s_dot / n * mu_frac;
    vecsub(m, b, Ax, rp, 1);
    vecsub(n, c, ATy, rd, 1);
    vecsub_inplace(n, s, rd, 1);
    *rp_norm = vec_L2_norm(rp, m) / (1 + c_norm);
    *rd_norm = vec_L2_norm(rd, n) / (1 + b_norm);

    *normalized_comp = *x_s_dot / (n * (1 + fabs(dot(c, x, n))));
}

void csr_to_dense_inplace(const CSRMatrix *csr, double *dense, int nrows, int ncols)
{
    for (int i = 0; i < csr->rows; ++i)
    {
        for (int j = csr->row_ptr[i]; j < csr->row_ptr[i + 1]; ++j)
        {
            int col = csr->col_idx[j];
            double val = csr->val[j];
            dense[i * ncols + col] = val;
        }
    }
}

// GR is a dense matrix
void IPM_with_preconditioner_csr(const CSRMatrix *A, const double *A_dense, const double *b,
                                 const double *c, double *x0, double *y0, double *s0, int m, int n,
                                 int max_col, bool use_mehrotra, bool use_ldl_solve, bool use_gpu,
                                 bool verbose)
{
    if (verbose)
    {
        printf(" Iter |      mu      | Primal feas norm |  Dual feas norm  |  Comp. "
               "slackness  |   Duality Gap \n");
        printf("---------------------------------------------------------------------"
               "---------------------------\n");
    }

    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)calloc(m, sizeof(double));
    double *s = (double *)malloc(n * sizeof(double));

    double *Ax = (double *)malloc(m * sizeof(double));
    double *ATy = (double *)malloc(n * sizeof(double));
    double *theta = (double *)malloc(n * sizeof(double));
    double *reg_theta = (double *)malloc(n * sizeof(double));
    double *GR = (double *)calloc(m * m, sizeof(double));

    double *rp = (double *)malloc(m * sizeof(double));
    double *rd = (double *)malloc(n * sizeof(double));

    double *g = (double *)malloc(m * sizeof(double));
    double *g_int_term = (double *)malloc(n * sizeof(double));
    double *ATdy = (double *)malloc(n * sizeof(double));
    double *dx = (double *)malloc(n * sizeof(double));
    double *dy = (double *)calloc(m, sizeof(double));
    double *ds = (double *)malloc(n * sizeof(double));
    double *dx_aff = (double *)malloc(n * sizeof(double));
    double *dy_aff = (double *)calloc(m, sizeof(double));
    double *ds_aff = (double *)malloc(n * sizeof(double));

    double const reg_p = 1e-12; // primal regularization
    double const reg_d = 1e-8;  // dual regularization

    double const b_norm = vec_L2_norm(b, m);
    double const c_norm = vec_L2_norm(c, n);
    double x_s_dot, mu;
    double rp_norm, rd_norm, normalized_comp;
    double prim_obj = NAN, dual_obj = NAN;
    double alpha_p = 0.0; // step sizes for primal variables
    double alpha_d = 0.0; // step sizes for dual variables
    CSRMatrix AT = csr_transpose(A);

    for (int i = 0; i < n; i++)
    {
        x[i] = x0[i];
        s[i] = s0[i];
    }
    for (int i = 0; i < m; i++)
    {
        y[i] = y0[i];
    }

    int iter = 0;
    update_residuals_and_mu_csr(A, &AT, Ax, ATy, b, c, x, y, s, rp, &rp_norm, rd, &rd_norm,
                                &normalized_comp, m, n, b_norm, c_norm, &x_s_dot, &mu, 1.0,
                                use_gpu);
    if (verbose)
    {
        printf("%4d  |  %10.4e  |  %14.4e  |  %14.4e  |  %14.4e   |  %14.4e\n", iter, mu, rp_norm,
               rd_norm, normalized_comp, dot(x, c, n) - dot(b, y, m));
    }

    while ((rp_norm > TOL || rd_norm > TOL || normalized_comp > TOL) && iter < MAX_ITER)
    {

        for (int i = 0; i < n; i++)
        {
            double safe_si = fmax(s[i], 1e-12);
            theta[i] = x[i] / safe_si;
            reg_theta[i] = 1 / (1 / theta[i] + reg_p);
        }

        if (use_gpu)
        {
            CSRMatrix AD;
            CSRMatrix GR_csr;
            copy_csr_matrix(A, &AD);
            for (int i = 0; i < AD.rows; ++i)
            {
                for (int idx = AD.row_ptr[i]; idx < AD.row_ptr[i + 1]; ++idx)
                {
                    int j = A->col_idx[idx];
                    AD.val[idx] *= reg_theta[j];
                }
            }
            GR_csr = cuda_matmul_csr(&AD, &AT);
            freeCSRmat(&AD);
            csr_to_dense_inplace(&GR_csr, GR, m, m);
            freeCSRmat(&GR_csr);
        }
        else
        {
            ADAT_prod(A_dense, reg_theta, GR, m, n);
        }
        for (int i = 0; i < m; i++)
        {
            GR[i * m + i] += reg_d;
        }
        if (use_mehrotra)
        {
            double sigma = 0.0;
            // Affine direction
            for (int i = 0; i < n; i++)
            {
                g_int_term[i] = theta[i] * rd[i] + x[i];
            }
            if (use_gpu)
            {
                cuda_matvec_csr(A, g_int_term, g);
            }
            else
            {
                matvec_csr(A, g_int_term, g, false);
            }
            vecadd_inplace(m, rp, g, 1);
            pcg_solve(GR, dy_aff, g, m, 1e-12, 10000, max_col, false, use_ldl_solve, use_gpu);
            if (use_gpu)
            {
                cuda_matvec_csr(&AT, dy_aff, ATdy);
            }
            else
            {
                matvec_csr(A, dy_aff, ATdy, true);
            }
            vecsub(n, rd, ATdy, ds_aff, 1);
            for (int i = 0; i < n; i++)
            {
                dx_aff[i] = -x[i] - theta[i] * ds_aff[i];
            }

            double alpha_p_aff, alpha_d_aff;
            compute_step_sizes(x, dx_aff, s, ds_aff, n, &alpha_p_aff, &alpha_d_aff, true);

            // Corrector step
            double mu_aff = 0.0;
            for (int i = 0; i < n; ++i)
            {
                double x_new = x[i] + alpha_p_aff * dx_aff[i];
                double s_new = s[i] + alpha_d_aff * ds_aff[i];
                mu_aff += x_new * s_new;
            }
            mu_aff /= n;
            sigma = fmin(1.0, pow(mu_aff / mu, 3.0));

            for (int i = 0; i < n; i++)
            {
                double safe_si = fmax(s[i], 1e-12);
                g_int_term[i] = dx_aff[i] * ds_aff[i] / safe_si - mu * sigma / safe_si;
            }

            if (use_gpu)
            {
                cuda_matvec_csr(A, g_int_term, g);
            }
            else
            {
                matvec_csr(A, g_int_term, g, false);
            }
            pcg_solve(GR, dy, g, m, 1e-12, 10000, max_col, false, use_ldl_solve, use_gpu);
            if (use_gpu)
            {
                cuda_matvec_csr(&AT, dy, ATdy);
            }
            else
            {
                matvec_csr(A, dy, ATdy, true);
            }
            for (int i = 0; i < n; i++)
            {
                ds[i] = -ATdy[i];
            }
            for (int i = 0; i < n; i++)
            {
                double safe_si = fmax(s[i], 1e-12);
                dx[i] = (sigma * mu - ds_aff[i] * dx_aff[i]) / safe_si - theta[i] * ds[i];
            }

            vecadd_inplace(n, dx_aff, dx, 1.0);
            vecadd_inplace(m, dy_aff, dy, 1.0);
            vecadd_inplace(n, ds_aff, ds, 1.0);

            compute_step_sizes(x, dx, s, ds, n, &alpha_p, &alpha_d, false);
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                g_int_term[i] = theta[i] * rd[i] + x[i] - mu / s[i];
            }
            if (use_gpu)
            {
                cuda_matvec_csr(A, g_int_term, g);
            }
            else
            {
                matvec_csr(A, g_int_term, g, false);
            }
            vecadd_inplace(m, rp, g, 1);
            pcg_solve(GR, dy, g, m, 1e-12, 10000, max_col, false, use_ldl_solve, use_gpu);

            // s search direction
            if (use_gpu)
            {
                cuda_matvec_csr(&AT, dy, ATdy);
            }
            else
            {
                matvec_csr(A, dy, ATdy, true);
            }
            vecsub(n, rd, ATdy, ds, 1);

            // x search direction
            for (int i = 0; i < n; i++)
            {
                dx[i] = mu / s[i] - x[i] - theta[i] * ds[i]; //
            }
            compute_step_sizes(x, dx, s, ds, n, &alpha_p, &alpha_d, false);
        }
        alpha_p *= ALPHA;
        alpha_d *= ALPHA;
        alpha_p = fmin(1.0, alpha_p);
        alpha_d = fmin(1.0, alpha_d);

        // variable update
        vecadd_inplace(n, dx, x, alpha_p);
        vecadd_inplace(m, dy, y, alpha_d);
        vecadd_inplace(n, ds, s, alpha_d);

        update_residuals_and_mu_csr(A, &AT, Ax, ATy, b, c, x, y, s, rp, &rp_norm, rd, &rd_norm,
                                    &normalized_comp, m, n, b_norm, c_norm, &x_s_dot, &mu, 1.0,
                                    use_gpu);
        prim_obj = dot(x, c, n);
        dual_obj = dot(b, y, m);
        if (verbose)
        {
            printf("%4d  |  %10.4e  |  %14.4e  |  %14.4e  |  %14.4e   |  %14.4e\n", iter + 1, mu,
                   rp_norm, rd_norm, normalized_comp, prim_obj - dual_obj);
        }
        iter++;
    }
    printf("Final primal objective: %f\n", prim_obj);

    freeCSRmat(&AT);
    free(GR);
    free(x);
    free(y);
    free(s);
    free(dx);
    free(dy);
    free(ds);
    free(dx_aff);
    free(dy_aff);
    free(ds_aff);
    free(Ax);
    free(ATy);
    free(ATdy);
    free(theta);
    free(reg_theta);
    free(rp);
    free(rd);
    free(g);
    free(g_int_term);
}
