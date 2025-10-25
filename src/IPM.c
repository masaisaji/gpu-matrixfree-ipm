#include "IPM.h"
#include "cg_solver.h"
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

/* Find a such that
 * a_p = max{a: x[i] + a_p * dx[i] >= 0 for all i} and
 * a_d = max{a: s[i] + a_d * ds[i] >= 0 for all i}
 *
 * Since x and s stay positive, we only need to consider cases
 * where dx and ds are negaive. If all elements are positive,
 * set alpha = 1 instead of infinity.
 * */
void compute_step_sizes(const double *x, const double *dx, const double *s, const double *ds, int n,
                        double *alpha_p_out, double *alpha_d_out, bool for_aff_dir)
{
    double alpha_p = DBL_MAX;
    double alpha_d = DBL_MAX;
    double min_step = for_aff_dir ? 1.0 : EPS;

    for (int i = 0; i < n; ++i)
    {
        if (dx[i] < 0.0)
        {
            double a_p = -x[i] / dx[i];
            if (a_p < alpha_p)
                alpha_p = a_p;
        }
        if (ds[i] < 0.0)
        {
            double a_d = -s[i] / ds[i];
            if (a_d < alpha_d)
                alpha_d = a_d;
        }
    }

    // No negative dx or ds cases
    if (alpha_p == DBL_MAX)
        alpha_p = min_step;
    if (alpha_d == DBL_MAX)
        alpha_d = min_step;

    *alpha_p_out = alpha_p;
    *alpha_d_out = alpha_d;
}

void update_residuals_and_mu(const double *A, double *Ax, double *ATy, const double *b,
                             const double *c, const double *x, const double *y, const double *s,
                             double *rp, double *rp_norm, double *rd, double *rd_norm,
                             double *normalized_comp, const int m, const int n, const double b_norm,
                             const double c_norm, double *x_s_dot, double *mu, double mu_frac)
{
    matvec(A, x, Ax, m, n, false);
    matvec(A, y, ATy, m, n, true);
    *x_s_dot = dot(x, s, n);
    *mu = *x_s_dot / n * mu_frac;
    vecsub(m, b, Ax, rp, 1);
    vecsub(n, c, ATy, rd, 1);
    vecsub_inplace(n, s, rd, 1);
    *rp_norm = vec_L2_norm(rp, m) / (1 + c_norm);
    *rd_norm = vec_L2_norm(rd, n) / (1 + b_norm);

    *normalized_comp = *x_s_dot / (n * (1 + fabs(dot(c, x, n))));
}

void IPM_with_preconditioner(double *A, double *b, double *c, double *x0, double *y0, double *s0,
                             int m, int n, int max_col, bool use_mehrotra, bool use_ldl_solve,
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
    update_residuals_and_mu(A, Ax, ATy, b, c, x, y, s, rp, &rp_norm, rd, &rd_norm, &normalized_comp,
                            m, n, b_norm, c_norm, &x_s_dot, &mu, 1.0);
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

        ADAT_prod(A, reg_theta, GR, m, n);
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
            matvec(A, g_int_term, g, m, n, false);
            vecadd_inplace(m, rp, g, 1);
            pcg_solve(GR, dy_aff, g, m, 1e-12, 10000, max_col, false, use_ldl_solve, false);
            matvec(A, dy_aff, ATdy, m, n, true);
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
            matvec(A, g_int_term, g, m, n, false);
            pcg_solve(GR, dy, g, m, 1e-12, 10000, max_col, false, use_ldl_solve, false);
            matvec(A, dy, ATdy, m, n, true);
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
            matvec(A, g_int_term, g, m, n, false);
            vecadd_inplace(m, rp, g, 1);
            pcg_solve(GR, dy, g, m, 1e-12, 10000, max_col, false, use_ldl_solve, false);
            // solve_spd_system(GR, dy, g, m); // for debugging

            // s search direction
            matvec(A, dy, ATdy, m, n, true);
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

        update_residuals_and_mu(A, Ax, ATy, b, c, x, y, s, rp, &rp_norm, rd, &rd_norm,
                                &normalized_comp, m, n, b_norm, c_norm, &x_s_dot, &mu, 1.0);
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
    free(GR);
    free(g);
    free(g_int_term);
}
