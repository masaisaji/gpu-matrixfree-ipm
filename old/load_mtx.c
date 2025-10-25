#include "cs.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int row;
    int col;
    double val;
} Triplet;

int triplet_cmp(const void *a, const void *b) {
    Triplet *ta = (Triplet *)a;
    Triplet *tb = (Triplet *)b;
    if (ta->row != tb->row) return ta->row - tb->row;
    return ta->col - tb->col;
}

int load_mtx_to_csr(const char *filename,
                    double **values,
                    int **col_indices,
                    int **row_ptr,
                    int *m, int *n, int *nnz) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Could not open file %s\n", filename);
        return 1;
    }

    char header[256];
    if (!fgets(header, sizeof(header), f)) {
        fclose(f);
        return 1;
    }

    int is_array = strstr(header, "array") != NULL;
    rewind(f);

    if (is_array) {
        // Custom dense array matrix parser
        char line[256];
        do {
            if (!fgets(line, sizeof(line), f)) {
                fclose(f);
                return 1;
            }
        } while (line[0] == '%');

        sscanf(line, "%d %d", m, n);
        *nnz = (*m) * (*n);

        *values = (double *)malloc(*nnz * sizeof(double));
        *col_indices = (int *)malloc(*nnz * sizeof(int));
        *row_ptr = (int *)calloc(*m + 1, sizeof(int));

        int count = 0;
        for (int i = 0; i < *m; i++) {
            (*row_ptr)[i + 1] = (*row_ptr)[i] + *n;
            for (int j = 0; j < *n; j++) {
                if (fscanf(f, "%lf", &(*values)[count]) != 1) {
                    fprintf(stderr, "Error reading matrix value\n");
                    fclose(f);
                    return 1;
                }
                (*col_indices)[count] = j;
                count++;
            }
        }

        fclose(f);
        return 0;
    } else {
        // Patch file: rewrite integer values to real if needed, including header
        FILE *temp = tmpfile();
        if (!temp) {
            fclose(f);
            fprintf(stderr, "Could not create temporary file for matrix patching.\n");
            return 1;
        }

        char line[512];
        int line_num = 0;
        int is_data_section = 0;

        while (fgets(line, sizeof(line), f)) {
            if (line_num == 0 && strstr(line, "integer")) {
                // Rewrite the header to say "real" instead of "integer"
                fprintf(temp, "%%MatrixMarket matrix coordinate real general\n");
                line_num++;
                continue;
            }
            if (line[0] == '%') {
                fputs(line, temp);
                line_num++;
                continue;
            }
            if (!is_data_section) {
                // Copy size line exactly as-is (integers only)
                fputs(line, temp);
                is_data_section = 1;
            } else {
                int i, j;
                double v;
                if (sscanf(line, "%d %d %lf", &i, &j, &v) == 3) {
                    fprintf(temp, "%d %d %.1f\n", i, j, v);
                } else {
                    int iv, jv, vv;
                    if (sscanf(line, "%d %d %d", &iv, &jv, &vv) == 3) {
                        fprintf(temp, "%d %d %.1f\n", iv, jv, (double)vv);
                    } else {
                        fputs(line, temp); // fallback
                    }
                }
            }
            line_num++;
        }

             // Debug: Print the first 5 lines of the patched matrix file before cs_load
        rewind(temp);
        char debug_line[512];
        int debug_count = 0;
        printf("DEBUG: Printing the first 5 lines of the patched matrix file:\n");
        while (fgets(debug_line, sizeof(debug_line), temp) && debug_count < 5) {
            printf("DEBUG: Line %d: %s", debug_count++, debug_line);
        }
        rewind(temp);  // Go back to the start for cs_load

        // Debug: Print the first 5 matrix entries to verify data
        printf("DEBUG: Printing first 5 matrix entries:\n");
        int entry_count = 0;
        while (fgets(debug_line, sizeof(debug_line), temp) && entry_count < 5) {
            printf("DEBUG: Entry %d: %s", entry_count++, debug_line);
        }
        rewind(temp);  // Go back to the start for cs_load

        // Now load the matrix
        cs *T = cs_load(temp);
        fclose(f);

        if (!T || T->m == 0 || T->n == 0 || T->nz == 0) {
            fprintf(stderr, "cs_load failed or empty matrix: %s (m=%d, n=%d, nz=%d)\n", filename,
                    T ? T->m : -1, T ? T->n : -1, T ? T->nz : -1);
            if (T) cs_spfree(T);
            fclose(temp);
            return 1;
        }


        *m = T->m;
        *n = T->n;
        *nnz = T->nz;

        Triplet *triplets = (Triplet *)malloc(T->nz * sizeof(Triplet));
        for (int i = 0; i < T->nz; i++) {
            triplets[i].row = T->i[i];
            triplets[i].col = T->p[i];
            triplets[i].val = T->x[i];
        }

        qsort(triplets, T->nz, sizeof(Triplet), triplet_cmp);

        *values = (double *)malloc(T->nz * sizeof(double));
        *col_indices = (int *)malloc(T->nz * sizeof(int));
        *row_ptr = (int *)calloc((*m + 1), sizeof(int));

        for (int i = 0; i < T->nz; i++) {
            (*values)[i] = triplets[i].val;
            (*col_indices)[i] = triplets[i].col;
            (*row_ptr)[triplets[i].row + 1]++;
        }

        for (int i = 0; i < *m; i++) {
            (*row_ptr)[i + 1] += (*row_ptr)[i];
        }

        cs_spfree(T);
        free(triplets);
        fclose(temp);
        return 0;
    }
}

int load_mtx_vector(const char *filename, double **vec, int *length) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Could not open vector file %s\n", filename);
        return 1;
    }

    char header[256];
    if (!fgets(header, sizeof(header), f)) {
        fclose(f);
        return 1;
    }

    int is_array = strstr(header, "array") != NULL;

    // Skip comments
    char line[256];
    do {
        if (!fgets(line, sizeof(line), f)) {
            fclose(f);
            return 1;
        }
    } while (line[0] == '%');

    if (is_array) {
        int rows, cols;
        sscanf(line, "%d %d", &rows, &cols);
        if (cols != 1) {
            fprintf(stderr, "Expected column vector, got %d cols\n", cols);
            fclose(f);
            return 1;
        }

        *vec = (double *)malloc(rows * sizeof(double));
        *length = rows;
        for (int i = 0; i < rows; i++) {
            if (fscanf(f, "%lf", &(*vec)[i]) != 1) {
                fprintf(stderr, "Unexpected format in vector file.\n");
                fclose(f);
                return 1;
            }
        }

        fclose(f);
        return 0;
    } else {
        rewind(f);
        cs *T = cs_load(f);
        fclose(f);
        if (!T || T->n != 1) {
            fprintf(stderr, "cs_load failed or not a single-column vector: %s\n", filename);
            if (T) cs_spfree(T);
            return 1;
        }

        *length = T->m;
        *vec = (double *)calloc(*length, sizeof(double));
        for (int i = 0; i < T->nz; i++) {
            int row = T->i[i];
            double val = T->x[i];
            (*vec)[row] = val;
        }

        cs_spfree(T);
        return 0;
    }
}
