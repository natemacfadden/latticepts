#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>
#include <time.h>

#define BOX_ENUM_IMPLEMENTATION
#include "box_enum.h"

int main(int argc, char *argv[])
{
    // Manwe:
    int dim = 7;
    int N_hyps = 73;
    int H[] = {0,  -5, 0,  3,  3,  0,  -1,
               0,  0,  0,  0,  -1, -2, -3,
               0,  0,  0,  -1, 0,  -3, -5,
               0,  0,  1,  0,  -1, -2, -3,
               0,  0,  -1, -1, 0,  -3, -4,
               0,  0,  0,  0,  0,  -1, -2,
               0,  0,  0,  0,  -1, -2, -3,
               0,  0,  0,  -1, -1, -5, -8,
               0,  0,  1,  0,  0,  0,  -1,
               0,  0,  0,  0,  0,  0,  -1,
               0,  0,  0,  -1, 0,  0,  -2,
               0,  1,  0,  -1, 1,  0,  -1,
               -1, 0,  0,  0,  2,  0,  -1,
               0,  -1, 0,  0,  0,  -4, -6,
               0,  0,  -4, -1, 0,  0,  -2,
               -1, 0,  -2, 0,  0,  0,  -1,
               -1, 0,  0,  0,  0,  -2, -3,
               0,  0,  -1, 0,  0,  -2, -2,
               0,  1,  -1, -2, 0,  -4, -6,
               0,  -1, 1,  0,  0,  0,  -2,
               0,  -3, 0,  1,  3,  0,  -1,
               0,  -1, 0,  0,  3,  0,  -1,
               0,  -1, 0,  0,  0,  -3, -4,
               0,  -1, 0,  0,  0,  0,  -2,
               0,  -1, 0,  0,  2,  0,  -1,
               0,  0,  0,  -1, -1, 0,  -3,
               0,  -1, 1,  0,  0,  -2, -4,
               0,  0,  0,  0,  0,  0,  -1,
               0,  -1, -1, 0,  0,  -2, -2,
               0,  0,  0,  0,  -1, 0,  -1,
               0,  -1, 0,  0,  0,  -3, -5,
               -1, 0,  0,  0,  0,  0,  -1,
               0,  0,  0,  0,  1,  0,  -1,
               0,  0,  1,  0,  0,  -2, -4,
               0,  1,  0,  -1, 0,  -1, -2,
               0,  0,  0,  0,  -1, 0,  -1,
               0,  0,  0,  -1, 0,  -4, -7,
               0,  0,  1,  0,  1,  0,  -1,
               0,  -1, 0,  0,  0,  -2, -3,
               0,  0,  0,  0,  0,  -1, -1,
               0,  0,  1,  0,  -1, -1, -2,
               0,  0,  1,  0,  2,  0,  -2,
               0,  0,  -1, 0,  0,  0,  0,
               0,  0,  0,  0,  -1, 0,  -1,
               -3, 0,  0,  0,  2,  0,  -1,
               0,  -1, 0,  0,  3,  0,  -2,
               5,  0,  0,  -2, -2, 0,  -1,
               -1, -1, 0,  1,  1,  0,  0,
               0,  -1, 0,  0,  4,  0,  -2,
               0,  0,  -1, 0,  0,  -1, -1,
               0,  0,  1,  0,  1,  0,  -2,
               0,  0,  0,  -1, 0,  0,  -2,
               0,  -1, 0,  0,  0,  0,  -2,
               0,  -1, -3, 0,  0,  0,  -2,
               0,  0,  1,  0,  0,  -1, -3,
               0,  0,  0,  -1, 0,  -4, -6,
               0,  0,  -1, -2, 0,  0,  -4,
               0,  1,  0,  -1, -1, -2, -3,
               0,  0,  -1, -1, 3,  0,  -1,
               0,  -1, -1, 0,  2,  0,  0,
               -1, 0,  0,  0,  0,  0,  -1,
               3,  0,  0,  -1, -1, 0,  0,
               3,  0,  0,  -2, 0,  0,  -1,
               0,  0,  0,  0,  -1, -1, -1,
               0,  0,  0,  -1, 0,  0,  -2,
               0,  0,  -1, -2, 0,  -7, -11,
               0,  0,  1,  0,  0,  -1, -2,
               5,  0,  0,  -2, 0,  0,  -1,
               0,  0,  0,  0,  0,  -1, -1,
               0,  0,  -1, 0,  -1, 0,  -1,
               0,  0,  1,  0,  -1, 0,  -1,
               -1, 0,  0,  0,  1,  0,  0,
               0,  0,  0,  -1, 0,  0,  -3};
    long max_N_out = 10000000000;
    long max_N_iter = 1000000000000;

    // read box size B
    if (argc != 2) {
        fprintf(stderr, "Usage: %s B\n", argv[0]);
        fprintf(stderr, "  Enumerate lattice points in [-B,B]^%d satisfying H @ x >= 1\n", dim);
        fprintf(stderr, "  Example: %s 3\n", argv[0]);
        return 1;
    }

    char *str_end;
    errno = 0;
    long val = strtol(argv[1], &str_end, 10);
    if (errno == ERANGE || *str_end != '\0' || val <= 0 || val > INT_MAX) {
        fprintf(stderr, "Invalid B: must be a positive integer\n");
        return 1;
    }
    int B = (int)val;

    // initialize output array
    int32_t *out = malloc((size_t)max_N_out * dim * sizeof(int32_t));
    if (!out) {
        perror("malloc out");
        return 1;
    }

    // do the enumeration
    long N_out = 0;

    clock_t t_start = clock();
    int rc = _box_enum_c(
        out,
        &N_out,
        dim,
        B,
        H,
        1,
        N_hyps,
        max_N_out,
        max_N_iter);
    clock_t t_end = clock();
    double eval_time = (double)(t_end - t_start) / CLOCKS_PER_SEC;

    if (rc != 0) {
        fprintf(stderr, "_box_enum_c failed (%d)\n", rc);
    } else {
        printf("Generated %ld vectors in %fs\n", N_out, eval_time);
    }

    // free memory
    free(out);
    return 0;
}
