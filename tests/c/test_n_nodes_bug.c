/* =============================================================================
 * Regression test for the *N_nodes early-exit bug in _box_enum_c.
 *
 * Bug: when set_bounds() returns 0 for the root component (the very first
 * call, at sp=0), _box_enum_c does `goto end`, which skips the
 * `*N_nodes = 1; // count the root` assignment that lives inside the main
 * loop. As a result *N_nodes is NEVER written and the caller observes
 * whatever it happened to initialize the variable to (garbage in C).
 *
 * The header documents N_nodes as "the number of nodes visited in the search
 * tree (including the root)", so even when the search dies immediately the
 * root has been visited and *N_nodes must be 1.
 *
 * Strategy: poison the caller's N_nodes with a sentinel before the call. A
 * correct implementation overwrites it with 1; the buggy one leaves the
 * sentinel untouched.
 *
 * Build & run:
 *     make test_n_nodes_bug && ./test_n_nodes_bug
 * Exit code 0 = all pass, 1 = a failure (bug present).
 * ===========================================================================*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define BOX_ENUM_IMPLEMENTATION
#include "box_enum.h"

#define SENTINEL (-999L)

static int failures = 0;

static void check_long(const char *what, long got, long want) {
    if (got == want) {
        printf("  PASS: %-40s got %ld\n", what, got);
    } else {
        printf("  FAIL: %-40s got %ld, expected %ld\n", what, got, want);
        failures++;
    }
}

/* Run one enumeration with N_nodes pre-poisoned to SENTINEL. */
static int run(int dim, int B, int *H, int *rhs, int N_hyps,
               long *N_out, long *N_nodes) {
    long max_N_out  = 1000000;
    long max_N_node = 1000000000000L;
    int32_t *out = malloc((size_t)max_N_out * dim * sizeof(int32_t));

    *N_out   = SENTINEL;
    *N_nodes = SENTINEL;   /* poison: a correct impl must overwrite this */

    int rc = _box_enum_c(out, N_out, N_nodes, dim, B,
                         H, rhs, N_hyps, max_N_out, max_N_node);
    free(out);
    return rc;
}

int main(void) {
    long N_out, N_nodes;
    int rc;

    /* -------------------------------------------------------------------
     * Case 1 (THE BUG): root dies immediately.
     *
     * dim=1, B=2, constraints  x >= 1  AND  -x >= 1  (i.e. x>=1 and x<=-1).
     * With dim=1 there is no slack, so set_bounds at the root yields
     * lo=1, hi=-1 -> 0 candidates -> `goto end` -> *N_nodes never set.
     *
     * Expected: status 0, 0 points found, but the root still counts so
     * N_nodes == 1.
     * ----------------------------------------------------------------- */
    printf("Case 1: empty-at-root (dim=1, x>=1 AND -x>=1)\n");
    {
        int H[]   = {1, -1};   /* shape (2,1) */
        int rhs[] = {1, 1};
        rc = run(/*dim=*/1, /*B=*/2, H, rhs, /*N_hyps=*/2, &N_out, &N_nodes);
        check_long("status", rc, 0);
        check_long("N_out", N_out, 0);
        check_long("N_nodes (root counts)", N_nodes, 1);
    }

    /* -------------------------------------------------------------------
     * Case 2 (regression guard): non-empty search still reports N_nodes.
     *
     * dim=1, B=2, no hyperplane constraints -> enumerate {-2,-1,0,1,2}.
     * For N_hyps=0 the documented node count is ((2B+1)^(dim+1)-1)/(2B)
     * = (5^2 - 1)/4 = 6.
     * ----------------------------------------------------------------- */
    printf("Case 2: non-empty (dim=1, B=2, no constraints)\n");
    {
        rc = run(/*dim=*/1, /*B=*/2, NULL, NULL, /*N_hyps=*/0,
                 &N_out, &N_nodes);
        check_long("status", rc, 0);
        check_long("N_out", N_out, 5);
        check_long("N_nodes", N_nodes, 6);
    }

    /* -------------------------------------------------------------------
     * Case 3 (the bug, higher dim): root dies in dim=2.
     *
     * The trigger only needs the *first* (highest-index) component's bounds
     * to be empty. Make component index dim-1 unsatisfiable directly:
     * x1 >= 1 AND -x1 >= 1 (the x0 column is all-zero so it adds no slack).
     * ----------------------------------------------------------------- */
    printf("Case 3: empty-at-root (dim=2, x1>=1 AND -x1>=1)\n");
    {
        int H[]   = {0, 1,
                     0, -1};   /* shape (2,2): constrains only component 1 */
        int rhs[] = {1, 1};
        rc = run(/*dim=*/2, /*B=*/2, H, rhs, /*N_hyps=*/2, &N_out, &N_nodes);
        check_long("status", rc, 0);
        check_long("N_out", N_out, 0);
        check_long("N_nodes (root counts)", N_nodes, 1);
    }

    printf("\n%s (%d failure%s)\n",
           failures ? "TESTS FAILED" : "ALL TESTS PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
