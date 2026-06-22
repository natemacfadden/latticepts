#ifndef BOX_ENUM_H
#define BOX_ENUM_H

// HEADER
// ======
#include <stdint.h>

/*
Enumerate lattice points ``vec`` obeying ``H @ vec >= rhs`` and ``|vec_i| <= B``
using Kannan's algorithm.

Prefer to sort the columns of ``H`` so that stricter constraints come first.
(This is automatically done in the .pyx file). See `set_bounds` for the logic.

Parameters
----------
out : int32_t*
    Output buffer. Must be pre-allocated to hold at least max_N_out * dim
    elements. Lattice points are written in row-major order.
N_out : long*
    Written with the number of lattice points found.
N_nodes : long*
    Written with the number of nodes visited in the search tree (including
    the root). For N_hyps=0 (no hyperplane constraints) this equals
    ((2B+1)^(n+1)-1)/(2B).
dim : int
    Dimension of the problem.
B : int
    Box half-width: each component satisfies |vec_i| <= B.
H : int*
    Constraint matrix of shape (N_hyps, dim), in row-major order. Each row
    defines one inequality ``H[i] @ vec >= rhs[i]``.
rhs : int*
    Right-hand side of each hyperplane constraint, of length N_hyps.
    May be NULL when N_hyps == 0.
N_hyps : int
    Number of hyperplane constraints (rows of H).
max_N_out : long
    Maximum number of output lattice points. Enumeration stops early if
    reached.
max_N_nodes : long
    Maximum number of search tree nodes to visit. Enumeration stops early
    if reached.

Returns
-------
int
    Status code:
        0  : success
       -1  : dim > 256 (unsupported)
       -2  : exceeded max_N_out outputs
       -3  : exceeded max_N_nodes
       -4  : N_hyps too large (constraint buffers would overflow the stack)
*/
int _box_enum_c(
    int32_t * restrict out,
    long * restrict N_out,
    long * restrict N_nodes,
    int dim,
    int B,
    int * restrict H,
    int * restrict rhs,
    int N_hyps,
    long max_N_out,
    long max_N_nodes,
    int primitive
);


// IMPLEMENTATION
// ==============
#ifdef BOX_ENUM_IMPLEMENTATION

#define MAX_SUPPORTED_DIM 256
// budget for the N_hyps-sized stack VLAs in _box_enum_c; inputs exceeding it
// return status -4 instead of overflowing the stack
#define MAX_CONSTRAINT_STACK_BYTES (4 * 1024 * 1024)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #define DEBUG  // uncomment to enable debug logging
#ifdef DEBUG
    #define DEBUG_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
    #define DEBUG_LOG(...) ((void)0)
#endif

static inline int max_int(int a, int b) {
    return a > b ? a : b;
}
static inline int min_int(int a, int b) {
    return a < b ? a : b;
}

// Exact integer ceil/floor division (any signs, b != 0) for the rare
// |numer| >= 2^53 case where set_bounds' double ceil/floor loses precision
static inline int64_t floor_div_i64(int64_t a, int64_t b) {
    int64_t q = a / b, r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}
static inline int64_t ceil_div_i64(int64_t a, int64_t b) {
    int64_t q = a / b, r = a % b;
    return (r != 0 && ((r < 0) == (b < 0))) ? q + 1 : q;
}

// Kannan vec[i] bound setting helper
static inline int set_bounds(
    int sp,
    int i,
    int dim,
    int N_hyps,
    int B,
    int * restrict H,
    int * restrict rhs,
    int64_t * restrict stack_partial_sum,
    int64_t * restrict abssum,
    int32_t * restrict stack_val_min,
    int32_t * restrict stack_val_len)
{
    /*
    Compute the iteration bounds for component i at stack depth sp, writing
    results into stack_val_min[sp] and stack_val_len[sp].

    Parameters
    ----------
    sp : int
        Current stack depth.
    i : int
        Component index being bounded.
    dim, N_hyps, B, H, rhs : (see _box_enum_c; rhs[j] is the bound for constraint j)
    stack_partial_sum : int64_t*
        Partial dot products ``H @ vec`` for components already fixed.
    abssum : int64_t*
        Precomputed prefix sums of |H[:,k]| for each constraint.
    stack_val_min : int32_t*
        Written with the minimum value to try for vec[i].
    stack_val_len : int32_t*
        Written with the number of candidates to try for vec[i].

    Returns
    -------
    int
        Number of candidates (stack_val_len[sp]).
    */
    int lo = -B;
    int hi =  B;

    // cut by each hyperplane
    for (int j=0; j<N_hyps; ++j) {
        /*
        Imposes constraint dot(H[j,:], vec) >= rhs[j].

        Split:
            rhs[j] <= dot(H[j,:i],vec[:i])
                    + H[j,i]*vec[i]
                    + dot(H[j,i+1:],vec[i+1:])
        the third term is stack_partial_sum[j], so
            rhs[j] <= dot(H[j,:i],vec[:i])
                    + H[j,i]*vec[i]
                    + stack_partial_sum[j].
        The maximum value possible of dot(H[j,:i],vec[:i]) is
            dot(H[j,:i],vec[:i]) <= B*sum(abs(H[j,:i])).
        Thus
            rhs[j] - stack_partial_sum[j] - B*sum(abs(H[j,:i])) <=  H[j,i]*vec[i].

        The term B * sum(abs(H[j,:i])) is the 'slack'. It is where any looseness
        in our bounds can arise from. Ideally, then, you'd want
        sum(abs(H[j,:i])) to be minimized to have as tight bounds as possible.

        We obviously can't sort each row/constraint individually since that'd
        muddle the components of vec. What we can do is minimize the net slack,
        sum(abs(H[:,:i])). This is done by ordering *columns* in increasing
        L1-norm, which is the rationale behind the column sorting.
        */
        int h = H[j*dim + i];
        if (h == 0){
            continue;
        }

        // Use int64 to avoid overflow: partial_sum ~ H*B, abssum*B can both exceed 2^31
        int64_t numer = (int64_t)rhs[j]
                      - stack_partial_sum[sp*N_hyps + j]
                      - (int64_t)B * (int64_t)abssum[j*(dim+1) + i];

        // |numer| <= |rhs| + B*sum|H[j]| stays under 2^53 for any feasible
        // box, so the double ceil/floor is exact and ~1.5x faster than int64
        // division on this hot path; guard the rare adversarial |numer| >= 2^53
        // case with an exact integer fallback so the bound is never off-by-one
        const int64_t FP_EXACT = ((int64_t)1 << 53);
        if (h>0){
            int64_t v = (numer >= -FP_EXACT && numer <= FP_EXACT)
                        ? (int64_t)ceil((double)numer/h)
                        : ceil_div_i64(numer, (int64_t)h);
            lo = max_int(lo, v > (int64_t)hi ? hi + 1 : (int)v);
        } else {
            int64_t v = (numer >= -FP_EXACT && numer <= FP_EXACT)
                        ? (int64_t)floor((double)numer/h)
                        : floor_div_i64(numer, (int64_t)h);
            hi = min_int(hi, v < (int64_t)lo ? lo - 1 : (int)v);
        }
    }

    // store the data to recreate the interval
    int num = 0;
    if (hi >= lo) {
        num = hi - lo + 1;
    }
    stack_val_min[sp] = lo;
    stack_val_len[sp] = num;

    // debug print statement
    DEBUG_LOG("Set bounds for %d to %d->%d+%d\n", sp, lo, lo, num);

    return num;
}

// custom Kannan code for lattice vector generation
int _box_enum_c(
    int32_t * restrict out,
    long * restrict N_out,
    long * restrict N_nodes,
    int dim,
    int B,
    int * restrict H,
    int * restrict rhs,
    int N_hyps,
    long max_N_out,
    long max_N_nodes,
    int primitive)
{
    /* (see header doc) */
    // check dimensions are reasonable
    // -------------------------------
    if (dim > MAX_SUPPORTED_DIM) {
        *N_out = 0;
        return -1;
    }

    // the N_hyps-sized VLAs below (stack_partial_sum, abssum) live on the stack;
    // reject inputs whose footprint would overflow a typical ~8 MB stack instead
    // of crashing with no recoverable error
    if ((long)N_hyps * (dim + 1) * (long)(sizeof(int64_t) + sizeof(int))
            > MAX_CONSTRAINT_STACK_BYTES) {
        *N_out = 0;
        return -4;
    }

    // define variables
    // ----------------
    int status = 0;
    *N_nodes = 1;  // the root counts even when the search dies before the loop

    // define arrays
    int32_t vec[dim];

    int32_t stack_i[dim+1];
    int32_t stack_pos[dim+1];

    int32_t stack_val_len[dim+1];
    int32_t stack_val_min[dim+1];

    int64_t stack_partial_sum[N_hyps*(dim+1)];
    memset(stack_partial_sum, 0, sizeof(stack_partial_sum));

    // output/stack pointer
    long op = 0;
    int sp = 0;

    // misc helpers
    int64_t abssum[N_hyps*(dim+1)];
    for (int j=0; j<N_hyps; ++j) {
        abssum[j*(dim+1) + 0] = 0;

        for (int k=0; k<dim; ++k) {
            abssum[j*(dim+1) + k+1] = abssum[j*(dim+1) + k] + abs(H[j*dim + k]);
        }

        // an all-zero row j is the constant constraint 0 >= rhs[j]; set_bounds
        // skips zero coefficients so the row is otherwise ignored, hence if
        // rhs[j] > 0 the constraint is unsatisfiable and the feasible set is empty
        if (abssum[j*(dim+1) + dim] == 0 && rhs[j] > 0) {
            *N_out = 0;
            return 0;
        }
    }

    // initialize stack
    // ----------------
    stack_i[sp]   = dim-1;
    stack_pos[sp] = 0;

    int k = set_bounds(
            sp,
            dim-1,
            dim,
            N_hyps,
            B,
            H,
            rhs,
            stack_partial_sum,
            abssum,
            stack_val_min,
            stack_val_len);
    if (k == 0) {
        goto end;
    }

    // iterate over the stack
    // ----------------------
    int i;
    int pos;

    while (sp >= 0) {

        // read from the stack
        i    = stack_i[sp];
        pos  = stack_pos[sp];

        // debug print statement
        DEBUG_LOG("Setting component-%d for op=%ld, sp=%d, pos=%d\n",
                  i, op, sp, pos);

        // save if node is complete
        // if i==-1, then we have fully written vec
        if (i == -1) {
            // primitive filter (gcd of |components| == 1); only when requested
            if (primitive) {
                int32_t g = 0;
                for (int t = 0; t < dim; t++) {
                    int32_t a = vec[t] < 0 ? -vec[t] : vec[t];
                    while (a) { int32_t r = g % a; g = a; a = r; }
                    if (g == 1) break;   // gcd can only stay 1 from here; stop early
                }
                if (g != 1) { sp--; continue; }   // not primitive: skip (don't count/emit)
            }

            if (op >= max_N_out) {
                status = -2;
                goto end;
            }

            // out == NULL  ->  count-only (dry run): tally op without writing
            if (out != NULL) memcpy(&out[op * dim], vec, dim * sizeof(int32_t));

            op++;

            // kill node
            sp--;
            continue;
        }

        // check if we exhausted values for this component
        if (pos == stack_val_len[sp]) {
            sp--;
            continue;
        }

        // set vec[i]
        int veci = stack_val_min[sp] + pos;
        vec[i] = veci;

        DEBUG_LOG("Set     component-%d for op=%ld, sp=%d, pos=%d to %d\n",
                  i, op, sp, pos, veci);

        // advance pos for next iteration
        stack_pos[sp] += 1;

        // passes cuts -> push next depth :)
        sp += 1;
        (*N_nodes)++;
        if (*N_nodes > max_N_nodes) {
            DEBUG_LOG("QUITTING DUE TO TOO MANY NODES/ITERATIONS\n");
            status = -3;
            goto end;
        }
        stack_i[sp]       = i-1;
        stack_pos[sp]     = 0;

        // update the partial sums
        for (int j = 0; j<N_hyps; ++j) {
            int64_t prev = stack_partial_sum[(sp-1)*N_hyps + j];
            stack_partial_sum[sp*N_hyps+j] = prev + (int64_t)H[j*dim + i] * (int64_t)veci;
        }

        if (i > 0) {
            set_bounds(
                sp,
                i-1,
                dim,
                N_hyps,
                B,
                H,
                rhs,
                stack_partial_sum,
                abssum,
                stack_val_min,
                stack_val_len);
        }
    }

    DEBUG_LOG("DONE\n");

    end:
        *N_out = op;
        return status;
}

#undef MAX_SUPPORTED_DIM
#undef MAX_CONSTRAINT_STACK_BYTES

#endif // BOX_ENUM_IMPLEMENTATION

#endif // BOX_ENUM_H
