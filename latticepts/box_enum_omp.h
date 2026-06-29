#ifndef BOX_ENUM_OMP_H
#define BOX_ENUM_OMP_H

// Optional OpenMP parallel enumeration path for box_enum (counting + materializing)
//
// Kept separate from box_enum.h on purpose: box_enum.h is a self-contained,
// dependency-free, single-file serial kernel (vendored standalone, e.g. in
// Macaulay2), and we don't want to dilute that artifact with parallel code;
// this header layers opt-in parallel enumeration on top of it, and is only built
// with the parallel path when the extension is compiled with -fopenmp
// (LATTICEPTS_OPENMP=1); without -fopenmp every call here forwards to the
// serial kernel, so the default build is byte-identical to box_enum.h
//
// enum_branch (and the surrounding setup) below deliberately re-implement the
// per-branch DFS and prologue from box_enum.h's serial loop rather than
// refactoring the serial kernel to expose a shared worker -- that refactor
// would entangle and grow the standalone artifact; this duplication is the
// deliberate price of keeping box_enum.h pristine

#include "box_enum.h"   // _box_enum_c, set_bounds, the i64 div helpers, types

// near-drop-in for _box_enum_c: same point count and status code (and same
// point set except under truncation), but it does not early-exit; it
// enumerates every branch fully and applies the max_N_out / max_N_nodes caps
// afterward, so N_nodes reflects the full search (the serial path bails early
// and reports fewer) and a tight max_N_nodes will not abort a call here;
// built with -fopenmp, both counting (out == NULL) and materialization run in
// parallel over the independent top-level branches; materialization uses a
// two-pass count-then-fill so each branch writes a disjoint output slice, while
// a serial build forwards every call to box_enum.h's _box_enum_c
int _box_enum_c_omp(
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


#ifdef BOX_ENUM_IMPLEMENTATION

#ifdef _OPENMP
#include <omp.h>

// box_enum.h #undef's these at the end of its implementation, so restate the
// two defensive limits here -- kept identical to box_enum.h on purpose
#define OMP_MAX_SUPPORTED_DIM 256
#define OMP_MAX_CONSTRAINT_STACK_BYTES (4 * 1024 * 1024)

// Branch worker: enumerate one top-level branch (vec[dim-1] pinned to v0) on
// thread-local scratch, returning the branch's lattice-point count; if out is
// non-NULL it also materializes each point at global index base + (local index),
// skipping any index >= max_N_out (the truncation cap); H/rhs/abssum are
// read-only / shared, and the per-node step mirrors box_enum.h's serial loop
static long enum_branch(
    int32_t * restrict out, long base, long max_N_out,
    int v0, int dim, int N_hyps, int B, int primitive,
    int * restrict H, int * restrict rhs, int64_t * restrict abssum,
    int32_t * restrict vec,
    int32_t * restrict stack_i, int32_t * restrict stack_pos,
    int32_t * restrict stack_val_min, int32_t * restrict stack_val_len,
    int64_t * restrict stack_partial_sum, long * restrict nodes)
{
    long cnt = 0;
    vec[dim-1] = v0;
    for (int j = 0; j < N_hyps; ++j)
        stack_partial_sum[1*N_hyps + j] = (int64_t)H[j*dim + (dim-1)] * (int64_t)v0;

    if (dim == 1) {                 // vec fully set by the top component
        if (primitive) { int32_t a = v0 < 0 ? -v0 : v0; if (a != 1) return 0; }
        if (out != NULL && base < max_N_out)
            memcpy(&out[base*dim], vec, dim * sizeof(int32_t));
        return 1;
    }

    int sp = 1;
    stack_i[sp] = dim - 2;
    stack_pos[sp] = 0;
    if (set_bounds(sp, dim-2, dim, N_hyps, B, H, rhs, stack_partial_sum,
                   abssum, stack_val_min, stack_val_len) == 0)
        return 0;

    while (sp >= 1) {
        int i = stack_i[sp], pos = stack_pos[sp];
        if (i == -1) {
            if (primitive) {
                int32_t g = 0;
                for (int t = 0; t < dim; t++) {
                    int32_t a = vec[t] < 0 ? -vec[t] : vec[t];
                    while (a) { int32_t r = g % a; g = a; a = r; }
                    if (g == 1) break;
                }
                if (g != 1) { sp--; continue; }
            }
            if (out != NULL) {              // materialize into this branch's slice
                long gi = base + cnt;
                if (gi < max_N_out) memcpy(&out[gi*dim], vec, dim * sizeof(int32_t));
            }
            cnt++; sp--; continue;
        }
        if (pos == stack_val_len[sp]) { sp--; continue; }
        int veci = stack_val_min[sp] + pos;
        vec[i] = veci;
        stack_pos[sp] += 1;
        sp += 1;
        (*nodes)++;
        stack_i[sp] = i - 1;
        stack_pos[sp] = 0;
        for (int j = 0; j < N_hyps; ++j) {
            int64_t prev = stack_partial_sum[(sp-1)*N_hyps + j];
            stack_partial_sum[sp*N_hyps + j] = prev + (int64_t)H[j*dim + i] * (int64_t)veci;
        }
        if (i > 0)
            set_bounds(sp, i-1, dim, N_hyps, B, H, rhs, stack_partial_sum,
                       abssum, stack_val_min, stack_val_len);
    }
    return cnt;
}

int _box_enum_c_omp(
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
    // ---- prologue mirrors _box_enum_c ----
    if (dim > OMP_MAX_SUPPORTED_DIM) { *N_out = 0; return -1; }
    if ((long)N_hyps * (dim + 1) * (long)(sizeof(int64_t) + sizeof(int))
            > OMP_MAX_CONSTRAINT_STACK_BYTES) { *N_out = 0; return -4; }

    *N_nodes = 1;  // the root counts even when the search dies before fan-out

    int64_t abssum[N_hyps*(dim+1)];
    for (int j = 0; j < N_hyps; ++j) {
        abssum[j*(dim+1) + 0] = 0;
        for (int k = 0; k < dim; ++k)
            abssum[j*(dim+1) + k+1] = abssum[j*(dim+1) + k] + abs(H[j*dim + k]);
        // all-zero row is the constant constraint 0 >= rhs[j]; unsat if rhs>0
        if (abssum[j*(dim+1) + dim] == 0 && rhs[j] > 0) { *N_out = 0; return 0; }
    }

    // top-level bounds: vec[dim-1] in [min0, min0+len0) indexes the independent subtrees
    int32_t top_vmin[dim+1], top_vlen[dim+1];
    int64_t top_ps[N_hyps*(dim+1)];
    memset(top_ps, 0, sizeof(top_ps));
    if (set_bounds(0, dim-1, dim, N_hyps, B, H, rhs, top_ps, abssum,
                   top_vmin, top_vlen) == 0) { *N_out = 0; return 0; }
    int min0 = top_vmin[0];
    int len0 = top_vlen[0];

    // Materialize needs each branch's count up front to lay out disjoint output
    // slices; count-only just needs the total -- cnt[] is len0 longs (tiny)
    long *cnt = NULL;
    if (out != NULL) {
        cnt = (long *)malloc((size_t)len0 * sizeof(long));
        if (cnt == NULL)   // a few KB-MB; failing is near-impossible, but stay safe
            return _box_enum_c(out, N_out, N_nodes, dim, B, H, rhs, N_hyps,
                               max_N_out, max_N_nodes, primitive);
    }

    // ---- Pass 1: count every branch (the whole job when out == NULL) ----
    long total = 0, nodes = 0;
    #pragma omp parallel reduction(+:total) reduction(+:nodes)
    {
        int32_t t_vec[dim];
        int32_t t_si[dim+1], t_sp[dim+1], t_vmin[dim+1], t_vlen[dim+1];
        int64_t t_ps[N_hyps*(dim+1)];
        // dynamic,1 won a schedule sweep on the Manwe count and materialize; fine
        // chunks matter here -- adjacent branches have correlated cost, so coarse
        // or guided schedules imbalance badly (2-3x slower in the sweep)
        #pragma omp for schedule(dynamic, 1)
        for (int b = 0; b < len0; b++) {
            long nb = 0;
            long c = enum_branch(NULL, 0, 0, min0 + b, dim, N_hyps, B, primitive,
                                 H, rhs, abssum, t_vec, t_si, t_sp,
                                 t_vmin, t_vlen, t_ps, &nb);
            if (cnt != NULL) cnt[b] = c;
            total += c;
            nodes += nb + 1;   // +1 for this branch's top-level node
        }
    }
    *N_nodes += nodes;         // *N_nodes already counts the root; one traversal's worth

    // ---- Pass 2: materialize into disjoint slices (out != NULL only) ----
    if (out != NULL) {
        // exclusive prefix sum: branch b writes to out[cnt[b] .. cnt[b]+count_b)
        long acc = 0;
        for (int b = 0; b < len0; b++) { long c = cnt[b]; cnt[b] = acc; acc += c; }
        #pragma omp parallel
        {
            int32_t t_vec[dim];
            int32_t t_si[dim+1], t_sp[dim+1], t_vmin[dim+1], t_vlen[dim+1];
            int64_t t_ps[N_hyps*(dim+1)];
            #pragma omp for schedule(dynamic, 1)
            for (int b = 0; b < len0; b++) {
                long nb = 0;   // nodes already counted in Pass 1; don't double-count
                enum_branch(out, cnt[b], max_N_out, min0 + b, dim, N_hyps, B,
                            primitive, H, rhs, abssum, t_vec, t_si, t_sp,
                            t_vmin, t_vlen, t_ps, &nb);
            }
        }
        free(cnt);
    }

    // truncation + status match the serial kernel exactly
    *N_out = (total > max_N_out) ? max_N_out : total;
    int status = 0;
    if (total > max_N_out)      status = -2;
    if (*N_nodes > max_N_nodes) status = -3;
    return status;
}

#undef OMP_MAX_SUPPORTED_DIM
#undef OMP_MAX_CONSTRAINT_STACK_BYTES

#else  // !_OPENMP : serial build -> forward everything to the serial kernel

int _box_enum_c_omp(
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
    return _box_enum_c(out, N_out, N_nodes, dim, B, H, rhs, N_hyps,
                       max_N_out, max_N_nodes, primitive);
}

#endif // _OPENMP

#endif // BOX_ENUM_IMPLEMENTATION

#endif // BOX_ENUM_OMP_H
