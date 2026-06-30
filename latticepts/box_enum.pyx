# box_enum.pyx
# Cython wrapper for box_enum

# import C types
# --------------
from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, free
import numpy as np

# declare the external C functions
# --------------------------------
# box_enum_omp.h wraps box_enum.h and exposes both entry points: _box_enum_c is
# the serial kernel; _box_enum_c_omp runs counting and materialization in
# parallel when built with -fopenmp (LATTICEPTS_OPENMP=1), else forwards to the
# serial kernel. The `parallel` kwarg below picks between them at call time, so a
# single OpenMP build can run either back to back (e.g. for benchmarking)
cdef extern from "box_enum_omp.h":
    int _box_enum_c_omp(
        int32_t * out,
        long * N_out,
        long * N_nodes,
        int dim,
        int B,
        int * H,
        int * rhs,
        int N_hyps,
        long max_N_out,
        long max_N_nodes,
        int primitive
    )

cdef extern from "box_enum.h":
    int _box_enum_c(
        int32_t * out,
        long * N_out,
        long * N_nodes,
        int dim,
        int B,
        int * H,
        int * rhs,
        int N_hyps,
        long max_N_out,
        long max_N_nodes,
        int primitive
    )

# Python-exposed wrapper
# ----------------------
def box_enum(B: int,
                int[:, ::1] H,
                rhs,
                long max_N_out,
                long max_N_nodes = -1,
                bint count_only = False,
                bint primitive = False,
                bint parallel = True) -> tuple:
    """
    Enumerate lattice points ``vec`` obeying ``H @ vec >= rhs`` and
    ``|vec_i| <= B`` using Kannan's algorithm.

    Columns of ``H`` are internally sorted by L1 norm (strictest constraints
    first) before calling the C kernel, then the output is un-sorted before
    returning.

    Parameters
    ----------
    B : int
        Box half-width: each component satisfies ``|vec_i| <= B``.
    H : int[:, ::1] of shape (N_hyps, dim)
        Hyperplane constraint matrix. Each row defines one inequality
        ``H[i] @ vec >= rhs[i]``.
    rhs : int or array-like of int, shape (N_hyps,)
        Right-hand side of the hyperplane constraints. A scalar is broadcast
        to all constraints (i.e. ``H @ vec >= rhs`` uniformly).
    max_N_out : long
        Maximum number of output vectors to materialize (the output-buffer
        size). Bounds materialization only; with ``count_only=True`` it does
        not cap the returned count.
    max_N_nodes : long, optional
        Maximum number of search tree nodes to visit (where N_nodes =
        N_iterations + 1, counting the root). Defaults to
        ``((2*B+1)**(dim+1) - 1) // (2*B)``, the node count for N_hyps=0
        (no hyperplane constraints), i.e. the maximum possible for this B.
    count_only : bool, optional
        If True, tally the feasible points without materializing them (no
        output buffer is allocated) and return the true count. ``max_N_out``
        is ignored in this mode; bound the search with ``max_N_nodes``
        (status -3) if needed.
    parallel : bool, optional
        If True (default), use the OpenMP path, which parallelizes counting and
        materialization over all available threads (cap with ``OMP_NUM_THREADS``)
        when built with ``LATTICEPTS_OPENMP=1``; in a serial build it is
        equivalent to the serial kernel. If False, always use the single-threaded
        serial kernel, e.g. as a single-thread baseline for benchmarking.

    Returns
    -------
    out : ndarray of shape (N, dim), dtype int32
        Lattice points satisfying all constraints.
    status : int
        Status code:
            0  : success
           -1  : dim > 256 (unsupported)
           -2  : exceeded max_N_out outputs
           -3  : exceeded max_N_nodes
           -4  : N_hyps too large (constraint buffers would overflow the stack)
    N_nodes : int
        Number of search tree nodes visited (including the root), where
        N_nodes = N_iterations + 1.

    Notes
    -----
    Under truncation (status ``-2``, when the feasible set exceeds
    ``max_N_out``) only the returned count and status are guaranteed equal
    across the serial and parallel paths; the specific points may differ. The
    serial kernel emits the first ``max_N_out`` points in DFS order, the OpenMP
    path in per-branch order. Size the buffer to the full count (e.g. a
    ``count_only`` pass, then fill) to get the complete, path-independent set.

    The parallel path also does not early-exit: it enumerates every branch
    fully and applies the ``max_N_out`` / ``max_N_nodes`` caps afterward, while
    the serial path bails as soon as a cap is hit. So ``N_nodes`` reflects the
    full search under ``parallel=True`` and the truncated search under
    ``parallel=False`` (they differ), and a tight ``max_N_nodes`` does not abort
    early in parallel. The count and status code still agree.
    """
    # read some inputs
    cdef int dim    = H.shape[1]
    cdef int N_hyps = H.shape[0]
    cdef long N_out = 0
    cdef long N_nodes = 0
    cdef int status

    # normalize rhs: scalar broadcasts to all constraints
    if np.ndim(rhs) == 0:
        rhs_np = np.full(N_hyps, rhs, dtype=np.int32)
    else:
        rhs_np = np.asarray(rhs, dtype=np.int32)
        if rhs_np.shape[0] != N_hyps:
            raise ValueError(
                f"rhs length {rhs_np.shape[0]} != N_hyps {N_hyps}")

    cdef int[::1] rhs_view
    cdef int *rhs_ptr
    if N_hyps > 0:
        rhs_view = rhs_np
        rhs_ptr = &rhs_view[0]
    else:
        rhs_ptr = NULL

    # allocate output arrays (count_only -> no buffer; kernel just tallies)
    cdef int32_t *c_out = NULL
    cdef size_t n_elem, n_bytes
    if not count_only:
        # guard the allocation: max_N_out * dim * sizeof(int32_t) is computed in
        # size_t and can wrap to a near-zero malloc, after which the kernel writes
        # out of bounds (a wrapped-to-0 size is non-NULL on glibc, so the check
        # below would not catch it)
        if max_N_out <= 0:
            raise ValueError("max_N_out must be positive")
        n_elem = <size_t>max_N_out * <size_t>dim
        if dim > 0 and n_elem // <size_t>dim != <size_t>max_N_out:
            raise MemoryError("max_N_out * dim overflows size_t")
        n_bytes = n_elem * sizeof(int32_t)
        if n_bytes // sizeof(int32_t) != n_elem:
            raise MemoryError("output buffer size overflows size_t")
        c_out = <int32_t *>malloc(n_bytes)
        if c_out == NULL:
            raise MemoryError("Failed to allocate c_out")

    # ensure H is sorted to have strict constraints coming first
    H_np = np.asarray(H)
    col_l1_norm  = np.sum(np.abs(H_np), axis=0)
    sort_inds    = np.argsort(col_l1_norm)
    undo_sort_np = np.argsort(sort_inds).astype(np.int32)
    H_np = H_np[:, sort_inds]
    H_np = np.ascontiguousarray(H_np, dtype=np.int32)

    cdef int[:, ::1] H_view
    cdef int *H_ptr
    if N_hyps > 0:
        H_view = H_np
        H_ptr = &H_view[0, 0]
    else:
        H_ptr = NULL

    cdef int32_t[::1] undo_sort_view = undo_sort_np

    if max_N_nodes == -1:
        trivial = ((2*B + 1)**(dim + 1) - 1) // (2*B)
        max_N_nodes = min(trivial, 9_200_000_000_000_000_000)  # cap at ~LONG_MAX

    # call the C kernel: parallel path by default; parallel=False forces serial
    if parallel:
        status = _box_enum_c_omp(
            c_out, &N_out, &N_nodes, dim, B, H_ptr, rhs_ptr, N_hyps,
            max_N_out, max_N_nodes, 1 if primitive else 0)
    else:
        status = _box_enum_c(
            c_out, &N_out, &N_nodes, dim, B, H_ptr, rhs_ptr, N_hyps,
            max_N_out, max_N_nodes, 1 if primitive else 0)

    # count-only dry run: no buffer to unpack, just return the tally
    if count_only:
        return int(N_out), status, N_nodes

    if N_out == 0:
        free(c_out)
        return np.empty((0, dim), dtype=np.int32), status, N_nodes

    # unsort
    out = np.empty((N_out, dim), dtype=np.int32)
    cdef int32_t[:, ::1] out_view = out
    cdef long i
    cdef int j

    for i in range(N_out):
        for j in range(dim):
            out_view[i, j] = c_out[i*dim + undo_sort_view[j]]

    # free C memory
    free(c_out)

    return out, status, N_nodes
