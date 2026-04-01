# box_enum.pyx
# Cython wrapper for box_enum

# import C types
# --------------
from libc.stdint cimport int32_t
from libc.stdlib cimport malloc, free
import numpy as np

# declare the external C function
# -------------------------------
cdef extern from "box_enum.h":
    int _box_enum_c(
        int32_t * out,
        long * N_out,
        int dim,
        int B,
        int * H,
        int rhs,
        int N_hyps,
        long max_N_out,
        long max_N_iter
    )

# Python-exposed wrapper
# ----------------------
def box_enum(B: int,
                int[:, ::1] H,
                int rhs,
                long max_N_out,
                long max_N_iter = -1) -> tuple[np.ndarray, int]:
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
        ``H[i] @ vec >= rhs``.
    rhs : int
        Minimum value each hyperplane constraint must satisfy. Inclusive.
    max_N_out : long
        Maximum number of output vectors allowed.
    max_N_iter : long, optional
        Maximum number of Kannan iterations. Defaults to ``1000 * max_N_out``.

    Returns
    -------
    out : ndarray of shape (N, dim), dtype int32
        Lattice points satisfying all constraints.
    status : int
        Status code:
            0  : success
           -1  : dim > 256 (unsupported)
           -2  : exceeded max_N_out outputs
           -3  : exceeded max_N_iter iterations
    """
    # read some inputs
    cdef int dim    = H.shape[1]
    cdef int N_hyps = H.shape[0]
    cdef long N_out = 0
    cdef int status

    # allocate output arrays
    cdef int32_t *c_out = <int32_t *>malloc(max_N_out * dim * sizeof(int32_t))
    if c_out == NULL:
        raise MemoryError("Failed to allocate c_out")

    # ensure H is sorted to have strict constraints coming first
    H_np = np.asarray(H)
    col_l1_norm  = np.sum(np.abs(H_np), axis=0)
    sort_inds    = np.argsort(col_l1_norm)
    undo_sort_np = np.argsort(sort_inds).astype(np.int32)
    H_np = H_np[:, sort_inds]
    H_np = np.ascontiguousarray(H_np, dtype=np.int32)

    cdef int[:, ::1] H_view = H_np
    cdef int *H_ptr = &H_view[0, 0]

    cdef int32_t[::1] undo_sort_view = undo_sort_np

    if max_N_iter == -1:
        max_N_iter = 1000*max_N_out

    # call the C function
    status = _box_enum_c(
        c_out,
        &N_out,
        dim,
        B,
        H_ptr,
        rhs,
        N_hyps,
        max_N_out,
        max_N_iter
    );

    if N_out == 0:
        free(c_out)
        return np.empty((0, dim), dtype=np.int32), status

    # unsort
    out = np.empty((N_out, dim), dtype=np.int32)
    cdef int32_t[:, ::1] out_view = out
    cdef int i, j

    for i in range(N_out):
        for j in range(dim):
            out_view[i, j] = c_out[i*dim + undo_sort_view[j]]

    # free C memory
    free(c_out)

    return out, status
