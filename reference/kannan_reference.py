# =============================================================================
#    Copyright (C) 2026  Nate MacFadden for the Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Reference njit implementation of Kannan's algorithm. For
#               learning only - read alongside latticepts/box_enum.h.
#
# Note:         numba requires numpy <= 2.3. If your environment has a newer
#               numpy, this file will not import. numba is not a dependency
#               of latticepts itself.
# -----------------------------------------------------------------------------

# external imports
from numba import njit
import numpy as np

from numpy.typing import ArrayLike

# Kannan box method (points in cones)
# -----------------------------------
@njit
def kannan_box_mat_njit(
        B: int,
        linmat: "ArrayLike",
        linmin: int,
        max_N_out: int = 1_000_000,
        max_N_iter: int = 1_000_000_000,
        COORD_BUFF_SIZE: int = 2048) -> "ArrayLike":
    """
    Enumerate all nonzero integer vectors vec satisfying
        -B <= vec[i] <= B  (box constraint)
        linmat @ vec >= linmin  (linear constraint)

    Named "Kannan box" after R. Kannan's lattice enumeration strategy: the
    search space is an L-inf box [-B, B]^dim. At each DFS depth i, the
    feasible range for vec[i] is tightened per linear constraint row j:
      - `stack_partial_sum[j]` accumulates the contribution of already-fixed
        coordinates (indices > i) to row j of linmat @ vec
      - `abssum[j, i]` is the sum of |linmat[j, k]| for k < i, giving the
        maximum possible contribution from the still-unfixed coordinates
        assuming each is bounded by B
    Together these yield a tighter lower bound on vec[i] per constraint,
    pruning branches where no completion can satisfy linmat @ vec >= linmin.

    This is an iterative (DFS) branch-and-bound implementation using an
    explicit stack. Columns of linmat are sorted by L1 norm so that tighter
    constraints are applied first.

    Parameters
    ----------
    B : int
        Box half-width bound.
    linmat : ArrayLike
        Linear constraint matrix, shape (n_constraints, dim).
    linmin : int
        Minimum value for each linear constraint (applied row-wise).
    max_N_out : int, optional
        Maximum number of output vectors.
    max_N_iter : int, optional
        Maximum number of stack iterations before early exit.
    COORD_BUFF_SIZE : int, optional
        Size of the per-depth candidate value buffer.

    Returns
    -------
    out : np.ndarray, shape (N, dim)
        Vectors satisfying all constraints.
    Niter : int
        Number of stack iterations performed.
    """
    dim        = linmat.shape[1]

    # sort the columns of linmat so stricter components come first
    # ------------------------------------------------------------
    col_l1_norm = np.sum(np.abs(linmat), axis=0)
    sort_inds   = np.argsort(col_l1_norm)
    undo_sort   = np.argsort(sort_inds)

    linmat = linmat[:,sort_inds]

    # output object
    # -------------
    out = np.empty((max_N_out, dim), dtype=np.int64)

    # output pointer
    op  = 0

    # internal vector that gets built/written to output
    vec = np.zeros(dim, dtype=np.int64)

    # compute helper variable
    # -----------------------
    abssum = np.empty((linmat.shape[0],linmat.shape[1]+1), np.int64)
    for j in range(abssum.shape[0]):
        abssum[j,0] = 0#abs(linmat[j,0])

        for i in range(dim):
            abssum[j,i+1] = abssum[j,i] + abs(abs(linmat[j,i]))

    # stack variables
    # ---------------
    # stack pointer
    sp = 0

    # max stack depth
    MAX_DEPTH = dim+1

    # stack arrays: i, pos, remaining_Q, nonzero, candidate values
    stack_i      = np.empty(MAX_DEPTH, np.int64)
    stack_pos    = np.empty(MAX_DEPTH, np.int64)

    # vec[i] candidate arrays per depth (preallocate maximum possible size)
    stack_val_len= np.zeros(MAX_DEPTH, np.int64) # number of candidates
    stack_vals   = np.empty((MAX_DEPTH, COORD_BUFF_SIZE), np.int64) # candidates

    # misc helper
    # stack_partial_sum[sp][j] = \sum_{ k > stack_i[sp] } linmat[j][k] * vec[k]
    stack_partial_sum = np.zeros((MAX_DEPTH, linmat.shape[0]), np.int64)

    # initialize stack
    # ----------------
    stack_i[sp]    = dim-1
    stack_pos[sp]  = 0

    _kannan_box_mat_set_coord_candidates(
        sp,
        dim-1,
        B,
        linmat,
        linmin,
        stack_partial_sum,
        abssum,
        stack_vals,
        stack_val_len,
        COORD_BUFF_SIZE
    )

    # process stack until empty
    # -------------------------
    Niter = 0
    while sp >= 0:
        Niter += 1
        if Niter >= max_N_iter:
            break
        # read values
        i    = stack_i[sp]
        pos  = stack_pos[sp]

        # check if node is completed
        # --------------------------
        # if i==-1, then we have fully written vec
        if i == -1:
            if op >= max_N_out:
                break
            out[op, :] = vec
            op += 1

            # kill node
            sp -= 1
            continue

        # check if current depth is completed
        # -----------------------------------
        if pos == stack_val_len[sp]:
            # kill node
            sp -= 1
            continue

        # pick candidate veci
        # -------------------
        veci   = stack_vals[sp, pos]
        vec[i] = veci

        # advance pos for next iteration
        stack_pos[sp] += 1

        # passes cuts -> push next depth :)
        sp += 1
        stack_i[sp]       = i-1
        stack_pos[sp]     = 0

        for j in range(linmat.shape[0]):
            stack_partial_sum[sp,j] = stack_partial_sum[sp-1,j] + linmat[j,i]*vec[i]

        if i >= 1:
            _kannan_box_mat_set_coord_candidates(
                sp,
                i-1,
                B,
                linmat,
                linmin,
                stack_partial_sum,
                abssum,
                stack_vals,
                stack_val_len,
                COORD_BUFF_SIZE
            )

    return out[:op, undo_sort], Niter

@njit
def _kannan_box_mat_set_coord_candidates(
    sp,
    i,
    B,
    linmat,
    linmin,
    stack_partial_sum,
    abssum,
    stack_vals,
    stack_val_len,
    COORD_BUFF_SIZE) -> int:
    """
    ***For use only in kannan_box_mat_njit.***

    Sets the candidate values for vec[i] satisfying -B <= vec[i] <= B and
    the linear constraints linmat @ vec >= linmin, given partial sums for
    indices j > i.

    Parameters
    ----------
    sp : int
        Stack pointer, used to index into stack_vals and stack_val_len.
    i : int
        Current coordinate index being enumerated.
    B : int
        Box half-width bound.
    linmat : ArrayLike
        Linear constraint matrix.
    linmin : int
        Minimum value for the linear constraints.
    stack_partial_sum : ArrayLike
        Partial sums of linmat @ vec for indices j > i.
    abssum : ArrayLike
        Precomputed cumulative absolute column sums of linmat.
    stack_vals : ArrayLike
        Output buffer to store candidate values for vec[i].
    stack_val_len : ArrayLike
        Output buffer to store the count of candidate values.
    COORD_BUFF_SIZE : int
        Maximum number of candidates per depth.

    Returns
    -------
    int
        The number of candidate values written to stack_vals[sp].
    """
    lo = -B
    hi = B

    # For each linear constraint row j, derive a bound on vec[i].
    # We need: linmat[j,i]*vec[i] + partial_j + (remaining contribution) >= linmin
    # In the worst case the remaining contribution (indices k < i) is at most
    # B * abssum[j, i]. So a necessary condition on vec[i] alone is:
    #   linmat[j,i]*vec[i] >= linmin - partial_j - B*abssum[j,i]  =: numer
    # If linmat[j,i] > 0 this gives a lower bound; if < 0, an upper bound.
    for j in range(linmat.shape[0]):
        if linmat[j,i] == 0:
            continue

        numer = linmin - stack_partial_sum[sp,j] - B*abssum[j,i]

        h = linmat[j,i]
        if h>0:
            lo = max(lo, int(np.ceil(numer/h)))
        else:
            hi = min(hi, int(np.floor(numer/h)))

    # compute the number of veci to iterate over
    k = hi - lo + 1
    if k <= 0:
        stack_val_len[sp] = 0
        return 0
    if k > COORD_BUFF_SIZE:
        msg = f"Assumed |hi-lo| <= {COORD_BUFF_SIZE}, but got {k}"
        raise ValueError(msg)

    for t in range(k):
        stack_vals[sp, t] = lo + t
    stack_val_len[sp] = k

    return k
