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
# Description:  This module contains a wrapper of box_enum that increases the
#               box size B until N lattice points are found. Optionally reduces
#               the points by GCDs.
# -----------------------------------------------------------------------------

# external imports
import numpy as np
import warnings

from numpy.typing import ArrayLike

# local imports (import the function explicitly, not the submodule, so type
# checkers resolve box_enum to the callable rather than the module)
from .box_enum import box_enum

def enum_lattice_points(
    H: ArrayLike,
    rhs: "int | ArrayLike",
    min_N_pts: int,
    primitive: bool = False,
    max_B: int = 10_000,
    min_efficiency: float = 1e-6,
    count_only: bool = False,
    verbosity: int = 0,
    max_N_out: "int | None" = None) -> "np.ndarray | tuple[int, int]":
    """
    Generate (optionally primitive) lattice points in
        {x in Z^dim : H @ x >= rhs}
    using a branch-and-bound search (Kannan). The core logic is in box_enum.

    Parameters
    ----------
    H : array-like of shape (N_hyps, dim)
        Integer hyperplane matrix.
    rhs : int or array-like of int, shape (N_hyps,)
        Right-hand side of the hyperplane constraints. A scalar is broadcast
        to all rows (uniform offset); a vector allows per-constraint bounds,
        enabling enumeration of lattice points in general convex polyhedra.
    min_N_pts : int
        Minimum number of lattice points to return.
    primitive : bool, optional
        If True, only return vectors with GCD(x) = 1. Defaults to False.
    max_B : int, optional
        Maximum box size to search. If reached without finding min_N_pts
        points, returns however many were found. Defaults to 10_000.
    min_efficiency : float, optional
        Minimum permitted efficiency r = N_nodes_dense / N_nodes_seen, where
        N_nodes_dense = sum_{k=0}^{dim} min_N_pts^{k/dim} is the node count
        for N_hyps=0 (no hyperplane constraints). r=1 tolerates only a
        fully dense box; r=0 imposes no limit. Defaults to 1e-6.
    count_only : bool, optional
        If True, do not materialize the points; return a ``(B, N)`` tuple of
        the box half-width B reached and the point count N. Defaults to False.
    verbosity : int, optional
        The verbosity level. >= 1 prints per-iteration diagnostics
        (fill_fraction, exploration_fraction, efficiency). >= 2 also prints
        attempt-level progress messages. Defaults to 0 (silent).
    max_N_out : int or None, optional
        Output buffer size. If None, the buffer is sized exactly to the number
        of points in the final box (done by first just counting the number of
        points, then materializing them at the end). If, in contrast, an int is
        given, that value is used as the buffer for every trial box (faster, but
        dangerous with memory).

    Returns
    -------
    pts : ndarray of shape (N, dim)
        Lattice points satisfying H @ x >= rhs, where N >= `min_N_pts`
        unless max_B was reached. With `count_only=True`, instead returns a
        `(B, N)` tuple of ints: the box half-width B reached and the point
        count N (no points are materialized).
    """
    if min_N_pts <= 0:
        raise ValueError(f"min_N_pts must be > 0, got {min_N_pts}.")

    # whether to begin with a dry run
    dry_running = count_only or (max_N_out is None)
    if dry_running:
        max_N_out = 2**62 # effectively uncapped
    assert max_N_out is not None  # None implies dry_running, capped just above

    H = np.asarray(H, dtype=np.int32)
    dim = H.shape[1]
    N_hyps = H.shape[0]

    if np.ndim(rhs) == 0:
        rhs = np.full(N_hyps, rhs, dtype=np.int32)
    else:
        rhs = np.asarray(rhs, dtype=np.int32)
        if rhs.shape[0] != N_hyps:
            raise ValueError(f"rhs length {rhs.shape[0]} != N_hyps {N_hyps}")

    # Smallest B such that an unconstrained box (N_hyps=0) could contain
    # min_N_pts points: (2B+1)^dim >= min_N_pts => B >= (min_N_pts^{1/dim}-1)/2
    B = max(1, int((min_N_pts**(1.0/dim) - 1) / 2))

    # Node budget: minimum nodes to find min_N_pts points with N_hyps=0
    # (no hyperplane constraints), N_nodes_dense = sum_{k=0}^{dim} min_N_pts^{k/dim}.
    N_nodes_dense = sum(min_N_pts**(k/dim) for k in range(dim + 1))
    if min_efficiency <= 0:
        max_N_nodes = -1  # no limit; box_enum will use its own default
    else:
        max_N_nodes = max(1_000_000, int(np.floor(N_nodes_dense / min_efficiency)))

    # get the lattice points
    _B_INT_MAX = 2**31 - 1  # box_enum takes C int
    Bs_fit   = []
    Npts_fit = []

    i        = -1
    Nlast    = 0
    stop_why = None
    best_pts = np.empty((0, dim), dtype=np.int32)
    while True:
        i += 1
        if verbosity >= 2:
            print(f"Attempt #{i}: computing lattice pts in box |x_i| <= {B}...",
                  flush=True)

        # the actual work (box_enum takes C int; cap B at INT_MAX)
        _res, status, N_nodes_seen = box_enum(
            B=min(B, _B_INT_MAX),
            H=H,
            rhs=rhs,
            max_N_out=max_N_out,
            max_N_nodes=max_N_nodes,
            count_only=dry_running,
            primitive=primitive,
        )

        # N: how many points this box has -- the count directly on a count-only
        # dry run, or the length of the materialized array otherwise.
        N = _res if dry_running else len(_res)
        if verbosity >= 1:
            N_nodes_B = ((2*B + 1)**(dim + 1) - 1) // (2*B)
            fill_fraction = N / (2*B + 1)**dim
            exploration_fraction = N_nodes_seen / N_nodes_B
            efficiency = N_nodes_dense / N_nodes_seen if N_nodes_seen > 0 else 0.0
            print(f"B={B}: N_out={N}, "
                  f"fill_fraction={fill_fraction:.3e}, "
                  f"exploration_fraction={exploration_fraction:.3e}, "
                  f"efficiency={efficiency:.3e}",
                  flush=True)
        if status == -1:
            raise ValueError(f"dim={H.shape[1]} > 256 (unsupported by box_enum)")
        elif status == -4:
            raise ValueError(
                f"N_hyps={H.shape[0]} too large (box_enum's constraint buffers "
                f"would overflow the stack)")
        elif status == -2:
            warnings.warn(f"exceeded max_N_out={max_N_out} outputs")
        elif status == -3:
            warnings.warn(
                f"exceeded max_N_nodes={max_N_nodes} "
                f"(= floor(N_nodes_dense / min_efficiency), N_nodes = N_iterations + 1)"
            )

        # when materializing during the search (caller supplied max_N_out), keep
        # the largest point set found across the B iterations
        if not dry_running and N > len(best_pts):
            best_pts = _res  # == pts here; _res keeps mypy from seeing Optional

        # check if done
        if N >= min_N_pts:
            break
        if B >= _B_INT_MAX:
            stop_why = f"B={B} reached INT_MAX={_B_INT_MAX} (box_enum C int limit)"
            break
        if B >= max_B:
            stop_why = f"B={B} reached max_B={max_B}"
            break
        if verbosity >= 2:
            print(f"Attempt #{i}: found {N} lattice pts. Compare to ", end=" ")
            print(f"previous iteration ({Nlast})...",
                  flush=True)

        # save data for estimating next bounds B to try
        if N > 0 and N > Nlast:
            Nlast = N
            Bs_fit.append(np.log(B))
            Npts_fit.append(np.log(N))

        # these constants are chosen for performance after a light scan
        # no effect on correctness (for reasonable values)
        magicA = 1.5
        magicB = 200
        magicC = 0.05

        # guess the B to scale it to using some fitting:
        # log(N) = m log(B) + b
        # log(N1)-log(N0) = m(log(B1)-log(B0))
        # (ensure there are at least 3x data points. Otherwise, fit empirically
        #  untrustworthy)
        if len(Bs_fit) > 2:  # require >=3 points before trusting the fit
            m = (Npts_fit[-1]-Npts_fit[-2])/(Bs_fit[-1]-Bs_fit[-2])
            # Inflate slope to underestimate the next B
            m *= magicA

            Bguess = (np.log(min_N_pts)-Npts_fit[-1])/m + Bs_fit[-1]
            Bguess = np.exp(min(Bguess, np.log(max_B) if max_B > 0 else 0))
            # With few points the log-log fit is noisy, so cap the step
            # at some % of B to avoid large jumps on unreliable extrapolation
            if N <= magicB:
                Bstep  = min(Bguess - B, magicC*B)
            else:
                Bstep = Bguess - B
            Bstep = int(np.ceil(Bstep))
            if Bstep <= 0:
                B += min(3, int(np.ceil(magicC*B)))
            else:
                B += Bstep
        else:
            # be very conservative with B if we have few points
            B += min(3, int(np.ceil(magicC*B)))

    if count_only:
        # dry run: return the box B reached and the count N
        return B, N

    if dry_running and N > 0:
        # default path: materialize once at the final box B, sized exactly to
        # the N points it contains (no oversized buffer)
        best_pts, _, _ = box_enum(
            B=min(B, _B_INT_MAX),
            H=H,
            rhs=rhs,
            max_N_out=N,
            max_N_nodes=max_N_nodes,
            count_only=False,
            primitive=primitive,
        )

    if len(best_pts) < min_N_pts:
        msg = f"returning {len(best_pts)} points, fewer than min_N_pts={min_N_pts}"
        if stop_why is not None:
            msg += f"; stopped because {stop_why}"
        warnings.warn(msg)
    return best_pts


def min_B_for(H, rhs, min_N_pts, primitive, max_B=10_000, verbosity=0):
    """
    Dry run: determine the smallest box size B such that enum_lattice_points
    generates min_N_pts lattice points.

    Parameters
    ----------
    H, rhs, min_N_pts, primitive, max_B, verbosity
        See :func:`enum_lattice_points` -- identical meaning.

    Returns
    -------
    (B, N) : tuple of int
        The smallest box half-width B that yields at least min_N_pts points,
        and the resulting point count N.
    """
    return enum_lattice_points(H, rhs, min_N_pts, primitive=primitive,
                               max_B=max_B, count_only=True,
                               verbosity=verbosity)
