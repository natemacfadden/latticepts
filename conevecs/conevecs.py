# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
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

# local imports
from . import box_enum

def enum_lattice_points(
    H: ArrayLike,
    rhs: int,
    min_N_pts: int,
    primitive: bool = False,
    max_B: int = 10_000,
    verbosity: int = 0) -> np.ndarray:
    """
    Generate (optionally primitive) lattice points in
        {x in Z^dim : H @ x >= rhs}
    using a branch-and-bound search (Kannan). The core logic is in box_enum.

    Parameters
    ----------
    H : array-like of shape (N_hyps, dim)
        Integer hyperplane matrix.
    rhs : int
        Right-hand side of the inequality H @ x >= rhs.
    min_N_pts : int
        Minimum number of lattice points to return.
    primitive : bool, optional
        If True, only return vectors with GCD(x) = 1. Defaults to False.
    max_B : int, optional
        Maximum box size to search. If reached without finding min_N_pts
        points, returns however many were found. Defaults to 10_000.
    verbosity : int, optional
        The verbosity level. Higher is more verbose. Defaults to 0.

    Returns
    -------
    pts : ndarray of shape (N, dim)
        Lattice points satisfying H @ x >= rhs, where N >= `min_N_pts`
        unless max_B was reached.
    """
    if min_N_pts <= 0:
        raise ValueError(f"min_N_pts must be > 0, got {min_N_pts}.")

    # box_enum has a safety max number of iterations, outputs
    # set both to a large number
    max_N_out  = max(10_000, 10*min_N_pts)
    max_N_iter = max(1_000_000, 1_000_000*min_N_pts)

    H = np.asarray(H, dtype=np.int32)

    # get the lattice points
    Bs_fit   = []
    Npts_fit = []

    i     = -1
    B     = 1
    Nlast = 0
    while True:
        i += 1
        if verbosity >= 1:
            print(f"Attempt #{i}: computing lattice pts in box |x_i| <= {B}...",
                  flush=True)

        # the actual work
        pts, status = box_enum(
            B=B,
            H=H,
            rhs=rhs,
            max_N_out=max_N_out,
            max_N_iter=max_N_iter
        )
        if status == -1:
            raise ValueError(f"dim={H.shape[1]} > 256 (unsupported by box_enum)")
        elif status == -2:
            warnings.warn(f"exceeded max_N_out={max_N_out} outputs")
        elif status == -3:
            warnings.warn(f"exceeded max_N_iter={max_N_iter} iterations")

        # remove points with nontrivial GCDs
        if primitive and (len(pts) > 0):
            gcds = np.gcd.reduce(pts, axis=1)
            pts  = pts[gcds == 1]
        N = len(pts)

        # check if done
        if N >= min_N_pts:
            break
        if B >= max_B:
            if verbosity >= 1:
                print(f"Reached max_B={max_B} with {N} points. Stopping.")
            break
        if verbosity >= 1:
            print(f"Attempt #{i}: found {N} lattice pts. Compare to ", end=" ")
            print(f"previous iteration ({Nlast})...",
                  flush=True)

        # save data for estimating next bounds B to try
        if N > 0 and N > Nlast:
            Nlast = N
            Bs_fit.append(np.log(B))
            Npts_fit.append(np.log(N))

        # guess the B to scale it to using some fitting:
        # log(N) = m log(B) + b
        # log(N1)-log(N0) = m(log(B1)-log(B0))
        # (ensure there are at least 3x data points. Otherwise, fit empirically
        #  untrustworthy)
        if len(Bs_fit) > 2:  # require >=3 points before trusting the fit
            m = (Npts_fit[-1]-Npts_fit[-2])/(Bs_fit[-1]-Bs_fit[-2])
            # Inflate slope by 1.5x to underestimate the next B
            m *= 1.5

            Bguess = (np.log(min_N_pts)-Npts_fit[-1])/m + Bs_fit[-1]
            Bguess = np.exp(Bguess)
            # With few points the log-log fit is noisy, so cap the step
            # at 5% of B to avoid large jumps on unreliable extrapolation
            if N <= 200:
                Bstep  = min(Bguess - B, 0.05*B)
            else:
                Bstep = Bguess - B
            Bstep = int(np.ceil(Bstep))
            if Bstep <= 0:
                B += min(3, int(np.ceil(0.05*B)))
            else:
                B += Bstep
        else:
            # be very conservative with B if we have few points
            B += min(3, int(np.ceil(0.05*B)))

    if len(pts) < min_N_pts:
        warnings.warn(
            f"returning {len(pts)} points, fewer than min_N_pts={min_N_pts}"
        )
    return pts
