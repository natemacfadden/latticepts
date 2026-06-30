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

import itertools
import math

import numpy as np

try:
    import PyNormaliz
except ImportError:
    PyNormaliz = None

try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None

# =============================================================================
# Shared test helpers.
# =============================================================================

def _sort_rows(pts):
    return pts[np.lexsort(pts.T[::-1])]


def _brute_force(H, B, rhs, primitive=False):
    """Dependency-free oracle: all x in [-B,B]^dim with H @ x >= rhs (and
    GCD(|x|)==1 if primitive). O((2B+1)^dim) -- keep B and dim small."""
    H = np.asarray(H, dtype=np.int64)
    dim = H.shape[1]
    rhs_v = (np.asarray(rhs, dtype=np.int64) if np.ndim(rhs)
             else np.full(H.shape[0], rhs, dtype=np.int64))
    out = []
    for x in itertools.product(range(-B, B + 1), repeat=dim):
        xv = np.array(x, dtype=np.int64)
        if np.all(H @ xv >= rhs_v):
            if primitive and math.gcd(*(abs(int(v)) for v in xv)) != 1:
                continue
            out.append(tuple(int(v) for v in xv))
    return sorted(out)


def _run_normaliz(H, B, rhs):
    """
    Encode H @ x >= rhs and |x_i| <= B as inhomogeneous ineqs for PyNormaliz

    Each row [a | -rhs[i]] encodes a @ x >= rhs[i] in the inhomogeneous format

    Box constraints |x_i| <= B are added as pairs x_i <= B, -x_i <= B, encoded
    as [e_i | B] and [-e_i | B].
    """
    dim = H.shape[1]
    rhs_vec = np.broadcast_to(rhs, (H.shape[0],))

    # hyperplane constraints
    ineqs = [list(map(int, row)) + [-int(rhs_vec[i])] for i, row in enumerate(H)]

    # box constraints
    for i in range(dim):
        row_p = [0]*dim + [B]; row_p[i] =  1; ineqs.append(row_p)
        row_m = [0]*dim + [B]; row_m[i] = -1; ineqs.append(row_m)

    # find the points
    cone = PyNormaliz.Cone(inhom_inequalities=ineqs)
    pts_raw = cone.LatticePoints()
    if not pts_raw:
        return np.empty((0, dim), dtype=np.int64)

    # Normaliz appends a homogenizing coordinate; strip it if present
    pts = np.array(pts_raw, dtype=np.int64)
    if pts.shape[1] == dim + 1:
        pts = pts[:, :dim]
    return pts


def _run_cpsat(H, B, rhs):
    """
    Encode H @ x >= rhs and |x_i| <= B as a CP-SAT problem and enumerate
    all solutions via a callback

    Box constraints are implicit in the variable bounds [-B, B]
    """
    dim = H.shape[1]
    rhs_vec = np.broadcast_to(rhs, (H.shape[0],))

    # build model
    model = cp_model.CpModel()
    xs = [model.new_int_var(-B, B, f'x{i}') for i in range(dim)]
    for j, row in enumerate(H):
        model.add(sum(int(row[i]) * xs[i] for i in range(dim)) >= int(rhs_vec[j]))

    # collect all solutions
    solutions = []

    class _Collector(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            solutions.append([self.value(xs[i]) for i in range(dim)])

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.solve(model, _Collector())

    if not solutions:
        return np.empty((0, dim), dtype=np.int64)
    return np.array(solutions, dtype=np.int64)
