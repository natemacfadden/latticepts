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


def _run_normaliz(H, B, rhs):
    """
    Encode H @ x >= rhs and |x_i| <= B as inhomogeneous ineqs for PyNormaliz

    Each row [a | -rhs] encodes a @ x >= rhs in the inhomogeneous format

    Box constraints |x_i| <= B are added as pairs x_i <= B, -x_i <= B, encoded
    as [e_i | B] and [-e_i | B].
    """
    dim = H.shape[1]

    # hyperplane constraints
    ineqs = [list(map(int, row)) + [-rhs] for row in H]

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

    # build model
    model = cp_model.CpModel()
    xs = [model.new_int_var(-B, B, f'x{i}') for i in range(dim)]
    for row in H:
        model.add(sum(int(row[i]) * xs[i] for i in range(dim)) >= rhs)

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
