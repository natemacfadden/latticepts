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
import pytest

from conevecs import box_enum

# the following imports are only needed for testing
try:
    import PyNormaliz
    HAS_NORMALIZ = True
except ImportError:
    HAS_NORMALIZ = False

try:
    from ortools.sat.python import cp_model
    HAS_CPSAT = True
except ImportError:
    HAS_CPSAT = False

# =============================================================================
# Hard-coded 'Manwe' data (from https://arxiv.org/abs/2406.13751)
# Extracted from CYTools.
# =============================================================================

H = np.array([
    [ 0, -5,  0,  3,  3,  0, -1],
    [ 0,  0,  0,  0, -1, -2, -3],
    [ 0,  0,  0, -1,  0, -3, -5],
    [ 0,  0,  1,  0, -1, -2, -3],
    [ 0,  0, -1, -1,  0, -3, -4],
    [ 0,  0,  0,  0,  0, -1, -2],
    [ 0,  0,  0,  0, -1, -2, -3],
    [ 0,  0,  0, -1, -1, -5, -8],
    [ 0,  0,  1,  0,  0,  0, -1],
    [ 0,  0,  0,  0,  0,  0, -1],
    [ 0,  0,  0, -1,  0,  0, -2],
    [ 0,  1,  0, -1,  1,  0, -1],
    [-1,  0,  0,  0,  2,  0, -1],
    [ 0, -1,  0,  0,  0, -4, -6],
    [ 0,  0, -4, -1,  0,  0, -2],
    [-1,  0, -2,  0,  0,  0, -1],
    [-1,  0,  0,  0,  0, -2, -3],
    [ 0,  0, -1,  0,  0, -2, -2],
    [ 0,  1, -1, -2,  0, -4, -6],
    [ 0, -1,  1,  0,  0,  0, -2],
    [ 0, -3,  0,  1,  3,  0, -1],
    [ 0, -1,  0,  0,  3,  0, -1],
    [ 0, -1,  0,  0,  0, -3, -4],
    [ 0, -1,  0,  0,  0,  0, -2],
    [ 0, -1,  0,  0,  2,  0, -1],
    [ 0,  0,  0, -1, -1,  0, -3],
    [ 0, -1,  1,  0,  0, -2, -4],
    [ 0,  0,  0,  0,  0,  0, -1],
    [ 0, -1, -1,  0,  0, -2, -2],
    [ 0,  0,  0,  0, -1,  0, -1],
    [ 0, -1,  0,  0,  0, -3, -5],
    [-1,  0,  0,  0,  0,  0, -1],
    [ 0,  0,  0,  0,  1,  0, -1],
    [ 0,  0,  1,  0,  0, -2, -4],
    [ 0,  1,  0, -1,  0, -1, -2],
    [ 0,  0,  0,  0, -1,  0, -1],
    [ 0,  0,  0, -1,  0, -4, -7],
    [ 0,  0,  1,  0,  1,  0, -1],
    [ 0, -1,  0,  0,  0, -2, -3],
    [ 0,  0,  0,  0,  0, -1, -1],
    [ 0,  0,  1,  0, -1, -1, -2],
    [ 0,  0,  1,  0,  2,  0, -2],
    [ 0,  0, -1,  0,  0,  0,  0],
    [ 0,  0,  0,  0, -1,  0, -1],
    [-3,  0,  0,  0,  2,  0, -1],
    [ 0, -1,  0,  0,  3,  0, -2],
    [ 5,  0,  0, -2, -2,  0, -1],
    [-1, -1,  0,  1,  1,  0,  0],
    [ 0, -1,  0,  0,  4,  0, -2],
    [ 0,  0, -1,  0,  0, -1, -1],
    [ 0,  0,  1,  0,  1,  0, -2],
    [ 0,  0,  0, -1,  0,  0, -2],
    [ 0, -1,  0,  0,  0,  0, -2],
    [ 0, -1, -3,  0,  0,  0, -2],
    [ 0,  0,  1,  0,  0, -1, -3],
    [ 0,  0,  0, -1,  0, -4, -6],
    [ 0,  0, -1, -2,  0,  0, -4],
    [ 0,  1,  0, -1, -1, -2, -3],
    [ 0,  0, -1, -1,  3,  0, -1],
    [ 0, -1, -1,  0,  2,  0,  0],
    [-1,  0,  0,  0,  0,  0, -1],
    [ 3,  0,  0, -1, -1,  0,  0],
    [ 3,  0,  0, -2,  0,  0, -1],
    [ 0,  0,  0,  0, -1, -1, -1],
    [ 0,  0,  0, -1,  0,  0, -2],
    [ 0,  0, -1, -2,  0, -7,-11],
    [ 0,  0,  1,  0,  0, -1, -2],
    [ 5,  0,  0, -2,  0,  0, -1],
    [ 0,  0,  0,  0,  0, -1, -1],
    [ 0,  0, -1,  0, -1,  0, -1],
    [ 0,  0,  1,  0, -1,  0, -1],
    [-1,  0,  0,  0,  1,  0,  0],
    [ 0,  0,  0, -1,  0,  0, -3],
], dtype=np.int32)

dim = H.shape[1]

MAX_N_OUT  = 10_000_000_000
MAX_N_ITER = 1_000_000_000_000

DILATIONS      = list(range(1, 11))
RHS_VALUES     = [-1, 0, 1, 2]
COMPARISON_B   = 6  # max B for cross-method validation (reference methods are slow)

# Ground truth counts: {rhs: [count for B in DILATIONS]}.
# Populated after cross-method validation passes. Run test_manwe_vs_normaliz
# and test_manwe_vs_cpsat first, then add entries here.
EXPECTATIONS = {
    -1: [153, 2306, 13867, 52080, 152469, 376299, 824687, 1652696, 3090110, 5463157],
     0: [38, 452, 2729, 11400, 36865, 101060, 242831, 529518, 1066292, 2015438],
     1: [0, 0, 15, 284, 2001, 8886, 30235, 85239, 209450, 464518],
     2: [0, 0, 0, 0, 0, 83, 888, 4964, 19262, 59154],
}

# =============================================================================
# Helper functions.
# =============================================================================

def _sort_rows(pts):
    return pts[np.lexsort(pts.T[::-1])]

def _run_normaliz(B, rhs):
    """
    Encode H @ x >= rhs and |x_i| <= B as inhomogeneous ineqs for PyNormaliz

    Each row [a | -rhs] encodes a @ x >= rhs in the inhomogeneous format

    Box constraints |x_i| <= B are added as pairs x_i <= B, -x_i <= B, encoded
    as [e_i | B] and [-e_i | B].
    """
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

def _run_cpsat(B, rhs):
    """
    Encode H @ x >= rhs and |x_i| <= B as a CP-SAT problem and enumerate
    all solutions via a callback

    Box constraints are implicit in the variable bounds [-B, B]
    """
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


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.parametrize("rhs_val", list(EXPECTATIONS.keys()))
def test_manwe_counts(rhs_val):
    for B, expected in zip(DILATIONS, EXPECTATIONS[rhs_val]):
        out, status = box_enum(B=B, H=H, rhs=rhs_val, max_N_out=MAX_N_OUT,
                               max_N_iter=MAX_N_ITER)

        assert status == 0
        assert out.shape[0] == expected, \
            f"rhs={rhs_val}, B={B}: got {out.shape[0]}, expected {expected}"


@pytest.mark.parametrize("rhs_val", RHS_VALUES)
@pytest.mark.skipif(not HAS_NORMALIZ, reason="PyNormaliz not installed")
def test_manwe_vs_normaliz(rhs_val):
    out_kan, status = box_enum(B=COMPARISON_B, H=H, rhs=rhs_val,
                               max_N_out=MAX_N_OUT, max_N_iter=MAX_N_ITER)
    assert status == 0

    out_norm = _run_normaliz(COMPARISON_B, rhs_val)
    assert out_kan.shape[0] == out_norm.shape[0], \
        f"rhs={rhs_val}, B={COMPARISON_B}: box_enum={out_kan.shape[0]}, normaliz={out_norm.shape[0]}"

    np.testing.assert_array_equal(
        _sort_rows(out_kan.astype(np.int64)),
        _sort_rows(out_norm),
    )


@pytest.mark.parametrize("rhs_val", RHS_VALUES)
@pytest.mark.skipif(not HAS_CPSAT, reason="ortools not installed")
def test_manwe_vs_cpsat(rhs_val):
    out_kan, status = box_enum(B=COMPARISON_B, H=H, rhs=rhs_val,
                               max_N_out=MAX_N_OUT, max_N_iter=MAX_N_ITER)
    assert status == 0

    out_cp = _run_cpsat(COMPARISON_B, rhs_val)
    assert out_kan.shape[0] == out_cp.shape[0], \
        f"rhs={rhs_val}, B={COMPARISON_B}: box_enum={out_kan.shape[0]}, cpsat={out_cp.shape[0]}"

    np.testing.assert_array_equal(
        _sort_rows(out_kan.astype(np.int64)),
        _sort_rows(out_cp),
    )


# =============================================================================
# Benchmarks
# =============================================================================

def test_bench_box_enum(benchmark):
    benchmark(box_enum, B=10, H=H, rhs=1,
              max_N_out=MAX_N_OUT, max_N_iter=MAX_N_ITER)
