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
import time

from conevecs import box_enum

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
rhs = 1

dim = H.shape[1]
max_N_out  = 10_000_000_000
max_N_iter = 1_000_000_000_000


def primitive_filter(pts):
    if len(pts) == 0:
        return pts
    gcds = np.gcd.reduce(np.abs(pts), axis=1)
    nonzero = np.any(pts != 0, axis=1)
    return pts[(gcds == 1) & nonzero]


def run_normaliz(B):
    ineqs = [list(map(int, row)) + [-rhs] for row in H]
    for i in range(dim):
        row_p = [0]*dim + [B]; row_p[i] =  1; ineqs.append(row_p)
        row_m = [0]*dim + [B]; row_m[i] = -1; ineqs.append(row_m)
    cone = PyNormaliz.Cone(inhom_inequalities=ineqs)
    pts_raw = cone.LatticePoints()
    if not pts_raw:
        return np.empty((0, dim), dtype=np.int64)
    pts = np.array(pts_raw, dtype=np.int64)
    if pts.shape[1] == dim + 1:
        pts = pts[:, :dim]
    return primitive_filter(pts)


def run_cpsat(B):
    model = cp_model.CpModel()
    xs = [model.new_int_var(-B, B, f'x{i}') for i in range(dim)]
    for row in H:
        model.add(sum(int(row[i]) * xs[i] for i in range(dim)) >= rhs)
    solutions = []

    class _Collector(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            solutions.append([self.value(xs[i]) for i in range(dim)])

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.solve(model, _Collector())
    if not solutions:
        return np.empty((0, dim), dtype=np.int64)
    return primitive_filter(np.array(solutions, dtype=np.int64))


TIMEOUT = 5.0

if not HAS_NORMALIZ or not HAS_CPSAT:
    missing = [name for name, flag in [("PyNormaliz", HAS_NORMALIZ), ("ortools", HAS_CPSAT)] if not flag]
    print(f"NI: {', '.join(missing)} not installed")

print(f"{'B':>3}  {'N':>8}  {'box_enum':>11}  {'normaliz':>11}  {'cpsat':>11}")
print("-" * 52)

skip_box      = None
skip_normaliz = "NI" if not HAS_NORMALIZ else None
skip_cpsat    = "NI" if not HAS_CPSAT    else None


def _fmt(elapsed):
    return f"{elapsed:>10.3f}s"

def _skip_fmt(label):
    return f"{label:>11}"


for B in range(1, 30+1):
    if all(s is not None for s in (skip_box, skip_normaliz, skip_cpsat)):
        break

    if skip_box is None:
        t0 = time.perf_counter()
        out, status = box_enum(B=B, H=H, rhs=rhs, max_N_out=max_N_out, max_N_iter=max_N_iter)
        elapsed = time.perf_counter() - t0
        t_box = _fmt(elapsed)
        n_str = f"{out.shape[0]:>8}"
        if elapsed > TIMEOUT:
            skip_box = "TO"
    else:
        t_box = _skip_fmt(skip_box)
        n_str = f"{'TO':>8}"

    if skip_normaliz is None:
        t0 = time.perf_counter()
        run_normaliz(B)
        elapsed = time.perf_counter() - t0
        t_normaliz = _fmt(elapsed)
        if elapsed > TIMEOUT:
            skip_normaliz = "TO"
    else:
        t_normaliz = _skip_fmt(skip_normaliz)

    if skip_cpsat is None:
        t0 = time.perf_counter()
        run_cpsat(B)
        elapsed = time.perf_counter() - t0
        t_cpsat = _fmt(elapsed)
        if elapsed > TIMEOUT:
            skip_cpsat = "TO"
    else:
        t_cpsat = _skip_fmt(skip_cpsat)

    print(f"{B:>3}  {n_str}  {t_box}  {t_normaliz}  {t_cpsat}")
