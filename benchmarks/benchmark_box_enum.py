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

import os
import numpy as np

from latticepts import box_enum

from _bench import timed_median

# the following imports are only needed for benchmarking
try:
    import PyNormaliz
    HAS_NORMALIZ = True
except ImportError as e:
    HAS_NORMALIZ = False
    print(f"PyNormaliz import failed: {e}")

try:
    from ortools.sat.python import cp_model
    HAS_CPSAT = True
except ImportError as e:
    HAS_CPSAT = False
    print(f"ortools import failed: {e}")

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError as e:
    HAS_MPL = False
    print(f"matplotlib import failed: {e}")

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
rhs = 1

MAX_N_OUT   = 100_000_000
MAX_N_NODES = 1_000_000_000_000

# =============================================================================
# Reference implementations
# =============================================================================

def run_normaliz(B):
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


def run_cpsat(B, num_workers=1):
    """
    Encode H @ x >= rhs and |x_i| <= B as a CP-SAT problem and enumerate
    all solutions via a callback

    Box constraints are implicit in the variable bounds [-B, B]
    num_workers sets the CP-SAT search-worker count (1 = single-threaded)
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
    solver.parameters.num_workers = num_workers
    solver.solve(model, _Collector())

    if not solutions:
        return np.empty((0, dim), dtype=np.int64)
    return np.array(solutions, dtype=np.int64)


# =============================================================================
# Benchmark
# =============================================================================

TIMEOUT = 5.0

def _fmt(elapsed):
    return f"{elapsed:>10.3f}"

def _skip_fmt(label):
    return f"{label:>10}"


if __name__ == "__main__":
    if not HAS_NORMALIZ or not HAS_CPSAT:
        missing = [name for name, flag
                   in [("PyNormaliz", HAS_NORMALIZ), ("ortools", HAS_CPSAT)]
                   if not flag]
        print(f"NI: {', '.join(missing)} not installed")

    # Single-threaded comparison: each tool gets one thread, so none is helped
    # or hurt by its own parallelism (thread scaling lives in benchmark_threads.py).
    # latticepts uses parallel=False (serial kernel), CP-SAT num_workers=1, and
    # Normaliz is pinned to 1 thread here -- independent of OMP_NUM_THREADS
    if HAS_NORMALIZ:
        PyNormaliz.NmzSetNumberOfNormalizThreads(1)

    print(f"{'B':>3}  {'N':>9}  {'fill_frac':>9}  {'expl_frac':>9}  {'effic':>9}  "
          f"{'box_enum(s)':>10}  {'normaliz(s)':>10}  {'cpsat(s)':>10}")
    print("-" * 85)

    skip_box      = None
    skip_normaliz = "NI" if not HAS_NORMALIZ else None
    skip_cpsat    = "NI" if not HAS_CPSAT    else None

    # per-curve plot data: N (x), median time, and lo/hi for error bars
    Ns  = {k: [] for k in ("box", "norm", "cpsat")}
    ts  = {k: [] for k in ("box", "norm", "cpsat")}
    los = {k: [] for k in ("box", "norm", "cpsat")}
    his = {k: [] for k in ("box", "norm", "cpsat")}

    for B in range(1, 30+1):
        if all(s is not None for s in (skip_box, skip_normaliz, skip_cpsat)):
            break

        # point count + node stats (same for every tool); count_only avoids a
        # needless multi-GB materialize just to learn N
        N, _, N_nodes_seen = box_enum(B=B, H=H, rhs=rhs, max_N_out=MAX_N_OUT,
                                      max_N_nodes=MAX_N_NODES, count_only=True)
        N_nodes_B = ((2*B + 1)**(dim + 1) - 1) // (2*B)
        fill_fraction = N / (2*B + 1)**dim
        exploration_fraction = N_nodes_seen / N_nodes_B
        N_nodes_dense = (sum(N**(k/dim) for k in range(dim + 1)) if N > 0 else 0.0)
        efficiency = N_nodes_dense / N_nodes_seen if N_nodes_seen > 0 else 0.0

        # latticepts, single-threaded (parallel=False -> the serial kernel)
        if skip_box is None:
            e, lo, hi = timed_median(box_enum, B=B, H=H, rhs=rhs,
                                     max_N_out=MAX_N_OUT, max_N_nodes=MAX_N_NODES,
                                     parallel=False)
            Ns["box"].append(N); ts["box"].append(e)
            los["box"].append(lo); his["box"].append(hi)
            t_box = _fmt(e)
            if e > TIMEOUT:
                skip_box = "TO"
        else:
            t_box = _skip_fmt(skip_box)

        if skip_normaliz is None:
            e, lo, hi = timed_median(run_normaliz, B)
            Ns["norm"].append(N); ts["norm"].append(e)
            los["norm"].append(lo); his["norm"].append(hi)
            t_norm = _fmt(e)
            if e > TIMEOUT:
                skip_normaliz = "TO"
        else:
            t_norm = _skip_fmt(skip_normaliz)

        if skip_cpsat is None:
            e, lo, hi = timed_median(run_cpsat, B)
            Ns["cpsat"].append(N); ts["cpsat"].append(e)
            los["cpsat"].append(lo); his["cpsat"].append(hi)
            t_cpsat = _fmt(e)
            if e > TIMEOUT:
                skip_cpsat = "TO"
        else:
            t_cpsat = _skip_fmt(skip_cpsat)

        print(f"{B:>3}  {N:>9}  {fill_fraction:>9.2e}  {exploration_fraction:>9.2e}  "
              f"{efficiency:>9.2e}  {t_box}  {t_norm}  {t_cpsat}")

    # =============================================================================
    # Plot
    # =============================================================================

    if HAS_MPL and ts["box"]:
        fig, ax = plt.subplots(figsize=(7, 4))

        def _yerr(med, lo, hi):
            med = np.asarray(med, dtype=float)
            return [med - np.asarray(lo, dtype=float),
                    np.asarray(hi, dtype=float) - med]

        def _curve(key, fmt, color, label):
            if ts[key]:
                ax.errorbar(Ns[key], ts[key],
                            yerr=_yerr(ts[key], los[key], his[key]),
                            fmt=fmt, color=color, capsize=3, label=label)

        _curve("box",   'o-',  'steelblue', 'latticepts')
        _curve("norm",  's--', 'tomato',    'Normaliz')
        _curve("cpsat", '^--', 'goldenrod', 'CP-SAT')

        ax.set_xlabel('N')
        ax.set_ylabel('time (s)')
        ax.set_title('Lattice point enumeration: latticepts vs reference solvers\n'
                     '(single-threaded; Manwe 7d example, arXiv:2406.13751)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        plt.tight_layout()
        out = os.path.join(os.path.dirname(__file__), '..', 'docs',
                           'benchmark_box_enum.png')
        plt.savefig(out, dpi=150)
        plt.show()
        print(f"Saved {out}")
