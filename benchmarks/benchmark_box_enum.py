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
import time

from latticepts import box_enum

# the following imports are only needed for benchmarking
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

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

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


def run_cpsat(B):
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
# Benchmark
# =============================================================================

TIMEOUT = 5.0

def _fmt(elapsed):
    return f"{elapsed:>10.3f}s"

def _skip_fmt(label):
    return f"{label:>11}"


if not HAS_NORMALIZ or not HAS_CPSAT:
    missing = [name for name, flag in [("PyNormaliz", HAS_NORMALIZ), ("ortools", HAS_CPSAT)] if not flag]
    print(f"NI: {', '.join(missing)} not installed")

print(f"{'B':>3}  {'N':>8}  {'fill_frac':>9}  {'expl_frac':>9}  {'effic':>9}  {'box_enum':>11}  {'normaliz':>11}  {'cpsat':>11}")
print("-" * 85)

skip_box      = None
skip_normaliz = "NI" if not HAS_NORMALIZ else None
skip_cpsat    = "NI" if not HAS_CPSAT    else None

plot_N        = []
plot_t_box    = []
plot_t_norm   = []
plot_t_cpsat  = []

for B in range(1, 30+1):
    if all(s is not None for s in (skip_box, skip_normaliz, skip_cpsat)):
        break

    if skip_box is None:
        t0 = time.perf_counter()
        out, status, N_nodes_seen = box_enum(B=B, H=H, rhs=rhs, max_N_out=MAX_N_OUT, max_N_nodes=MAX_N_NODES)
        elapsed = time.perf_counter() - t0
        N = out.shape[0]
        N_nodes_B = ((2*B + 1)**(dim + 1) - 1) // (2*B)
        fill_fraction = N / (2*B + 1)**dim
        exploration_fraction = N_nodes_seen / N_nodes_B
        N_nodes_dense = sum(N**(k/dim) for k in range(dim + 1)) if N > 0 else 0.0
        efficiency = N_nodes_dense / N_nodes_seen if N_nodes_seen > 0 else 0.0
        t_box = _fmt(elapsed)
        n_str  = f"{N:>8}"
        cw_str = f"{fill_fraction:>9.2e}"
        fd_str = f"{exploration_fraction:>9.2e}"
        ef_str = f"{efficiency:>9.2e}"
        plot_N.append(N)
        plot_t_box.append(elapsed)
        if elapsed > TIMEOUT:
            skip_box = "TO"
    else:
        t_box = _skip_fmt(skip_box)
        n_str  = f"{'TO':>8}"
        cw_str = f"{'':>9}"
        fd_str = f"{'':>9}"
        ef_str = f"{'':>9}"

    if skip_normaliz is None:
        t0 = time.perf_counter()
        run_normaliz(B)
        elapsed_norm = time.perf_counter() - t0
        t_normaliz = _fmt(elapsed_norm)
        plot_t_norm.append(elapsed_norm)
        if elapsed_norm > TIMEOUT:
            skip_normaliz = "TO"
    else:
        t_normaliz = _skip_fmt(skip_normaliz)

    if skip_cpsat is None:
        t0 = time.perf_counter()
        run_cpsat(B)
        elapsed_cp = time.perf_counter() - t0
        t_cpsat = _fmt(elapsed_cp)
        plot_t_cpsat.append(elapsed_cp)
        if elapsed_cp > TIMEOUT:
            skip_cpsat = "TO"
    else:
        t_cpsat = _skip_fmt(skip_cpsat)

    print(f"{B:>3}  {n_str}  {cw_str}  {fd_str}  {ef_str}  {t_box}  {t_normaliz}  {t_cpsat}")

# =============================================================================
# Plot
# =============================================================================

if HAS_MPL and plot_N:
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(plot_N, plot_t_box, 'o-', color='steelblue', label='latticepts (box_enum)')
    if plot_t_norm:
        ax.plot(plot_N[:len(plot_t_norm)], plot_t_norm, 's--', color='tomato',   label='Normaliz')
    if plot_t_cpsat:
        ax.plot(plot_N[:len(plot_t_cpsat)], plot_t_cpsat, '^--', color='goldenrod', label='CP-SAT')

    ax.set_xlabel('N')
    ax.set_ylabel('time (s)')
    ax.set_title('Lattice point enumeration: latticepts vs reference solvers\n'
                 'Manwe, 7d example from arXiv:2406.13751')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'docs', 'benchmark_box_enum.png')
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved {out}")
