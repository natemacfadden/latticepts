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
# Two benchmarks:
#   1. h11 scaling: latticepts vs CYTools vs Normaliz vs CP-SAT,
#      h11 = 6..491
#   2. cube dimension scaling: latticepts vs CYTools vs Normaliz vs CP-SAT,
#      dim = 2..14
#
# Saves plots to docs/benchmark_h11.png and docs/benchmark_dim.png.
# CYTools, PyNormaliz, and ortools are optional; comparisons are skipped
# if not available.

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    import PyNormaliz
    HAS_NORMALIZ = True
except ImportError:
    HAS_NORMALIZ = False
    print("PyNormaliz not available; skipping Normaliz comparisons.")

from latticepts import box_enum

from _bench import timed_median

try:
    from cytools import fetch_polytopes
    from cytools.polytope import saturating_lattice_pts  # CYTools routine; implements the Braun/SageMath algorithm
    HAS_CYTOOLS = True
except ImportError:
    HAS_CYTOOLS = False
    print("CYTools not available; skipping CYTools comparisons.")

try:
    from ortools.sat.python import cp_model
    HAS_CPSAT = True
except ImportError:
    HAS_CPSAT = False
    print("ortools not available; skipping CP-SAT comparisons.")

DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')
TIMEOUT  = 60  # seconds; skip remaining calls for a method if exceeded


# =============================================================================
# Shared helpers
# =============================================================================

def _run_normaliz_poly(H, rhs):
    """Enumerate lattice points in the bounded polytope H @ x >= rhs via
    Normaliz."""
    ineqs = [list(map(int, H[i])) + [-int(rhs[i])] for i in range(len(rhs))]
    cone  = PyNormaliz.Cone(inhom_inequalities=ineqs)
    pts   = cone.LatticePoints()
    if not pts:
        return 0
    pts = np.array(pts, dtype=np.int64)
    if pts.shape[1] == H.shape[1] + 1:
        pts = pts[:, :H.shape[1]]
    return len(pts)


def _run_cpsat_poly(H, B, rhs):
    """Enumerate lattice points in the polytope H @ x >= rhs with
    |x_i| <= B via CP-SAT."""
    dim = H.shape[1]
    model = cp_model.CpModel()
    xs = [model.new_int_var(-B, B, f'x{i}') for i in range(dim)]
    for j in range(len(rhs)):
        model.add(sum(int(H[j, i]) * xs[i] for i in range(dim)) >= int(rhs[j]))

    solutions = []

    class _Collector(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            solutions.append([self.value(xs[i]) for i in range(dim)])

    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.solve(model, _Collector())
    return len(solutions)


def _unpack(xs, vals):
    """
    From parallel lists of x-values and (median, lo, hi)-or-None timing tuples,
    drop the None entries and return the x-values, the median array, and
    [lo_err, hi_err] arrays suitable for matplotlib errorbar.
    """
    X, med, lo, hi = [], [], [], []
    for x, v in zip(xs, vals):
        if v is None:
            continue
        X.append(x); med.append(v[0]); lo.append(v[1]); hi.append(v[2])
    med = np.asarray(med)
    yerr = [med - np.asarray(lo), np.asarray(hi) - med]
    return X, med, yerr


# =============================================================================
# Benchmark 1: h11 scaling
# =============================================================================

def run_h11_benchmark():
    h11s = list(range(6, 491+1, 5))

    time_latticepts = []
    time_cytools    = []
    time_normaliz   = []
    time_cpsat      = []

    skip_cytools  = not HAS_CYTOOLS
    skip_normaliz = not HAS_NORMALIZ
    skip_cpsat    = not HAS_CPSAT

    for h11 in h11s:
        print(f"h11={h11}", end='\r')

        if not HAS_CYTOOLS:
            time_latticepts.append(None)
            time_cytools.append(None)
            time_normaliz.append(None)
            time_cpsat.append(None)
            continue

        polys = fetch_polytopes(h11=h11, limit=1)
        if len(polys) == 0:
            time_latticepts.append(None)
            time_cytools.append(None)
            time_normaliz.append(None)
            time_cpsat.append(None)
            continue

        p     = polys[0]
        pts   = p.points(optimal=True)
        ineqs = p._ineqs_optimal

        H   = np.ascontiguousarray(ineqs[:, :-1], dtype=np.int32)
        rhs = (-ineqs[:, -1]).astype(np.int32)
        B   = int(np.max(np.abs(pts)))

        n_lp = len(box_enum(B=B, H=H, rhs=rhs,
                            max_N_out=1_000_000, max_N_nodes=-1)[0])
        time_latticepts.append(
            timed_median(box_enum, B=B, H=H, rhs=rhs,
                         max_N_out=1_000_000, max_N_nodes=-1))

        if not skip_cytools:
            n_ct = len(saturating_lattice_pts(
                pts_in=pts, ineqs=ineqs, dim=4)[0])
            assert n_lp == n_ct
            res = timed_median(saturating_lattice_pts,
                               pts_in=pts, ineqs=ineqs, dim=4)
            time_cytools.append(res)
            if res[0] > TIMEOUT:
                skip_cytools = True
        else:
            time_cytools.append(None)

        if not skip_normaliz:
            n_nm = _run_normaliz_poly(H, rhs)
            assert n_lp == n_nm
            res = timed_median(_run_normaliz_poly, H, rhs)
            time_normaliz.append(res)
            if res[0] > TIMEOUT:
                skip_normaliz = True
        else:
            time_normaliz.append(None)

        if not skip_cpsat:
            n_cp = _run_cpsat_poly(H, B, rhs)
            assert n_lp == n_cp
            res = timed_median(_run_cpsat_poly, H, B, rhs)
            time_cpsat.append(res)
            if res[0] > TIMEOUT:
                skip_cpsat = True
        else:
            time_cpsat.append(None)

        print(f"h11={h11}  N={n_lp}", end='\r')

    print()
    return h11s, time_latticepts, time_cytools, time_normaliz, time_cpsat


def plot_h11(h11s, time_latticepts, time_cytools, time_normaliz, time_cpsat):
    fig, ax = plt.subplots(figsize=(7, 4))

    Xlp, med_lp, ye_lp = _unpack(h11s, time_latticepts)
    ax.errorbar(Xlp, med_lp, yerr=ye_lp, fmt='o', color='steelblue',
                ms=4, capsize=2, linestyle='none', label='latticepts')
    Xnm, med_nm, ye_nm = _unpack(h11s, time_normaliz)
    if Xnm:
        ax.errorbar(Xnm, med_nm, yerr=ye_nm, fmt='s', color='tomato',
                    ms=4, capsize=2, linestyle='none', label="Normaliz")
    Xcp, med_cp, ye_cp = _unpack(h11s, time_cpsat)
    if Xcp:
        ax.errorbar(Xcp, med_cp, yerr=ye_cp, fmt='^', color='goldenrod',
                    ms=4, capsize=2, linestyle='none', label="CP-SAT")
    Xct, med_ct, ye_ct = _unpack(h11s, time_cytools)
    if Xct:
        ax.errorbar(Xct, med_ct, yerr=ye_ct, fmt='D', color='mediumpurple',
                    ms=4, capsize=2, linestyle='none', label="CYTools (Braun alg.)")

    ax.set_xlabel('h11')
    ax.set_ylabel('time to compute lattice points (s)')
    ax.set_yscale('log')
    ax.set_title('Lattice point enumeration: latticepts vs reference solvers\n'
                 '4D reflexive polytopes, h11 = 6..491')
    ax.legend()
    plt.tight_layout()

    out = os.path.join(DOCS_DIR, 'benchmark_h11.png')
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


# =============================================================================
# Benchmark 2: cube dimension scaling
# =============================================================================

def cube_Hrhs(length, dim):
    H   = np.vstack([ np.eye(dim, dtype=np.int32),
                     -np.eye(dim, dtype=np.int32)])
    rhs = np.full(2*dim, -(length // 2), dtype=np.int32)
    B   = length // 2
    return H, rhs, B


def cube_verts(length, dim):
    verts = []
    for i in range(2**dim):
        bits = bin(i)[2:].zfill(dim)
        verts.append([length * int(c) - length // 2 for c in bits])
    return np.array(verts, dtype=np.int32)


def run_dim_benchmark():
    dims   = list(range(2, 14+1, 2))
    length = 2

    time_latticepts = []
    time_cytools    = []
    time_normaliz   = []
    time_cpsat      = []
    num             = []

    skip_cytools  = not HAS_CYTOOLS
    skip_normaliz = not HAS_NORMALIZ
    skip_cpsat    = not HAS_CPSAT

    for dim in dims:
        print(f"dim={dim}", end='\r')

        H, rhs, B = cube_Hrhs(length, dim)

        n_lp = len(box_enum(B=B, H=H, rhs=rhs,
                            max_N_out=10_000_000, max_N_nodes=-1)[0])
        time_latticepts.append(
            timed_median(box_enum, B=B, H=H, rhs=rhs,
                         max_N_out=10_000_000, max_N_nodes=-1))
        num.append(n_lp)

        if not skip_cytools:
            ineqs = np.hstack([H, -rhs.reshape(-1, 1)])
            pts   = cube_verts(length, dim)
            n_ct  = len(saturating_lattice_pts(
                pts_in=pts, ineqs=ineqs, dim=dim)[0])
            assert n_lp == n_ct
            res = timed_median(saturating_lattice_pts,
                               pts_in=pts, ineqs=ineqs, dim=dim)
            time_cytools.append(res)
            if res[0] > TIMEOUT:
                skip_cytools = True
        else:
            time_cytools.append(None)

        if not skip_normaliz:
            n_nm = _run_normaliz_poly(H, rhs)
            assert n_lp == n_nm
            res = timed_median(_run_normaliz_poly, H, rhs)
            time_normaliz.append(res)
            if res[0] > TIMEOUT:
                skip_normaliz = True
        else:
            time_normaliz.append(None)

        if not skip_cpsat:
            n_cp = _run_cpsat_poly(H, B, rhs)
            assert n_lp == n_cp
            res = timed_median(_run_cpsat_poly, H, B, rhs)
            time_cpsat.append(res)
            if res[0] > TIMEOUT:
                skip_cpsat = True
        else:
            time_cpsat.append(None)

        print(f"dim={dim}  N={n_lp}", end='\r')

    print()
    return dims, num, time_latticepts, time_cytools, time_normaliz, time_cpsat


def plot_dim(dims, num, time_latticepts, time_cytools, time_normaliz,
             time_cpsat, length):
    fig, ax = plt.subplots(figsize=(7, 4))

    Xlp, med_lp, ye_lp = _unpack(num, time_latticepts)
    ax.errorbar(Xlp, med_lp, yerr=ye_lp, fmt='o-', color='steelblue',
                capsize=3, label='latticepts', zorder=3)

    Xnm, med_nm, ye_nm = _unpack(num, time_normaliz)
    if Xnm:
        ax.errorbar(Xnm, med_nm, yerr=ye_nm, fmt='s--', color='tomato',
                    capsize=3, label="Normaliz", zorder=3)

    Xcp, med_cp, ye_cp = _unpack(num, time_cpsat)
    if Xcp:
        ax.errorbar(Xcp, med_cp, yerr=ye_cp, fmt='^--', color='goldenrod',
                    capsize=3, label="CP-SAT", zorder=3)

    Xct, med_ct, ye_ct = _unpack(num, time_cytools)
    if Xct:
        ax.errorbar(Xct, med_ct, yerr=ye_ct, fmt='D--', color='mediumpurple',
                    capsize=3, label="CYTools (Braun alg.)", zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('# lattice points')
    ax.set_ylabel('time (s)')
    ax.set_title(f'Lattice point enumeration: latticepts vs reference solvers\n'
                 f'length-{length} hypercube, dim = 2..14')
    ax.legend()
    plt.tight_layout()

    out = os.path.join(DOCS_DIR, 'benchmark_dim.png')
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h11',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Run h11 scaling benchmark')
    parser.add_argument('--dim',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Run cube dimension scaling benchmark')
    args = parser.parse_args()

    if args.h11:
        print("=== h11 benchmark ===")
        h11s, t_lp, t_ct, t_nm, t_cp = run_h11_benchmark()
        plot_h11(h11s, t_lp, t_ct, t_nm, t_cp)

    if args.dim:
        print("=== cube dimension benchmark ===")
        dims, num, t_lp, t_ct, t_nm, t_cp = run_dim_benchmark()
        plot_dim(dims, num, t_lp, t_ct, t_nm, t_cp, length=2)
