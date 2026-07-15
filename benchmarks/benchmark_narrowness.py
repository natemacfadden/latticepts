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
# Benchmark enum_lattice_points runtime vs cone narrowness using a pairwise
# cone family.  For each ordered pair (i,j), the constraint a*x_i - b*x_j >= 0
# is added, giving effective ratio c = a/b and x_i/x_j in [1/c, c].
# c -> 1: cone collapses to the diagonal (narrowest).
# c -> inf: no constraint (widest).

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

from latticepts import enum_lattice_points
from _bench import timed_median

def _fmt(elapsed):
    return f"{elapsed:>10.3f}"

# =============================================================================
# Parameters
# =============================================================================

dim     = 4
B_start = 25
N       = (2*B_start + 1)**dim  # exact integer, avoids float rounding in B_dense
rhs     = 0
MAX_B   = 10**12

# =============================================================================
# Cone construction
# =============================================================================

def make_H(a, b):
    """Pairwise constraint matrix: a*x_i - b*x_j >= 0 for all i != j."""
    rows = []
    for i in range(dim):
        for j in range(dim):
            if i != j:
                row = np.zeros(dim, dtype=np.int32)
                row[i] =  a
                row[j] = -b
                rows.append(row)
    return np.array(rows, dtype=np.int32)

# =============================================================================
# Build c values: 30 targets log-spaced in c-1 from 1e-6 to 3
# =============================================================================

c_targets = 1 + np.logspace(-6, np.log10(3), 30)
AB_VALUES = []
for c in c_targets:
    frac = Fraction(c).limit_denominator(1_000_000)
    AB_VALUES.append((frac.numerator, frac.denominator))

C_EFF   = [a / b for a, b in AB_VALUES]
H_empty = np.empty((0, dim), dtype=np.int32)

# =============================================================================
# Run
# =============================================================================


if __name__ == "__main__":
    print(f"dim={dim}, N={N:,}, rhs={rhs}")
    print("# c = a/b = pairwise constraint ratio  (a*x_i - b*x_j >= 0 for all i!=j  =>  x_i/x_j in [1/c, c])")
    print("# log(c-1) spreads out the near-1 regime  (c=1 narrowest: all x_i equal; large c: wide)")
    print(f"{'log(c-1)':>10}  {'(a/b)':>22}  {'median(s)':>10}  {'lo(s)':>10}  {'hi(s)':>10}")
    print("-" * 70)

    med_u, lo_u, hi_u = timed_median(enum_lattice_points, H=H_empty, rhs=rhs,
                                     min_N_pts=N, verbosity=0, max_B=MAX_B)
    print(f"{'--':>10}  {'no constraints':>22}  "
          f"{_fmt(med_u)}  {_fmt(lo_u)}  {_fmt(hi_u)}")

    times   = []
    los     = []
    his     = []
    for (a, b), c in zip(AB_VALUES, C_EFF):
        med, lo, hi = timed_median(enum_lattice_points, H=make_H(a, b), rhs=rhs,
                                   min_N_pts=N, verbosity=0, max_B=MAX_B)
        times.append(med)
        los.append(lo)
        his.append(hi)
        print(f"{np.log(c-1):>10.3f}  {f'({a}/{b})':>22}  "
              f"{_fmt(med)}  {_fmt(lo)}  {_fmt(hi)}")

    # =============================================================================
    # Plot
    # =============================================================================

    x = np.log(np.array(C_EFF) - 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(x, times, yerr=[np.array(times) - np.array(los),
                                 np.array(his) - np.array(times)],
                fmt='o-', color='steelblue', capsize=3, label='pairwise cone')
    ax.axhline(med_u, color='gray', linestyle='--', label='no constraints')
    # fill a faint band for the unconstrained interval
    ax.axhspan(lo_u, hi_u, color='gray', alpha=0.06)
    ax.set_xlabel('$\\log(c - 1)$  (narrower $\\leftarrow$  wider $\\rightarrow$)')
    ax.set_ylabel('time (s)')
    ax.set_title(f'enum_lattice_points: time vs cone narrowness  (dim={dim}, N={N:,})')
    ax.legend()
    plt.tight_layout()
    plt.savefig('benchmark_narrowness.png', dpi=150)
    plt.show()
    print("saved benchmark_narrowness.png")
