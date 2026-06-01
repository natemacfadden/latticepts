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

import numpy as np
import time

from latticepts import enum_lattice_points

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

rhs = 1

N_VALUES = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

# =============================================================================
# Benchmark
# =============================================================================

TIMEOUT = 60.0

def _fmt(elapsed):
    return f"{elapsed:>10.3f}s"

def _skip_fmt(label):
    return f"{label:>11}"

def _fmt_N(n):
    if n >= 1_000_000:   return f"{n // 1_000_000}m"
    if n >= 1_000:       return f"{n // 1_000}k"
    return str(n)


print(f"{'N':>5}  {'N_out':>6}  {'enum_lattice_pts':>18}")
print("-" * 35)

skip = None

for N in N_VALUES:
    n_str = _fmt_N(N)

    if skip is not None:
        print(f"{n_str:>5}  {'TO':>6}  {_skip_fmt(skip)}")
        continue

    t0 = time.perf_counter()
    pts = enum_lattice_points(H=H, rhs=rhs, min_N_pts=N)
    elapsed = time.perf_counter() - t0

    print(f"{n_str:>5}  {_fmt_N(len(pts)):>6}  {_fmt(elapsed)}")

    if elapsed > TIMEOUT:
        skip = "TO"
