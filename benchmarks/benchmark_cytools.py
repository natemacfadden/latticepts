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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from latticepts import enum_lattice_points

# the following imports are only needed for benchmarking
try:
    from cytools import Cone
    HAS_CYTOOLS = True
except ImportError:
    HAS_CYTOOLS = False

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

rhs = 1

N_VALUES = [100, 1_000, 10_000, 100_000, 1_000_000]

# =============================================================================
# Benchmark
# =============================================================================

TIMEOUT = 30.0

def _fmt_N(n):
    if n >= 1_000_000: return f"{n // 1_000_000}m"
    if n >= 1_000:     return f"{n // 1_000}k"
    return str(n)

def _fmt_t(elapsed, label="TO"):
    if elapsed is None: return f"{label:>10}"
    return f"{elapsed:>10.3f}s"

def _run_with_timeout(fn, *args, **kwargs):
    """
    Run fn(*args, **kwargs) with a wall-clock timeout.

    Returns (elapsed, result, None) on success,
            (None, None, None)     on timeout,
            (None, None, exc)      on error.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        t0 = time.perf_counter()
        try:
            result  = future.result(timeout=TIMEOUT)
            elapsed = time.perf_counter() - t0
            return elapsed, result, None
        except FuturesTimeoutError:
            return None, None, None
        except Exception as e:
            return None, None, e


def _run_latticepts(N):
    return enum_lattice_points(H=H, rhs=rhs, min_N_pts=N)


def _run_cytools(N):
    c = Cone(hyperplanes=H.astype(int))
    return c.find_lattice_points(min_points=N, c=rhs)


if not HAS_CYTOOLS:
    print("NI: CYTools not installed — skipping comparison")

col_w = 12
header = f"{'N':>5}  {'latticepts':>{col_w}}  {'cytools':>{col_w}}"
print(header)
print("-" * len(header))

skip_latticepts = None
skip_cytools  = None if HAS_CYTOOLS else "NI"

# store timings for optional plot
t_latticepts = []
t_cytools  = []

for N in N_VALUES:
    n_str = _fmt_N(N)

    if skip_latticepts is None:
        elapsed_c, _, err_c = _run_with_timeout(_run_latticepts, N)
        t_latticepts.append(elapsed_c)
        if err_c is not None:
            print(f"latticepts error at N={n_str}: {err_c}")
            skip_latticepts = "ERR"
        elif elapsed_c is None:
            skip_latticepts = "TO"
    else:
        elapsed_c = None
        t_latticepts.append(None)

    if skip_cytools is None:
        elapsed_y, _, err_y = _run_with_timeout(_run_cytools, N)
        t_cytools.append(elapsed_y)
        if err_y is not None:
            print(f"cytools error at N={n_str}: {err_y}")
            skip_cytools = "ERR"
        elif elapsed_y is None:
            skip_cytools = "TO"
    else:
        elapsed_y = None
        t_cytools.append(None)

    print(f"{n_str:>5}  {_fmt_t(elapsed_c, skip_latticepts or 'TO')}  {_fmt_t(elapsed_y, skip_cytools or 'TO')}")

# =============================================================================
# Plot
# =============================================================================

if HAS_MPL and HAS_CYTOOLS:
    fig, ax = plt.subplots()

    xs = np.array(N_VALUES)

    # latticepts
    mask_c = np.array([t is not None for t in t_latticepts])
    if mask_c.any():
        ax.plot(xs[mask_c], np.array(t_latticepts)[mask_c],
                marker='o', label='latticepts')

    # cytools
    mask_y = np.array([t is not None for t in t_cytools])
    if mask_y.any():
        ax.plot(xs[mask_y], np.array(t_cytools)[mask_y],
                marker='s', label='cytools')

    # timeout line
    ax.axhline(TIMEOUT, color='gray', linestyle='--', linewidth=0.8,
               label=f'timeout ({TIMEOUT:.0f}s)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N (lattice points requested)')
    ax.set_ylabel('time (s)')
    ax.set_title("Manwe: latticepts vs CYTools")
    ax.legend()
    plt.tight_layout()
    plt.savefig('benchmark_cytools.png', dpi=150)
    print("\nPlot saved to benchmark_cytools.png")
    plt.show()
