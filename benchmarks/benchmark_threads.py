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
# -----------------------------------------------------------------------------
# Description:  latticepts thread-scaling on a bounded cube |x_i| <= B.
#               The cube is the clean test for parallel scale-up: latticepts
#               parallelizes across the top coordinate's values, so the number
#               of top-level branches (2B+1) is a hard ceiling on the speedup.
#               Here B=10 gives 2B+1 = 21 branches, above the core count, so the
#               measured scaling is limited by cores / memory bandwidth rather
#               than by the geometry.
#
#               Each thread count is timed in a fresh subprocess (OMP_NUM_THREADS
#               must be set before the OpenMP runtime initializes); OMP_WAIT_POLICY
#               is passive so idle threads sleep instead of busy-spinning.
# -----------------------------------------------------------------------------

import os
import sys
import json
import subprocess

import numpy as np

THREADS = [1, 2, 4, 6, 8, 12]

DIM = 6
B   = 10   # |x_i| <= 10 in 6D: (21)^6 = 85.8M points, 21 top-level branches (> cores)


def _worker(n):
    """Time count + materialize on the bounded cube at the env-fixed thread count."""
    import time
    from latticepts.box_enum import box_enum
    H = np.zeros((0, DIM), dtype=np.int32)   # pure box |x_i| <= B, no extra constraints

    def best(fn, k=4):
        b = 9e9
        for _ in range(k):
            t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
        return b

    res = {
        "count": best(lambda: box_enum(B=B, H=H, rhs=0, max_N_out=10**18,
                                       count_only=True, parallel=True)),
        "mater": best(lambda: box_enum(B=B, H=H, rhs=0,
                                       max_N_out=(2 * B + 1) ** DIM,
                                       parallel=True)),
    }
    print(json.dumps(res))


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _worker(int(sys.argv[2]))
        sys.exit(0)

    data = {}
    for n in THREADS:
        env = dict(os.environ, OMP_NUM_THREADS=str(n), OMP_WAIT_POLICY="passive")
        out = subprocess.run([sys.executable, os.path.abspath(__file__),
                              "--worker", str(n)],
                             env=env, capture_output=True, text=True)
        try:
            data[n] = json.loads(out.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            sys.exit(f"worker (threads={n}) failed:\n{out.stdout}\n{out.stderr}")
        print(f"threads={n:2d}  "
              + "  ".join(f"{k}={v:.3f}s" for k, v in data[n].items()))

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib not available; timings printed above")

    curves = [("count", "latticepts (count)",       "steelblue", "o-"),
              ("mater", "latticepts (materialize)", "navy",      "o--")]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(THREADS, THREADS, color="0.6", ls=":", label="ideal (linear)")
    for key, label, color, fmt in curves:
        t1 = data[THREADS[0]][key]
        speedup = [t1 / data[n][key] for n in THREADS]
        ax.plot(THREADS, speedup, fmt, color=color, label=label)

    ax.set_xlabel("threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title("latticepts thread scaling on a bounded cube\n"
                 f"$|x_i| \\leq {B}$ in {DIM}D ({(2*B+1)**DIM/1e6:.0f}M points)")
    ax.set_xticks(THREADS)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "docs",
                       "benchmark_threads.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
