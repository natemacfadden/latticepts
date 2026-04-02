import numpy as np
import time
import matplotlib.pyplot as plt
from fractions import Fraction
from conevecs import enum_lattice_points

# Pairwise cone family: for each ordered pair (i,j), i!=j, add row a*x_i - b*x_j >= 0.
# Effective ratio c = a/b; constraint x_i/x_j in [b/a, a/b] = [1/c, c].
# c=1: all components must be equal (narrowest).
# c->inf: no constraint (widest).

dim     = 4
B_start = 25
N       = (2*B_start + 1)**dim  # exact integer, avoids float rounding in B_dense
rhs     = 0

def make_H(a, b):
    """Each row: a*x_i - b*x_j >= 0, effective ratio c = a/b."""
    rows = []
    for i in range(dim):
        for j in range(dim):
            if i != j:
                row = np.zeros(dim, dtype=np.int32)
                row[i] =  a
                row[j] = -b
                rows.append(row)
    return np.array(rows, dtype=np.int32)

# 10 target c values log-spaced from 1+1e-6 to 4, approximated as rationals
c_targets = 1 + np.logspace(-6, np.log10(3), 10)
AB_VALUES = []
for c in c_targets:
    frac = Fraction(c).limit_denominator(1_000_000)
    AB_VALUES.append((frac.numerator, frac.denominator))

C_EFF   = [a / b for a, b in AB_VALUES]
H_empty = np.empty((0, dim), dtype=np.int32)

print(f"dim={dim}, N={N}, rhs={rhs}")
print(f"c values: {[f'log(c-1)={np.log(c-1):.3f} ({a}/{b})' for (a,b), c in zip(AB_VALUES, C_EFF)]}\n")

# time unconstrained baseline
t0 = time.perf_counter()
enum_lattice_points(H=H_empty, rhs=rhs, min_N_pts=N, verbosity=0, max_B=10**12)
t_unconstrained = time.perf_counter() - t0
print(f"no constraints: {t_unconstrained:.3f}s")

# time each (a, b) pair
times = []
for (a, b), c in zip(AB_VALUES, C_EFF):
    t0 = time.perf_counter()
    enum_lattice_points(H=make_H(a, b), rhs=rhs, min_N_pts=N, verbosity=0, max_B=10**12)
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f"log(c-1)={np.log(c-1):.3f} ({a}/{b}): {elapsed:.3f}s")

# plot: x-axis = log(c-1)
x = np.log(np.array(C_EFF) - 1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, times, 'o-', color='steelblue', label='pairwise cone')
ax.axhline(t_unconstrained, color='gray', linestyle='--', label='no constraints')
ax.set_xlabel('$\\log(c - 1)$  (narrower $\\leftarrow$  wider $\\rightarrow$)')
ax.set_ylabel('time (s)')
ax.set_title(f'enum_lattice_points: time vs cone width  (dim={dim}, N={N:,})')
ax.legend()
plt.tight_layout()
plt.savefig('scratch_cone_width.png', dpi=150)
plt.show()
print("saved scratch_cone_width.png")
