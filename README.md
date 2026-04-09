# latticepts
*[Nate MacFadden](https://github.com/natemacfadden), Liam McAllister Group, Cornell*

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.19406651.svg)](https://doi.org/10.5281/zenodo.19406651)

Fast lattice point enumeration for convex polyhedra. A C/Cython implementation of Kannan's algorithm, significantly beating Normaliz and OR-Tools CP-SAT in speed for certain problems. As one performance example: `latticepts` generates ~107M lattice points in the strict interior of an example 7D cone (['Manwe'](https://arxiv.org/abs/2406.13751)) in ~23s. See [the benchmarks](#benchmarks) for benchmarking plots.

## Citation

If you use `latticepts` in your research, please cite it:

```bibtex
@software{latticepts,
  author  = {MacFadden, Nate},
  title   = {latticepts},
  doi     = {10.5281/zenodo.19406651},
  url     = {https://github.com/natemacfadden/latticepts},
  orcid   = {0000-0002-8481-3724},
}
```

## Description

More explicitly, `latticepts` enumerates lattice points

$$ \\{x\in\mathbb{Z}^{\text{dim}}: Hx\geq\text{rhs}\\} $$

for $H\in\mathbb{Z}^{N_\text{hyps}\times\text{dim}}$ and $\text{rhs}\in\mathbb{Z}^{N_\text{hyps}}$. Here each row of $H$ is an inward-facing facet normal and the corresponding entry of $\text{rhs}$ is its offset. Cones correspond to $\text{rhs}=0$; 'stretched cones' (e.g., for finding strict interior points) correspond to $\text{rhs} > 0$; polyhedra to general nonzero $\text{rhs}$.

## Limitations

- Maximum dimension: 256 (returns an error if `dim > 256`). For convex cones, `latticepts` excels at low-dimensions. It can become sluggish in comparison to alternatives at higher-dimensions (well before 256)
- Windows is not supported: the C kernel uses C99 variable-length arrays, which MSVC does not support

## Installation

```
pip install -e .
```

Requires a C compiler and Cython. NumPy must be installed first.

## Algorithm Notes

This repo contains a Cython wrapper of a C implementation of [Kannan's algorithm](https://doi.org/10.1287/moor.12.3.415). See [this webpage](https://cseweb.ucsd.edu/~daniele/Lattice/Enum.html) for some other relevant work (not by me). The specific implementation in this repo is for latttice point enumeration in square boxes $|x_i|\leq B$ for $B\geq 1$. I.e.,

$$ \\{x\in\mathbb{Z}^{\text{dim}}: Hx\geq\text{rhs} \text{ and } |x|_\infty \leq B\\}. $$

It is a [short algorithm](https://github.com/natemacfadden/latticepts/blob/main/latticepts/box_enum.h), only $\leq 350$ lines - I encourage you to read it.

A helper method is provided in case the user wants $N$ points but doesn't care about box size. In this case, boxes of increasing sizes are studied until $\geq N$ lattice points are found.

## Benchmarks

**Convex cones:** runtime vs requested number of interior lattice points (geometry 'Manwe', h11=491, 7D, https://arxiv.org/abs/2406.13751):

<p align="center">
  <img src="https://raw.githubusercontent.com/natemacfadden/latticepts/main/docs/benchmark_box_enum.png" alt="Runtime vs N on the Manwe example: latticepts outperforms PyNormaliz and OR-Tools CP-SAT"/>
</p>

**Polytopes:** runtime vs size of 4D reflexive polytope (measured by h11, one polytope per h11 value, h11 = 6..491):

<p align="center">
  <img src="https://raw.githubusercontent.com/natemacfadden/latticepts/main/docs/benchmark_h11.png" alt="Runtime vs h11 for 4D reflexive polytopes"/>
</p>

**More polytopes:** runtime vs dimension of length-2 hypercubes (dim = 2..14; x-axis is $3^{\text{dim}}$):

<p align="center">
  <img src="https://raw.githubusercontent.com/natemacfadden/latticepts/main/docs/benchmark_dim.png" alt="Runtime vs dimension for the length-2 hypercube"/>
</p>

## Usage

The primary interface is `enum_lattice_points`, which handles box sizing automatically. This was designed for the purpose of enumerating lattice points in convex cones, which are unbounded. See the following example of how ro use this to get lattice points in the strict (since $rhs=1$) interior of a convex cone:

```python
import numpy as np
from latticepts import enum_lattice_points

H   = np.array([[1, 2], [3, -1]], dtype=np.int32)
rhs = 1

# Find at least 1000 lattice points in {x : H @ x >= rhs}
pts = enum_lattice_points(H=H, rhs=rhs, min_N_pts=1000)

# Optionally restrict to primitive vectors (GCD = 1)
pts = enum_lattice_points(H=H, rhs=rhs, min_N_pts=1000, primitive=True)
```

`box_enum` allows direct control over the box size $B$ instead of the number of lattice points. I.e., to enumerate all lattice points in $\\{x: Hx \geq \text{rhs},\\ |x|_\infty \leq B\\}$:

```python
from latticepts import box_enum

pts, status, N_nodes = box_enum(B=5, H=H, rhs=rhs, max_N_out=10_000)
# status: 0 = success, -1 = dim>256, -2 = hit max_N_out, -3 = hit max_N_nodes
# (statuses are also explained in the docstring)
```

`box_enum` is well suited to lattice point enumeration in polytopes (assuming an H-representation is known). For example, if one knows a bounding box of the polytope (if you have a V-representation, this is trivial: $B = \max|v_i|$ over all vertices), then the lattice point enumeration can be done as follows. This example is of the $h^{1,1}=491$ 4D reflexive polytope:

```python
import numpy as np
from latticepts import box_enum

H   = np.array([[ 1,   0,   0,   0],
                [-15,  8,   6,   1],
                [-15,  8,   6,  -1],
                [ -1,  1,  -1,   0],
                [  0, -1,   0,   0]], dtype=np.int32)
rhs = np.array([-1, -1, -1, -1, -1], dtype=np.int32)
# has bounding box B = max(|vertices|) = 42 (basis-dependent)
B   = 42

# one can then get the lattice points via:
pts, status, N_nodes = box_enum(B=B, H=H, rhs=rhs, max_N_out=10_000)
```

## Organization

```
latticepts/
├── latticepts/
│   ├── box_enum.h                       # STB-style library for the Kannan enumeration
|   ├── box_enum.pyx                     # Cython wrapper
|   └── latticepts.py                    # the enum_lattice_points wrapper for box_enum
├── tests/
│   ├── conftest.py                      # shared test helpers (pytest)
│   ├── test_box_enum.py                 # tests of box_enum tests
│   ├── test_manwe.py                    # tests relating to 'Manwe' (arXiv:2406.13751)
│   ├── test_enum_lattice_points.py      # tests of enum_lattice_points
│   │
│   ├── benchmark_box_enum.py            # runtime vs B for the Manwe geometry (h11=491, 7D)
│   ├── benchmark_enum_lattice_points.py # runtime vs requested N for the Manwe cone (h11=491, 7D)
│   ├── benchmark_polytopes.py           # runtime vs h11 for 4D reflexive polytopes; runtime vs dimension for hypercubes
│   ├── benchmark_narrowness.py          # runtime vs narrowness for a 4D convex cone
│   │
|   └── c/                               # simple C-kernel tests (no Python interface)
├── pyproject.toml
└── setup.py
```
