Enumerates lattice points $\\{x\in\mathbb{Z}^{\text{dim}}: Hx>\text{rhs}\\}$ using [Kannan's algorithm](https://doi.org/10.1287/moor.12.3.415) for $H\in\mathbb{Z}^{N,\text{dim}}$ and $\text{rhs}\in\mathbb{Z}$. See [this webpage](https://cseweb.ucsd.edu/~daniele/Lattice/Enum.html) for some relevant work.

The main use case is for finding lattice points in convex cones, for which $H$ are the inwards-facing hyperplanes. If $\text{rhs}=0$, this will find lattice points in the cone, including its boundary. If $\text{rhs}=1$, then this only finds lattice points in the strict interior of the cone. 
