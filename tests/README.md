# Tests

## Running the tests

```
pip install -e ".[test]"
pytest tests/
```

## Benchmark scripts

| Script | Description |
|--------|-------------|
| `benchmark_polytopes.py` | Runtime vs dimension for the length-2 hypercube; runtime vs h11 for 4D reflexive polytopes |
| `benchmark_box_enum.py` | Runtime vs bounding box size B for the Manwe geometry (h11=491, 7D) |
| `benchmark_enum_lattice_points.py` | Runtime vs requested number of lattice points N in the strict interior of the Manwe cone (h11=491, 7D) |
| `benchmark_narrowness.py` | Runtime vs narrowness for a 4D convex cone |

## Optional competitor dependencies

The benchmark scripts compare against several reference implementations.
Install any or all of the following to enable comparisons:

### PyNormaliz
```
pip install PyNormaliz
```
On macOS and Linux, pip pulls a prebuilt wheel that bundles the [Normaliz](https://www.normaliz.uni-osnabrueck.de) C++ library, so no separate install is needed. Elsewhere, install the Normaliz C++ library first.

### OR-Tools (CP-SAT)
```
pip install ortools
```

### SageMath
```
pip install sagemath
```

