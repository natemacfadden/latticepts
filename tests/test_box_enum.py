# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
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
import pytest

from conevecs import box_enum
from conftest import _sort_rows, _run_normaliz, _run_cpsat

# the following imports are only needed for testing
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

# =============================================================================
# Test cases spanning dim=1 to dim=12.
#
# Each case has a semi-arbitrary H matrix and a fixed B chosen to keep runtimes
# modest. Cases ending in '_empty' have zero lattice points for rhs >= 1
# (e.g. H = [[1,0],[-1,0]] forces x >= rhs and -x >= rhs simultaneously).
#
# B is fixed per case (not swept), since the goal is correctness across
# dimensions rather than scaling behaviour (that's test_manwe.py).
# =============================================================================

TEST_CASES = {
    # -- dim 1 ----------------------------------------------------------------
    "dim1": {
        "H": np.array([[1]], dtype=np.int32),
        "B": 8,
    },
    "dim1_empty": {
        # x >= rhs AND -x >= rhs is unsatisfiable for rhs >= 1
        "H": np.array([[1], [-1]], dtype=np.int32),
        "B": 8,
    },
    # -- dim 2 ----------------------------------------------------------------
    "dim2": {
        "H": np.array([[1, 0], [0, 1], [1, 1]], dtype=np.int32),
        "B": 6,
    },
    "dim2_empty": {
        # x >= rhs AND -x >= rhs: empty for rhs >= 1
        "H": np.array([[1, 0], [-1, 0]], dtype=np.int32),
        "B": 6,
    },
    # -- dim 3 ----------------------------------------------------------------
    "dim3": {
        "H": np.array([
            [1,  0,  0],
            [0,  1,  0],
            [0,  0,  1],
            [1,  1, -1],
        ], dtype=np.int32),
        "B": 5,
    },
    "dim3_empty": {
        # x_0 >= rhs AND -x_0 >= rhs: empty for rhs >= 1
        "H": np.array([
            [ 1, 0, 0],
            [-1, 0, 0],
            [ 0, 1, 0],
        ], dtype=np.int32),
        "B": 5,
    },
    # -- dim 4 ----------------------------------------------------------------
    "dim4": {
        "H": np.array([
            [ 1,  1,  0, -1],
            [ 0,  1, -1,  0],
            [-1,  0,  1,  1],
            [ 0, -1,  0,  1],
        ], dtype=np.int32),
        "B": 4,
    },
    # -- dim 5 ----------------------------------------------------------------
    "dim5": {
        "H": np.array([
            [ 1,  0, -1,  0,  1],
            [ 0,  1,  1, -1,  0],
            [-1,  1,  0,  1, -1],
            [ 0,  0,  1,  0, -1],
        ], dtype=np.int32),
        "B": 3,
    },
    # -- dim 6 ----------------------------------------------------------------
    "dim6": {
        "H": np.array([
            [ 1,  0, -1,  0,  1, -1],
            [ 0,  1,  1, -1,  0,  1],
            [-1,  1,  0,  1, -1,  0],
            [ 1, -1,  0,  0,  1,  0],
        ], dtype=np.int32),
        "B": 3,
    },
    # -- dim 7 ----------------------------------------------------------------
    "dim7": {
        "H": np.array([
            [ 1,  0, -1,  0,  1, -1,  0],
            [ 0,  1,  1, -1,  0,  1, -1],
            [-1,  1,  0,  1, -1,  0,  1],
        ], dtype=np.int32),
        "B": 3,
    },
    # -- dim 8 ----------------------------------------------------------------
    "dim8": {
        "H": np.array([
            [ 1,  0, -1,  0,  1, -1,  0,  1],
            [ 0,  1,  1, -1,  0,  1, -1,  0],
            [-1,  1,  0,  1, -1,  0,  1, -1],
        ], dtype=np.int32),
        "B": 3,
    },
    # -- dim 9 ----------------------------------------------------------------
    "dim9": {
        "H": np.array([
            [ 1,  0, -1,  0,  1, -1,  0,  1, -1],
            [ 0,  1,  1, -1,  0,  1, -1,  0,  1],
            [-1,  1,  0,  1, -1,  0,  1, -1,  0],
        ], dtype=np.int32),
        "B": 2,
    },
    # -- dim 10 ---------------------------------------------------------------
    "dim10": {
        "H": np.array([
            [ 1,  0, -1,  0,  1, -1,  0,  1, -1,  0],
            [ 0,  1,  1, -1,  0,  1, -1,  0,  1, -1],
            [-1,  1,  0,  1, -1,  0,  1, -1,  0,  1],
        ], dtype=np.int32),
        "B": 2,
    },
    # -- dim 11 ---------------------------------------------------------------
    "dim11": {
        "H": np.array([
            [ 1,  0, -1,  0,  1, -1,  0,  1, -1,  0,  1],
            [ 0,  1,  1, -1,  0,  1, -1,  0,  1, -1,  0],
            [-1,  1,  0,  1, -1,  0,  1, -1,  0,  1, -1],
        ], dtype=np.int32),
        "B": 2,
    },
    # -- dim 12 ---------------------------------------------------------------
    "dim12": {
        "H": np.array([
            [ 1,  0, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1],
            [ 0,  1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1],
            [-1,  1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0],
            [ 1, -1,  0,  0,  1,  0, -1,  1,  0, -1,  0,  1],
        ], dtype=np.int32),
        "B": 2,
    },
}

RHS_VALUES        = [-1, 0, 1, 2]
COMPARISON_B      = 3    # max B for cross-method validation (reference methods are slow)
COMPARISON_CASES  = [k for k in TEST_CASES if not any(k.startswith(f"dim{d}") for d in range(8, 13))]

# Ground truth counts: {case_name: {rhs: count}}.
# Generated by running box_enum and validated against PyNormaliz and CP-SAT.
EXPECTATIONS = {
    "dim1":       {-1: 10, 0: 9, 1: 8, 2: 7},
    "dim1_empty": {-1:  3, 0: 1, 1: 0, 2: 0},
    "dim2":       {-1: 63, 0: 49, 1: 36, 2: 25},
    "dim2_empty": {-1: 39, 0: 13, 1:  0, 2:  0},
    "dim3":       {-1: 287, 0: 181, 1: 105, 2:  54},
    "dim3_empty": {-1: 231, 0:  66, 1:   0, 2:   0},
    "dim4":       {-1: 527, 0: 151, 1:  15, 2:   0},
    "dim5":       {-1: 2070,  0:  607, 1:  78, 2: 0},
    "dim6":       {-1: 10083, 0: 2399, 1: 238, 2: 0},
    "dim7":       {-1: 118773, 0: 51130, 1: 13883, 2: 1092},
    "dim8":       {-1: 748333, 0: 320427, 1: 86945, 2: 6846},
    "dim9":       {-1: 343150, 0: 119452, 1: 17232, 2:    0},
    "dim10":      {-1: 1562628, 0: 537426, 1:  76908, 2:    0},
    "dim11":      {-1: 7237193, 0: 2475717, 1: 353532, 2:    0},
    "dim12":      {-1: 19195196, 0: 4195682, 1: 292419, 2:    0},
}

MAX_N_OUT  = 10_000_000_000
MAX_N_NODES = 1_000_000_000_000

# =============================================================================
# Tests
# =============================================================================

@pytest.mark.parametrize("name", list(EXPECTATIONS.keys()))
@pytest.mark.parametrize("rhs_val", RHS_VALUES)
def test_counts(name, rhs_val):
    c = TEST_CASES[name]
    out, status, _ = box_enum(B=c["B"], H=c["H"], rhs=rhs_val,
                           max_N_out=MAX_N_OUT, max_N_nodes=MAX_N_NODES)
    assert status == 0
    expected = EXPECTATIONS[name][rhs_val]
    assert out.shape[0] == expected, \
        f"{name}, rhs={rhs_val}: got {out.shape[0]}, expected {expected}"


@pytest.mark.parametrize("name", COMPARISON_CASES)
@pytest.mark.parametrize("rhs_val", RHS_VALUES)
@pytest.mark.skipif(not HAS_NORMALIZ, reason="PyNormaliz not installed")
def test_vs_normaliz(name, rhs_val):
    c = TEST_CASES[name]
    B = min(c["B"], COMPARISON_B)
    out_kan, status, _ = box_enum(B=B, H=c["H"], rhs=rhs_val,
                               max_N_out=MAX_N_OUT, max_N_nodes=MAX_N_NODES)
    assert status == 0
    out_norm = _run_normaliz(c["H"], B, rhs_val)
    assert out_kan.shape[0] == out_norm.shape[0], \
        f"{name}, rhs={rhs_val}, B={B}: box_enum={out_kan.shape[0]}, normaliz={out_norm.shape[0]}"
    np.testing.assert_array_equal(
        _sort_rows(out_kan.astype(np.int64)),
        _sort_rows(out_norm),
    )


@pytest.mark.parametrize("name", COMPARISON_CASES)
@pytest.mark.parametrize("rhs_val", RHS_VALUES)
@pytest.mark.skipif(not HAS_CPSAT, reason="ortools not installed")
def test_vs_cpsat(name, rhs_val):
    c = TEST_CASES[name]
    B = min(c["B"], COMPARISON_B)
    out_kan, status, _ = box_enum(B=B, H=c["H"], rhs=rhs_val,
                               max_N_out=MAX_N_OUT, max_N_nodes=MAX_N_NODES)
    assert status == 0
    out_cp = _run_cpsat(c["H"], B, rhs_val)
    assert out_kan.shape[0] == out_cp.shape[0], \
        f"{name}, rhs={rhs_val}, B={B}: box_enum={out_kan.shape[0]}, cpsat={out_cp.shape[0]}"
    np.testing.assert_array_equal(
        _sort_rows(out_kan.astype(np.int64)),
        _sort_rows(out_cp),
    )
