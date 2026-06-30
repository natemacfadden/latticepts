# Type stub for the compiled Cython extension latticepts.box_enum.
# Keep the signature in sync with box_enum() in box_enum.pyx.
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Return element 0 is the point array when materializing, or the integer count
# when count_only=True -- hence Any (the shape depends on a runtime flag).
def box_enum(
    B: int,
    H: NDArray[np.int32],
    rhs: int | NDArray[np.int32],
    max_N_out: int,
    max_N_nodes: int = ...,
    count_only: bool = ...,
    primitive: bool = ...,
    parallel: bool = ...,
) -> tuple[Any, int, int]: ...
