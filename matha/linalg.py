# Core routines live in the C extension matha/_linalg.c (plain C, no BLAS:
# fast on small matrices, NumPy wins on large ones). 1-D / 2-D float64 only.

import numpy as np
from numpy.linalg import det, inv, solve, norm  # re-exported as matha.det etc.

from ._linalg import (
    matrix_multiply,
    identity,
    matrix_determinant,
    matrix_inverse,
    solve_linear_system,
)


def transpose(A: np.ndarray) -> np.ndarray:
    """
    Transpose of a matrix A (A^T). Returns a view, no copy, so C adds nothing here.

    Args:
        A: The input vector (1D) or matrix (2D).
    """
    if A.ndim not in [1, 2]:
        raise ValueError("Input must be a vector (1D) or a matrix (2D).")
    return A.T
