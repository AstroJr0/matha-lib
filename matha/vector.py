# All functions live in the C extension matha/_vector.c.
# Inputs are converted to float64 arrays, so int input gives float64 output.
# dot_product / angle_between need 1-D vectors.

import numpy as np

from ._vector import (
    add,
    subtract,
    scalar_multiply,
    dot_product,
    cross_product,
    magnitude,
    normalize,
    angle_between,
)

# Type hint for vectors
Vector = np.ndarray
