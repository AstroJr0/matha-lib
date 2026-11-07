# Most functions live in the C extension matha/_stats.c. Data is converted to
# float64, so results are always float. mode() needs hashing of arbitrary
# values and corr_matrix() is a thin NumPy call, so both stay in Python.

from collections import Counter
from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike

from ._stats import (
    mean,
    median,
    data_range,
    variance,
    std_dev,
    z_score,
    percentile,
    quartiles,
    iqr,
    covariance,
    correlation,
    binomial_pmf,
    binomial_cdf,
    geometric_pmf,
    geometric_cdf,
    poisson_pmf,
    poisson_cdf,
    normal_pdf,
    normal_cdf,
)

# Type alias for clarity
Data = Union[List[float], np.ndarray]


def mode(data: Data) -> List[float]:
    """
    Mode(s) (most frequent value(s)) of the data; a list, since data can be multimodal.
    """
    if not len(data):
        return []

    freq = Counter(data)
    max_count = max(freq.values())

    return [val for val, count in freq.items() if count == max_count]


def corr_matrix(data: ArrayLike) -> np.ndarray:
    """
    Correlation matrix of a 2D dataset (features as columns, observations as rows).
    """
    return np.corrcoef(data, rowvar=False)
