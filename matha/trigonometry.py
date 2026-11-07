# sin/cos/tan, the reciprocals, law_of_cosines and pythagorean_identity live in
# the C extension matha/_trig.c (the in_degrees flag is handled there, so the
# old _angle_handler decorator is gone). The rest are plain math.* functions
# and are bound to them directly. All are scalar-only, as before.

import math
import numpy as np
from typing import Union

from ._trig import (
    sin,
    cos,
    tan,
    cosec,
    sec,
    cot,
    law_of_cosines,
    pythagorean_identity,
)

# simple angle conversation 

def to_radians(degrees: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts an angle from degrees to radians."""
    return degrees * (math.pi / 180.0)

def to_degrees(radians: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Converts an angle from radians to degrees."""
    return radians * (180.0 / math.pi)

# inverse trig alias 

asin = math.asin
acos = math.acos
atan = math.atan
atan2 = math.atan2   # atan2(y, x): 4-quadrant arctan

# aloas for hyperbolic 

sinh = math.sinh
cosh = math.cosh
tanh = math.tanh

# alias for inverse hyperbolic

asinh = math.asinh
acosh = math.acosh   # x >= 1
atanh = math.atanh   # -1 < x < 1

# some aliases 

arcsine = asin
arccosine = acos
arctangent = atan
arctangent2 = atan2

# hyperbolic aliases
hsin = sinh
hcos = cosh
htan = tanh
inverse_hyperbolic_sin = asinh
inverse_hyperbolic_cos = acosh
inverse_hyperbolic_tan = atanh

"""
    In any trigonometric function, if the value is in degrees,
    in your function call, set the keyword argument `in_degrees=True` :)
"""
