from .algebra import *
from .calculas import *
from .geometry import *
from .linalg import *
from .numTheory import *
from .stats import *
from .trigonometry import *
from .regression import *
from .machine_learning import *
from .complex import *
from .vector import *
from .special_functions import *
from .graph import *
from .solvers import *

__version__ = "2.0.0"

# Define __all__ for explicit export control (good practice)
__all__ = [
    # List functions/classes you guys want to export here, 
    # but using '*' imports above is often simpler for utility libraries. its efficent :)
]
