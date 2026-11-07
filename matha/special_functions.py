import numpy as np
import math
# assume 'integral' is available for the definition form
try:
    from calculas import integral 
except ImportError:
    # if calculas isn't available, we cannot run the integral definition
    integral = None
    
# Rely on Scipy for highly optimized and accurate implementations
try:
    from scipy.special import gamma, beta, erf, erfc
except ImportError:
    # scipy is not available so we rely only on the definitions
    gamma = None
    beta = None
    erf = None
    erfc = None

# gamma, beta, erf, erfc above stay importable as matha.gamma etc. (scipy's own).
# The matha functions below use the C backend, except beta_function.
from ._special import gamma_raw as _gamma, erf as _erf, erfc as _erfc, stirling as _stirling


# gamma and beta funcs

def gamma_function(z: float, use_integral_approx: bool = False, n: int = 10000) -> float:
    """
    Calculates the Gamma function, Γ(z). 
    It is a generalization of the factorial function to complex and non-integer numbers:
    Γ(n) = (n-1)! for positive integers n. 

[Image of Gamma Function curve]

    
    Args:
        z: The argument (must not be a non-positive integer).
        use_integral_approx: If True, uses the numerical integral definition 
                             (less accurate than the internal Scipy implementation).
        n: Number of intervals for integral approximation.
        
    Returns:
        The value of Γ(z).
        
    Raises:
        ValueError: If z is a non-positive integer.
    """
    if z <= 0 and int(z) == z:
        raise ValueError("Gamma function is not defined for non-positive integers.")
    
    if use_integral_approx and integral is not None:
        # Integral definition: Γ(z) = ∫[0, ∞] t^(z-1) * e^(-t) dt
        f = lambda x: x**(z - 1) * np.exp(-x)
        # Using a large but finite upper bound (50) for approximation
        return integral(f, 0, 50, n)
    return _gamma(z)


def beta_function(x: float, y: float) -> float:
    """
    Calculates the Beta function, B(x, y). 
    It is closely related to the Gamma function: B(x, y) = Γ(x)Γ(y) / Γ(x+y).
    It is also known as the Euler integral of the first kind. 
    
    Args:
        x: The first parameter (must be positive).
        y: The second parameter (must be positive).
        
    Returns:
        The value of B(x, y).
        
    Raises:
        ValueError: If x or y is not positive.
    """
    if x <= 0 or y <= 0:
        raise ValueError("Parameters x and y must be positive.")
        
    if beta is not None:
        return beta(x, y)
    else:
        # fallback using Gamma function relation (if Scipy Beta is missing)
        try:
            return gamma_function(x) * gamma_function(y) / gamma_function(x + y)
        except:
            raise ImportError("Scipy is required to calculate the Beta function.")


# error func

def error_function(x: float) -> float:
    """
    Calculates the Error Function, erf(x).
    It is a non-elementary sigmoid function that arises in probability, statistics, 
    and diffusion physics. 
    
    Args:
        x: The argument.
        
    Returns:
        The value of erf(x).
    """
    return _erf(x)


def complementary_error_function(x: float) -> float:
    """
    Calculates the Complementary Error Function, erfc(x), defined as 1 - erf(x).
    It is particularly useful for large x, where erf(x) is near 1.
    
    Args:
        x: The argument.
        
    Returns:
        The value of erfc(x).
    """
    return _erfc(x)


# other special

def sterling_approx_factorial(n: int) -> float:
    """
    Calculates the factorial (n!) using Stirling's approximation. 
    Useful for very large n where direct calculation overflows standard floats.
    Formula: n! ≈ sqrt(2πn) * (n/e)^n 
    
    Args:
        n: The integer whose factorial is to be approximated.
        
    Returns:
        The approximate value of n!.
    """
    return _stirling(n)

