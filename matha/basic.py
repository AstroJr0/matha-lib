import numpy as np
import math

from calculas import derivative 

def solve_equation(f, x0: float, tol: float = 1e-7, max_iter: int = 100):
    """
    Finds a root of the equation f(x) = 0 using the Newton-Raphson Method.
    Requires a derivative function (assumed to be imported as 'derivative').
    
    Args:
        f: The function for which to find the root.
        x0: The initial guess for the root.
        tol: The convergence tolerance.
        max_iter: The maximum number of iterations.
        
    Returns:
        The approximate root of f(x).
        
    Raises:
        ValueError: If the derivative is zero or maximum iterations are reached.
    """
    x = x0
    for _ in range(max_iter):
        f_x = f(x)
        f_prime_x = derivative(f, x) 
        
        if abs(f_prime_x) < tol: 
            # Check if derivative is effectively zero (near-horizontal tangent)
            raise ValueError("Derivative is too close to zero. The method failed.")
            
        # Newton-Raphson iteration: x_new = x - f(x) / f'(x)
        x_new = x - f_x / f_prime_x
        
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
        
    raise ValueError(f"Maximum iterations ({max_iter}) reached. No solution found within tolerance.")


def solve_equation_bisection(f, a: float, b: float, tol: float = 1e-7, max_iter: int = 100):
    """
    Finds the root of f(x) = 0 within the interval [a, b] using the Bisection Method.
    This method guarantees convergence if a root is bracketed. 
    
    Args:
        f: The function for which to find the root.
        a: The lower bound of the interval.
        b: The upper bound of the interval.
        tol: The convergence tolerance.
        max_iter: The maximum number of interval halving iterations.
        
    Returns:
        The approximate root of f(x).
        
    Raises:
        ValueError: If the interval does not bracket a root (f(a) and f(b) same sign).
    """
    if f(a) * f(b) >= 0:
        raise ValueError("Function must have opposite signs at the interval endpoints [a, b] to bracket a root.")

    for _ in range(max_iter):
        c = (a + b) / 2 # Calculate the midpoint
        f_c = f(c)

        if abs(f_c) < tol:
            return c

        # Determine which sub-interval contains the root
        if f(a) * f_c < 0:
            b = c
        else:
            a = c
            
    # Return the midpoint of the final, small interval
    return (a + b) / 2 

# === ===

def integrate_trapezoid(f, a: float, b: float, n: int = 100):
    """
    Numerically approximates the definite integral of f(x) from a to b
    using the Trapezoidal Rule. 
    
    Args:
        f: The function to integrate (y = f(x)).
        a: The lower limit of integration.
        b: The upper limit of integration.
        n: The number of sub-intervals (trapezoids).
        
    Returns:
        The approximate value of the definite integral.
    """
    if n <= 0:
        raise ValueError("Number of sub-intervals (n) must be positive.")
        
    h = (b - a) / n  # Width of each trapezoid
    
    # Sum the function values at the interior points (f(x1) + f(x2) + ... )
    sum_interior = 0.0
    for i in range(1, n):
        x_i = a + i * h
        sum_interior += f(x_i)
        
    # Formula: h/2 * [f(a) + f(b) + 2 * sum(f(x_i))]
    integral = (h / 2) * (f(a) + f(b) + 2 * sum_interior)
    
    return integral

# === other utils ===

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamps a value to be strictly within a specified range [min_val, max_val].
    
    Args:
        value: The input value.
        min_val: The minimum allowed value.
        max_val: The maximum allowed value.
        
    Returns:
        The clamped value.
    """
    return max(min_val, min(value, max_val))
