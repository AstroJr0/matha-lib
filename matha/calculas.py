import numpy as np
import math
from math import inf as Infinity

def integral(f, a: float, b: float, n: int = 2000) -> float:
    """
    Numerically approximates the definite integral of f(x) from a to b 
    using the Composite Trapezoidal Rule. 
    
    Handles finite limits and large values for 'infinity' approximation.

    Args:
        f: The function to integrate. Must accept a NumPy array of x values.
        a: The lower limit of integration.
        b: The upper limit of integration.
        n: The number of sub-intervals (trapezoids).
        
    Returns:
        The approximate value of the definite integral.
    """
    # Handle pseudo-infinity by setting large numerical bounds
    if a == -Infinity:
        a = -1e6
    if b == Infinity:
        b = 1e6

    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    
    # Composite Trapezoidal Rule: h/2 * [y0 + 2*(y1 + ... + yn-1) + yn]
    return (h / 2) * (y[0] + 2 * np.sum(y[1:n]) + y[n])


def improper_integral(f, limit: float, direction: str = "right", n: int = 2000) -> float:
    """
    Numerically approximates improper integrals by setting a large finite limit.
    
    Args:
        f: The function to integrate.
        limit: The finite endpoint.
        direction: 'right' (integrate from limit to infinity) or 
                   'left' (integrate from negative infinity to limit).
        n: The number of sub-intervals.
        
    Returns:
        The approximate value of the improper integral.
    """
    if direction == "right":
        return integral(f, limit, Infinity, n)
    elif direction == "left":
        return integral(f, -Infinity, limit, n)
    else:
        raise ValueError("Direction must be 'right' or 'left'.")


def derivative(f, x: float, h: float = 1e-6) -> float:
    """
    Calculates the first derivative of f(x) at point x using the 
    Central Difference Formula (most accurate for first derivative).
    
    Args:
        f: The function to differentiate.
        x: The point at which to evaluate the derivative.
        h: The step size.
        
    Returns:
        The approximate value of f'(x).
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def derivative_left(f, x: float, h: float = 1e-6) -> float:
    """Calculates the first derivative using the Forward Difference Formula."""
    return (f(x) - f(x - h)) / h


def derivative_right(f, x: float, h: float = 1e-6) -> float:
    """Calculates the first derivative using the Backward Difference Formula."""
    return (f(x + h) - f(x)) / h


def second_derivative(f, x: float, h: float = 1e-5) -> float:
    """
    Calculates the second derivative of f(x) at point x using the 
    Central Difference Formula for the second derivative.
    """
    # Formula: (f(x+h) - 2f(x) + f(x-h)) / h^2
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)

def nth_derivative(f, x: float, n: int, h: float = 1e-5) -> float:
    """
    Calculates the n-th derivative of f(x) at point x using a recursive 
    Central Difference approximation.
    
    Args:
        f: The function to differentiate.
        x: The point at which to evaluate.
        n: The order of the derivative (must be >= 1).
        h: The step size.
        
    Returns:
        The approximate value of the n-th derivative f^(n)(x).
    """
    if n < 1:
        raise ValueError("Order of derivative (n) must be 1 or greater.")
    if n == 1:
        return derivative(f, x, h)
    if n == 2:
        return second_derivative(f, x, h)
    
    # Recursive application of the central difference formula
    return (nth_derivative(f, x + h, n - 1, h) - 
            nth_derivative(f, x - h, n - 1, h)) / (2 * h)

# === ===
def limit(f, x0: float, direction: str = "both", h: float = 1e-6) -> float:
    """
    Approximates the limit of f(x) as x approaches x0.
    
    Args:
        f: The function to evaluate the limit for.
        x0: The value being approached.
        direction: 'both', 'left', or 'right'.
        h: The small offset used for approximation.
        
    Returns:
        The approximate limit value.
        
    Raises:
        ValueError: If direction is 'both' and left/right limits do not match.
    """

    # Handle limits at infinity
    if x0 == Infinity:
        return f(1e6)
    if x0 == -Infinity:
        return f(-1e6)

    left = f(x0 - h)
    right = f(x0 + h)

    if direction == "left":
        return left
    if direction == "right":
        return right

    if np.isclose(left, right):
        return (left + right) / 2 # Return the average
    raise ValueError("Left and right limits do not match for 'both' direction.")


def maclaurin(f, n_terms: int = 5, h: float = 1e-5):
    """
    Generates the coefficients for the Maclaurin series expansion of f(x).
    
    The Maclaurin series is a Taylor series expansion around x=0:
    f(x) ≈ Σ [f^(n)(0) / n!] * x^n 
    
    Args:
        f: The function to expand.
        n_terms: The number of terms (coefficients) to generate (a0 to a_{n-1}).
        h: The step size for numerical derivatives.
        
    Returns:
        A list of coefficients: [a0, a1, a2, ...].
    """
    terms = []
    for n in range(n_terms):
        # Calculate the n-th derivative at x=0
        deriv_n = nth_derivative(f, 0, n, h) if n > 0 else f(0)
        
        # Coefficient = f^(n)(0) / n!
        coef = deriv_n / math.factorial(n)
        terms.append(coef)
    return terms


def partial_derivative(f, vars: list, index: int, h: float = 1e-6) -> float:
    """
    Calculates the partial derivative of f with respect to the variable 
    at the specified index.
    
    Args:
        f: The multivariable function (f(v1, v2, ...)).
        vars: A list of the current variable values [v1, v2, ...].
        index: The index of the variable to differentiate by.
        h: The step size.
        
    Returns:
        The approximate value of the partial derivative.
    """
    shifted = vars.copy()
    shifted[index] += h
    forward = f(*shifted)

    shifted[index] -= 2 * h # Return to original, then go backward
    backward = f(*shifted)

    # Central difference approximation
    return (forward - backward) / (2 * h)


def gradient(f, vars: list) -> np.ndarray:
    """
    Calculates the gradient vector (∇f) of a scalar field f.
    
    The gradient is a vector where each component is the partial derivative
    with respect to one variable. 
    """
    return np.array([partial_derivative(f, vars, i) for i in range(len(vars))])


def divergence(F: list, vars: list) -> float:
    """
    Calculates the divergence (∇·F) of a vector field F = [F1, F2, F3].
    
    Divergence = ∂F1/∂x + ∂F2/∂y + ∂F3/∂z.
    
    Args:
        F: A list of functions representing the components of the vector field.
        vars: The list of current variable values [x, y, z, ...].
        
    Returns:
        The scalar divergence value.
    """
    if len(F) != len(vars):
        raise ValueError("Number of field components must match number of variables.")
        
    return np.sum([
        partial_derivative(lambda *v: F[i](*v), vars, i)
        for i in range(len(F))
    ])


def curl(F: list, vars: list) -> np.ndarray:
    """
    Calculates the curl (∇×F) of a 3D vector field F = [Fx, Fy, Fz].
    
    Args:
        F: A list of three functions [Fx, Fy, Fz].
        vars: The list of three current variable values [x, y, z].
        
    Returns:
        The vector curl [curl_x, curl_y, curl_z]. 

[Image of Curl vector field]

    """
    if len(F) != 3 or len(vars) != 3:
        raise ValueError("Curl is defined for 3D vector fields (3 components, 3 variables).")
        
    Fx, Fy, Fz = F

    # Curl components: (dFz/dy - dFy/dz), (dFx/dz - dFz/dx), (dFy/dx - dFx/dy)
    return np.array([
        partial_derivative(lambda *v: Fz(*v), vars, 1) -
        partial_derivative(lambda *v: Fy(*v), vars, 2),

        partial_derivative(lambda *v: Fx(*v), vars, 2) -
        partial_derivative(lambda *v: Fz(*v), vars, 0),

        partial_derivative(lambda *v: Fy(*v), vars, 0) -
        partial_derivative(lambda *v: Fx(*v), vars, 1),
    ])


def arc_length(f, a: float, b: float, n: int = 2000) -> float:
    """
    Calculates the arc length of a function f(x) from x=a to x=b.
    
    Uses the integral formula: L = ∫[a, b] sqrt(1 + (f'(x))^2) dx
    
    Args:
        f: The function whose length is being measured.
        a: The start point on the x-axis.
        b: The end point on the x-axis.
        n: The number of sub-intervals for integration.
        
    Returns:
        The approximate arc length.
    """
    # Create the integrand function g(x) = sqrt(1 + (f'(x))^2)
    def integrand(x):
        # Must handle single value or array inputs for integration
        if isinstance(x, np.ndarray):
            return np.sqrt(1 + derivative(f, x)**2)
        else:
            return math.sqrt(1 + derivative(f, x)**2)
            
    return integral(integrand, a, b, n)
