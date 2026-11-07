# --------------------------
# @Author : AstroJr0
# @Date : 17-12-25
# --------------------------

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.integrate import solve_ivp
import cmath

def solve_system(coefficients, constants):
    """
    Solves a system of linear equations Ax = b.
    Args:
      param coefficients: 2D array-like (Matrix A)
      param constants: 1D array-like (Vector b)
    Returns: 
      Solution vector x
    """
    try:
        A = np.array(coefficients)
        b = np.array(constants)
        return np.linalg.solve(A, b).tolist()
    except np.linalg.LinAlgError:
        return "Error: Matrix is singular or not square."

  
def find_root(func, initial_guess):
    """
    Finds the root of a non-linear equation func(x) = 0.
    Args:
      The function to solve (e.g., lambda x: x**2 - 4)
      Starting point for the search
    Returns:
      The value of x where the function is zero
    """
    root = fsolve(func, initial_guess)
    return float(root[0])


def solve_quadratic(a, b, c, list=False):
    """
    Solves ax^2 + bx + c = 0 using the quadratic formula.
    Handles real and complex roots.

    Args:
      values of a,b,c from ax^2 + bx + c
    Returns:
      Two Solutions of x 
    """
    # Calculate the discriminant
    d = (b**2) - (4*a*c)
    
    # Find two solutions
    sol1 = (-b - cmath.sqrt(d)) / (2*a)
    sol2 = (-b + cmath.sqrt(d)) / (2*a)

    if list:
      return [sol1, sol2]
    else:
      return (sol1, sol2)


def solve_ode(func, t_span, y0):
    """
    Solves an Ordinary Differential Equation dy/dt = func(t, y).
    :param func: Function defining the derivative (t, y)
    :param t_span: Tuple of (start_time, end_time)
    :param y0: List of initial conditions
    :return: Object containing 't' (time steps) and 'y' (values)
    """
    sol = solve_ivp(func, t_span, y0, t_eval=np.linspace(t_span[0], t_span[1], 100))
    return sol.t, sol.y


def solve_polynomial(coeffs):
    """
    Finds all roots of a polynomial.
    :param coeffs: List of coefficients from highest power to lowest.
           Example: x^3 - 6x^2 + 11x - 6  -> [1, -6, 11, -6]
    :return: Array of real or complex roots
    """
    return np.roots(coeffs).tolist()

def find_minimum(func, initial_guess):
    """
    Finds the x-value that minimizes the given function.
    :param func: The function to minimize
    :param initial_guess: Starting point
    :return: The x-value where the function is at its minimum
    """
    res = minimize(func, initial_guess)
    return float(res.x[0])

def solve_least_squares(x_data, y_data):
    """
    Finds the line of best fit (y = mx + c) for a set of data.
    :return: Slope (m) and Intercept (c)
    """
    A = np.vstack([x_data, np.ones(len(x_data))]).T
    m, c = np.linalg.lstsq(A, y_data, rcond=None)[0]
    return {"slope": m, "intercept": c}
