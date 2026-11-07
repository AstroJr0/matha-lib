import numpy as np
from ._ml import (
    mse as _mse, mae as _mae, r2 as _r2, linear_regression as _linreg,
    gradient_descent as _gd, predict as _predict,
)

def mse(y_true, y_pred) -> float:
    """
    Calculates Mean Squared Error (MSE).
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        
    Returns:
        The MSE value.
    """
    return _mse(y_true, y_pred)

def mae(y_true, y_pred) -> float:
    """
    Calculates Mean Absolute Error (MAE).
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        
    Returns:
        The MAE value.
    """
    return _mae(y_true, y_pred)

def r_squared(y_true, y_pred) -> float:
    """
    Calculates the R-squared (coefficient of determination) score.
    Represents the proportion of variance in the dependent variable explained by the model.
    
    
    Args:
        y_true: Actual target values.
        y_pred: Predicted values.
        
    Returns:
        The R-squared score (best is 1.0).
    """
    return _r2(y_true, y_pred)

def linear_regression_analytical(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculates the optimal weights (coefficients) for Simple Linear Regression 
    using the closed-form Normal Equation.
    
    For a linear model y = mX + b:
    - m = Sum((Xi - mean(X)) * (Yi - mean(Y))) / Sum((Xi - mean(X))^2)
    - b = mean(Y) - m * mean(X)
    
    Args:
        X: The feature array (1D).
        y: The target array (1D).
        
    Returns:
        A tuple (slope, intercept) or (m, b).
        
    Raises:
        ValueError: If X has zero variance.
    """
    return _linreg(X, y)

def gradient_descent(x: np.ndarray, y: np.ndarray, lr: float = 0.01, iterations: int = 1000) -> tuple[float, float]: 
    """
    Finds the optimal slope (m) and intercept (b) for Simple Linear Regression 
    using the Gradient Descent optimization algorithm.
    
    
    Args:
        x: The feature array (1D).
        y: The target array (1D).
        lr: Learning rate (step size).
        iterations: Number of training steps.
        
    Returns:
        A tuple (m, b) representing the optimized slope and intercept.
    """
    return _gd(x, y, lr, iterations)

def predict(X: np.ndarray, m: float, b: float) -> np.ndarray:
    """
    Makes predictions using a simple linear regression model (y = mX + b).
    
    Args:
        X: The input feature array.
        m: The model's slope (coefficient).
        b: The model's intercept.
        
    Returns:
        The predicted values array (y_pred).
    """
    return _predict(X, m, b)
