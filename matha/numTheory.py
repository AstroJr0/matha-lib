# Everything here lives in the C extension matha/_numtheory.c, except
# fibonacci(): its result grows past 64 bits, so Python big ints stay.
#
# C side: arguments must fit in signed 64-bit, else OverflowError.

from ._numtheory import (
    is_prime,
    prime_factors,
    gcd,
    extended_gcd,
    lcm,
    mod_pow,
    mod_inverse,
    totient,
    is_perfect,
    nth_prime,
)


def _fib_fast_doubling(n: int) -> tuple:
    """Fast doubling: returns (F(n), F(n+1)) in O(log n)."""
    if n == 0:
        return (0, 1)

    a, b = _fib_fast_doubling(n // 2)  # F(k), F(k+1) with k = n // 2
    c = a * (2 * b - a)  # F(2k)
    d = a * a + b * b    # F(2k+1)

    if n % 2 == 0:
        return (c, d)
    return (d, c + d)


def fibonacci(n: int) -> int:
    """
    n-th Fibonacci number (F(0)=0, F(1)=1) via fast doubling, O(log n).
    Arbitrary precision, so it stays in Python.
    """
    if n < 0:
        raise ValueError("Fibonacci sequence is typically defined for non-negative indices.")
    return _fib_fast_doubling(n)[0]
  
