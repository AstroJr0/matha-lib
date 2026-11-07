# cmath, abs, complex and operator are already C builtins, so the plain
# functions are bound to them directly so C is not harmed here.
import cmath
import operator

# type hinting for Complex Number
Complex = complex

# ceartion and properties

create = complex            # create(real, imag) -> real + imag*j
polar = cmath.polar         # (magnitude, phase_in_radians)


def conjugate(z: Complex) -> Complex:
    """Returns the complex conjugate z* (a - bj)."""
    return z.conjugate()


# functions used for basic arithmetic 

add = operator.add
subtract = operator.sub
multiply = operator.mul


def divide(z1: Complex, z2: Complex) -> Complex:
    if z2 == 0:
        raise ZeroDivisionError("Cannot divide by zero complex number.")
    return z1 / z2


# commonly used mathmatical functions 

magnitude = abs             # |z|
phase = cmath.phase         # phase angle in radians
complex_sqrt = cmath.sqrt
complex_log = cmath.log

# alias of complex trig functions

c_sin = cmath.sin
c_cos = cmath.cos
c_tan = cmath.tan


# reciprocals
def c_cosec(z: Complex) -> Complex:
    return 1.0 / cmath.sin(z)


def c_sec(z: Complex) -> Complex:
    return 1.0 / cmath.cos(z)


def c_cot(z: Complex) -> Complex:
    return 1.0 / cmath.tan(z)


# al9as of complex hyperbolic functions
c_sinh = cmath.sinh
c_cosh = cmath.cosh
c_tanh = cmath.tanh

# alias of inverse hyperbolic functions

c_asinh = cmath.asinh
c_acosh = cmath.acosh
c_atanh = cmath.atanh

# alias of inverse trig functions

c_asin = cmath.asin
c_acos = cmath.acos
c_atan = cmath.atan
