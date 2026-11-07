import inspect
import math
import pathlib
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import matha
from matha import (algebra, calculas, complex as mcomplex, constants, geometry,
                   graph, linalg, machine_learning as ml, numTheory, regression,
                   solvers, special_functions as sf, stats, trigonometry as trig,
                   vector)

A = pytest.approx


# basics
def test_sanity():
    assert matha.__version__


def test_complex_exported():
    z = matha.create(3, 4)
    assert (z.real, z.imag) == (3, 4)


# numTheory
def test_is_prime():
    assert [n for n in range(20) if numTheory.is_prime(n)] == [2, 3, 5, 7, 11, 13, 17, 19]
    assert numTheory.is_prime(2**61 - 1) is True
    assert numTheory.is_prime(2**61 + 1) is False


def test_prime_factors():
    assert numTheory.prime_factors(28) == {2: 2, 7: 1}


def test_gcd_lcm_ext():
    assert numTheory.gcd(12, 18) == 6
    assert numTheory.lcm(4, 6) == 12
    g, x, y = numTheory.extended_gcd(240, 46)
    assert g == 2 and 240 * x + 46 * y == 2


def test_modular():
    assert numTheory.mod_pow(2, 10, 1000) == 24
    assert numTheory.mod_inverse(3, 11) == 4
    assert numTheory.totient(36) == 12


def test_perfect_nth_fib():
    assert numTheory.is_perfect(28) and not numTheory.is_perfect(27)
    assert numTheory.nth_prime(1) == 2
    assert numTheory.nth_prime(2000) == 17389
    assert numTheory.fibonacci(10) == 55
    assert numTheory.fibonacci(90) == 2880067194370816120
    with pytest.raises(ValueError):
        numTheory.fibonacci(-1)


# vector
def test_vector_ops():
    a, b = [1, 2, 3], [4, 5, 6]
    np.testing.assert_allclose(vector.add(a, b), [5, 7, 9])
    np.testing.assert_allclose(vector.subtract(b, a), [3, 3, 3])
    np.testing.assert_allclose(vector.scalar_multiply(a, 2), [2, 4, 6])
    assert vector.dot_product(a, b) == 32
    np.testing.assert_allclose(vector.cross_product([1, 0, 0], [0, 1, 0]), [0, 0, 1])
    assert vector.magnitude([3, 4]) == 5
    np.testing.assert_allclose(vector.normalize([3, 4]), [0.6, 0.8])
    assert vector.angle_between([1, 0], [0, 1]) == A(math.pi / 2)


def test_vector_errors():
    with pytest.raises(ValueError):
        vector.add([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        vector.normalize([0, 0])


def test_vector_class():
    assert matha.Vector is vector.Vector


# linalg (linear algebra)
M2 = [[1, 2], [3, 4]]


def test_linalg():
    np.testing.assert_allclose(linalg.matrix_multiply(M2, M2), [[7, 10], [15, 22]])
    np.testing.assert_allclose(linalg.identity(2), np.eye(2))
    assert linalg.matrix_determinant(M2) == A(-2)
    assert linalg.det(M2) == A(-2)
    np.testing.assert_allclose(linalg.matrix_inverse(M2), [[-2, 1], [1.5, -0.5]])
    np.testing.assert_allclose(linalg.inv(M2), [[-2, 1], [1.5, -0.5]])
    np.testing.assert_allclose(linalg.solve_linear_system(M2, [5, 6]), [-4, 4.5])
    np.testing.assert_allclose(linalg.solve(M2, [5, 6]), [-4, 4.5])
    assert linalg.norm([3, 4]) == A(5)
    np.testing.assert_allclose(linalg.transpose(np.array(M2)), [[1, 3], [2, 4]])


# stats
D = [2, 4, 4, 4, 5, 5, 7, 9]


def test_descriptive():
    assert stats.mean(D) == 5
    assert stats.median(D) == 4.5
    assert stats.data_range(D) == 7
    assert stats.variance(D, sample=False) == A(4)
    assert stats.variance(D) == A(np.var(D, ddof=1))
    assert stats.std_dev(D, sample=False) == A(2)
    assert stats.std_dev(D) == A(np.std(D, ddof=1))
    assert stats.z_score(9, D, sample=False) == A(2)
    assert stats.percentile(D, 50) == A(4.5)
    assert tuple(stats.quartiles(D)) == A((4.0, 4.5, 5.5))
    assert stats.iqr(D) == A(1.5)
    assert stats.mode([1, 2, 2, 3]) == [2]


def test_bivariate():
    x, y = [1, 2, 3], [2, 4, 6]
    assert stats.covariance(x, y) == A(2)
    assert stats.correlation(x, y) == A(1)
    m = stats.corr_matrix(np.array([[1, 2], [2, 4], [3, 6.5]]))
    assert m.shape == (2, 2) and m[0, 0] == A(1)


def test_distributions():
    assert stats.binomial_pmf(2, 4, 0.5) == A(0.375)
    assert stats.binomial_cdf(2, 4, 0.5) == A(0.6875)
    assert stats.geometric_pmf(3, 0.5) == A(0.125)
    assert stats.geometric_cdf(3, 0.5) == A(0.875)
    assert stats.poisson_pmf(0, 1) == A(math.exp(-1))
    assert stats.poisson_cdf(1, 1) == A(2 / math.e)
    assert stats.normal_pdf(0) == A(1 / math.sqrt(2 * math.pi))
    assert stats.normal_cdf(0) == A(0.5)
    assert stats.normal_cdf(1.96) == A(0.9750021, abs=1e-6)


# geometry
SQ = [(0, 0), (1, 0), (1, 1), (0, 1)]


def test_geometry():
    assert geometry.distance((0, 0), (3, 4)) == 5
    assert tuple(geometry.midpoint((0, 0), (2, 4))) == (1, 2)
    assert geometry.slope([(0, 0), (2, 4)]) == 2
    assert geometry.perimeter(SQ) == 4
    assert geometry.area(SQ) == 1
    assert tuple(geometry.centroid(SQ)) == (0.5, 0.5)
    assert geometry.circle_area(1) == A(math.pi)
    assert geometry.circle_circumference(1) == A(2 * math.pi)


def test_plot_polygon(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    geometry.plot_polygon(SQ)
    plt.close("all")


# trigonometry 

@pytest.mark.parametrize("fn,deg,want", [
    ("sin", 90, 1), ("cos", 60, 0.5), ("tan", 45, 1),
    ("cosec", 30, 2), ("sec", 60, 2), ("cot", 45, 1),
])
def test_trig_degrees(fn, deg, want):
    assert getattr(trig, fn)(deg, in_degrees=True) == A(want)
    assert getattr(trig, fn)(math.radians(deg)) == A(want)


def test_trig_misc():
    assert trig.law_of_cosines(3, 4, 90, True) == A(5)
    assert trig.pythagorean_identity(0.7) == A(1)
    assert trig.to_radians(180) == A(math.pi)
    assert trig.to_degrees(math.pi) == A(180)


@pytest.mark.parametrize("name,ref,arg", [
    ("asin", math.asin, 0.5), ("acos", math.acos, 0.5), ("atan", math.atan, 0.5),
    ("sinh", math.sinh, 0.5), ("cosh", math.cosh, 0.5), ("tanh", math.tanh, 0.5),
    ("asinh", math.asinh, 0.5), ("acosh", math.acosh, 1.5), ("atanh", math.atanh, 0.5),
    ("arcsine", math.asin, 0.5), ("arccosine", math.acos, 0.5),
    ("arctangent", math.atan, 0.5), ("hsin", math.sinh, 0.5),
    ("hcos", math.cosh, 0.5), ("htan", math.tanh, 0.5),
    ("inverse_hyperbolic_sin", math.asinh, 0.5),
    ("inverse_hyperbolic_cos", math.acosh, 1.5),
    ("inverse_hyperbolic_tan", math.atanh, 0.5),
])
def test_trig_math_bound(name, ref, arg):
    assert getattr(trig, name)(arg) == A(ref(arg))


def test_trig_atan2():
    assert trig.atan2(1, 1) == A(math.pi / 4)
    assert trig.arctangent2(1, 1) == A(math.pi / 4)


# complex
def test_complex_basic():
    assert mcomplex.create(3, 4) == 3 + 4j
    assert mcomplex.conjugate(1 + 2j) == 1 - 2j
    assert mcomplex.add(1, 1j) == 1 + 1j
    assert mcomplex.subtract(1, 1j) == 1 - 1j
    assert mcomplex.multiply(1j, 1j) == -1
    assert mcomplex.divide(1 + 1j, 1j) == 1 - 1j
    assert mcomplex.magnitude(3 + 4j) == 5
    assert mcomplex.phase(1j) == A(math.pi / 2)
    assert mcomplex.polar(1j) == A((1.0, math.pi / 2))
    assert mcomplex.complex_sqrt(-1) == 1j
    assert mcomplex.complex_log(1) == 0


@pytest.mark.parametrize("name,ref", [
    ("c_sin", "sin"), ("c_cos", "cos"), ("c_tan", "tan"), ("c_sinh", "sinh"),
    ("c_cosh", "cosh"), ("c_tanh", "tanh"), ("c_asinh", "asinh"),
    ("c_acosh", "acosh"), ("c_atanh", "atanh"), ("c_asin", "asin"),
    ("c_acos", "acos"), ("c_atan", "atan"),
])
def test_complex_cmath_bound(name, ref):
    import cmath
    z = 0.3 + 0.4j
    assert getattr(mcomplex, name)(z) == A(getattr(cmath, ref)(z))


def test_complex_reciprocals():
    z = 1 + 1j
    import cmath
    assert mcomplex.c_cosec(z) == A(1 / cmath.sin(z))
    assert mcomplex.c_sec(z) == A(1 / cmath.cos(z))
    assert mcomplex.c_cot(z) == A(1 / cmath.tan(z))


def test_complex_class():
    assert mcomplex.Complex is not None


# algebra
def test_algebra():
    assert algebra.factorial(5) == 120
    assert algebra.nPr(5, 2) == 20
    assert algebra.nCr(5, 2) == 10
    assert algebra.log(8, 2) == A(3)
    assert algebra.ln(math.e) == A(1)
    assert algebra.exp(0) == 1
    assert algebra.sqrt(9) == 3
    assert algebra.pow(2, 10) == 1024
    assert algebra.gcd(12, 18) == 6
    assert algebra.lcm(4, 6) == 12


def test_binomial():
    assert algebra.binomial_expand(3) == [(1, 3, 0), (3, 2, 1), (3, 1, 2), (1, 0, 3)]
    assert algebra.binomial_to_string_optimized(algebra.binomial_expand(2)) == "x^2 + 2xy + y^2"


def test_algebra_quadratic():
    r = algebra.solve_quadratic(1, -3, 2)
    assert sorted(x.real for x in r) == A([1, 2])
    assert set(algebra.solve_quadratic(1, 0, 1)) == {1j, -1j}


# calculas
def test_integral_arc():
    assert calculas.integral(lambda x: x ** 2, 0, 3) == A(9, abs=1e-4)
    assert calculas.arc_length(lambda x: x, 0, 1) == A(math.sqrt(2), abs=1e-6)


@pytest.mark.xfail(reason="improper_integral raises TypeError (numpy scalar conversion)")
def test_improper_integral():
    assert calculas.improper_integral(lambda x: math.exp(-x), 0) == A(1, abs=1e-3)
    assert calculas.improper_integral(lambda x: math.exp(x), 0, "left") == A(1, abs=1e-3)


def test_derivatives():
    f = lambda x: x ** 2
    assert calculas.derivative(f, 3) == A(6, abs=1e-4)
    assert calculas.derivative_left(f, 3) == A(6, abs=1e-4)
    assert calculas.derivative_right(f, 3) == A(6, abs=1e-4)
    assert calculas.second_derivative(lambda x: x ** 3, 2) == A(12, abs=1e-2)
    assert calculas.nth_derivative(lambda x: x ** 4, 1, 2) == A(12, abs=1e-2)


def test_limit_maclaurin():
    assert calculas.limit(lambda x: math.sin(x) / x, 0) == A(1, abs=1e-6)
    assert calculas.limit(lambda x: abs(x) / x, 0, "right") == A(1)
    c = calculas.maclaurin(math.exp, 3)
    assert c[:3] == A([1, 1, 0.5], abs=1e-3)


def test_multivariable():
    f = lambda x, y: x * y
    assert calculas.partial_derivative(f, [2, 3], 0) == A(3, abs=1e-4)
    np.testing.assert_allclose(calculas.gradient(f, [2, 3]), [3, 2], atol=1e-4)
    assert calculas.divergence([lambda x, y: x, lambda x, y: y], [1, 2]) == A(2, abs=1e-4)
    z = lambda x, y, w: 0.0
    np.testing.assert_allclose(
        calculas.curl([lambda x, y, w: -y, lambda x, y, w: x, z], [1, 2, 0]),
        [0, 0, 2], atol=1e-4)


# solvers
def test_solvers():
    np.testing.assert_allclose(solvers.solve_system([[2, 1], [1, 3]], [3, 5]), [0.8, 1.4])
    assert solvers.find_root(lambda x: x ** 2 - 2, 1) == A(math.sqrt(2))
    assert sorted(x.real for x in solvers.solve_quadratic(1, -3, 2)) == A([1, 2])
    assert len(solvers.solve_quadratic(1, -3, 2, list=True)) == 2
    assert sorted(solvers.solve_polynomial([1, -3, 2])) == A([1, 2])
    assert solvers.find_minimum(lambda x: (x - 2) ** 2, 0) == A(2, abs=1e-4)
    r = solvers.solve_least_squares([1, 2, 3], [2, 4, 6])
    assert r["slope"] == A(2) and r["intercept"] == A(0, abs=1e-9)


def test_solve_ode():
    t, y = solvers.solve_ode(lambda t, y: -y, (0, 1), [1])
    assert np.ravel(y)[-1] == A(math.exp(-1), rel=1e-3)


# regression
def test_regression():
    yt, yp = np.array([1.0, 2, 3]), np.array([1.0, 2, 5])
    assert regression.mse(yt, yp) == A(4 / 3)
    assert regression.mae(yt, yp) == A(2 / 3)
    assert regression.r_squared(yt, yp) == A(1 - 4 / 2)
    X = np.array([[1.0], [2], [3], [4]])
    y = 2 * X[:, 0] + 1
    m, b = regression.linear_regression_analytical(X[:, 0], y)
    assert m == A(2) and b == A(1)
    m, b = regression.gradient_descent(X[:, 0], y, lr=0.05, iterations=5000)
    assert m == A(2, abs=1e-2) and b == A(1, abs=1e-2)
    np.testing.assert_allclose(regression.predict(X[:, 0], 2, 1), y)


# simple machine learning func's test
def test_ml_losses():
    p, a = np.array([1.0, 2, 3]), np.array([1.0, 2, 5])
    assert ml.mse(p, a) == A(4 / 3)
    assert ml.rmse(p, a) == A(math.sqrt(4 / 3))
    assert ml.mae(p, a) == A(2 / 3)
    assert ml.r2_score(p, a) == A(1 - 4 / ((a - a.mean()) ** 2).sum())
    assert ml.binary_cross_entropy(np.array([0.9, 0.1]), np.array([1, 0])) == A(-math.log(0.9))


def test_ml_classification():
    assert ml.accuracy_score([1, 0, 1], [1, 1, 1]) == A(2 / 3)
    cm = ml.confusion_matrix_basic([1, 0, 1, 0], [1, 1, 0, 0])
    assert (cm["TP"], cm["TN"], cm["FP"], cm["FN"]) == (1, 1, 1, 1)


def test_ml_activations():
    assert ml.sigmoid(0) == A(0.5)
    np.testing.assert_array_equal(ml.relu(np.array([-1, 2])), [0, 2])
    s = ml.softmax(np.array([1.0, 2, 3]))
    assert s.sum() == A(1) and s[2] > s[1] > s[0]


def test_ml_data():
    Xtr, Xte, ytr, yte = ml.train_test_split(np.arange(10).reshape(5, 2), np.arange(5))
    assert (Xtr.shape, Xte.shape, ytr.shape, yte.shape) == ((4, 2), (1, 2), (4,), (1,))
    np.testing.assert_allclose(ml.min_max_scale(np.array([[1.0, 2], [3, 4]])), [[0, 0], [1, 1]])


@pytest.mark.xfail(reason="min_max_scale fails on 1-D input (item assignment)")
def test_min_max_scale_1d():
    np.testing.assert_allclose(ml.min_max_scale(np.array([1.0, 2, 3])), [0, 0.5, 1])


# special
def test_special():
    assert sf.gamma_function(5) == A(24)
    assert sf.gamma_function(0.5) == A(math.sqrt(math.pi))
    assert sf.gamma_function(5, use_integral_approx=True) == A(24, rel=1e-2)
    assert sf.beta_function(2, 3) == A(1 / 12)
    assert sf.error_function(0) == 0
    assert sf.error_function(1) == A(math.erf(1))
    assert sf.complementary_error_function(1) == A(math.erfc(1))
    assert sf.sterling_approx_factorial(10) == A(3628800, rel=1e-2)


def test_special_type_errors():
    with pytest.raises(TypeError):
        sf.error_function(None)


# ------------------------------------------------------------ constants
@pytest.mark.parametrize("name,val", [
    ("PI", math.pi), ("TAU", math.tau), ("e", math.e), ("EULER", math.e),
    ("GOLDEN_RATIO", (1 + 5 ** 0.5) / 2), ("PHI", (1 + 5 ** 0.5) / 2),
    ("PHI_CONJUGATE", (1 - 5 ** 0.5) / 2),
    ("EULER_MASCHERONI", 0.5772156649015329), ("GAMMA", 0.5772156649015329),
    ("LN2", math.log(2)), ("LN10", math.log(10)),
    ("LOG2E", math.log2(math.e)), ("LOG10E", math.log10(math.e)),
    ("EPSILON", 1e-10), ("C", 299792458), ("SPEED_OF_LIGHT", 299792458),
    ("G", 6.6743e-11), ("H", 6.62607015e-34), ("PLANCK_CONSTANT", 6.62607015e-34),
    ("K_BOLTZMANN", 1.380649e-23), ("NA", 6.02214076e23),
    ("AVOGADRO_NUMBER", 6.02214076e23), ("R", 8.314462618),
])
def test_constants(name, val):
    assert getattr(constants, name) == A(val, rel=1e-12)


def test_constants_special():
    assert constants.Infinity == math.inf
    assert constants.NAN != constants.NAN


# graph
@pytest.fixture
def noshow(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


RNG = np.random.default_rng(0)
X, Y, Z = RNG.normal(size=50), RNG.normal(size=50), RNG.normal(size=50)


@pytest.mark.parametrize("name,args,kw", [
    ("histogram", (X,), {}),
    ("boxplot", (X,), {}),
    ("violin_plot", (X,), {}),
    ("density_plot", (X,), {}),
    ("hexbin_plot", (X, Y), {}),
    ("scatter_plot", (X, Y), {}),
    ("scatter_3d", (X, Y, Z), {}),
    ("scatter_matrix", (np.c_[X, Y, Z],), {}),
    ("plot_function", (np.sin, -3, 3), {}),
    ("plot_multiple_functions", ([np.sin, np.cos], -3, 3), {}),
    ("plot_points", ([(0, 0), (1, 2), (2, 1)],), {}),
    ("plot_root", (lambda x: x ** 2 - 2, 2 ** 0.5), {}),
    ("plot_regression", ([1, 2, 3], [2, 4, 6], 2, 0), {}),
    ("plot_ode", (np.linspace(0, 1, 20), np.exp(-np.linspace(0, 1, 20))), {}),
    ("plot_complex_2d", ([1 + 1j, -1j, 2],), {}),
    ("plot_complex_surface", (lambda z: z ** 2,), {"res": 20}),
])
def test_graph_smoke(noshow, name, args, kw):
    getattr(graph, name)(*args, **kw)


# guard
def test_every_function_is_tested():
    """Fails when a public function is added without a test mentioning it."""
    src = pathlib.Path(__file__).read_text()
    skip = {"Vector", "Complex"}
    missing = []
    for mod in (algebra, calculas, geometry, graph, linalg, numTheory, stats, trig,
                regression, ml, vector, sf, solvers, mcomplex):
        for n, f in vars(mod).items():
            if n.startswith("_") or n in skip or not callable(f) or inspect.isclass(f):
                continue
            home = getattr(f, "__module__", "") or ""
            if home.startswith(("matha", "math", "cmath", "numpy.linalg", "builtins", "operator")) is False:
                continue
            if not re.search(r"\b%s\b" % re.escape(n), src):
                missing.append(f"{mod.__name__.split('.')[-1]}.{n}")
    assert not missing, missing
