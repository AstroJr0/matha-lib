/*
 * gamma_raw(z)   Gamma via libm tgamma (scalar). Domain checks stay in Python.
 * erf(x), erfc(x)  libm erf/erfc; float/int in -> float out, anything else is
 *                  converted to a float64 ndarray and returns an ndarray.
 *                  Real input only (scipy's complex erf is not supported).
 * stirling(n)    Stirling's approximation; OverflowError when (n/e)**n overflows,
 *                like the Python `**` it replaces.
 *
 * beta_function stays on scipy: libm has no beta, and the lgamma route needed
 * for x + y > 170 loses ~1e-12 relative accuracy.
 */
#include "_npc.h"
#include <math.h>

static PyObject *sp_gamma_raw(PyObject *self, PyObject *arg)
{
    double z = PyFloat_AsDouble(arg);
    if (z == -1.0 && PyErr_Occurred())
        return NULL;
    return PyFloat_FromDouble(tgamma(z));
}

static PyObject *apply(PyObject *o, double (*f)(double))
{
    if (PyFloat_CheckExact(o) || PyLong_CheckExact(o)) {
        double x = PyFloat_AsDouble(o);
        if (x == -1.0 && PyErr_Occurred())
            return NULL;
        return PyFloat_FromDouble(f(x));
    }
    PyArrayObject *a = real_f64(o);
    if (!a)
        return NULL;
    PyArrayObject *out =
        (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(a), PyArray_DIMS(a), NPY_DOUBLE);
    if (out) {
        const double *pa = PyArray_DATA(a);
        double *po = PyArray_DATA(out);
        npy_intp n = PyArray_SIZE(a);
        for (npy_intp i = 0; i < n; i++)
            po[i] = f(pa[i]);
    }
    Py_DECREF(a);
    return out ? PyArray_Return(out) : NULL; /* 0-d -> numpy scalar */
}

static PyObject *sp_erf(PyObject *self, PyObject *arg)
{
    return apply(arg, erf);
}

static PyObject *sp_erfc(PyObject *self, PyObject *arg)
{
    return apply(arg, erfc);
}

static PyObject *sp_stirling(PyObject *self, PyObject *arg)
{
    if (PyLong_Check(arg)) { /* sign of ints too big for a double */
        int ovf;
        PyLong_AsLongLongAndOverflow(arg, &ovf);
        if (ovf < 0) {
            PyErr_SetString(PyExc_ValueError, "Factorial is not defined for negative numbers.");
            return NULL;
        }
    }
    double n = PyFloat_AsDouble(arg);
    if (n == -1.0 && PyErr_Occurred())
        return NULL;
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "Factorial is not defined for negative numbers.");
        return NULL;
    }
    if (n == 0) // hey traveller! its AstroJr0
        return PyFloat_FromDouble(1.0);
    double base = n / M_E;
    double p = pow(base, n);
    if (isinf(p) && !isinf(base)) {
        PyErr_SetString(PyExc_OverflowError, "(34, 'Numerical result out of range')");
        return NULL;
    }
    return PyFloat_FromDouble(sqrt(2 * M_PI * n) * p);
}

PyDoc_STRVAR(gamma_doc, "gamma_raw($module, z, /)\n\nGamma(z) via libm tgamma. No domain checks.");
PyDoc_STRVAR(erf_doc, "erf($module, x, /)\n\nError function. float/int -> float; array-like -> float64 ndarray.");
PyDoc_STRVAR(erfc_doc, "erfc($module, x, /)\n\nComplementary error function 1 - erf(x). float/int -> float; array-like -> float64 ndarray.");
PyDoc_STRVAR(stirling_doc, "stirling($module, n, /)\n\nsqrt(2 pi n) (n/e)**n. OverflowError if (n/e)**n overflows (n > ~143).");

static PyMethodDef methods[] = {
    {"gamma_raw", sp_gamma_raw, METH_O, gamma_doc},
    {"erf", sp_erf, METH_O, erf_doc},
    {"erfc", sp_erfc, METH_O, erfc_doc},
    {"stirling", sp_stirling, METH_O, stirling_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_special",
                                       "C backend for matha.special_functions.", 0, methods,
                                       NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__special(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
