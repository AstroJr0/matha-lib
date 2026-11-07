/* shared NumPy C-API glue for the matha C extensions.
 * Include once per extension (single translation unit). */
#ifndef MATHA_NPC_H
#define MATHA_NPC_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* Any array-like -> C-contiguous, aligned float64 array (new reference).
 * Ints/lists are converted; complex raises TypeError (unsafe cast). */
static inline PyArrayObject *as_f64(PyObject *o)
{
    return (PyArrayObject *)PyArray_FROM_OTF(o, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
}

/* bool/int/float array-like -> C-contiguous float64 array (new reference).
 * None, str, object and complex input raise TypeError instead of being coerced. */
static inline PyArrayObject *real_f64(PyObject *o)
{
    PyArrayObject *a = (PyArrayObject *)PyArray_FROM_O(o);
    if (!a)
        return NULL;
    if (!(PyArray_ISBOOL(a) || PyArray_ISINTEGER(a) || PyArray_ISFLOAT(a))) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_TypeError, "input must be real numbers");
        return NULL;
    }
    PyArrayObject *d = (PyArrayObject *)PyArray_FROM_OTF((PyObject *)a, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    Py_DECREF(a);
    return d;
}

/* C double -> np.float64 scalar (new reference) */
static inline PyObject *np_f64(double v)
{
    PyArray_Descr *d = PyArray_DescrFromType(NPY_DOUBLE);
    if (!d)
        return NULL;
    PyObject *r = PyArray_Scalar(&v, d, NULL);
    Py_DECREF(d);
    return r;
}

#endif
