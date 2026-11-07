/*
 * Plain C loops (i-k-j matmul, LU with partial pivoting). No BLAS/LAPACK,
 * so this wins on small matrices and loses to NumPy on large ones.
 * Inputs are converted to float64; 1-D / 2-D only.
 */
#include "_npc.h"
#include <math.h>
#include <string.h>

#define KW (PyCFunction)(void (*)(void))

static void raise_singular(void)
{
    PyObject *mod = PyImport_ImportModule("numpy.linalg");
    if (!mod)
        return;
    PyObject *exc = PyObject_GetAttrString(mod, "LinAlgError");
    Py_DECREF(mod);
    if (!exc)
        return;
    PyErr_SetString(exc, "Singular matrix");
    Py_DECREF(exc);
}

/* LU in place, row-major n x n. Returns 1 if an exact zero pivot is hit. */
static int lu_decomp(double *a, npy_intp n, npy_intp *piv, int *sign)
{
    *sign = 1;
    for (npy_intp k = 0; k < n; k++) {
        npy_intp p = k;
        double mx = fabs(a[k * n + k]);
        for (npy_intp i = k + 1; i < n; i++) {
            double v = fabs(a[i * n + k]);
            if (v > mx) {
                mx = v;
                p = i;
            }
        }
        piv[k] = p;
        if (mx == 0.0)
            return 1;
        if (p != k) {
            for (npy_intp j = 0; j < n; j++) {
                double t = a[k * n + j];
                a[k * n + j] = a[p * n + j];
                a[p * n + j] = t;
            }
            *sign = -*sign;
        }
        double d = a[k * n + k];
        for (npy_intp i = k + 1; i < n; i++) {
            double f = a[i * n + k] / d;
            a[i * n + k] = f;
            if (f != 0.0)
                for (npy_intp j = k + 1; j < n; j++)
                    a[i * n + j] -= f * a[k * n + j];
        }
    }
    return 0;
}

/* Solve LU x = P b in place; x is n x m row-major. */
static void lu_solve(const double *lu, const npy_intp *piv, npy_intp n, double *x, npy_intp m)
{
    for (npy_intp k = 0; k < n; k++) {
        if (piv[k] != k) {
            for (npy_intp j = 0; j < m; j++) {
                double t = x[k * m + j];
                x[k * m + j] = x[piv[k] * m + j];
                x[piv[k] * m + j] = t;
            }
        }
    }
    for (npy_intp i = 1; i < n; i++)
        for (npy_intp k = 0; k < i; k++) {
            double f = lu[i * n + k];
            if (f != 0.0)
                for (npy_intp j = 0; j < m; j++)
                    x[i * m + j] -= f * x[k * m + j];
        }
    for (npy_intp i = n - 1; i >= 0; i--) {
        for (npy_intp k = i + 1; k < n; k++) {
            double f = lu[i * n + k];
            if (f != 0.0)
                for (npy_intp j = 0; j < m; j++)
                    x[i * m + j] -= f * x[k * m + j];
        }
        double d = lu[i * n + i];
        for (npy_intp j = 0; j < m; j++)
            x[i * m + j] /= d;
    }
}

static int check_square(PyArrayObject *A)
{
    if (PyArray_NDIM(A) != 2 || PyArray_DIMS(A)[0] != PyArray_DIMS(A)[1]) {
        PyErr_SetString(PyExc_ValueError, "Input matrix must be square (N x N).");
        return -1;
    }
    return 0;
}

/* copy of A's data plus pivot array, for in-place LU. */
static int alloc_lu(PyArrayObject *A, double **lu, npy_intp **piv)
{
    npy_intp n = PyArray_DIMS(A)[0];
    *lu = PyMem_Malloc((size_t)(n * n) * sizeof(double) + 1);
    *piv = PyMem_Malloc((size_t)n * sizeof(npy_intp) + 1);
    if (!*lu || !*piv) {
        PyMem_Free(*lu);
        PyMem_Free(*piv);
        PyErr_NoMemory();
        return -1;
    }
    memcpy(*lu, PyArray_DATA(A), (size_t)(n * n) * sizeof(double));
    return 0;
}

// matrix multiply
static PyObject *l_matrix_multiply(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"A", "B", NULL};
    PyObject *oa, *ob, *res = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:matrix_multiply", kwlist, &oa, &ob))
        return NULL;
    PyArrayObject *A = as_f64(oa), *B = A ? as_f64(ob) : NULL;
    if (!B)
        goto done;

    int na = PyArray_NDIM(A), nb = PyArray_NDIM(B);
    if (na < 1 || na > 2 || nb < 1 || nb > 2) {
        PyErr_SetString(PyExc_ValueError, "matrix_multiply supports 1-D and 2-D arrays only.");
        goto done;
    }
    npy_intp *da = PyArray_DIMS(A), *db = PyArray_DIMS(B);
    npy_intp am = na == 2 ? da[0] : 1, ak = da[na - 1];
    npy_intp bk = db[0], bn = nb == 2 ? db[1] : 1;
    if (ak != bk) {
        PyErr_Format(PyExc_ValueError,
                     "Inner dimensions must match for matrix multiplication. "
                     "A has %zd columns, B has %zd rows.",
                     (Py_ssize_t)ak, (Py_ssize_t)bk);
        goto done;
    }

    const double *pa = PyArray_DATA(A), *pb = PyArray_DATA(B);
    if (na == 1 && nb == 1) {
        double s = 0;
        for (npy_intp k = 0; k < ak; k++)
            s += pa[k] * pb[k];
        res = PyFloat_FromDouble(s);
        goto done;
    }

    npy_intp od[2];
    int nd = 0;
    if (na == 2)
        od[nd++] = am;
    if (nb == 2)
        od[nd++] = bn;
    PyArrayObject *C = (PyArrayObject *)PyArray_ZEROS(nd, od, NPY_DOUBLE, 0);
    if (!C)
        goto done;
    double *pc = PyArray_DATA(C);
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < am; i++)
        for (npy_intp k = 0; k < ak; k++) {
            double aik = pa[i * ak + k];
            const double *brow = pb + k * bn;
            double *crow = pc + i * bn;
            for (npy_intp j = 0; j < bn; j++)
                crow[j] += aik * brow[j];
        }
    Py_END_ALLOW_THREADS
    res = (PyObject *)C;
done:
    Py_XDECREF(A);
    Py_XDECREF(B);
    return res;
}

// identity
static PyObject *l_identity(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"n", NULL};
    Py_ssize_t n;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "n:identity", kwlist, &n))
        return NULL;
    if (n < 0) {
        PyErr_SetString(PyExc_ValueError, "negative dimensions are not allowed");
        return NULL;
    }
    npy_intp d[2] = {n, n};
    PyArrayObject *eye = (PyArrayObject *)PyArray_ZEROS(2, d, NPY_DOUBLE, 0);
    if (!eye)
        return NULL;
    double *p = PyArray_DATA(eye);
    for (npy_intp i = 0; i < n; i++)
        p[i * n + i] = 1.0;
    return (PyObject *)eye;
}

// determinant
static PyObject *l_determinant(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"A", NULL};
    PyObject *o, *res = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:matrix_determinant", kwlist, &o))
        return NULL;
    PyArrayObject *A = as_f64(o);
    if (!A)
        return NULL;
    if (check_square(A) < 0)
        goto done;
    {
        npy_intp n = PyArray_DIMS(A)[0];
        double *lu;
        npy_intp *piv;
        if (alloc_lu(A, &lu, &piv) < 0)
            goto done;
        int sign, singular;
        double det = 1.0;
        Py_BEGIN_ALLOW_THREADS
        singular = lu_decomp(lu, n, piv, &sign);
        if (singular) {
            det = 0.0;
        }
        else {
            det = sign;
            for (npy_intp i = 0; i < n; i++)
                det *= lu[i * n + i];
        }
        Py_END_ALLOW_THREADS
        PyMem_Free(lu);
        PyMem_Free(piv);
        res = PyFloat_FromDouble(det);
    }
done:
    Py_DECREF(A);
    return res;
}

// inverse
static PyObject *l_inverse(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"A", NULL};
    PyObject *o, *res = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:matrix_inverse", kwlist, &o))
        return NULL;
    PyArrayObject *A = as_f64(o);
    if (!A)
        return NULL;
    if (check_square(A) < 0)
        goto done;
    {
        npy_intp n = PyArray_DIMS(A)[0];
        double *lu;
        npy_intp *piv;
        if (alloc_lu(A, &lu, &piv) < 0)
            goto done;
        npy_intp d[2] = {n, n};
        PyArrayObject *X = (PyArrayObject *)PyArray_ZEROS(2, d, NPY_DOUBLE, 0);
        if (!X) {
            PyMem_Free(lu);
            PyMem_Free(piv);
            goto done;
        }
        double *x = PyArray_DATA(X);
        for (npy_intp i = 0; i < n; i++)
            x[i * n + i] = 1.0;
        int singular;
        Py_BEGIN_ALLOW_THREADS
        singular = lu_decomp(lu, n, piv, &(int){0});
        if (!singular)
            lu_solve(lu, piv, n, x, n);
        Py_END_ALLOW_THREADS
        PyMem_Free(lu);
        PyMem_Free(piv);
        if (singular) {
            Py_DECREF(X);
            raise_singular();
        }
        else {
            res = (PyObject *)X;
        }
    }
done:
    Py_DECREF(A);
    return res;
}

// solve
static PyObject *l_solve(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"A", "b", NULL};
    PyObject *oa, *ob, *res = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:solve_linear_system", kwlist, &oa, &ob))
        return NULL;
    PyArrayObject *A = as_f64(oa), *B = A ? as_f64(ob) : NULL, *X = NULL;
    if (!B)
        goto done;
    if (PyArray_NDIM(A) != 2 || PyArray_DIMS(A)[0] != PyArray_DIMS(A)[1] ||
        PyArray_NDIM(B) < 1 || PyArray_NDIM(B) > 2 ||
        PyArray_DIMS(A)[0] != PyArray_DIMS(B)[0]) {
        PyErr_SetString(PyExc_ValueError, "A must be square (N x N) and b must have N rows.");
        goto done;
    }
    {
        npy_intp n = PyArray_DIMS(A)[0];
        npy_intp m = PyArray_NDIM(B) == 2 ? PyArray_DIMS(B)[1] : 1;
        double *lu;
        npy_intp *piv;
        if (alloc_lu(A, &lu, &piv) < 0)
            goto done;
        X = (PyArrayObject *)PyArray_NewCopy(B, NPY_CORDER);
        if (!X) {
            PyMem_Free(lu);
            PyMem_Free(piv);
            goto done;
        } // made by AstroJr0
        int singular;
        Py_BEGIN_ALLOW_THREADS
        singular = lu_decomp(lu, n, piv, &(int){0});
        if (!singular)
            lu_solve(lu, piv, n, PyArray_DATA(X), m);
        Py_END_ALLOW_THREADS
        PyMem_Free(lu);
        PyMem_Free(piv);
        if (singular) {
            Py_DECREF(X);
            raise_singular();
        }
        else {
            res = (PyObject *)X;
        }
    }
done:
    Py_XDECREF(A);
    Py_XDECREF(B);
    return res;
}

PyDoc_STRVAR(matrix_multiply_doc, "matrix_multiply($module, /, A, B)\n\nMatrix product A @ B for 1-D / 2-D arrays. Raises ValueError if A's columns != B's rows.");
PyDoc_STRVAR(identity_doc, "identity($module, /, n)\n\nn x n identity matrix.");
PyDoc_STRVAR(determinant_doc, "matrix_determinant($module, /, A)\n\nDeterminant of a square matrix (LU, partial pivoting).");
PyDoc_STRVAR(inverse_doc, "matrix_inverse($module, /, A)\n\nInverse of a square matrix. Raises numpy.linalg.LinAlgError if singular.");
PyDoc_STRVAR(solve_doc, "solve_linear_system($module, /, A, b)\n\nSolve A x = b (b is N or N x M). Raises numpy.linalg.LinAlgError if A is singular.");

static PyMethodDef methods[] = {
    {"matrix_multiply", KW l_matrix_multiply, METH_VARARGS | METH_KEYWORDS, matrix_multiply_doc},
    {"identity", KW l_identity, METH_VARARGS | METH_KEYWORDS, identity_doc},
    {"matrix_determinant", KW l_determinant, METH_VARARGS | METH_KEYWORDS, determinant_doc},
    {"matrix_inverse", KW l_inverse, METH_VARARGS | METH_KEYWORDS, inverse_doc},
    {"solve_linear_system", KW l_solve, METH_VARARGS | METH_KEYWORDS, solve_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_linalg",
                                       "C backend for matha.linalg.", 0, methods,
                                       NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__linalg(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
