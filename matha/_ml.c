/* Metrics are single-pass with Neumaier-compensated sums (more accurate than
 * NumPy's pairwise sum, so results agree to ~1e-15 relative, not bit-for-bit).
 * Metric inputs broadcast like NumPy; results are np.float64 like before.
 * Inputs must be bool/int/float (None / str / object arrays raise TypeError).
 *
 * linear_regression / gradient_descent need x and y of the same shape.
 * All functions are positional-only; the Python wrappers keep the signatures
 * and docstrings.
 *
 * Not ported (NumPy is already as fast or faster): sigmoid / softmax / relu
 * (SIMD exp), accuracy_score / confusion_matrix_basic (string labels),
 * train_test_split, min_max_scale.
 */
#include "_npc.h"
#include <math.h>

#define FAST(f) (PyCFunction)(void (*)(void))(f)

typedef struct {
    double s, c, plain;
} Acc;
#define ACC0 {0.0, 0.0, 0.0}

static inline void acc_add(Acc *a, double v)
{
    double t = a->s + v;
    if (fabs(a->s) >= fabs(v))
        a->c += (a->s - t) + v;
    else
        a->c += (v - t) + a->s;
    a->s = t;
    a->plain += v;
}

static inline double acc_val(const Acc *a)
{
    double r = a->s + a->c;
    return isfinite(r) ? r : a->plain;
}

static PyObject *nargs_error(const char *name, Py_ssize_t want, Py_ssize_t got)
{
    PyErr_Format(PyExc_TypeError, "%s() takes %zd positional arguments but %zd were given",
                 name, want, got);
    return NULL;
}

static int same_shape(PyArrayObject *a, PyArrayObject *b)
{
    int nd = PyArray_NDIM(a);
    if (nd != PyArray_NDIM(b))
        return 0;
    for (int i = 0; i < nd; i++)
        if (PyArray_DIMS(a)[i] != PyArray_DIMS(b)[i])
            return 0;
    return 1;
}

/* New references to two same-shape, C-contiguous float64 arrays (NumPy broadcasting). */
static int broadcast_pair(PyArrayObject *a, PyArrayObject *b, PyArrayObject **A, PyArrayObject **B)
{
    if (same_shape(a, b)) {
        Py_INCREF(a);
        Py_INCREF(b);
        *A = a;
        *B = b;
        return 0;
    }
    PyObject *np = PyImport_ImportModule("numpy");
    if (!np)
        return -1;
    PyObject *res = PyObject_CallMethod(np, "broadcast_arrays", "OO", (PyObject *)a, (PyObject *)b);
    Py_DECREF(np);
    if (!res)
        return -1;
    PyObject *va = PySequence_GetItem(res, 0), *vb = va ? PySequence_GetItem(res, 1) : NULL;
    Py_DECREF(res);
    if (!vb) {
        Py_XDECREF(va);
        return -1;
    }
    *A = (PyArrayObject *)PyArray_FROM_OTF(va, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    *B = *A ? (PyArrayObject *)PyArray_FROM_OTF(vb, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY) : NULL;
    Py_DECREF(va);
    Py_DECREF(vb);
    if (!*B) {
        Py_XDECREF(*A);
        return -1;
    }
    return 0;
}

static int pair(PyObject *oa, PyObject *ob, PyArrayObject **A, PyArrayObject **B)
{
    PyArrayObject *a = real_f64(oa), *b = a ? real_f64(ob) : NULL;
    int rc = -1;
    if (b)
        rc = broadcast_pair(a, b, A, B);
    Py_XDECREF(a);
    Py_XDECREF(b);
    return rc;
}

// metrics
static double mse_of(const double *a, const double *b, npy_intp n)
{
    Acc s = ACC0;
    for (npy_intp i = 0; i < n; i++) {
        double d = b[i] - a[i];
        acc_add(&s, d * d);
    }
    return n ? acc_val(&s) / (double)n : NAN;
}

static PyObject *ml_mse(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
        return nargs_error("mse", 2, nargs);
    PyArrayObject *A, *B;
    if (pair(args[0], args[1], &A, &B) < 0)
        return NULL;
    double r = mse_of(PyArray_DATA(A), PyArray_DATA(B), PyArray_SIZE(A));
    Py_DECREF(A);
    Py_DECREF(B);
    return np_f64(r);
}

static PyObject *ml_rmse(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
        return nargs_error("rmse", 2, nargs);
    PyArrayObject *A, *B;
    if (pair(args[0], args[1], &A, &B) < 0)
        return NULL;
    double r = sqrt(mse_of(PyArray_DATA(A), PyArray_DATA(B), PyArray_SIZE(A)));
    Py_DECREF(A);
    Py_DECREF(B);
    return np_f64(r);
}

static PyObject *ml_mae(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
        return nargs_error("mae", 2, nargs);
    PyArrayObject *A, *B;
    if (pair(args[0], args[1], &A, &B) < 0)
        return NULL;
    const double *a = PyArray_DATA(A), *b = PyArray_DATA(B);
    npy_intp n = PyArray_SIZE(A);
    Acc s = ACC0;
    for (npy_intp i = 0; i < n; i++)
        acc_add(&s, fabs(a[i] - b[i]));
    double r = n ? acc_val(&s) / (double)n : NAN;
    Py_DECREF(A);
    Py_DECREF(B);
    return np_f64(r);
}

/* r2(y_true, y_pred): 0.0 if the targets have zero variance */ 
static PyObject *ml_r2(PyObject *self, PyObject *const *args, Py_ssize_t nargs) // its AstroJr0
{
    if (nargs != 2)
        return nargs_error("r2", 2, nargs);
    PyArrayObject *T = real_f64(args[0]), *P = T ? real_f64(args[1]) : NULL, *A = NULL, *B = NULL;
    PyObject *res = NULL;
    if (!P)
        goto done;
    if (broadcast_pair(T, P, &A, &B) < 0)
        goto done;
    {
        const double *a = PyArray_DATA(A), *b = PyArray_DATA(B);
        npy_intp n = PyArray_SIZE(A);
        Acc res_s = ACC0;
        for (npy_intp i = 0; i < n; i++) {
            double d = a[i] - b[i];
            acc_add(&res_s, d * d);
        }
        /* total sum of squares uses y_true as given, not its broadcast copy */
        const double *t = PyArray_DATA(T);
        npy_intp nt = PyArray_SIZE(T);
        Acc ms = ACC0, tot = ACC0;
        for (npy_intp i = 0; i < nt; i++)
            acc_add(&ms, t[i]);
        double mean = nt ? acc_val(&ms) / (double)nt : NAN;
        for (npy_intp i = 0; i < nt; i++) {
            double d = t[i] - mean;
            acc_add(&tot, d * d);
        }
        double ss_tot = acc_val(&tot);
        res = ss_tot == 0 ? PyFloat_FromDouble(0.0) : np_f64(1 - acc_val(&res_s) / ss_tot);
    }
done:
    Py_XDECREF(T);
    Py_XDECREF(P);
    Py_XDECREF(A);
    Py_XDECREF(B);
    return res;
}

static PyObject *ml_bce(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3)
        return nargs_error("bce", 3, nargs);
    double eps = PyFloat_AsDouble(args[2]);
    if (eps == -1.0 && PyErr_Occurred())
        return NULL;
    PyArrayObject *A, *B; /* A = predicted probabilities, B = labels */
    if (pair(args[0], args[1], &A, &B) < 0)
        return NULL;
    const double *pr = PyArray_DATA(A), *y = PyArray_DATA(B);
    npy_intp n = PyArray_SIZE(A);
    double lo = eps, hi = 1 - eps;
    Acc s = ACC0;
    for (npy_intp i = 0; i < n; i++) {
        double p = pr[i];
        if (p < lo)
            p = lo; /* np.clip: NaN passes through, lo > hi gives hi */
        if (p > hi)
            p = hi;
        acc_add(&s, y[i] * log(p) + (1 - y[i]) * log(1 - p));
    }
    double r = n ? -(acc_val(&s) / (double)n) : NAN;
    Py_DECREF(A);
    Py_DECREF(B);
    return np_f64(r);
}

// simple linear regression
static PyObject *ml_linreg(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 2)
        return nargs_error("linear_regression_analytical", 2, nargs);
    PyArrayObject *X = as_f64(args[0]), *Y = X ? as_f64(args[1]) : NULL;
    PyObject *res = NULL;
    if (!Y)
        goto done;
    if (!same_shape(X, Y)) {
        PyErr_SetString(PyExc_ValueError, "X and y must have the same shape.");
        goto done;
    }
    {
        const double *x = PyArray_DATA(X), *y = PyArray_DATA(Y);
        npy_intp n = PyArray_SIZE(X);
        Acc sx = ACC0, sy = ACC0;
        for (npy_intp i = 0; i < n; i++) {
            acc_add(&sx, x[i]);
            acc_add(&sy, y[i]);
        }
        double xm = n ? acc_val(&sx) / (double)n : NAN, ym = n ? acc_val(&sy) / (double)n : NAN;
        Acc num = ACC0, den = ACC0;
        for (npy_intp i = 0; i < n; i++) {
            double dx = x[i] - xm;
            acc_add(&num, dx * (y[i] - ym));
            acc_add(&den, dx * dx);
        }
        double d = acc_val(&den);
        if (d == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "Cannot perform linear regression: Feature X has zero variance "
                            "(all values are the same).");
            goto done;
        }
        double m = acc_val(&num) / d;
        res = Py_BuildValue("(NN)", np_f64(m), np_f64(ym - m * xm));
    }
done:
    Py_XDECREF(X);
    Py_XDECREF(Y);
    return res;
}

/* one gradient pass: sum(x * r) and sum(r) with r = y - (m*x + b).
 * 4 independent accumulators per 512-element block (fast, pipelines well),
 * block totals added into a Neumaier sum (keeps the error ~1 ulp). */
static void gd_pass(const double *x, const double *y, npy_intp n, double m, double b,
                    double *sxr, double *sr)
{
    Acc A = ACC0, B = ACC0;
    for (npy_intp s = 0; s < n; s += 512) {
        npy_intp e = s + 512 < n ? s + 512 : n, i = s;
        double a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (; i + 4 <= e; i += 4) {
            double r0 = y[i] - (m * x[i] + b), r1 = y[i + 1] - (m * x[i + 1] + b);
            double r2 = y[i + 2] - (m * x[i + 2] + b), r3 = y[i + 3] - (m * x[i + 3] + b);
            a0 += x[i] * r0;
            a1 += x[i + 1] * r1;
            a2 += x[i + 2] * r2;
            a3 += x[i + 3] * r3;
            c0 += r0;
            c1 += r1;
            c2 += r2;
            c3 += r3;
        }
        for (; i < e; i++) {
            double r = y[i] - (m * x[i] + b);
            a0 += x[i] * r;
            c0 += r;
        }
        acc_add(&A, (a0 + a1) + (a2 + a3));
        acc_add(&B, (c0 + c1) + (c2 + c3));
    }
    *sxr = acc_val(&A);
    *sr = acc_val(&B);
}

static PyObject *ml_gradient_descent(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 4)
        return nargs_error("gradient_descent", 4, nargs);
    PyArrayObject *X = as_f64(args[0]), *Y = X ? as_f64(args[1]) : NULL;
    PyObject *res = NULL;
    if (!Y)
        goto done;
    double lr = PyFloat_AsDouble(args[2]);
    if (lr == -1.0 && PyErr_Occurred())
        goto done;
    Py_ssize_t iters = PyNumber_AsSsize_t(args[3], PyExc_OverflowError);
    if (iters == -1 && PyErr_Occurred())
        goto done;
    if (PyArray_NDIM(X) == 0) {
        PyErr_SetString(PyExc_TypeError, "len() of unsized object");
        goto done;
    }
    if (iters <= 0) {
        res = Py_BuildValue("(dd)", 0.0, 0.0);
        goto done;
    }
    npy_intp n = PyArray_DIMS(X)[0];
    if (n == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "division by zero");
        goto done;
    }
    if (!same_shape(X, Y)) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same shape.");
        goto done;
    }
    {
        const double *x = PyArray_DATA(X), *y = PyArray_DATA(Y);
        npy_intp size = PyArray_SIZE(X);
        const double c = -2.0 / (double)n;
        double m = 0.0, b = 0.0;
        Py_BEGIN_ALLOW_THREADS
        for (Py_ssize_t it = 0; it < iters; it++) {
            double sxr, sr;
            gd_pass(x, y, size, m, b, &sxr, &sr);
            double dm = c * sxr, db = c * sr;
            m -= lr * dm;
            b -= lr * db;
        }
        Py_END_ALLOW_THREADS
        res = Py_BuildValue("(NN)", np_f64(m), np_f64(b));
    }
done:
    Py_XDECREF(X);
    Py_XDECREF(Y);
    return res;
}

static PyObject *ml_predict(PyObject *self, PyObject *const *args, Py_ssize_t nargs)
{
    if (nargs != 3)
        return nargs_error("predict", 3, nargs);
    double m = PyFloat_AsDouble(args[1]);
    if (m == -1.0 && PyErr_Occurred())
        return NULL;
    double b = PyFloat_AsDouble(args[2]);
    if (b == -1.0 && PyErr_Occurred())
        return NULL;
    PyArrayObject *X = as_f64(args[0]);
    if (!X)
        return NULL;
    PyArrayObject *out =
        (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(X), PyArray_DIMS(X), NPY_DOUBLE);
    if (out) {
        const double *x = PyArray_DATA(X);
        double *o = PyArray_DATA(out);
        npy_intp n = PyArray_SIZE(X);
        for (npy_intp i = 0; i < n; i++)
            o[i] = m * x[i] + b;
    }
    Py_DECREF(X);
    return out ? PyArray_Return(out) : NULL;
}

static PyMethodDef methods[] = {
    {"mse", FAST(ml_mse), METH_FASTCALL, "mse(a, b, /)\n\nMean squared error (broadcasting)."},
    {"rmse", FAST(ml_rmse), METH_FASTCALL, "rmse(a, b, /)\n\nRoot mean squared error."},
    {"mae", FAST(ml_mae), METH_FASTCALL, "mae(a, b, /)\n\nMean absolute error."},
    {"r2", FAST(ml_r2), METH_FASTCALL, "r2(y_true, y_pred, /)\n\nR-squared; 0.0 if y_true has zero variance."},
    {"bce", FAST(ml_bce), METH_FASTCALL, "bce(pred, actual, epsilon, /)\n\nBinary cross entropy with clipping to [epsilon, 1 - epsilon]."},
    {"linear_regression", FAST(ml_linreg), METH_FASTCALL, "linear_regression(X, y, /)\n\nClosed-form (slope, intercept). X and y must have the same shape."},
    {"gradient_descent", FAST(ml_gradient_descent), METH_FASTCALL, "gradient_descent(x, y, lr, iterations, /)\n\nFit y = m*x + b by gradient descent on MSE, starting from (0, 0)."},
    {"predict", FAST(ml_predict), METH_FASTCALL, "predict(X, m, b, /)\n\nm * X + b with scalar m, b."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_ml",
                                       "C backend for matha.regression and matha.machine_learning.",
                                       0, methods, NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__ml(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
