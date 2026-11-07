/*
 * this is a C backend for matha.stats
 *
 * Data (list / tuple / 1-D ndarray) is converted to float64, so results are
 * always float (data_range([1, 5]) -> 4.0). Sums use Neumaier compensation,
 * like sum() in CPython >= 3.12. mode() and corr_matrix() stay in Python.
 *
 * Distributions: exact small-n formulas (same arithmetic as the old Python),
 * lgamma-based log-space for large n / k so they no longer overflow.
 * p must be in [0, 1] and lam >= 0 (ValueError otherwise).
 */
#include "_npc.h"
#include <math.h>
#include <string.h>

#define KW (PyCFunction)(void (*)(void))

// data conversion

typedef struct {
    double *p;
    Py_ssize_t n;
    double *heap;
    PyArrayObject *arr;
} Vec;

static int vec_get(PyObject *o, Vec *v)
{
    v->p = NULL;
    v->n = 0;
    v->heap = NULL;
    v->arr = NULL;
    if (PyArray_Check(o)) {
        PyArrayObject *a = as_f64(o);
        if (!a)
            return -1;
        if (PyArray_NDIM(a) != 1) {
            Py_DECREF(a);
            PyErr_SetString(PyExc_ValueError, "data must be 1-D");
            return -1;
        }
        v->arr = a;
        v->p = PyArray_DATA(a);
        v->n = PyArray_SIZE(a);
        return 0;
    }
    PyObject *seq = PySequence_Fast(o, "data must be a sequence of numbers");
    if (!seq)
        return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    double *buf = PyMem_Malloc((size_t)n * sizeof(double) + 1);
    if (!buf) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        if (i >= PySequence_Fast_GET_SIZE(seq)) {
            PyErr_SetString(PyExc_RuntimeError, "sequence changed size during iteration");
            goto fail;
        }
        PyObject *it = PySequence_Fast_GET_ITEM(seq, i);
        Py_INCREF(it);
        double x = PyFloat_AsDouble(it);
        Py_DECREF(it);
        if (x == -1.0 && PyErr_Occurred())
            goto fail;
        buf[i] = x;
    }
    Py_DECREF(seq);
    v->heap = buf;
    v->p = buf;
    v->n = n;
    return 0;
fail:
    PyMem_Free(buf);
    Py_DECREF(seq);
    return -1;
}

static void vec_free(Vec *v)
{
    PyMem_Free(v->heap);
    Py_XDECREF(v->arr);
}

// Neumaier compensated accumulator

typedef struct {
    double s, c, plain;
} Acc;

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

static double sum_of(const Vec *v)
{
    Acc a = {0, 0, 0};
    for (Py_ssize_t i = 0; i < v->n; i++)
        acc_add(&a, v->p[i]);
    return acc_val(&a);
}

static double mean_of(const Vec *v)
{
    return v->n == 0 ? NAN : sum_of(v) / (double)v->n;
}

static double var_of(const Vec *v, int sample)
{
    Py_ssize_t n = v->n;
    if (n < 1 || (sample && n < 2))
        return NAN;
    double mu = mean_of(v);
    Acc a = {0, 0, 0};
    for (Py_ssize_t i = 0; i < n; i++) {
        double d = v->p[i] - mu;
        acc_add(&a, d * d);
    }
    return acc_val(&a) / (double)(sample ? n - 1 : n);
}

// sorting or percentile

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    int nx = isnan(x), ny = isnan(y);
    if (nx || ny)
        return nx - ny; /* NaN sorts last */
    return (x > y) - (x < y);
}

static double *sorted_copy(const Vec *v)
{
    double *d = PyMem_Malloc((size_t)v->n * sizeof(double) + 1);
    if (!d) {
        PyErr_NoMemory();
        return NULL;
    }
    memcpy(d, v->p, (size_t)v->n * sizeof(double));
    qsort(d, (size_t)v->n, sizeof(double), cmp_d);
    return d;
}

static int at(Py_ssize_t idx, Py_ssize_t n, Py_ssize_t *out)
{
    if (idx < 0)
        idx += n;
    if (idx < 0 || idx >= n) {
        PyErr_SetString(PyExc_IndexError, "list index out of range");
        return -1;
    }
    *out = idx;
    return 0;
}

static int pct_sorted(const double *d, Py_ssize_t n, double p, double *res)
{
    double k = (double)(n - 1) * (p / 100.0);
    if (isnan(k)) {
        PyErr_SetString(PyExc_ValueError, "cannot convert float NaN to integer");
        return -1;
    }
    if (isinf(k)) {
        PyErr_SetString(PyExc_OverflowError, "cannot convert float infinity to integer");
        return -1;
    }
    double fk = floor(k), ck = ceil(k);
    if (fabs(fk) > 4e18 || fabs(ck) > 4e18) {
        PyErr_SetString(PyExc_IndexError, "list index out of range");
        return -1;
    }
    Py_ssize_t f, c;
    if (fk == ck) {
        if (at((Py_ssize_t)k, n, &f) < 0)
            return -1;
        *res = d[f];
        return 0;
    }
    if (at((Py_ssize_t)fk, n, &f) < 0 || at((Py_ssize_t)ck, n, &c) < 0)
        return -1;
    *res = d[f] * (ck - k) + d[c] * (k - fk);
    return 0;
}

// python facing -> descriptive stats

static PyObject *s_mean(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", NULL};
    PyObject *o;
    Vec v;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:mean", kwlist, &o) || vec_get(o, &v) < 0)
        return NULL;
    double r = mean_of(&v);
    vec_free(&v);
    return PyFloat_FromDouble(r);
}

static PyObject *s_median(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", NULL};
    PyObject *o;
    Vec v;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:median", kwlist, &o) || vec_get(o, &v) < 0)
        return NULL;
    PyObject *res = NULL;
    if (v.n == 0) {
        PyErr_SetString(PyExc_IndexError, "list index out of range");
    }
    else {
        double *d = sorted_copy(&v);
        if (d) {
            Py_ssize_t mid = v.n / 2;
            res = PyFloat_FromDouble(v.n % 2 == 0 ? (d[mid - 1] + d[mid]) / 2 : d[mid]);
            PyMem_Free(d);
        }
    }
    vec_free(&v);
    return res;
}

static PyObject *s_data_range(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", NULL};
    PyObject *o;
    Vec v;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:data_range", kwlist, &o) || vec_get(o, &v) < 0)
        return NULL;
    double r = NAN;
    if (v.n > 0) {
        double lo = v.p[0], hi = v.p[0];
        for (Py_ssize_t i = 1; i < v.n; i++) {
            if (v.p[i] < lo)
                lo = v.p[i];
            if (v.p[i] > hi)
                hi = v.p[i];
        }
        r = hi - lo;
    }
    vec_free(&v);
    return PyFloat_FromDouble(r);
}

static PyObject *s_variance(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", "sample", NULL};
    PyObject *o;
    int sample = 1;
    Vec v;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|p:variance", kwlist, &o, &sample) ||
        vec_get(o, &v) < 0)
        return NULL;
    double r = var_of(&v, sample);
    vec_free(&v);
    return PyFloat_FromDouble(r);
}

static PyObject *s_std_dev(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", "sample", NULL};
    PyObject *o;
    int sample = 1;
    Vec v;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|p:std_dev", kwlist, &o, &sample) ||
        vec_get(o, &v) < 0)
        return NULL;
    double r = sqrt(var_of(&v, sample));
    vec_free(&v);
    return PyFloat_FromDouble(r);
}

static PyObject *s_z_score(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"x", "data", "sample", NULL};
    double x;
    PyObject *o;
    int sample = 1;
    Vec v;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "dO|p:z_score", kwlist, &x, &o, &sample) ||
        vec_get(o, &v) < 0)
        return NULL;
    double mu = mean_of(&v), sigma = sqrt(var_of(&v, sample));
    vec_free(&v);
    return PyFloat_FromDouble(sigma == 0 ? NAN : (x - mu) / sigma);
}

// quantiles 

static PyObject *s_percentile(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", "p", NULL};
    PyObject *o, *res = NULL;
    double p, r;
    Vec v;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "Od:percentile", kwlist, &o, &p) ||
        vec_get(o, &v) < 0)
        return NULL;
    double *d = sorted_copy(&v);
    if (d) {
        if (pct_sorted(d, v.n, p, &r) == 0)
            res = PyFloat_FromDouble(r);
        PyMem_Free(d);
    }
    vec_free(&v);
    return res;
}

static int quartiles_of(PyObject *o, double q[3])
{
    Vec v;
    if (vec_get(o, &v) < 0)
        return -1;
    int rc = -1;
    double *d = sorted_copy(&v);
    if (d) {
        rc = (pct_sorted(d, v.n, 25, &q[0]) | pct_sorted(d, v.n, 50, &q[1]) |
              pct_sorted(d, v.n, 75, &q[2]));
        if (rc != 0)
            rc = -1;
        PyMem_Free(d);
    }
    vec_free(&v);
    return rc;
}

static PyObject *s_quartiles(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", NULL};
    PyObject *o;
    double q[3];
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:quartiles", kwlist, &o) ||
        quartiles_of(o, q) < 0)
        return NULL;
    return Py_BuildValue("(ddd)", q[0], q[1], q[2]);
}

static PyObject *s_iqr(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"data", NULL};
    PyObject *o;
    double q[3];
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:iqr", kwlist, &o) || quartiles_of(o, q) < 0)
        return NULL;
    return PyFloat_FromDouble(q[2] - q[0]);
}

// correlation

static int cov_of(const Vec *x, const Vec *y, double *out)
{
    if (x->n != y->n) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same length.");
        return -1;
    }
    Py_ssize_t n = x->n;
    if (n < 2) {
        *out = NAN;
        return 0;
    }
    double mx = mean_of(x), my = mean_of(y);
    Acc a = {0, 0, 0};
    for (Py_ssize_t i = 0; i < n; i++)
        acc_add(&a, (x->p[i] - mx) * (y->p[i] - my));
    *out = acc_val(&a) / (double)(n - 1);
    return 0;
}

static PyObject *cov_corr(PyObject *args, PyObject *kw, int corr)
{
    static char *kwlist[] = {"x", "y", NULL};
    PyObject *ox, *oy, *res = NULL;
    Vec x, y;
    if (!PyArg_ParseTupleAndKeywords(args, kw, corr ? "OO:correlation" : "OO:covariance",
                                     kwlist, &ox, &oy))
        return NULL;
    if (vec_get(ox, &x) < 0)
        return NULL;
    if (vec_get(oy, &y) < 0) {
        vec_free(&x);
        return NULL;
    }
    double r;
    if (corr) {
        double sx = sqrt(var_of(&x, 1)), sy = sqrt(var_of(&y, 1));
        if (sx == 0 || sy == 0) {
            res = PyFloat_FromDouble(NAN); /* undefined for zero variance */
        }
        else if (cov_of(&x, &y, &r) == 0) {
            res = PyFloat_FromDouble(r / (sx * sy));
        }
    }
    else if (cov_of(&x, &y, &r) == 0) {
        res = PyFloat_FromDouble(r);
    }
    vec_free(&x);
    vec_free(&y);
    return res;
}

static PyObject *s_covariance(PyObject *self, PyObject *args, PyObject *kw)
{
    return cov_corr(args, kw, 0);
}

static PyObject *s_correlation(PyObject *self, PyObject *args, PyObject *kw)
{
    return cov_corr(args, kw, 1);
}

// discrete distributions

static int check_p(double p)
{
    if (!(p >= 0.0 && p <= 1.0)) {
        PyErr_SetString(PyExc_ValueError, "p must be in [0, 1].");
        return -1;
    }
    return 0;
}

static double binom_pmf_d(long long k, long long n, double p)
{
    if (k < 0 || k > n)
        return 0.0;
    if (p == 0.0)
        return k == 0 ? 1.0 : 0.0;
    if (p == 1.0)
        return k == n ? 1.0 : 0.0;
    if (n <= 60) { /* exact comb, same arithmetic order as comb * p**k * (1-p)**(n-k) */
        unsigned long long c = 1;
        long long m = k < n - k ? k : n - k;
        for (long long i = 0; i < m; i++)
            c = c * (unsigned long long)(n - i) / (unsigned long long)(i + 1);
        return (double)c * pow(p, (double)k) * pow(1 - p, (double)(n - k));
    }
    double lg = lgamma((double)n + 1) - lgamma((double)k + 1) - lgamma((double)(n - k) + 1) +
                (double)k * log(p) + (double)(n - k) * log1p(-p);
    return exp(lg);
}

static int get_kn(PyObject *args, PyObject *kw, const char *fmt, long long *k, long long *n,
                  double *p)
{
    static char *kwlist[] = {"k", "n", "p", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kw, fmt, kwlist, k, n, p))
        return -1;
    if (*k < 0 || *n < 0) {
        PyErr_SetString(PyExc_ValueError,
                        *n < 0 ? "n must be a non-negative integer"
                               : "k must be a non-negative integer");
        return -1;
    }
    return 0;
}

static PyObject *s_binomial_pmf(PyObject *self, PyObject *args, PyObject *kw)
{
    long long k, n;
    double p;
    if (get_kn(args, kw, "LLd:binomial_pmf", &k, &n, &p) < 0 || check_p(p) < 0)
        return NULL;
    return PyFloat_FromDouble(binom_pmf_d(k, n, p));
}

static PyObject *s_binomial_cdf(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"k", "n", "p", NULL};
    long long k, n;
    double p;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "LLd:binomial_cdf", kwlist, &k, &n, &p))
        return NULL;
    if (k < 0)
        return PyFloat_FromDouble(0.0);
    if (k >= n)
        return PyFloat_FromDouble(1.0);
    if (check_p(p) < 0)
        return NULL;
    Acc a = {0, 0, 0};
    for (long long i = 0; i <= k; i++)
        acc_add(&a, binom_pmf_d(i, n, p));
    return PyFloat_FromDouble(acc_val(&a));
}

static PyObject *s_geometric_pmf(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"k", "p", NULL};
    long long k;
    double p;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "Ld:geometric_pmf", kwlist, &k, &p))
        return NULL;
    if (k < 1)
        return PyFloat_FromDouble(0.0);
    if (check_p(p) < 0)
        return NULL;
    return PyFloat_FromDouble(pow(1 - p, (double)(k - 1)) * p);
}

static PyObject *s_geometric_cdf(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"k", "p", NULL};
    long long k;
    double p;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "Ld:geometric_cdf", kwlist, &k, &p))
        return NULL;
    if (k < 1)
        return PyFloat_FromDouble(0.0);
    if (check_p(p) < 0)
        return NULL;
    return PyFloat_FromDouble(1 - pow(1 - p, (double)k));
}

static double poisson_pmf_d(long long k, double lam)
{
    if (k < 0)
        return 0.0;
    if (lam == 0.0)
        return k == 0 ? 1.0 : 0.0;
    if (k <= 20) { /* exact factorial, same arithmetic as the old Python */
        unsigned long long f = 1;
        for (long long i = 2; i <= k; i++)
            f *= (unsigned long long)i;
        return (pow(lam, (double)k) * exp(-lam)) / (double)f;
    }
    return exp((double)k * log(lam) - lam - lgamma((double)k + 1));
}

static PyObject *s_poisson_pmf(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"k", "lam", NULL};
    long long k;
    double lam;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "Ld:poisson_pmf", kwlist, &k, &lam))
        return NULL;
    if (k < 0)
        return PyFloat_FromDouble(0.0);
    if (!(lam >= 0.0)) {
        PyErr_SetString(PyExc_ValueError, "lam must be non-negative.");
        return NULL;
    }
    return PyFloat_FromDouble(poisson_pmf_d(k, lam));
}

static PyObject *s_poisson_cdf(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"k", "lam", NULL};
    long long k;
    double lam;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "Ld:poisson_cdf", kwlist, &k, &lam))
        return NULL;
    if (k < 0)
        return PyFloat_FromDouble(0.0);
    if (!(lam >= 0.0)) {
        PyErr_SetString(PyExc_ValueError, "lam must be non-negative.");
        return NULL;
    }
    Acc a = {0, 0, 0};
    for (long long i = 0; i <= k; i++)
        acc_add(&a, poisson_pmf_d(i, lam));
    return PyFloat_FromDouble(acc_val(&a));
}

// continuous distributions

static const char *SIGMA_MSG = "Standard deviation (sigma) must be positive.";

static inline double npdf(double x, double mu, double sigma)
{
    double factor = 1 / (sigma * sqrt(2 * M_PI));
    double d = x - mu;
    return factor * exp(-(d * d) / (2 * sigma * sigma));
}

static PyObject *s_normal_pdf(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"x", "mu", "sigma", NULL};
    PyObject *ox;
    double mu = 0, sigma = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O|dd:normal_pdf", kwlist, &ox, &mu, &sigma))
        return NULL;
    if (sigma <= 0) {
        PyErr_SetString(PyExc_ValueError, SIGMA_MSG);
        return NULL;
    }
    if (PyArray_Check(ox)) { /* vectorised use: ndarray in, ndarray out */
        PyArrayObject *a = as_f64(ox);
        if (!a)
            return NULL;
        PyArrayObject *out =
            (PyArrayObject *)PyArray_SimpleNew(PyArray_NDIM(a), PyArray_DIMS(a), NPY_DOUBLE);
        if (out) {
            const double *pa = PyArray_DATA(a);
            double *po = PyArray_DATA(out);
            npy_intp n = PyArray_SIZE(a);
            for (npy_intp i = 0; i < n; i++)
                po[i] = npdf(pa[i], mu, sigma);
        }
        Py_DECREF(a);
        return (PyObject *)out;
    }
    double x = PyFloat_AsDouble(ox);
    if (x == -1.0 && PyErr_Occurred())
        return NULL;
    return PyFloat_FromDouble(npdf(x, mu, sigma));
}

static PyObject *s_normal_cdf(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"x", "mu", "sigma", NULL};
    double x, mu = 0, sigma = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "d|dd:normal_cdf", kwlist, &x, &mu, &sigma))
        return NULL;
    if (sigma <= 0) {
        PyErr_SetString(PyExc_ValueError, SIGMA_MSG);
        return NULL;
    }
    return PyFloat_FromDouble(0.5 * (1 + erf((x - mu) / (sigma * sqrt(2.0)))));
}

// module

#define DOC(name, sig, text) PyDoc_STRVAR(name##_doc, #name "($module, /, " sig ")\n--\n\n" text)
DOC(mean, "data", "Arithmetic mean (nan for empty data).");
DOC(median, "data", "Median. Raises IndexError for empty data.");
DOC(data_range, "data", "max - min (nan for empty data).");
DOC(variance, "data, sample=True", "Sample (n-1) or population (n) variance; nan if too few points.");
DOC(std_dev, "data, sample=True", "Square root of variance().");
DOC(z_score, "x, data, sample=True", "(x - mean) / std_dev; nan if std_dev is 0.");
DOC(percentile, "data, p", "p-th percentile (0..100), linear interpolation (type 7, like NumPy / R).");
DOC(quartiles, "data", "(Q1, Q2, Q3).");
DOC(iqr, "data", "Interquartile range Q3 - Q1.");
DOC(covariance, "x, y", "Sample covariance. ValueError if lengths differ; nan for n < 2.");
DOC(correlation, "x, y", "Pearson correlation; nan if either input has zero variance.");
DOC(binomial_pmf, "k, n, p", "P(X = k) for Binomial(n, p). p in [0, 1].");
DOC(binomial_cdf, "k, n, p", "P(X <= k) for Binomial(n, p).");
DOC(geometric_pmf, "k, p", "P(X = k): first success on trial k (k >= 1).");
DOC(geometric_cdf, "k, p", "P(X <= k) = 1 - (1-p)**k.");
DOC(poisson_pmf, "k, lam", "P(X = k) for Poisson(lam), lam >= 0.");
DOC(poisson_cdf, "k, lam", "P(X <= k) for Poisson(lam).");
DOC(normal_pdf, "x, mu=0, sigma=1", "Normal density. x may be a float or an ndarray (returns ndarray).");
DOC(normal_cdf, "x, mu=0, sigma=1", "Normal cumulative distribution via erf.");

#define M(name) {#name, KW s_##name, METH_VARARGS | METH_KEYWORDS, name##_doc}

static PyMethodDef methods[] = {
    M(mean),          M(median),        M(data_range),    M(variance),     M(std_dev),
    M(z_score),       M(percentile),    M(quartiles),     M(iqr),          M(covariance),
    M(correlation),   M(binomial_pmf),  M(binomial_cdf),  M(geometric_pmf), M(geometric_cdf),
    M(poisson_pmf),   M(poisson_cdf),   M(normal_pdf),    M(normal_cdf),   {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_stats",
                                       "C backend for matha.stats.", 0, methods,
                                       NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__stats(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
