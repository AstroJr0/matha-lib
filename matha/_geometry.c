/* Points are any sequence indexable as p[0], p[1] (tuple, list, ndarray) and
 * are computed in double precision. slope / centroid unpack exactly two
 * coordinates per point, like the old `x, y = point`.
 * plot_polygon stays in Python (Matplotlib).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

#define KW (PyCFunction)(void (*)(void))

// point access
static int get_double(PyObject *o, double *out)
{
    double v = PyFloat_AsDouble(o);
    if (v == -1.0 && PyErr_Occurred())
        return -1;
    *out = v;
    return 0;
}

// p[0], p[1]
static int xy_index(PyObject *p, double *x, double *y)
{
    PyObject *a = PySequence_GetItem(p, 0);
    if (!a)
        return -1;
    int rc = get_double(a, x);
    Py_DECREF(a);
    if (rc < 0)
        return -1;
    PyObject *b = PySequence_GetItem(p, 1);
    if (!b)
        return -1;
    rc = get_double(b, y);
    Py_DECREF(b);
    return rc;
}

// x, y = p
static int xy_unpack(PyObject *p, double *x, double *y)
{
    PyObject *s = PySequence_Fast(p, "cannot unpack non-iterable point");
    if (!s)
        return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(s);
    int rc = -1;
    if (n < 2)
        PyErr_Format(PyExc_ValueError, "not enough values to unpack (expected 2, got %zd)", n);
    else if (n > 2)
        PyErr_SetString(PyExc_ValueError, "too many values to unpack (expected 2)");
    else
        rc = (get_double(PySequence_Fast_GET_ITEM(s, 0), x) < 0 ||
              get_double(PySequence_Fast_GET_ITEM(s, 1), y) < 0) ? -1 : 0;
    Py_DECREF(s);
    return rc;
}

typedef struct {
    double *x, *y;
    Py_ssize_t n;
} Poly;

static int poly_load(PyObject *points, int unpack, Poly *P)
{
    P->x = P->y = NULL;
    P->n = 0;
    PyObject *seq = PySequence_Fast(points, "points must be a sequence of (x, y) points");
    if (!seq)
        return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    double *buf = PyMem_Malloc((size_t)(2 * n + 1) * sizeof(double));
    if (!buf) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *it = PySequence_Fast_GET_ITEM(seq, i);
        Py_INCREF(it);
        int rc = unpack ? xy_unpack(it, &buf[i], &buf[n + i]) : xy_index(it, &buf[i], &buf[n + i]);
        Py_DECREF(it);
        if (rc < 0) {
            PyMem_Free(buf);
            Py_DECREF(seq);
            return -1;
        }
    }
    Py_DECREF(seq);
    P->x = buf;
    P->y = buf + n;
    P->n = n;
    return 0;
}

static void poly_free(Poly *P)
{
    PyMem_Free(P->x); /* x is the start of the single allocation */
}

// Neumaier sum (what sum() does for floats in CPython >= 3.12)
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

// basic
static PyObject *g_distance(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"p1", "p2", NULL};
    PyObject *o1, *o2;
    double x1, y1, x2, y2;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:distance", kwlist, &o1, &o2) ||
        xy_index(o1, &x1, &y1) < 0 || xy_index(o2, &x2, &y2) < 0)
        return NULL;
    double dx = x2 - x1, dy = y2 - y1;
    return PyFloat_FromDouble(sqrt(dx * dx + dy * dy));
}

static PyObject *g_midpoint(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"p1", "p2", NULL};
    PyObject *o1, *o2;
    double x1, y1, x2, y2;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:midpoint", kwlist, &o1, &o2) ||
        xy_index(o1, &x1, &y1) < 0 || xy_index(o2, &x2, &y2) < 0)
        return NULL;
    return Py_BuildValue("(dd)", (x1 + x2) / 2, (y1 + y2) / 2);
}

static PyObject *g_slope(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"points", NULL};
    PyObject *op;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:slope", kwlist, &op))
        return NULL;
    Py_ssize_t n = PySequence_Size(op);
    if (n < 0)
        return NULL;
    if (n < 2) {
        PyErr_SetString(PyExc_ValueError, "At least two points are required to calculate slope.");
        return NULL;
    }
    PyObject *p0 = PySequence_GetItem(op, 0), *p1 = p0 ? PySequence_GetItem(op, 1) : NULL;
    double x1, y1, x2, y2;
    int rc = (p1 && xy_unpack(p0, &x1, &y1) == 0 && xy_unpack(p1, &x2, &y2) == 0) ? 0 : -1;
    Py_XDECREF(p0);
    Py_XDECREF(p1);
    if (rc < 0)
        return NULL;
    if (x2 == x1)
        return PyFloat_FromDouble(INFINITY); /* vertical line */
    return PyFloat_FromDouble((y2 - y1) / (x2 - x1));
}

// polygons
static PyObject *g_perimeter(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"points", NULL};
    PyObject *op;
    Poly P;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:perimeter", kwlist, &op) ||
        poly_load(op, 0, &P) < 0)
        return NULL;
    double peri = 0.0;
    for (Py_ssize_t i = 0; i < P.n; i++) {
        Py_ssize_t j = (i + 1) % P.n;
        double dx = P.x[j] - P.x[i], dy = P.y[j] - P.y[i];
        peri += sqrt(dx * dx + dy * dy);
    }
    poly_free(&P);
    return PyFloat_FromDouble(peri);
}

static PyObject *g_area(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"points", NULL};
    PyObject *op;
    Poly P;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:area", kwlist, &op) ||
        poly_load(op, 0, &P) < 0)
        return NULL;
    Acc s1 = {0, 0, 0}, s2 = {0, 0, 0};
    for (Py_ssize_t i = 0; i < P.n; i++) {
        Py_ssize_t j = (i + 1) % P.n;
        acc_add(&s1, P.x[i] * P.y[j]);
        acc_add(&s2, P.y[i] * P.x[j]);
    }
    double r = fabs(acc_val(&s1) - acc_val(&s2)) / 2;
    poly_free(&P);
    return PyFloat_FromDouble(r);
}

static PyObject *g_centroid(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"points", NULL};
    PyObject *op;
    Poly P;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:centroid", kwlist, &op) ||
        poly_load(op, 1, &P) < 0)
        return NULL;
    double A = 0.0, Cx = 0.0, Cy = 0.0;
    for (Py_ssize_t i = 0; i < P.n; i++) {
        Py_ssize_t j = (i + 1) % P.n;
        double cross = P.x[i] * P.y[j] - P.x[j] * P.y[i];
        A += cross;
        Cx += (P.x[i] + P.x[j]) * cross;
        Cy += (P.y[i] + P.y[j]) * cross;
    }
    poly_free(&P);
    A *= 0.5;
    if (A == 0) {
        PyErr_SetString(PyExc_ValueError, "Polygon has zero area (points might be collinear).");
        return NULL;
    }
    return Py_BuildValue("(dd)", Cx / (6.0 * A), Cy / (6.0 * A));
}

// circle
static PyObject *g_circle_area(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"radius", NULL};
    double r;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "d:circle_area", kwlist, &r))
        return NULL;
    return PyFloat_FromDouble(M_PI * (r * r));
}

static PyObject *g_circle_circumference(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"radius", NULL};
    double r;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "d:circle_circumference", kwlist, &r))
        return NULL;
    return PyFloat_FromDouble(2 * M_PI * r);
}

PyDoc_STRVAR(distance_doc, "distance($module, /, p1, p2)\n\nEuclidean distance between (x, y) points.");
PyDoc_STRVAR(midpoint_doc, "midpoint($module, /, p1, p2)\n\nMidpoint (mx, my) of two (x, y) points.");
PyDoc_STRVAR(slope_doc, "slope($module, /, points)\n\nSlope of the line through the first two points; inf for a vertical line.");
PyDoc_STRVAR(perimeter_doc, "perimeter($module, /, points)\n\nPerimeter of a polygon given as ordered (x, y) vertices.");
PyDoc_STRVAR(area_doc, "area($module, /, points)\n\nPolygon area by the shoelace formula.");
PyDoc_STRVAR(centroid_doc, "centroid($module, /, points)\n\nPolygon centroid (Cx, Cy). ValueError if the area is zero.");
PyDoc_STRVAR(circle_area_doc, "circle_area($module, /, radius)\n\npi * r**2.");
PyDoc_STRVAR(circle_circ_doc, "circle_circumference($module, /, radius)\n\n2 * pi * r.");

static PyMethodDef methods[] = {
    {"distance", KW g_distance, METH_VARARGS | METH_KEYWORDS, distance_doc},
    {"midpoint", KW g_midpoint, METH_VARARGS | METH_KEYWORDS, midpoint_doc},
    {"slope", KW g_slope, METH_VARARGS | METH_KEYWORDS, slope_doc},
    {"perimeter", KW g_perimeter, METH_VARARGS | METH_KEYWORDS, perimeter_doc},
    {"area", KW g_area, METH_VARARGS | METH_KEYWORDS, area_doc},
    {"centroid", KW g_centroid, METH_VARARGS | METH_KEYWORDS, centroid_doc},
    {"circle_area", KW g_circle_area, METH_VARARGS | METH_KEYWORDS, circle_area_doc},
    {"circle_circumference", KW g_circle_circumference, METH_VARARGS | METH_KEYWORDS, circle_circ_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_geometry",
                                       "C backend for matha.geometry.", 0, methods,
                                       NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__geometry(void)
{
    return PyModule_Create(&moduledef);
}
