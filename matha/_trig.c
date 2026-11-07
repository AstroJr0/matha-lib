/* Scalars only (same as the old math.* based code). in_degrees converts with
 * x * (pi / 180), exactly like to_radians(). sin/cos/tan of +-inf raise
 * ValueError("math domain error"), like math.sin.
 * Functions that were plain math.* calls are bound to math in the .py file.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

#define KW (PyCFunction)(void (*)(void))

static int angle(PyObject *ox, int deg, double *out)
{
    double x = PyFloat_AsDouble(ox);
    if (x == -1.0 && PyErr_Occurred())
        return -1;
    if (deg)
        x = x * (M_PI / 180.0);
    *out = x;
    return 0;
}

static int finite_arg(double x)
{
    if (isinf(x)) {
        PyErr_SetString(PyExc_ValueError, "math domain error");
        return -1;
    }
    return 0;
}

static int parse_arg(PyObject *args, PyObject *kw, const char *fmt, char **kwlist, double *x)
{
    PyObject *ox;
    int deg = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kw, fmt, kwlist, &ox, &deg))
        return -1;
    if (angle(ox, deg, x) < 0)
        return -1;
    return finite_arg(*x);
}

static int parse_x(PyObject *args, PyObject *kw, const char *fmt, double *x)
{
    static char *kwlist[] = {"x", "in_degrees", NULL};
    return parse_arg(args, kw, fmt, kwlist, x);
}

#define SIMPLE(name, expr)                                                      \
    static PyObject *t_##name(PyObject *self, PyObject *args, PyObject *kw)     \
    {                                                                           \
        double x;                                                               \
        if (parse_x(args, kw, "O|p:" #name, &x) < 0)                            \
            return NULL;                                                        \
        return PyFloat_FromDouble(expr);                                        \
    }

SIMPLE(sin, sin(x))
SIMPLE(cos, cos(x))
SIMPLE(tan, tan(x))

static PyObject *t_cosec(PyObject *self, PyObject *args, PyObject *kw)
{
    double x;
    if (parse_x(args, kw, "O|p:cosec", &x) < 0)
        return NULL;
    double s = sin(x);
    if (s == 0) {
        PyErr_SetString(PyExc_ValueError, "Cosecant is undefined (sin(x) is zero).");
        return NULL;
    }
    return PyFloat_FromDouble(1.0 / s);
}

static PyObject *t_sec(PyObject *self, PyObject *args, PyObject *kw)
{
    double x;
    if (parse_x(args, kw, "O|p:sec", &x) < 0)
        return NULL;
    double c = cos(x);
    if (c == 0) {
        PyErr_SetString(PyExc_ValueError, "Secant is undefined (cos(x) is zero).");
        return NULL;
    }
    return PyFloat_FromDouble(1.0 / c);
}

static PyObject *t_cot(PyObject *self, PyObject *args, PyObject *kw)
{
    double x;
    if (parse_x(args, kw, "O|p:cot", &x) < 0)
        return NULL;
    double s = sin(x);
    if (s == 0) {
        PyErr_SetString(PyExc_ValueError, "Cotangent is undefined (sin(x) is zero).");
        return NULL;
    }
    return PyFloat_FromDouble(cos(x) / s);
}

static PyObject *t_law_of_cosines(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"a", "b", "angle_c", "in_degrees", NULL};
    double a, b;
    PyObject *oc;
    int deg = 0, rc;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "ddO|p:law_of_cosines", kwlist, &a, &b, &oc, &deg))
        return NULL;
    double c;
    rc = angle(oc, deg, &c);
    if (rc < 0 || finite_arg(c) < 0)
        return NULL;
    double c_sq = a * a + b * b - 2 * a * b * cos(c);
    if (c_sq < 0)
        return PyFloat_FromDouble(0.0); // hi (no hate)
    return PyFloat_FromDouble(sqrt(c_sq));
}

static PyObject *t_pythagorean_identity(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"theta", "in_degrees", NULL};
    double x;
    if (parse_arg(args, kw, "O|p:pythagorean_identity", kwlist, &x) < 0)
        return NULL;
    double s = sin(x), c = cos(x);
    return PyFloat_FromDouble(s * s + c * c);
}

PyDoc_STRVAR(sin_doc, "sin($module, /, x, in_degrees=False)\n\nSine of x (radians, or degrees if in_degrees=True).");
PyDoc_STRVAR(cos_doc, "cos($module, /, x, in_degrees=False)\n\nCosine of x (radians, or degrees if in_degrees=True).");
PyDoc_STRVAR(tan_doc, "tan($module, /, x, in_degrees=False)\n\nTangent of x (radians, or degrees if in_degrees=True).");
PyDoc_STRVAR(cosec_doc, "cosec($module, /, x, in_degrees=False)\n\n1/sin(x). ValueError if sin(x) == 0.");
PyDoc_STRVAR(sec_doc, "sec($module, /, x, in_degrees=False)\n\n1/cos(x). ValueError if cos(x) == 0.");
PyDoc_STRVAR(cot_doc, "cot($module, /, x, in_degrees=False)\n\ncos(x)/sin(x). ValueError if sin(x) == 0.");
PyDoc_STRVAR(loc_doc, "law_of_cosines($module, /, a, b, angle_c, in_degrees=False)\n\nSide c from c^2 = a^2 + b^2 - 2ab cos(C). Negative c^2 (rounding) gives 0.0.");
PyDoc_STRVAR(pyid_doc, "pythagorean_identity($module, /, theta, in_degrees=False)\n\nsin(theta)^2 + cos(theta)^2 (1.0 up to rounding).");

static PyMethodDef methods[] = {
    {"sin", KW t_sin, METH_VARARGS | METH_KEYWORDS, sin_doc},
    {"cos", KW t_cos, METH_VARARGS | METH_KEYWORDS, cos_doc},
    {"tan", KW t_tan, METH_VARARGS | METH_KEYWORDS, tan_doc},
    {"cosec", KW t_cosec, METH_VARARGS | METH_KEYWORDS, cosec_doc},
    {"sec", KW t_sec, METH_VARARGS | METH_KEYWORDS, sec_doc},
    {"cot", KW t_cot, METH_VARARGS | METH_KEYWORDS, cot_doc},
    {"law_of_cosines", KW t_law_of_cosines, METH_VARARGS | METH_KEYWORDS, loc_doc},
    {"pythagorean_identity", KW t_pythagorean_identity, METH_VARARGS | METH_KEYWORDS, pyid_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {PyModuleDef_HEAD_INIT, "_trig",
                                       "C backend for matha.trigonometry.", 0, methods,
                                       NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__trig(void)
{
    return PyModule_Create(&moduledef);
}
