#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifndef __SIZEOF_INT128__
#error "matha._numtheory needs a compiler with __int128. Plz use GCC or Clang. Its adviced not to use MSVC. Its A Slop."
#endif

// def sum macro types
typedef int64_t i64;
typedef uint64_t u64;
typedef __int128 i128;
typedef unsigned __int128 u128;

// helpers
static int as_i64(PyObject *o, i64 *out)
{
    int ovf;
    long long v = PyLong_AsLongLongAndOverflow(o, &ovf);
    if (ovf) {
        PyErr_SetString(PyExc_OverflowError,
                        "integer outside the signed 64-bit range");
        return -1;
    }
    if (v == -1 && PyErr_Occurred())
        return -1;
    *out = (i64)v;
    return 0;
}

static PyObject *from_i128(i128 v)
{
    if (v < INT64_MIN || v > INT64_MAX) {
        PyErr_SetString(PyExc_OverflowError,
                        "result outside the signed 64-bit range");
        return NULL;
    }
    return PyLong_FromLongLong((long long)v);
}

/* Python-style floor division / modulo (b != 0) */
static i128 floordiv128(i128 a, i128 b)
{
    i128 q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        q--;
    return q;
}

static i128 floormod128(i128 a, i128 b)
{
    i128 r = a % b;
    if (r != 0 && ((r < 0) != (b < 0)))
        r += b;
    return r;
}

static u64 abs_u64(i64 v)
{
    return v < 0 ? (u64)0 - (u64)v : (u64)v;
}

static u64 gcd_u(u64 a, u64 b)
{
    while (b) {
        u64 t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static inline u64 mulmod(u64 a, u64 b, u64 m)
{
    return (u64)((u128)a * b % m);
}

static u64 powmod(u64 a, u64 e, u64 m)
{
    u64 r = 1 % m;
    a %= m;
    while (e) {
        if (e & 1)
            r = mulmod(r, a, m);
        a = mulmod(a, a, m);
        e >>= 1;
    }
    return r;
}

static u64 isqrt_u(u64 n)
{
    u64 r = (u64)sqrt((double)n);
    while (r * r > n)
        r--;
    while ((r + 1) * (r + 1) <= n)
        r++;
    return r;
}

/* ---------- primality: deterministic Miller-Rabin (exact for all u64) ---------- */

static const u64 MR_BASES[12] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};

static int is_prime_u(u64 n)
{
    if (n < 2)
        return 0;
    for (int i = 0; i < 12; i++) {
        if (n == MR_BASES[i])
            return 1;
        if (n % MR_BASES[i] == 0)
            return 0;
    }
    u64 d = n - 1;
    int s = 0;
    while (!(d & 1)) {
        d >>= 1;
        s++;
    }
    for (int i = 0; i < 12; i++) {
        u64 x = powmod(MR_BASES[i], d, n);
        if (x == 1 || x == n - 1)
            continue;
        int composite = 1;
        for (int r = 1; r < s; r++) {
            x = mulmod(x, x, n);
            if (x == n - 1) {
                composite = 0;
                break;
            }
        }
        if (composite)
            return 0;
    }
    return 1;
}

/* factorization: trial division + pollard rho */

/* n odd, composite, not a perfect square */
static u64 rho(u64 n)
{
    for (u64 c = 1;; c++) {
        u64 x = 2, y = 2, d = 1;
        while (d == 1) {
            x = (mulmod(x, x, n) + c) % n;
            y = (mulmod(y, y, n) + c) % n;
            y = (mulmod(y, y, n) + c) % n;
            d = gcd_u(x > y ? x - y : y - x, n);
        }
        if (d != n)
            return d;
    }
}

static void factor_rec(u64 n, u64 *out, int *cnt)
{
    if (n == 1)
        return;
    if (is_prime_u(n)) {
        out[(*cnt)++] = n;
        return;
    }
    u64 r = isqrt_u(n);
    u64 d = (r * r == n) ? r : rho(n);
    factor_rec(d, out, cnt);
    factor_rec(n / d, out, cnt);
}

static int cmp_u64(const void *a, const void *b)
{
    u64 x = *(const u64 *)a, y = *(const u64 *)b;
    return (x > y) - (x < y);
}

/* n >= 1. Writes primes with multiplicity, ascending. out needs 64 slots. */
static int factorize(u64 n, u64 *out)
{
    int cnt = 0;
    while (n % 2 == 0) {
        out[cnt++] = 2;
        n /= 2;
    }
    for (u64 f = 3; f < 1000 && f * f <= n; f += 2) {
        while (n % f == 0) {
            out[cnt++] = f;
            n /= f;
        }
    }
    if (n > 1)
        factor_rec(n, out, &cnt);
    qsort(out, (size_t)cnt, sizeof(u64), cmp_u64);
    return cnt;
}

/* extended gcd with python floor semantics */

static void egcd(i128 a, i128 b, i128 *g, i128 *x, i128 *y)
{
    if (a == 0) {
        *g = b;
        *x = 0;
        *y = 1;
        return;
    }
    i128 g1, x1, y1;
    egcd(floormod128(b, a), a, &g1, &x1, &y1);
    *g = g1;
    *x = y1 - floordiv128(b, a) * x1;
    *y = x1;
}

/* python facing functions */

#define KW (PyCFunction)(void (*)(void))

static PyObject *nt_is_prime(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"n", NULL};
    PyObject *o;
    i64 n;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:is_prime", kwlist, &o))
        return NULL;
    if (as_i64(o, &n) < 0)
        return NULL;
    return PyBool_FromLong(n > 1 && is_prime_u((u64)n));
}

static PyObject *nt_prime_factors(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"n", NULL};
    PyObject *o;
    i64 n;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:prime_factors", kwlist, &o))
        return NULL;
    if (as_i64(o, &n) < 0)
        return NULL;
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "Input must be a positive integer.");
        return NULL;
    }
    u64 f[64];
    int cnt;
    Py_BEGIN_ALLOW_THREADS
    cnt = factorize((u64)n, f);
    Py_END_ALLOW_THREADS

    PyObject *d = PyDict_New();
    if (!d)
        return NULL;
    for (int i = 0; i < cnt;) {
        int j = i;
        while (j < cnt && f[j] == f[i])
            j++;
        PyObject *k = PyLong_FromUnsignedLongLong(f[i]);
        PyObject *v = PyLong_FromLong(j - i);
        if (!k || !v || PyDict_SetItem(d, k, v) < 0) {
            Py_XDECREF(k);
            Py_XDECREF(v);
            Py_DECREF(d);
            return NULL;
        }
        Py_DECREF(k);
        Py_DECREF(v);
        i = j;
    }
    return d;
}

static PyObject *nt_gcd(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"a", "b", NULL};
    PyObject *oa, *ob;
    i64 a, b;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:gcd", kwlist, &oa, &ob))
        return NULL;
    if (as_i64(oa, &a) < 0 || as_i64(ob, &b) < 0)
        return NULL;
    return PyLong_FromUnsignedLongLong(gcd_u(abs_u64(a), abs_u64(b)));
}

static PyObject *nt_extended_gcd(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"a", "b", NULL};
    PyObject *oa, *ob;
    i64 a, b;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:extended_gcd", kwlist, &oa, &ob))
        return NULL;
    if (as_i64(oa, &a) < 0 || as_i64(ob, &b) < 0)
        return NULL;
    i128 g, x, y;
    egcd(a, b, &g, &x, &y);
    PyObject *pg = from_i128(g), *px = pg ? from_i128(x) : NULL,
             *py = px ? from_i128(y) : NULL;
    if (!py) {
        Py_XDECREF(pg);
        Py_XDECREF(px);
        return NULL;
    }
    PyObject *t = PyTuple_Pack(3, pg, px, py);
    Py_DECREF(pg);
    Py_DECREF(px);
    Py_DECREF(py);
    return t;
}

static PyObject *nt_lcm(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"a", "b", NULL};
    PyObject *oa, *ob;
    i64 a, b;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:lcm", kwlist, &oa, &ob))
        return NULL;
    if (as_i64(oa, &a) < 0 || as_i64(ob, &b) < 0)
        return NULL;
    if (a == 0 || b == 0)
        return PyLong_FromLong(0);
    u64 ua = abs_u64(a), ub = abs_u64(b);
    u128 r = (u128)(ua / gcd_u(ua, ub)) * ub;
    if (r > UINT64_MAX) {
        PyErr_SetString(PyExc_OverflowError, "lcm outside the 64-bit range");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong((u64)r);
}

static PyObject *nt_mod_pow(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"base", "exp", "mod", NULL};
    PyObject *ob, *oe, *om;
    i64 base, exp, mod;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OOO:mod_pow", kwlist, &ob, &oe, &om))
        return NULL;
    if (as_i64(ob, &base) < 0 || as_i64(oe, &exp) < 0 || as_i64(om, &mod) < 0)
        return NULL;
    if (exp < 0) {
        PyErr_SetString(PyExc_ValueError, "Exponent must be non-negative.");
        return NULL;
    }
    if (mod == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "integer modulo by zero");
        return NULL;
    }
    i128 result = 1, b = floormod128(base, mod);
    while (exp > 0) {
        if (exp & 1)
            result = floormod128(result * b, mod);
        exp >>= 1;
        b = floormod128(b * b, mod);
    }
    return from_i128(result);
}

static PyObject *nt_mod_inverse(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"a", "mod", NULL};
    PyObject *oa, *om;
    i64 a, mod;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "OO:mod_inverse", kwlist, &oa, &om))
        return NULL;
    if (as_i64(oa, &a) < 0 || as_i64(om, &mod) < 0)
        return NULL;
    i128 g, x, y;
    egcd(a, mod, &g, &x, &y);
    if (g != 1) {
        PyObject *pg = from_i128(g);
        if (!pg)
            return NULL;
        PyErr_Format(PyExc_ValueError,
                     "Modular inverse of %lld mod %lld does not exist (GCD is %S).",
                     (long long)a, (long long)mod, pg);
        Py_DECREF(pg);
        return NULL;
    }
    if (mod == 0) {
        PyErr_SetString(PyExc_ZeroDivisionError, "integer modulo by zero");
        return NULL;
    }
    return from_i128(floormod128(x, mod));
}

static PyObject *nt_totient(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"n", NULL};
    PyObject *o;
    i64 n;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:totient", kwlist, &o))
        return NULL;
    if (as_i64(o, &n) < 0)
        return NULL;
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "Input must be a positive integer.");
        return NULL;
    }
    u64 f[64], result = (u64)n;
    int cnt;
    Py_BEGIN_ALLOW_THREADS
    cnt = factorize((u64)n, f);
    Py_END_ALLOW_THREADS
    for (int i = 0; i < cnt; i++) {
        if (i == 0 || f[i] != f[i - 1])
            result -= result / f[i];
    }
    return PyLong_FromUnsignedLongLong(result);
}

static int perfect_u(u64 n)
{
    u128 s = 1;
    for (u64 i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            s += i;
            u64 q = n / i;
            if (q != i)
                s += q;
            if (s > n)
                return 0; /* s only grows */
        }
    }
    return s == n;
}

static PyObject *nt_is_perfect(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"n", NULL};
    PyObject *o;
    i64 n;
    int r;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:is_perfect", kwlist, &o))
        return NULL;
    if (as_i64(o, &n) < 0)
        return NULL;
    if (n < 2)
        Py_RETURN_FALSE;
    Py_BEGIN_ALLOW_THREADS
    r = perfect_u((u64)n);
    Py_END_ALLOW_THREADS
    return PyBool_FromLong(r);
}

/* odd-only sieve; p_n < n(ln n + ln ln n) for n >= 6 (Rosser) */
static u64 nth_prime_u(u64 n, int *err)
{
    if (n == 1)
        return 2;
    double dn = (double)n;
    u64 limit = n < 6 ? 15 : (u64)(dn * (log(dn) + log(log(dn)))) + 3;
    size_t m = (size_t)((limit + 1) / 2); /* index i <-> number 2i+1 */
    uint8_t *s = calloc(m, 1);
    if (!s) {
        *err = 1;
        return 0;
    }
    s[0] = 1;
    for (u64 i = 1; (2 * i + 1) * (2 * i + 1) <= limit; i++) {
        if (!s[i]) {
            u64 p = 2 * i + 1;
            for (u64 j = p * p / 2; j < m; j += p)
                s[j] = 1;
        }
    }
    u64 count = 1, ans = 0;
    for (size_t i = 1; i < m; i++) {
        if (!s[i] && ++count == n) {
            ans = 2 * (u64)i + 1;
            break;
        }
    }
    free(s);
    if (!ans)
        *err = 2;
    return ans;
}

static PyObject *nt_nth_prime(PyObject *self, PyObject *args, PyObject *kw)
{
    static char *kwlist[] = {"n", NULL};
    PyObject *o;
    i64 n;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "O:nth_prime", kwlist, &o))
        return NULL;
    if (as_i64(o, &n) < 0)
        return NULL;
    if (n < 1) {
        PyErr_SetString(PyExc_ValueError, "n must be a positive integer.");
        return NULL;
    }
    int err = 0;
    u64 p;
    Py_BEGIN_ALLOW_THREADS
    p = nth_prime_u((u64)n, &err);
    Py_END_ALLOW_THREADS
    if (err == 1)
        return PyErr_NoMemory();
    if (err == 2) {
        PyErr_SetString(PyExc_SystemError, "nth_prime: sieve bound too small");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(p);
}
PyDoc_STRVAR(is_prime_doc, "is_prime($module, /, n)\n\nReturn True if n is prime (deterministic Miller-Rabin, exact for n < 2**63).");
PyDoc_STRVAR(prime_factors_doc, "prime_factors($module, /, n)\n\nPrime factorization of a positive integer as {prime: exponent}, ascending.\nTrial division for small primes, Pollard rho for the rest. n < 2**63.");
PyDoc_STRVAR(gcd_doc, "gcd($module, /, a, b)\n\nGreatest common divisor (always >= 0).");
PyDoc_STRVAR(extended_gcd_doc, "extended_gcd($module, /, a, b)\n\nReturn (g, x, y) with a*x + b*y == g == gcd(a, b).");
PyDoc_STRVAR(lcm_doc, "lcm($module, /, a, b)\n\nLeast common multiple (0 if either argument is 0). Result must fit in 64 bits.");
PyDoc_STRVAR(mod_pow_doc, "mod_pow($module, /, base, exp, mod)\n\n(base ** exp) % mod by binary exponentiation. exp must be >= 0.");
PyDoc_STRVAR(mod_inverse_doc, "mod_inverse($module, /, a, mod)\n\nModular multiplicative inverse of a modulo mod.\nRaises ValueError if gcd(a, mod) != 1.");
PyDoc_STRVAR(totient_doc, "totient($module, /, n)\n\nEuler's totient phi(n) for a positive integer n.");
PyDoc_STRVAR(is_perfect_doc, "is_perfect($module, /, n)\n\nTrue if n equals the sum of its proper divisors (6, 28, 496, ...).");
PyDoc_STRVAR(nth_prime_doc, "nth_prime($module, /, n)\n\nThe n-th prime, 1-indexed (nth_prime(1) == 2). Uses an odd-only sieve.");

static PyMethodDef methods[] = {
    {"is_prime", KW nt_is_prime, METH_VARARGS | METH_KEYWORDS, is_prime_doc},
    {"prime_factors", KW nt_prime_factors, METH_VARARGS | METH_KEYWORDS, prime_factors_doc},
    {"gcd", KW nt_gcd, METH_VARARGS | METH_KEYWORDS, gcd_doc},
    {"extended_gcd", KW nt_extended_gcd, METH_VARARGS | METH_KEYWORDS, extended_gcd_doc},
    {"lcm", KW nt_lcm, METH_VARARGS | METH_KEYWORDS, lcm_doc},
    {"mod_pow", KW nt_mod_pow, METH_VARARGS | METH_KEYWORDS, mod_pow_doc},
    {"mod_inverse", KW nt_mod_inverse, METH_VARARGS | METH_KEYWORDS, mod_inverse_doc},
    {"totient", KW nt_totient, METH_VARARGS | METH_KEYWORDS, totient_doc},
    {"is_perfect", KW nt_is_perfect, METH_VARARGS | METH_KEYWORDS, is_perfect_doc},
    {"nth_prime", KW nt_nth_prime, METH_VARARGS | METH_KEYWORDS, nth_prime_doc},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_numtheory",
    "C backend for matha.numTheory.", 0, methods,
    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__numtheory(void)
{
    return PyModule_Create(&moduledef);
}
