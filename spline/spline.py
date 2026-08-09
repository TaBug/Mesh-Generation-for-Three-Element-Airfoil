import numpy as np
import sympy as sp
from scipy import sparse
from scipy import integrate
from scipy.sparse import linalg
import matplotlib.pyplot as plt

# Keyed by .gri boundary group index. The mesh writes the airfoil curves in the
# order main, slat, flap -- see msh2gri.py -- so group 2 is the slat and group 3
# is the flap. Getting this backwards silently projects nodes onto the wrong
# element, a full chord away.
AIRFOIL_NAMES = {1: 'main', 2: 'slat', 3: 'flap'}

SPLINE_FILES = {g: f'spline/spline_{name}.npy'
                for g, name in AIRFOIL_NAMES.items()}

# The curved boundary groups; the rest of the domain is the flat farfield.
AIRFOIL_GROUPS = tuple(sorted(SPLINE_FILES))

_cache = {}


def _loadSpline(bgroup, nSample=40):
    """Load, lambdify, and densely sample one airfoil spline (cached).

    Returns (sKnots, fx, fy, sSample, xySample), where fx[i]/fy[i] are fast
    numeric callables for segment i and xySample is a dense point cloud used to
    seed the nearest-point search.
    """
    if bgroup in _cache:
        return _cache[bgroup]
    if bgroup not in SPLINE_FILES:
        raise ValueError(f'bgroup {bgroup} is not a curved airfoil boundary')

    spline = np.load(SPLINE_FILES[bgroup], allow_pickle=True)
    sSym = sp.symbols('s')
    fx = [sp.lambdify(sSym, e, 'numpy') for e in spline[0]]
    fy = [sp.lambdify(sSym, e, 'numpy') for e in spline[1]]
    sKnots = np.asarray(spline[2], dtype=float)

    sSample, xySample = [], []
    for i in range(len(sKnots) - 1):
        lastSegment = (i == len(sKnots) - 2)
        t = np.linspace(sKnots[i], sKnots[i + 1], nSample, endpoint=lastSegment)
        sSample.append(t)
        xySample.append(np.column_stack((fx[i](t), fy[i](t))))
    sSample = np.concatenate(sSample)
    xySample = np.concatenate(xySample)

    _cache[bgroup] = (sKnots, fx, fy, sSample, xySample)
    return _cache[bgroup]


def evalSpline(bgroup, s):
    """Evaluate the spline of one airfoil at arclength parameter s."""
    sKnots, fx, fy, _, _ = _loadSpline(bgroup)
    s = float(np.clip(s, sKnots[0], sKnots[-1]))
    i = int(np.clip(np.searchsorted(sKnots, s) - 1, 0, len(fx) - 1))
    return float(fx[i](s)), float(fy[i](s))


def snapToBoundary(point, bgroup, tol=1e-12):
    """Project a point onto the true spline geometry of an airfoil element.

    Finds the nearest sampled point to seed a golden-section search on squared
    distance over the bracketing arclength interval. Purely geometric, so it
    works at any refinement level.

    Parameters
    ----------
    point : array_like
        (x, y) of the point to snap, e.g. an edge midpoint.
    bgroup : int
        Boundary group index: 1 = main, 2 = flap, 3 = slat.

    Returns
    -------
    ndarray
        (2,) array with the snapped coordinates.
    """
    sKnots, _, _, sSample, xySample = _loadSpline(bgroup)
    p = np.asarray(point, dtype=float).ravel()[:2]

    k = int(np.argmin(np.sum((xySample - p) ** 2, axis=1)))
    a = sSample[max(k - 1, 0)]
    b = sSample[min(k + 1, len(sSample) - 1)]

    def d2(s):
        x, y = evalSpline(bgroup, s)
        return (x - p[0]) ** 2 + (y - p[1]) ** 2

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = d2(c), d2(d)
    while (b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = d2(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = d2(d)

    return np.array(evalSpline(bgroup, 0.5 * (a + b)))


def solveForCoeff(s, x, dxds):
    """Build the per-segment cubic polynomials from nodal values and slopes."""
    sSym = sp.symbols('s')
    f = []
    for i in range(len(s) - 1):
        A = np.array([[s[i] ** 3, s[i] ** 2, s[i], 1],
                      [s[i + 1] ** 3, s[i + 1] ** 2, s[i + 1], 1],
                      [3 * s[i] ** 2, 2 * s[i], 1, 0],
                      [3 * s[i + 1] ** 2, 2 * s[i + 1], 1, 0]])
        b = np.array([x[i], x[i + 1], dxds[i], dxds[i + 1]])
        a = np.linalg.solve(A, b)
        f.append(a[0] * sSym ** 3 + a[1] * sSym ** 2 + a[2] * sSym + a[3])
    return f


def spline1d(x, s):
    """Solve the tridiagonal system for the nodal slopes dx/ds."""
    n = len(s)
    A = sparse.lil_matrix((n, n))

    A1 = 2 * (s[-1] - s[-2] + s[1] - s[0])
    C1 = s[-1] - s[-2]
    Bn = s[-1] - s[-2]
    An = 2 * (s[-2] - s[-3] + s[-1] - s[-2])
    A[0, 0] = A1
    A[0, 1] = C1
    A[n - 1, n - 1] = An
    A[n - 1, n - 2] = Bn

    D = np.zeros(n)
    for i in range(1, n - 1):
        delsi = s[i] - s[i - 1]
        delsim1 = s[i + 1] - s[i]
        A[i, i - 1] = delsi
        A[i, i] = 2 * (delsim1 + delsi)
        A[i, i + 1] = delsim1
        D[i] = 3 * ((x[i] - x[i - 1]) * delsi / delsim1
                    + (x[i + 1] - x[i]) * delsim1 / delsi)

    return linalg.spsolve(A.tocsr(), D)


def spline2d(coord):
    """Fit an arclength-parameterized cubic spline through a set of points."""
    x, y = coord[:, 0], coord[:, 1]

    s = np.zeros(len(x))
    for i in range(len(x) - 1):
        s[i + 1] = s[i] + np.hypot(x[i + 1] - x[i], y[i + 1] - y[i])

    sTrue = np.zeros(len(s))
    L1 = np.inf
    while L1 > 1e-10:
        dxds = spline1d(x, s)
        dyds = spline1d(y, s)

        for i in range(len(s) - 1):
            dels_i = (s[-1] - s[-2]) if i == 0 else (s[i] - s[i - 1])
            xp0 = (dxds[i] - (x[i + 1] - x[i]) / dels_i) * dels_i
            xp1 = (dxds[i + 1] - (x[i + 1] - x[i]) / dels_i) * dels_i
            yp0 = (dyds[i] - (y[i + 1] - y[i]) / dels_i) * dels_i
            yp1 = (dyds[i + 1] - (y[i + 1] - y[i]) / dels_i) * dels_i
            f_i = lambda t: np.sqrt(
                (x[i + 1] - x[i] + (1 - 4 * t + 3 * t ** 2) * xp0 + (-2 * t + 3 * t ** 2) * xp1) ** 2
                + (y[i + 1] - y[i] + (1 - 4 * t + 3 * t ** 2) * yp0 + (-2 * t + 3 * t ** 2) * yp1) ** 2)
            sTrue[i + 1] = sTrue[i] + integrate.quad(f_i, 0.0, 1.0)[0]

        L1 = np.sum(np.abs(s - sTrue))
        s = sTrue.copy()

    return np.array([solveForCoeff(s, x, dxds),
                     solveForCoeff(s, y, dyds), s], dtype=object)


def test():
    """Plot segment midpoints of the main-element spline as a sanity check."""
    sKnots, fx, fy, _, _ = _loadSpline(1)
    mid = np.array([[fx[i](0.5 * (sKnots[i] + sKnots[i + 1])),
                     fy[i](0.5 * (sKnots[i] + sKnots[i + 1]))]
                    for i in range(len(fx))])
    plt.plot(mid[:, 0], mid[:, 1], '.')
    plt.axis('equal')
    plt.show()


def main():
    """Refit every airfoil spline from its point file."""
    for g, name in sorted(AIRFOIL_NAMES.items()):
        pts = np.loadtxt(f'geometries/{name}.txt')
        np.save(SPLINE_FILES[g], spline2d(pts))
        print(f'group {g}: {name} -> {SPLINE_FILES[g]}')


if __name__ == "__main__":
    main()
    # test()
