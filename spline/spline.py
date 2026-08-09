import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import sympy as sp
import matplotlib.pyplot as plt
from scipy import integrate


# def l_i(p, x):
#     if p == 0:
#         l_ix = 1
#     elif p == 1:
#         l_ix = x
#     elif p == 2:
#         l_ix = 3 / 2 * x ** 2 - 1 / 2
#     else:
#         l_ix = 3 / 2 * x ** 3 - 1 / 2 * x
#     return l_ix
#
#
# def l_i_int(p, x):
#     if p == 0:
#         l_ix_int = x
#     elif p == 1:
#         l_ix_int = 1 / 2 * x ** 2
#     elif p == 2:
#         l_ix_int = 1 / 2 * x ** 3 - 1 / 2 * x
#     else:
#         l_ix_int = 5 / 8 * x ** 4 - 3 / 4 * x ** 2
#     return l_ix_int
#
#
# def quad(f, x1, x2):
#     A = np.zeros([4, 2])
#     x = np.array([x1, x2])
#     b = np.zeros([4, 1])
#     for i in range(2):
#         for j in range(2):
#             A[i, j] = l_i(i, x[j])
#         b[i] = l_i_int(i, 1) - l_i_int(i, -1)
#
#     w = np.dot(np.linalg.pinv(A), b)
#     output = 0
#     t = sp.symbols('t')
#     for q in range(len(w)):
#         output += f.subs(t, )

def slapToBoundary(B, bgroup, ib):
    if bgroup > 3:
        print('Warning! Wrong boundary to manipulate')

    if bgroup == 1:
        bIndex = ib[0][0]
        spline = np.load('spline/spline_main.npy', allow_pickle=True)
    elif bgroup == 2:
        bIndex = ib[0][0] - len(B[0])
        spline = np.load('spline/spline_flap.npy', allow_pickle=True)
    else:
        bIndex = ib[0][0] - len(B[0]) - len(B[1])
        spline = np.load('spline/spline_slat.npy', allow_pickle=True)
    fx = spline[0]
    fy = spline[1]
    sArray = spline[2]
    s_i = sArray[5 * bIndex] + (sArray[5 * bIndex + 5] - sArray[5 * bIndex]) / 2
    sIndex = np.argmax(sArray >= s_i)
    fx_i = fx[sIndex - 1]
    fy_i = fy[sIndex - 1]
    s = sp.symbols('s')
    x = fx_i.subs(s, s_i)
    y = fy_i.subs(s, s_i)
    coordNew = np.array([[x, y]])
    return coordNew


def solveForCoeff(s, x, dxds):
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
    n = len(s)
    # Initialize A matrix
    A = sparse.csr_matrix((n, n))

    # First row and last row
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
        if i == 1:
            delsim1 = s[-1] - s[-2]
        else:
            delsim1 = s[i - 1] - s[i - 2]
        Ai = 2 * (delsim1 + delsi)
        Bi = delsi
        Ci = delsim1
        D[i] = 3 * ((x[i] - x[i - 1]) * delsi / delsim1 + (x[i + 1] - x[i]) * delsim1 / delsi)
        A[i, i - 1] = Bi
        A[i, i] = Ai
        A[i, i + 1] = Ci
    dxds = linalg.spsolve(A, D)  # solution at all points
    return dxds


def spline2d(coord):
    # extract x and y coordinates
    x = coord[:, 0]
    y = coord[:, 1]

    # estimate the arc-length parameter at the nodes using simple linear distance
    s1 = 0
    s = np.zeros(len(coord[:, 0]))
    for i in range(len(coord[:, 0]) - 1):
        si1 = s1 + np.sqrt((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2)
        s[i + 1] = si1
        s1 = si1

    sTrue = np.zeros(len(s))
    L1 = np.Inf
    dxds = spline1d(x, s)
    dyds = spline1d(y, s)
    while L1 > 1e-10:
        # Spline x(s) and y(s) independently using the current estimate for S_i
        dxds = spline1d(x, s)
        dyds = spline1d(y, s)

        # loop over the airfoil segments
        for i in range(len(s) - 1):
            # Compute the true arc-length of the current curve
            if i == 0:
                dels_i = s[-1] - s[-2]
            else:
                dels_i = s[i] - s[i - 1]

            xPrime_i0 = (dxds[i] - (x[i + 1] - x[i]) / dels_i) * dels_i
            xPrime_i1 = (dxds[i + 1] - (x[i + 1] - x[i]) / dels_i) * dels_i
            yPrime_i0 = (dyds[i] - (y[i + 1] - y[i]) / dels_i) * dels_i
            yPrime_i1 = (dyds[i + 1] - (y[i + 1] - y[i]) / dels_i) * dels_i
            f_i = lambda t: np.sqrt(
                (x[i + 1] - x[i] + (1 - 4 * t + 3 * t ** 2) * xPrime_i0 + (-2 * t + 3 * t ** 2) * xPrime_i1) ** 2 + (
                        y[i + 1] - y[i] + (1 - 4 * t + 3 * t ** 2) * yPrime_i0 + (
                            -2 * t + 3 * t ** 2) * yPrime_i1) ** 2)
            integral = integrate.quadrature(f_i, 0.0, 1.0)
            sTrue[i + 1] = sTrue[i] + integral[0]
        L1 = np.sum(np.abs(s - sTrue))
        print(L1)
        s = sTrue

    fx = solveForCoeff(s, x, dxds)
    fy = solveForCoeff(s, y, dyds)
    return np.array([fx, fy, s], dtype=object)


def test():
    s_i = np.linspace(0, 2, 200)
    f = np.load('spline_main.npy', allow_pickle=True)
    fx = f[0]
    fy = f[1]
    s = f[2]
    sSym = sp.symbols('s')
    output = np.array([[]])
    for i in range(len(s) - 1):
        fx_i = fx[i].subs(sSym, s[i] + (s[i + 1] - s[i]) / 2)
        fy_i = fy[i].subs(sSym, s[i] + (s[i + 1] - s[i]) / 2)
        if output.size == 0:
            output = np.array([[fx_i, fy_i]])
        else:
            output = np.append(output, np.array([[fx_i, fy_i]]), axis=0)

    plt.scatter(output[:, 0], output[:, 1])
    plt.show()
    # s = sp.symbols('s')
    # f = solveForCoeff(np.array([0, 4]), np.array([1, 2]), np.array([1, 1]))
    # y = f[0].subs(s, 3)
    # print(y)


def main():
    maintxt = np.loadtxt('geometries/main.txt')
    slat = np.loadtxt('geometries/slat.txt')
    flap = np.loadtxt('geometries/flap.txt')
    np.save('spline_main.npy', spline2d(maintxt))
    np.save('spline_slat.npy', spline2d(slat))
    np.save('spline_flap.npy', spline2d(flap))


if __name__ == "__main__":
    main()
    test()
