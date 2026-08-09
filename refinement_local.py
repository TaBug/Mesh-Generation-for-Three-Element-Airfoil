import numpy as np

from matrices_generator import (readgri, genGri, getI2E, getB2E, edgeKey,
                                faceNodes)
from spline.spline import AIRFOIL_GROUPS, snapToBoundary


def _angle(apex, p, q):
    """Interior angle at `apex` in the triangle apex-p-q."""
    v1, v2 = p - apex, q - apex
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(np.clip(c, -1.0, 1.0))


def localRefine(x, y, r, fnameInput, fnameOutputRefine, fnameOutputSmooth,
                omega=0.3, nSmooth=5):
    """Refine the mesh around (x, y) within radius r, then smooth.

    Elements whose centroid lies within r are flagged; all of their edges are
    flagged for splitting; every element adjacent to a flagged edge is then
    refined with the 1-, 2-, or 3-edge template. New nodes are keyed on the
    edge they split, so the two elements sharing an edge always agree on the
    node and the mesh stays conforming. New nodes on the airfoil surfaces are
    snapped to the spline geometry.
    """
    mesh = readgri(fnameInput)
    V, E, B, Bname = mesh['V'], mesh['E'], mesh['B'], mesh['Bname']
    I2E = getI2E(fnameInput, False, mesh)
    B2E = getB2E(fnameInput, False, mesh)

    # ---- 1. flag elements whose centroid falls within r of the target point
    centroids = (V[E[:, 0] - 1] + V[E[:, 1] - 1] + V[E[:, 2] - 1]) / 3.0
    elemFlag = np.hypot(centroids[:, 0] - x, centroids[:, 1] - y) <= r

    # ---- 2. flag every edge of a flagged element, on BOTH adjacent elements
    # (marking both sides is what keeps the refined mesh conforming)
    split = np.zeros((len(E), 3), dtype=bool)
    for elemL, faceL, elemR, faceR in I2E:
        if elemFlag[elemL - 1] or elemFlag[elemR - 1]:
            split[elemL - 1, faceL - 1] = True
            split[elemR - 1, faceR - 1] = True
    for elem, face, _ in B2E:
        if elemFlag[elem - 1]:
            split[elem - 1, face - 1] = True

    # ---- 3. locate boundary edges so new nodes on them can be snapped
    bfaceOf = {}
    for ib, (elem, face, g) in enumerate(B2E):
        bfaceOf[edgeKey(*faceNodes(E[elem - 1], face))] = (int(g), int(ib))

    Vnew = [row.copy() for row in V]
    midOf = {}

    def midpoint(a, b):
        """Node index (1-based) of the midpoint of edge (a, b), created once."""
        k = edgeKey(a, b)
        if k in midOf:
            return midOf[k]

        coord = 0.5 * (V[k[0] - 1] + V[k[1] - 1])
        if k in bfaceOf:
            g, _ = bfaceOf[k]
            if g in AIRFOIL_GROUPS:
                coord = snapToBoundary(coord, g)

        Vnew.append(np.asarray(coord, dtype=float))
        midOf[k] = len(Vnew)
        return midOf[k]

    # ---- 4. apply the refinement templates
    Enew = []
    touched = set()          # nodes affected by refinement, for smoothing

    for iElem, elem in enumerate(E):
        n = [int(elem[0]), int(elem[1]), int(elem[2])]
        flagged = np.where(split[iElem])[0]

        if len(flagged) == 0:
            Enew.append(n)
            continue

        touched.update(n)

        if len(flagged) == 1:
            i = flagged[0]
            M = midpoint(*faceNodes(elem, i + 1))
            touched.add(M)
            Enew.append([n[i], n[(i + 1) % 3], M])
            Enew.append([n[i], M, n[(i + 2) % 3]])

        elif len(flagged) == 2:
            # k = the unsplit face; node k is the apex shared by both split edges
            k = 3 - flagged[0] - flagged[1]
            A, Bv, C = n[(k + 1) % 3], n[(k + 2) % 3], n[k]
            M_CA = midpoint(C, A)      # midpoint on edge C-A
            M_BC = midpoint(Bv, C)     # midpoint on edge B-C
            touched.update((M_CA, M_BC))

            # apex triangle
            Enew.append([M_BC, C, M_CA])

            # split the quad A-B-M_BC-M_CA across the larger of the two angles
            aA = _angle(V[A - 1], V[Bv - 1], V[C - 1])
            aB = _angle(V[Bv - 1], V[C - 1], V[A - 1])
            if aA >= aB:
                Enew.append([A, Bv, M_BC])
                Enew.append([A, M_BC, M_CA])
            else:
                Enew.append([A, Bv, M_CA])
                Enew.append([Bv, M_BC, M_CA])

        else:
            M = [midpoint(*faceNodes(elem, i + 1)) for i in range(3)]
            touched.update(M)
            Enew.append([n[0], M[2], M[1]])
            Enew.append([n[1], M[0], M[2]])
            Enew.append([n[2], M[1], M[0]])
            Enew.append([M[0], M[1], M[2]])

    # ---- 5. rebuild the boundary groups, splitting any edge that was refined
    Bnew = []
    for g, bGroup in enumerate(B, start=1):
        edges = []
        for edge in bGroup:
            a, b = int(edge[0]), int(edge[1])
            k = edgeKey(a, b)
            if k in midOf:
                m = midOf[k]
                edges.append([a, m])
                edges.append([m, b])
            else:
                edges.append([a, b])
        Bnew.append(np.array(edges, dtype=int))

    Vnew = np.array(Vnew, dtype=float)
    Enew = np.array(Enew, dtype=int)
    genGri(fnameOutputRefine, Vnew, Enew, Bnew, Bname)

    # ---- 6. smooth the affected interior nodes
    boundaryNodes = set()
    for bGroup in Bnew:
        boundaryNodes.update(int(v) for v in bGroup.ravel())

    neighbors = {}
    for tri in Enew:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            neighbors.setdefault(int(a), set()).add(int(b))
            neighbors.setdefault(int(b), set()).add(int(a))

    movable = [i for i in touched if i not in boundaryNodes and i in neighbors]
    for _ in range(nSmooth):
        for i in movable:
            nb = np.array(sorted(neighbors[i]), dtype=int)
            Vnew[i - 1] = (1 - omega) * Vnew[i - 1] + omega * Vnew[nb - 1].mean(axis=0)

    genGri(fnameOutputSmooth, Vnew, Enew, Bnew, Bname)
    return Vnew, Enew, Bnew


def main():
    localRefine(1, 0, 0.1, 'gri/all.gri',
                'gri/refined_local_all_trail.gri',
                'gri/smoothed_local_all_trail.gri')

    localRefine(0, 0, 0.1, 'gri/smoothed_local_all_trail.gri',
                'gri/refined_local_all.gri',
                'gri/smoothed_local_all.gri')


if __name__ == "__main__":
    main()
