import numpy as np

from matrices_generator import readgri, genGri, edgeKey, faceNodes
from spline.spline import AIRFOIL_GROUPS, snapToBoundary


def uniformRefine(fnameInput, fnameOutput):
    """Split every element into four sub-elements by bisecting all three edges.

    New nodes are keyed on the edge they split, so the two elements sharing an
    edge always agree on the midpoint node and the refined mesh stays
    conforming. Midpoints of boundary edges on the curved airfoil surfaces are
    projected onto the true spline geometry. Every edge is visited a constant
    number of times, so the cost is O(N).
    """
    mesh = readgri(fnameInput)
    V, E, B, Bname = mesh['V'], mesh['E'], mesh['B'], mesh['Bname']

    # boundary edge -> group index, so new nodes on the airfoil can be snapped
    groupOf = {}
    for g, bGroup in enumerate(B, start=1):
        for edge in bGroup:
            groupOf[edgeKey(edge[0], edge[1])] = g

    Vnew = [row.copy() for row in V]
    midOf = {}

    def midpoint(a, b):
        """Node index (1-based) of the midpoint of edge (a, b), created once."""
        k = edgeKey(a, b)
        if k in midOf:
            return midOf[k]

        coord = 0.5 * (V[k[0] - 1] + V[k[1] - 1])
        g = groupOf.get(k)
        if g in AIRFOIL_GROUPS:
            coord = snapToBoundary(coord, g)

        Vnew.append(np.asarray(coord, dtype=float))
        midOf[k] = len(Vnew)
        return midOf[k]

    # M[i] is the midpoint of local face i+1, which is opposite local node i
    Enew = []
    for elem in E:
        n0, n1, n2 = int(elem[0]), int(elem[1]), int(elem[2])
        M = [midpoint(*faceNodes(elem, f)) for f in (1, 2, 3)]
        Enew.append([n0, M[2], M[1]])
        Enew.append([n1, M[0], M[2]])
        Enew.append([n2, M[1], M[0]])
        Enew.append([M[0], M[1], M[2]])

    # every boundary edge belongs to a refined element, so each one is bisected
    Bnew = []
    for bGroup in B:
        edges = []
        for edge in bGroup:
            a, b = int(edge[0]), int(edge[1])
            m = midOf[edgeKey(a, b)]
            edges.append([a, m])
            edges.append([m, b])
        Bnew.append(np.array(edges, dtype=int))

    genGri(fnameOutput, np.array(Vnew, dtype=float),
           np.array(Enew, dtype=int), Bnew, Bname)


def main():
    """Generate the ~8k, ~32k, and ~128k refinements of the locally-refined mesh."""
    fname = 'gri/smoothed_local_all.gri'
    for label in ('8k', '32k', '128k'):
        fnameOutput = f'gri/refinement_uniform_all_{label}.gri'
        uniformRefine(fname, fnameOutput)
        fname = fnameOutput


if __name__ == "__main__":
    main()
