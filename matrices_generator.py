import os

import numpy as np

# Local face i (0-based) is opposite local node i, so it spans the other two
# nodes. Ordering is chosen so the traversal node1 -> node2 is counterclockwise
# within a counterclockwise element, making (dy, -dx) point out of the element.
LOCAL_FACE_NODES = ((1, 2), (2, 0), (0, 1))


def edgeKey(a, b):
    """Canonical key for an edge, independent of traversal direction."""
    a, b = int(a), int(b)
    return (a, b) if a < b else (b, a)


def faceNodes(elem, face):
    """Global nodes of local face `face` (1-based) of element `elem`."""
    i, j = LOCAL_FACE_NODES[face - 1]
    return int(elem[i]), int(elem[j])


def readgri(fname):
    """Parse a .gri mesh file into a Mesh dict.

    Returns
    -------
    dict
        'V' (Nn x 2 node coordinates), 'E' (Ne x 3 triangle vertices, 1-based),
        'B' (list of arrays, one per boundary group, each row a 1-based 2-node
        edge), 'Bname' (boundary group names).
    """
    with open(fname, 'r') as f:
        Nn, Ne, dim = [int(s) for s in f.readline().split()]
        V = np.array([[float(s) for s in f.readline().split()] for _ in range(Nn)])

        NB = int(f.readline())
        B = []
        Bname = []
        for _ in range(NB):
            s = f.readline().split()
            Nb = int(s[0])
            Bname.append(s[2])
            B.append(np.array([[int(t) for t in f.readline().split()] for _ in range(Nb)]))

        Ne0 = 0
        E = []
        while Ne0 < Ne:
            s = f.readline().split()
            ne = int(s[0])
            Ei = np.array([[int(t) for t in f.readline().split()] for _ in range(ne)])
            E = Ei if (Ne0 == 0) else np.concatenate((E, Ei), axis=0)
            Ne0 += ne

    return {'V': V, 'E': E, 'B': B, 'Bname': Bname}


def genGri(fnameOutput, V, E, B, Bname=None):
    """Write a mesh out to a .gri file. Inverse of readgri()."""
    with open(fnameOutput, 'w') as f:
        f.write(f"{len(V)} {len(E)} 2\n")
        for node in V:
            f.write(f"{node[0]} {node[1]}\n")

        f.write(f"{len(B)}\n")
        for i, bGroup in enumerate(B):
            title = Bname[i] if Bname is not None else f"bgroup{i + 1}"
            f.write(f"{len(bGroup)} 2 {title}\n")
            for bFace in bGroup:
                f.write(f"{bFace[0]} {bFace[1]}\n")

        f.write(f"{len(E)} 1 TriLagrange\n")
        for elem in E:
            f.write(f"{elem[0]} {elem[1]} {elem[2]}\n")


def _boundary_edge_map(B):
    """Map each boundary edge (sorted node pair) to its 1-based group index."""
    bmap = {}
    for ibGroup, bGroup in enumerate(B):
        for edge in bGroup:
            bmap[edgeKey(edge[0], edge[1])] = ibGroup + 1
    return bmap


def _element_edge_map(E):
    """Map each element edge (sorted node pair) to the (elem, face) pairs on it.

    One pass over all elements: every edge is visited exactly twice for an
    interior face and once for a boundary face, so the build is O(N).
    Indices in the values are 1-based.
    """
    emap = {}
    for i, elem in enumerate(E):
        for e, (a, b) in enumerate(LOCAL_FACE_NODES):
            emap.setdefault(edgeKey(elem[a], elem[b]), []).append((i + 1, e + 1))
    return emap


def _write_rows(path, rows, fmt):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for row in rows:
            f.write(fmt(row))


def getI2E(fnameInput, toOutput, Mesh=None):
    """Build the interior-face-to-element (I2E) matrix.

    Every element edge is hashed on its sorted node pair, so the two elements
    sharing an interior face collide in the same bucket. Cost is O(N).

    Returns
    -------
    ndarray
        Rows [elemL, faceL, elemR, faceR], 1-based, with L the smaller element
        index. Sorted by (elemL, faceL).
    """
    if Mesh is None:
        Mesh = readgri(fnameInput)
    E, B = Mesh['E'], Mesh['B']

    bmap = _boundary_edge_map(B)
    emap = _element_edge_map(E)

    rows = []
    for key, sides in emap.items():
        if key in bmap:
            continue
        if len(sides) != 2:
            raise ValueError(
                f"Interior edge {key} is adjacent to {len(sides)} element(s); "
                "mesh is non-conforming or a boundary group is incomplete."
            )
        (e1, f1), (e2, f2) = sides
        if e1 <= e2:
            rows.append([e1, f1, e2, f2])
        else:
            rows.append([e2, f2, e1, f1])

    output = np.array(rows, dtype=int).reshape(-1, 4)
    output = output[np.lexsort((output[:, 1], output[:, 0]))]

    if toOutput:
        name = os.path.splitext(os.path.basename(fnameInput))[0]
        _write_rows(f'matrices/I2E_{name}.txt', output,
                    lambda r: f'{r[0]} {r[1]} {r[2]} {r[3]}\n')

    return output


def getB2E(fnameInput, toOutput, Mesh=None):
    """Build the boundary-face-to-element (B2E) matrix.

    Uses the same element-edge hash map, so each boundary edge is resolved by a
    single dictionary lookup instead of a scan over all elements. Rows follow
    the boundary group and edge ordering in the .gri file.

    Returns
    -------
    ndarray
        Rows [elem, face, bgroup], 1-based.
    """
    if Mesh is None:
        Mesh = readgri(fnameInput)
    E, B = Mesh['E'], Mesh['B']

    emap = _element_edge_map(E)

    rows = []
    for ibGroup, bGroup in enumerate(B):
        for edge in bGroup:
            n1, n2 = int(edge[0]), int(edge[1])
            key = (min(n1, n2), max(n1, n2))
            sides = emap.get(key)
            if sides is None:
                raise ValueError(
                    f"Boundary edge ({n1}, {n2}) in group {ibGroup + 1} "
                    "does not belong to any element."
                )
            if len(sides) != 1:
                raise ValueError(
                    f"Boundary edge ({n1}, {n2}) in group {ibGroup + 1} has "
                    f"{len(sides)} adjacent elements; it is not on the boundary."
                )
            elem, face = sides[0]
            rows.append([elem, face, ibGroup + 1])

    output = np.array(rows, dtype=int).reshape(-1, 3)

    if toOutput:
        name = os.path.splitext(os.path.basename(fnameInput))[0]
        _write_rows(f'matrices/B2E_{name}.txt', output,
                    lambda r: f'{r[0]} {r[1]} {r[2]}\n')

    return output


def edgehash(fnameInput, toOutput, Mesh=None):
    """Compute unit normals and lengths for interior and boundary faces.

    Interior normals point from the L to the R element; boundary normals point
    out of the domain. Both are computed vectorized from the L (or only)
    element's local face node ordering.

    Returns
    -------
    tuple of ndarray
        (In, Bn, lIn, lBn).
    """
    if Mesh is None:
        Mesh = readgri(fnameInput)
    E, V = Mesh['E'], Mesh['V']

    I2E = getI2E(fnameInput, False, Mesh)
    B2E = getB2E(fnameInput, False, Mesh)

    def normals(elems, faces):
        if len(elems) == 0:
            return np.zeros((0, 2)), np.zeros(0)
        local = np.array(LOCAL_FACE_NODES)[faces - 1]      # (Nf, 2) local nodes
        glob = E[elems - 1][np.arange(len(elems))[:, None], local]
        p1 = V[glob[:, 0] - 1]
        p2 = V[glob[:, 1] - 1]
        d = p2 - p1
        l = np.hypot(d[:, 0], d[:, 1])
        n = np.column_stack((d[:, 1], -d[:, 0])) / l[:, None]
        return n, l

    In, lIn = normals(I2E[:, 0], I2E[:, 1])
    Bn, lBn = normals(B2E[:, 0], B2E[:, 1])

    if toOutput:
        name = os.path.splitext(os.path.basename(fnameInput))[0]
        _write_rows(f'matrices/In_{name}.txt', In, lambda r: f'{r[0]} {r[1]}\n')
        _write_rows(f'matrices/Bn_{name}.txt', Bn, lambda r: f'{r[0]} {r[1]}\n')

    return In, Bn, lIn, lBn


def area(fnameInput, toOutput, Mesh=None):
    """Compute the signed area of every triangular element (shoelace formula).

    Returns
    -------
    ndarray
        One area per element, in element order.
    """
    if Mesh is None:
        Mesh = readgri(fnameInput)
    E, V = Mesh['E'], Mesh['V']

    p0 = V[E[:, 0] - 1]
    p1 = V[E[:, 1] - 1]
    p2 = V[E[:, 2] - 1]
    areas = 0.5 * ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                   - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))

    if toOutput:
        name = os.path.splitext(os.path.basename(fnameInput))[0]
        _write_rows(f'matrices/area_{name}.txt', areas, lambda a: f'{a}\n')

    return areas


def getF2V(fnameInput, toOutput, Mesh=None):
    """Build the face-to-vertex (F2V) matrix, interior faces then boundary faces.

    Returns
    -------
    ndarray
        Rows [node1, node2] (global, 1-based), in I2E order followed by B2E order.
    """
    if Mesh is None:
        Mesh = readgri(fnameInput)
    E = Mesh['E']

    I2E = getI2E(fnameInput, False, Mesh)
    B2E = getB2E(fnameInput, False, Mesh)

    def verts(elems, faces):
        if len(elems) == 0:
            return np.zeros((0, 2), dtype=int)
        local = np.array(LOCAL_FACE_NODES)[faces - 1]
        return E[elems - 1][np.arange(len(elems))[:, None], local]

    output = np.vstack((verts(I2E[:, 0], I2E[:, 1]),
                        verts(B2E[:, 0], B2E[:, 1]))).astype(int)

    if toOutput:
        name = os.path.splitext(os.path.basename(fnameInput))[0]
        _write_rows(f'matrices/F2V_{name}.txt', output,
                    lambda r: f'{r[0]} {r[1]}\n')

    return output


def verify(fnameInput, Mesh=None):
    """Mesh verification test: sum of outward normal * length over each element.

    E_e = sum_i n_ei^outward * l_ei should be zero to machine precision.

    Returns
    -------
    float
        max_e |E_e| over all elements.
    """
    if Mesh is None:
        Mesh = readgri(fnameInput)
    E = Mesh['E']

    I2E = getI2E(fnameInput, False, Mesh)
    B2E = getB2E(fnameInput, False, Mesh)
    In, Bn, lIn, lBn = edgehash(fnameInput, False, Mesh)

    Esum = np.zeros((len(E), 2))
    # interior: normal points L -> R, so it is outward for L and inward for R
    np.add.at(Esum, I2E[:, 0] - 1, In * lIn[:, None])
    np.subtract.at(Esum, I2E[:, 2] - 1, In * lIn[:, None])
    # boundary: normal already points out of the domain
    np.add.at(Esum, B2E[:, 0] - 1, Bn * lBn[:, None])

    return np.max(np.hypot(Esum[:, 0], Esum[:, 1]))


def main():
    fname = 'gri/test.gri'
    Mesh = readgri(fname)
    print('I2E\n', getI2E(fname, True, Mesh))
    print('B2E\n', getB2E(fname, True, Mesh))
    In, Bn, lIn, lBn = edgehash(fname, True, Mesh)
    print('In\n', In)
    print('Bn\n', Bn)
    print('Area\n', area(fname, True, Mesh))
    print('max |E_e| =', verify(fname, Mesh))


if __name__ == "__main__":
    main()
