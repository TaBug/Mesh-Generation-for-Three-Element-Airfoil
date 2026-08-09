import os

import numpy as np


def readgri(fname):
    """Parse a .gri mesh file into a Mesh dict.

    Reads node coordinates, boundary edge groups (with names), and
    triangular element connectivity from the given .gri file.

    Parameters
    ----------
    fname : str
        Path to the .gri file.

    Returns
    -------
    dict
        Mesh with keys:
        'V' (Nn x 2 array of node coordinates),
        'E' (Ne x 3 array of triangle vertex indices, 1-based),
        'B' (list of arrays, one per boundary group, each row a 1-based
             2-node edge),
        'Bname' (list of boundary group name strings).
    """
    f = open(fname, 'r')
    Nn, Ne, dim = [int(s) for s in f.readline().split()]
    # read vertices
    V = np.array([[float(s) for s in f.readline().split()] for n in range(Nn)])
    # read boundaries
    NB = int(f.readline())
    B = []
    Bname = []
    for i in range(NB):
        s = f.readline().split()
        Nb = int(s[0])
        Bname.append(s[2])
        Bi = np.array([[int(s) for s in f.readline().split()] for n in range(Nb)])
        B.append(Bi)
    # read elements
    Ne0 = 0
    E = []
    while Ne0 < Ne:
        s = f.readline().split()
        ne = int(s[0])
        Ei = np.array([[int(s) for s in f.readline().split()] for n in range(ne)])
        E = Ei if (Ne0 == 0) else np.concatenate((E, Ei), axis=0)
        Ne0 += ne
    f.close()
    Mesh = {'V': V, 'E': E, 'B': B, 'Bname': Bname}
    return Mesh


def genGri(fnameOutput, V, E, B):
    """Write a mesh out to a .gri file.

    Inverse of readgri(): serializes node coordinates, boundary edge
    groups, and triangular element connectivity into the .gri text
    format.

    Parameters
    ----------
    fnameOutput : str
        Path of the .gri file to write.
    V : ndarray
        Nn x 2 array of node coordinates.
    E : ndarray
        Ne x 3 array of triangle vertex indices (1-based).
    B : list of ndarray
        One array per boundary group, each row a 1-based 2-node edge.
    """
    f = open(fnameOutput, 'w')
    nNode = len(V)
    nElemTot = len(E)
    f.write(f"{nNode} {nElemTot} 2\n")
    # node coordinates
    for node in V:
        f.write(f"{node[0]} {node[1]}\n")
    f.write(f"{len(B)}\n")

    # boundary faces
    for i, bGroup in enumerate(B):
        f.write(f"{len(bGroup)} 2 {i + 1}\n")
        for j, bFace in enumerate(bGroup):
            f.write(f"{bFace[0]} {bFace[1]}\n")

    # elements
    f.write(f"{len(E)} 1 TriLagrange\n")
    for i, elem in enumerate(E):
        f.write(f"{elem[0]} {elem[1]} {elem[2]}\n")


# map from interior faces to elements
def getI2E(fnameInput, toOutput):
    """Build the interior-face-to-element (I2E) connectivity matrix.

    Loads the mesh from fnameInput, then for every element edge that
    is not part of a boundary group, finds the neighboring element
    that shares it and records both sides' element/local-edge index.

    Parameters
    ----------
    fnameInput : str
        Path to the .gri file to read.
    toOutput : bool
        If True, also write the result to 'I2E.txt'.

    Returns
    -------
    ndarray
        Array of rows [elemL, faceL, elemR, faceR] (1-based indices),
        one per interior face.
    """
    Mesh = readgri(fnameInput)
    E = Mesh['E']
    B = Mesh['B']

    output = np.array([[]])
    faces = np.array([[]])

    for i, elem in enumerate(E):
        for e in range(3):
            if e == 0:
                node1 = 1
                node2 = 2
            elif e == 1:
                node1 = 2
                node2 = 0
            else:
                node1 = 0
                node2 = 1

            newFace = np.array([[elem[node1], elem[node2]]])
            for bGroup in B:
                if np.isin(bGroup, newFace).all(axis=1).any():
                    break
            else:
                if faces.size == 0:
                    faces = newFace
                    output = np.array([[i + 1, e + 1, 0, 0]])
                else:
                    isin = np.isin(faces, newFace).all(axis=1)
                    if isin.any():
                        iFace = np.where(isin)[0][0]
                        output[iFace][2] = i + 1
                        output[iFace][3] = e + 1
                    else:
                        faces = np.append(faces, newFace, axis=0)
                        face = np.array([[i + 1, e + 1, 0, 0]])
                        output = np.append(output, face, axis=0)
                continue

    if toOutput:
        inputName = os.path.splitext(os.path.basename(fnameInput))[0]
        with open(f'matrices/I2E_{inputName}.txt', 'w') as f:
            for i in range(len(output)):
                f.write(f'{int(output[i][0])} {int(output[i][1])} {int(output[i][2])} {int(output[i][3])}\n')
            f.close()

    return output


def getB2E(fnameInput, toOutput):
    """Build the boundary-face-to-element (B2E) connectivity matrix.

    Loads the mesh from fnameInput, then for every boundary edge in
    every boundary group, finds the element it belongs to and its
    local edge index within that element.

    Parameters
    ----------
    fnameInput : str
        Path to the .gri file to read.
    toOutput : bool
        If True, also write the result to 'B2E.txt'.

    Returns
    -------
    ndarray
        Array of rows [elem, face, bgroup] (1-based indices), one per
        boundary face.
    """
    Mesh = readgri(fnameInput)
    E = Mesh['E']
    B = Mesh['B']
    output = np.array([[]], dtype=int)
    iElem = len(E)
    for ibGroup, bGroup in enumerate(B):
        for nb in bGroup:
            for i, ne in enumerate(np.isin(E, nb)):
                if np.count_nonzero(ne == True) == 2:
                    iElem = i
                    break
            # iElem = np.where(np.any(E == nb, axis=1))
            node1 = nb[0]
            node2 = nb[1]
            elem = E[iElem]

            if np.where(elem == node1)[0].size == 0 or np.where(elem == node2)[0].size == 0:
                print(f'At least one node of boundary does not exist (element = {iElem + 1}, bgroup = {ibGroup + 1})')

            inode1 = np.where(elem == node1)[0][0]
            inode2 = np.where(elem == node2)[0][0]
            iface = 3 - inode1 - inode2

            newB = np.array([[int(iElem + 1), int(iface + 1), int(ibGroup + 1)]])
            if output.size == 0:
                output = newB
            else:
                output = np.append(output, newB, axis=0)

    if toOutput:
        inputName = os.path.splitext(os.path.basename(fnameInput))[0]
        with open(f'matrices/B2E_{inputName}.txt', 'w') as f:
            for i in range(len(output)):
                f.write(f'{int(output[i][0])} {int(output[i][1])} {int(output[i][2])}\n')
            f.close()

    return output


def edgehash(fnameInput, toOutput):
    """Compute unit normal vectors and lengths for interior and boundary faces.

    Loads the mesh and its I2E/B2E connectivity, then for each
    interior face and each boundary face computes the edge length and
    an outward-ish unit normal (rotated tangent) from the edge's two
    endpoint coordinates.

    Parameters
    ----------
    fnameInput : str
        Path to the .gri file to read.
    toOutput : bool
        If True, also write interior normals to 'In.txt' and boundary
        normals to 'Bn.txt' (written incrementally during the loops).

    Returns
    -------
    tuple of ndarray
        (In, Bn, lIn, lBn):
        In  -- interior face unit normals (Nin x 2),
        Bn  -- boundary face unit normals (Nbn x 2),
        lIn -- interior face lengths (Nin,),
        lBn -- boundary face lengths (Nbn,).
    """
    Mesh = readgri(fnameInput)
    E = Mesh['E']
    V = Mesh['V']
    I2E = getI2E(fnameInput, False)
    B2E = getB2E(fnameInput, False)
    In = np.array([[]])
    Bn = np.array([[]])
    lIn = np.array([])
    lBn = np.array([])
    
    inputName = os.path.splitext(os.path.basename(fnameInput))[0]

    for iface, face in enumerate(I2E):
        elemL = face[0]
        faceL = face[1]
        if faceL == 1:
            node1 = 1
            node2 = 2
        elif faceL == 2:
            node1 = 2
            node2 = 0
        else:
            node1 = 0
            node2 = 1
        node1Global = E[elemL - 1][node1]
        node2Global = E[elemL - 1][node2]
        node1c = V[node1Global - 1]
        node2c = V[node2Global - 1]
        l = np.sqrt((node2c[0] - node1c[0]) ** 2 + (node2c[1] - node1c[1]) ** 2)
        n = np.array([[(node2c[1] - node1c[1]) / l, -(node2c[0] - node1c[0]) / l]])
        if In.size == 0:
            In = n
        else:
            In = np.append(In, n, axis=0)
        lIn = np.append(lIn, l)

        if toOutput:
            with open(f'matrices/In_{inputName}.txt', 'w') as f:
                for Ini in In:
                    f.write(f'{Ini[0]} {Ini[1]}\n')
            f.close()

    for iface, face in enumerate(B2E):
        elem = face[0]
        faceLocal = face[1]
        if faceLocal == 1:
            node1 = 1
            node2 = 2
        elif faceLocal == 2:
            node1 = 2
            node2 = 0
        else:
            node1 = 0
            node2 = 1
        node1Global = E[elem - 1][node1]
        node2Global = E[elem - 1][node2]
        node1c = V[node1Global - 1]
        node2c = V[node2Global - 1]
        l = np.sqrt((node2c[0] - node1c[0]) ** 2 + (node2c[1] - node1c[1]) ** 2)
        n = np.array([[(node2c[1] - node1c[1]) / l, -(node2c[0] - node1c[0]) / l]])
        if Bn.size == 0:
            Bn = n
        else:
            Bn = np.append(Bn, n, axis=0)
        lBn = np.append(lBn, l)

        if toOutput:
            with open(f'matrices/Bn_{inputName}.txt', 'w') as f:
                for Bni in Bn:
                    f.write(f'{Bni[0]} {Bni[1]}\n')
            f.close()

    return In, Bn, lIn, lBn


# input: element matrix, node coordinate matrix
# output: element area matrix (index = element index)
def area(fnameInput, toOutput):
    """Compute the signed area of every triangular element in a mesh.

    Loads the mesh from fnameInput and evaluates the shoelace formula
    for each triangle's three vertices.

    Parameters
    ----------
    fnameInput : str
        Path to the .gri file to read.
    toOutput : bool
        If True, write one area per line to 'area.txt'.

    Returns
    -------
    None
        (Areas are computed into a local array but not returned;
        only written to file when toOutput is True.)
    """
    Mesh = readgri(fnameInput)
    E = Mesh['E']
    V = Mesh['V']

    areas = np.zeros(len(E))
    for i, ne in enumerate(E):
        coor0 = V[int(ne[0]) - 1]
        coor1 = V[int(ne[1]) - 1]
        coor2 = V[int(ne[2]) - 1]
        areas[i] = 1 / 2 * (coor0[0] * (coor1[1] - coor2[1]) + coor1[0] * (coor2[1] - coor0[1]) + coor2[0] * (
                coor0[1] - coor1[1]))

    if toOutput:
        with open('area.txt', 'w') as f:
            for areai in areas:
                f.write(f'{areai}\n')
        f.close()


def getF2V(fnameInput, toOutput):
    """Build the face-to-vertex (F2V) matrix for all interior and boundary faces.

    Loads the mesh and its I2E/B2E connectivity, then for each face
    (interior first, then boundary) maps the (element, local edge)
    pair to its two global node indices.

    Parameters
    ----------
    fnameInput : str
        Path to the .gri file to read.
    toOutput : bool
        If True, also write the result to 'F2V.txt'.

    Returns
    -------
    ndarray
        Array of rows [node1, node2] (global node indices), interior
        faces followed by boundary faces, in I2E/B2E order.
    """
    # read fnameInput
    mesh = readgri(fnameInput)
    E = mesh['E']

    # get interior and boundary face mapping matrices
    I2E = getI2E(fnameInput, False)
    B2E = getB2E(fnameInput, False)

    output = np.array([[]], dtype=int)
    # loop through interior faces
    for i, face in enumerate(I2E):
        elemL = face[0]
        faceL = face[1]
        node1 = E[elemL - 1][(faceL + 1) % 3 - 1]
        node2 = E[elemL - 1][(faceL - 1) % 3 - 1]
        newFace = np.array([[node1, node2]])
        if output.size == 0:
            output = newFace
        else:
            output = np.append(output, newFace, axis=0)

    # loop through boundary faces
    for i, face in enumerate(B2E):
        elem = face[0]
        face = face[1]
        node1 = E[elem - 1][(face + 1) % 3 - 1]
        node2 = E[elem - 1][(face - 1) % 3 - 1]
        newFace = np.array([[node1, node2]])
        if output.size == 0:
            output = newFace
        else:
            output = np.append(output, newFace, axis=0)

    if toOutput:
        inputName = os.path.splitext(os.path.basename(fnameInput))[0]
        with open(f'matrices/F2V_{inputName}.txt', 'w') as f:
            for face in output:
                f.write(f'{face[0]} {face[1]}\n')
        f.close()

    return output


def main():
    """Entry point: generate I2E/B2E connectivity files for sample meshes.

    Runs getI2E on 'localSmoothedAllTrail.gri' and getB2E on
    'localRefinedAll.gri', writing I2E.txt/B2E.txt, then prints the
    element count of 'localRefinedAll.gri' as a sanity check.
    """
    getI2E('gri/test.gri', True)
    getB2E('gri/test.gri', True)
    edgehash('gri/test.gri', True)
    area('gri/test.gri', False)
    getF2V('gri/test.gri', True)
    # print(len(readgri('localRefinedAll.gri')['E']))


if __name__ == "__main__":
    main()
