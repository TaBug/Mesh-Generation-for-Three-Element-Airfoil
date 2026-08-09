import numpy as np
from matrices_generator import readgri, genGri, getI2E, getB2E, edgehash, area, getF2V
from spline.spline import slapToBoundary


def uniformRefine(fnameInput, fnameOutput):
    mesh = readgri(fnameInput)
    E = mesh['E']
    V = mesh['V']
    B = mesh['B']
    B2E = getB2E(fnameInput, False)
    Ecopy = E.copy()
    Vcopy = V.copy()
    Bcopy = B.copy()

    for iElem in range(len(E)):
        # find the boundary face index
        ib = np.where(B2E[:, 0] == iElem + 1)

        # find the node indicis
        node1 = E[iElem][0]
        node2 = E[iElem][1]
        node3 = E[iElem][2]

        # create new nodes
        node1NewC = np.array(
            [[(V[node2 - 1][0] + V[node3 - 1][0]) / 2, (V[node2 - 1][1] + V[node3 - 1][1]) / 2]])
        node2NewC = np.array(
            [[(V[node1 - 1][0] + V[node3 - 1][0]) / 2, (V[node1 - 1][1] + V[node3 - 1][1]) / 2]])
        node3NewC = np.array(
            [[(V[node1 - 1][0] + V[node2 - 1][0]) / 2, (V[node1 - 1][1] + V[node2 - 1][1]) / 2]])

        # if there is a boundary face
        if ib[0].size != 0 and 1 <= B2E[ib[0][0]][2] <= 3:
            iNodeB = B2E[ib[0][0]][1]
            bgroup = B2E[ib[0][0]][2]
            nodeBC = slapToBoundary(B, bgroup, ib)
            if iNodeB == 1:
                node1NewC = nodeBC
            elif iNodeB == 2:
                node2NewC = nodeBC
            else:
                node3NewC = nodeBC

        # check if the nodes have been seen
        where1 = np.where(Vcopy == node1NewC)[0]
        where2 = np.where(Vcopy == node2NewC)[0]
        where3 = np.where(Vcopy == node3NewC)[0]

        if where1.size == 0:
            Vcopy = np.append(Vcopy, node1NewC, axis=0)
            node1New = len(Vcopy)
        else:
            node1New = where1[0] + 1
        if where2.size == 0:
            Vcopy = np.append(Vcopy, node2NewC, axis=0)
            node2New = len(Vcopy)
        else:
            node2New = where2[0] + 1
        if where3.size == 0:
            Vcopy = np.append(Vcopy, node3NewC, axis=0)
            node3New = len(Vcopy)
        else:
            node3New = where3[0] + 1

        # add new boundaries
        if ib[0].size != 0:
            iNodeB = B2E[ib[0][0]][1]
            bgroup = B2E[ib[0][0]][2]
            if iNodeB == 1:
                newB1 = np.array([[node2, node1New]])
                newB2 = np.array([[node1New, node3]])
            elif iNodeB == 2:
                newB1 = np.array([[node3, node2New]])
                newB2 = np.array([[node2New, node1]])
            else:
                newB1 = np.array([[node1, node3New]])
                newB2 = np.array([[node3New, node2]])

            bIndex = ib[0][0] - sum(len(B[i]) for i in range(bgroup - 1))
            Bcopy[bgroup - 1] = np.append(Bcopy[bgroup - 1], newB1, axis=0)
            Bcopy[bgroup - 1][bIndex] = newB2

        # create new elements
        elemNew1 = np.array([[node1New, node3, node2New]])
        elemNew2 = np.array([[node2New, node1, node3New]])
        elemNew3 = np.array([[node3New, node2, node1New]])
        elemNew4 = np.array([[node1New, node2New, node3New]])

        # add to the new E matrix
        Ecopy = np.append(Ecopy, elemNew1, axis=0)
        Ecopy = np.append(Ecopy, elemNew2, axis=0)
        Ecopy = np.append(Ecopy, elemNew3, axis=0)
        Ecopy[iElem] = elemNew4
    genGri(fnameOutput, Vcopy, Ecopy, Bcopy)


def main():
    uniformRefine('gri/smoothed_local_all.gri', 'gri/refined_uniform_all.gri')


if __name__ == "__main__":
    main()
