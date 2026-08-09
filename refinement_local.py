import numpy as np
from matrices_generator import readgri, genGri, getI2E, getB2E, edgehash, area, getF2V
from spline.spline import slapToBoundary


def localRefine(x, y, r, fnameInput, fnameOutputRefine, fnameOutputSmooth):
    mesh = readgri(fnameInput)
    I2E = getI2E(fnameInput, False)
    B2E = getB2E(fnameInput, False)
    E = mesh['E']
    V = mesh['V']
    B = mesh['B']
    Ecopy = E.copy()
    Vcopy = V.copy()
    Bcopy = B.copy()
    eleFlagged = np.zeros(len(E), dtype=int)
    edgeFlagged = np.zeros(len(I2E) + len(B2E), dtype=int)

    # flag the elements that fall in the range
    for iElem, elem in enumerate(E):
        xc = (V[elem[0] - 1][0] + V[elem[1] - 1][0] + V[elem[2] - 1][0]) / 3
        yc = (V[elem[0] - 1][1] + V[elem[1] - 1][1] + V[elem[2] - 1][1]) / 3
        if np.sqrt((xc - x) ** 2 + (yc - y) ** 2) <= r:
            eleFlagged[iElem] += 1

    # flag the faces of the flagged elements
    for iEdge, i2e in enumerate(I2E):
        if eleFlagged[i2e[0] - 1] == 1 or eleFlagged[i2e[2] - 1] == 1:
            edgeFlagged[iEdge] += 1

    for iEdge, b2e in enumerate(B2E):
        if eleFlagged[b2e[0] - 1] == 1:
            edgeFlagged[iEdge + len(I2E)] += 1

    # flag the elements that are adjacent to the flagged faces
    # record the local index of the face that is to be refined
    # interior faces
    eleRefine = np.zeros([len(E), 3], dtype=int)
    for iEdge, i2e in enumerate(I2E):
        if edgeFlagged[iEdge] == 1:
            elemL = i2e[0]
            faceL = i2e[1]
            elemR = i2e[2]
            faceR = i2e[3]
            eleRefine[elemL - 1][faceL - 1] += 1
            eleRefine[elemR - 1][faceR - 1] += 1

    # boundary faces
    for iEdge, b2e in enumerate(B2E):
        if edgeFlagged[iEdge + len(I2E)] == 1:
            elem = b2e[0]
            face = b2e[1]
            eleRefine[elem - 1][face - 1] += 1

    for iElem, elem in enumerate(eleRefine):
        # find the boundary face index
        ib = np.where(B2E[:, 0] == iElem + 1)
        # one edge flagged
        if sum(elem) == 1:
            index = np.where(elem == 1)[0][0]
            node1 = E[iElem][index]
            node2 = E[iElem][index - 2]
            node3 = E[iElem][index - 1]

            # if the edge flagged is boundary edge and on the airfoils
            if ib[0].size != 0 and B2E[ib[0][0]][1] == index + 1:
                bgroup = B2E[ib[0][0]][2]
                if 1 <= bgroup <= 3:
                    coordNew = slapToBoundary(B, bgroup, ib)
                else:
                    coordNew = np.array(
                        [[(V[node3 - 1][0] + V[node2 - 1][0]) / 2, (V[node3 - 1][1] + V[node2 - 1][1]) / 2]])
                Vcopy = np.append(Vcopy, coordNew, axis=0)
                nodeNew = len(Vcopy)

                # add new boundary
                bIndex = ib[0][0] - sum(len(B[i]) for i in range(bgroup - 1))
                Bcopy[bgroup - 1] = np.append(Bcopy[bgroup - 1], np.array([[node2, nodeNew]]), axis=0)
                Bcopy[bgroup - 1][bIndex] = np.array([node2, nodeNew])
            else:
                coordNew = np.array(
                    [[(V[node3 - 1][0] + V[node2 - 1][0]) / 2, (V[node3 - 1][1] + V[node2 - 1][1]) / 2]])

                # check if the new node has already been seen
                where = np.where(Vcopy == coordNew)[0]
                # never seen
                if where.size == 0:
                    Vcopy = np.append(Vcopy, coordNew, axis=0)
                    nodeNew = len(Vcopy)
                else:
                    nodeNew = where[0] + 1

            # split the element in two
            elemNew1 = np.array([[nodeNew, node1, node2]])
            elemNew2 = np.array([[nodeNew, node3, node1]])
            Ecopy[iElem] = elemNew1
            Ecopy = np.append(Ecopy, elemNew2, axis=0)

        # two edges flagged
        elif sum(elem) == 2:
            index = np.where(elem == 1)[0]
            node1 = E[iElem][index[0]]
            node2 = E[iElem][index[1]]
            node3 = E[iElem][3 - index[0] - index[1]]
            node1c = V[node1 - 1]
            node2c = V[node2 - 1]
            node3c = V[node3 - 1]

            # find the angle of node1
            v1 = node1c - node2c
            v2 = node1c - node3c
            angle1 = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

            # find the angle of node2
            v1 = node2c - node1c
            v2 = node2c - node3c
            angle2 = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

            # find the bigger angle
            if angle1 > angle2:
                nodeBig = node1
                nodeSml = node2
                iLocalBig = index[0]
                iLocalSml = index[1]
            else:
                nodeBig = node2
                nodeSml = node1
                iLocalBig = index[1]
                iLocalSml = index[0]

            # create new nodes
            # Smaller angle new node
            # if there is a boundary face, and it is smaller angle face
            if ib[0].size != 0 and B2E[ib[0][0]][1] == iLocalSml + 1:
                bgroup = B2E[ib[0][0]][2]
                if 1 <= B2E[ib[0][0]][2] <= 3:
                    nodeSmlc = slapToBoundary(B, bgroup, ib)
                else:
                    nodeSmlc = np.array([[(node3c[0] + V[nodeBig - 1][0]) / 2, (node3c[1] + V[nodeBig - 1][1]) / 2]])
                Vcopy = np.append(Vcopy, nodeSmlc, axis=0)
                nodeSmlNew = len(Vcopy)

                # add new boundary
                bIndex = ib[0][0] - sum(len(B[i]) for i in range(bgroup - 1))
                Bcopy[bgroup - 1] = np.append(Bcopy[bgroup - 1], np.array([[nodeBig, nodeSmlNew]]), axis=0)
                Bcopy[bgroup - 1][bIndex] = np.array([nodeSmlNew, node3])
            else:
                nodeSmlc = np.array([[(node3c[0] + V[nodeBig - 1][0]) / 2, (node3c[1] + V[nodeBig - 1][1]) / 2]])
                whereSml = np.where(Vcopy == nodeSmlc)[0]
                if whereSml.size == 0:
                    Vcopy = np.append(Vcopy, nodeSmlc, axis=0)
                    nodeSmlNew = len(Vcopy)
                else:
                    nodeSmlNew = whereSml[0] + 1

            # Bigger angle new node
            # if there is a boundary face, and it is bigger angle face
            if ib[0].size != 0 and B2E[ib[0][0]][1] == iLocalBig + 1:
                bgroup = B2E[ib[0][0]][2]
                if 1 <= B2E[ib[0][0]][2] <= 3:
                    nodeBigc = slapToBoundary(B, bgroup, ib)
                else:
                    nodeBigc = np.array([[(node3c[0] + V[nodeSml - 1][0]) / 2, (node3c[1] + V[nodeSml - 1][1]) / 2]])
                Vcopy = np.append(Vcopy, nodeBigc, axis=0)
                nodeBigNew = len(Vcopy)

                bIndex = ib[0][0] - sum(len(B[i]) for i in range(bgroup - 1))
                Bcopy[bgroup - 1] = np.append(Bcopy[bgroup - 1], np.array([[node3, nodeBigNew]]), axis=0)
                Bcopy[bgroup - 1][bIndex] = np.array([nodeBigNew, nodeSml])
            else:
                nodeBigc = np.array([[(node3c[0] + V[nodeSml - 1][0]) / 2, (node3c[1] + V[nodeSml - 1][1]) / 2]])
                whereBig = np.where(Vcopy == nodeBigc)[0]
                if whereBig.size == 0:
                    Vcopy = np.append(Vcopy, nodeBigc, axis=0)
                    nodeBigNew = len(Vcopy)
                else:
                    nodeBigNew = whereBig[0] + 1

            # create new elements
            elemNew1 = np.array([[nodeBigNew, nodeSml, nodeBig]])
            elemNew2 = np.array([[nodeBigNew, nodeBig, nodeSmlNew]])
            elemNew3 = np.array([[nodeBigNew, nodeSmlNew, node3]])

            # add to the new E matrix
            Ecopy = np.append(Ecopy, elemNew1, axis=0)
            Ecopy = np.append(Ecopy, elemNew2, axis=0)
            Ecopy[iElem] = elemNew3

        elif sum(elem) == 3:
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
    genGri(fnameOutputRefine, Vcopy, Ecopy, Bcopy)

    # smooth locally refined mesh
    omega = 0.3
    I2E = getI2E(fnameOutputRefine, False)
    F2V = getF2V(fnameOutputRefine, False)

    # create a list containing the 0 based indices of the elements that were affected by the refinement
    elemIndices = np.array([], dtype=int)
    for i, elem in enumerate(eleRefine):
        if sum(elem) != 0:
            elemIndices = np.append(elemIndices, i)
    for i in range(len(Ecopy) - len(E)):
        elemIndices = np.append(elemIndices, len(E) + i)

    # make a dictionary that mapp each node to its neighbors (1 based)
    neighbors = {}
    for i in range(len(I2E)):
        face = F2V[i]
        if face[0] not in F2V[len(I2E):-1]:
            if face[0] not in neighbors:
                neighbors[face[0]] = np.array([face[1]])
            else:
                neighbors[face[0]] = np.append(neighbors[face[0]], face[1])
        if face[1] not in F2V[len(I2E):-1]:
            if face[1] not in neighbors:
                neighbors[face[1]] = np.array([face[0]])
            else:
                neighbors[face[1]] = np.append(neighbors[face[1]], face[0])

    # smooth the elements in elemIndices
    # create the set that contains the nodes to be smoothed
    node_set = set()
    for elem in elemIndices:
        for node in Ecopy[elem]:
            node_set.add(node)

    for node in node_set:
        if node in neighbors:
            neighbor = neighbors[node]
            neighborX = np.array([0, 0])
            for nodeNeighbor in neighbor:
                neighborX = np.add(neighborX, Vcopy[nodeNeighbor - 1])
            newX = (1 - omega) * Vcopy[node - 1] + omega / len(neighbor) * neighborX
            Vcopy[node - 1] = newX

    genGri(fnameOutputSmooth, Vcopy, Ecopy, Bcopy)



def main():
    # refine trailing edge
    localRefine(1, 0, 0.1, 'gri/all.gri', 'gri/refined_local_all_trail.gri', 'gri/smoothed_local_all_trail.gri')

    # refine leading edge
    #localRefine(0, 0, 0.1, 'gri/all.gri', 'gri/refined_local_all_lead.gri', 'gri/smoothed_local_all_lead.gri')

    # refine leading edge
    localRefine(0, 0, 0.1, 'gri/smoothed_local_all_trail.gri', 'gri/refined_local_all.gri', 'gri/smoothed_local_all.gri')
    #localRefine(0, 0, 0.1, 'gri/refined_local_all.gri', 'gri/refined_local_all.gri', 'gri/smoothed_local_all.gri')
    # for fun
    #localRefine(99, 99, 50, 'gri/all.gri', 'gri/shitRefined.gri', 'gri/shitSmoothed.gri')


if __name__ == "__main__":
    main()
