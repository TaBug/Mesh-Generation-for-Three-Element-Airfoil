# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:47:46 2023

@author: liton
"""
import numpy as np


def msh2gri(fnameInput, fnameOutput):
    # input file
    with open(fnameInput, "r") as f:
        lines = f.readlines()

    maintxt = np.loadtxt('Airfoil Geometries/main.txt')
    flap = np.loadtxt('Airfoil Geometries/flap.txt')
    slat = np.loadtxt('Airfoil Geometries/slat.txt')
    refactor = 5

    # output .gri file
    with open(fnameOutput, "w") as f:
        iEntities = lines.index('$Entities\n')
        numPoints, numCurves, numSurfaces, numVolumes = map(int, lines[iEntities + 1].strip().split())
        iNodes = lines.index('$Nodes\n')
        numEntityBlocks, numNodes, minNodeTag, maxNodeTag = map(int, lines[iNodes + 1].strip().split())
        iElements = lines.index('$Elements\n')

        f.write('             ')
        # Node coordinates
        index = iNodes + 2
        for i in range(numPoints + numCurves + numSurfaces):
            entityDim, entityTag, parametric, numNodesInBlock = map(int, lines[index].strip().split())
            index += numNodesInBlock
            for j in range(numNodesInBlock):
                index += 1
                x, y, z = map(float, lines[index].strip().split())
                f.write(f"{x} {y}\n")
            index += 1

        # Boundaries
        nBGroup = 7
        f.write(f'{nBGroup}\n')
        index = iElements + 2 + numPoints * 2
        lengths = [int(len(maintxt)/refactor), int(len(flap)/refactor), int(len(slat)/refactor), 5, 5, 5, 5]
        tags = ['main', 'flap', 'slat', 'bot', 'right', 'top', 'left']
        for i in range(nBGroup):
            f.write(f"{lengths[i]} 2 {tags[i]}\n")
            for j in range(lengths[i]):
                index += 1
                elementTag, node1, node2 = map(int, lines[index].strip().split())
                f.write(f"{node1} {node2}\n")
                if i < 3:
                    index += 1
            if i > 2:
                index += 1


        # Elements
        nElemGroup = 1
        for elemGroup in range(nElemGroup):
            entityDim, entityTag, elementType, numElementsInBlock = map(int, lines[index].strip().split())
            f.write(f"{numElementsInBlock} 1 TriLagrange\n")
            for i in range(numElementsInBlock):
                index += 1
                elementTag, node1, node2, node3 = map(int, lines[index].strip().split())
                f.write(f"{node1} {node2} {node3}\n")

        string = f"{numNodes} {numElementsInBlock} 2\n"
        f.seek(0, 0)
        f.write(string)
        f.close()


if __name__ == "__main__":
    msh2gri('all.msh', 'all.gri')
