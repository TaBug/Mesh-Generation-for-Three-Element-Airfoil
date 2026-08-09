# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 20:47:46 2023

@author: liton
"""
import numpy as np

# The three airfoil elements are well separated in x (slat ~ -0.1, main ~ 0.5,
# flap ~ 1.15), so a boundary group can be identified from the mean x of its
# nodes. Naming the groups this way keeps the .gri labels correct regardless of
# the order Gmsh happens to emit the curves in; hard-coding that order is what
# previously swapped the slat and flap, silently snapping refined nodes onto the
# wrong spline a full chord away.
AIRFOIL_NAMES = ('main', 'slat', 'flap')
FARFIELD_NAMES = ('bot', 'right', 'top', 'left')


def msh2gri(fnameInput, fnameOutput):
    with open(fnameInput, "r") as f:
        lines = f.readlines()

    geometries = {name: np.loadtxt(f'geometries/{name}.txt')
                  for name in AIRFOIL_NAMES}
    refactor = 5

    nBFace = {int(len(pts) / refactor) for pts in geometries.values()}
    if len(nBFace) != 1:
        raise ValueError('airfoil point files must have equal point counts; '
                         'the boundary blocks are read positionally')
    nBFace = nBFace.pop()

    iEntities = lines.index('$Entities\n')
    numPoints, numCurves, numSurfaces, numVolumes = map(int, lines[iEntities + 1].strip().split())
    iNodes = lines.index('$Nodes\n')
    numEntityBlocks, numNodes, minNodeTag, maxNodeTag = map(int, lines[iNodes + 1].strip().split())
    iElements = lines.index('$Elements\n')

    # ---- node coordinates
    coords = []
    index = iNodes + 2
    for i in range(numPoints + numCurves + numSurfaces):
        entityDim, entityTag, parametric, numNodesInBlock = map(int, lines[index].strip().split())
        index += numNodesInBlock
        for j in range(numNodesInBlock):
            index += 1
            x, y, z = map(float, lines[index].strip().split())
            coords.append((x, y))
        index += 1

    # ---- boundary faces, three airfoil groups followed by the four box sides
    nBGroup = len(AIRFOIL_NAMES) + len(FARFIELD_NAMES)
    lengths = [nBFace] * len(AIRFOIL_NAMES) + [5] * len(FARFIELD_NAMES)
    index = iElements + 2 + numPoints * 2
    groups = []
    for i in range(nBGroup):
        edges = []
        for j in range(lengths[i]):
            index += 1
            elementTag, node1, node2 = map(int, lines[index].strip().split())
            edges.append((node1, node2))
            # each airfoil line is its own entity block, so skip its header
            if i < len(AIRFOIL_NAMES):
                index += 1
        if i >= len(AIRFOIL_NAMES):
            index += 1
        groups.append(edges)

    # ---- elements
    entityDim, entityTag, elementType, numElementsInBlock = map(int, lines[index].strip().split())
    elems = []
    for i in range(numElementsInBlock):
        index += 1
        elementTag, node1, node2, node3 = map(int, lines[index].strip().split())
        elems.append((node1, node2, node3))

    # ---- label the airfoil groups by geometry, not by emission order
    reference = {name: pts[:, 0].mean() for name, pts in geometries.items()}
    tags = []
    for edges in groups[:len(AIRFOIL_NAMES)]:
        x = np.mean([coords[n - 1][0] for edge in edges for n in edge])
        tags.append(min(reference, key=lambda name: abs(reference[name] - x)))
    if sorted(tags) != sorted(AIRFOIL_NAMES):
        raise ValueError(f'boundary groups did not match the airfoils one to '
                         f'one, got {tags}; check the .msh curve layout')
    tags += list(FARFIELD_NAMES)

    # ---- output .gri file
    with open(fnameOutput, "w") as f:
        f.write(f"{numNodes} {numElementsInBlock} 2\n")
        for x, y in coords:
            f.write(f"{x} {y}\n")

        f.write(f"{nBGroup}\n")
        for tag, edges in zip(tags, groups):
            f.write(f"{len(edges)} 2 {tag}\n")
            for node1, node2 in edges:
                f.write(f"{node1} {node2}\n")

        f.write(f"{numElementsInBlock} 1 TriLagrange\n")
        for node1, node2, node3 in elems:
            f.write(f"{node1} {node2} {node3}\n")


if __name__ == "__main__":
    msh2gri('msh/all.msh', 'gri/all.gri')
