"""Convert a .gri mesh to a Gmsh .msh file, the inverse of msh2gri.py.

A .gri holds only a mesh, with no underlying CAD geometry, so the entities
written here are synthetic: one curve per boundary group and one surface for
the triangles. Each gets a physical group named after the .gri boundary group,
which is what carries the `main`/`slat`/`flap`/farfield labels through Gmsh.
"""
import argparse
import glob
import os

from matrices_generator import readgri

# Gmsh element type codes for the two shapes a .gri can hold.
LINE2 = 1
TRI3 = 2

# Physical group name given to the triangles; the boundary groups keep their
# own .gri names.
INTERIOR_NAME = 'interior'


def _bbox(V, nodes):
    """Axis-aligned bounding box of 1-based `nodes`, as Gmsh entities want it."""
    xs = [V[n - 1][0] for n in nodes]
    ys = [V[n - 1][1] for n in nodes]
    return min(xs), min(ys), 0.0, max(xs), max(ys), 0.0


def _groupNames(B, Bname):
    """Boundary group names, filled in positionally when the .gri omits them."""
    if Bname is not None and len(Bname) == len(B):
        return list(Bname)
    return [f'bgroup{i + 1}' for i in range(len(B))]


def _write41(f, V, E, B, names):
    """Write MSH format 4.1, the version Gmsh writes for this project."""
    nB = len(B)
    # physical tags: 1..nB for the boundary curves, nB + 1 for the surface
    surfacePhys = nB + 1

    f.write('$MeshFormat\n4.1 0 8\n$EndMeshFormat\n')

    f.write('$PhysicalNames\n')
    f.write(f'{nB + 1}\n')
    for i, name in enumerate(names):
        f.write(f'1 {i + 1} "{name}"\n')
    f.write(f'2 {surfacePhys} "{INTERIOR_NAME}"\n')
    f.write('$EndPhysicalNames\n')

    # ---- entities: no points, one curve per boundary group, one surface.
    # Curves declare no bounding points, which is legal and keeps a mesh-only
    # file from inventing geometry it does not have.
    f.write('$Entities\n')
    f.write(f'0 {nB} 1 0\n')
    for i, bGroup in enumerate(B):
        minX, minY, minZ, maxX, maxY, maxZ = _bbox(V, bGroup.flatten())
        f.write(f'{i + 1} {minX} {minY} {minZ} {maxX} {maxY} {maxZ} '
                f'1 {i + 1} 0\n')
    minX, minY, minZ, maxX, maxY, maxZ = _bbox(V, range(1, len(V) + 1))
    curveTags = ' '.join(str(i + 1) for i in range(nB))
    f.write(f'1 {minX} {minY} {minZ} {maxX} {maxY} {maxZ} '
            f'1 {surfacePhys} {nB}{" " if nB else ""}{curveTags}\n')
    f.write('$EndEntities\n')

    # ---- nodes, all classified on the surface in a single block. Gmsh does
    # not require boundary nodes to be classified on the curves they lie on.
    f.write('$Nodes\n')
    f.write(f'1 {len(V)} 1 {len(V)}\n')
    f.write(f'2 1 0 {len(V)}\n')
    for n in range(1, len(V) + 1):
        f.write(f'{n}\n')
    for x, y in V:
        f.write(f'{x} {y} 0\n')
    f.write('$EndNodes\n')

    # ---- elements: boundary lines first, then the triangles, tagged with one
    # global counter so the tag range stays contiguous.
    numElements = sum(len(bGroup) for bGroup in B) + len(E)
    f.write('$Elements\n')
    f.write(f'{nB + 1} {numElements} 1 {numElements}\n')
    tag = 0
    for i, bGroup in enumerate(B):
        f.write(f'1 {i + 1} {LINE2} {len(bGroup)}\n')
        for node1, node2 in bGroup:
            tag += 1
            f.write(f'{tag} {node1} {node2}\n')
    f.write(f'2 1 {TRI3} {len(E)}\n')
    for node1, node2, node3 in E:
        tag += 1
        f.write(f'{tag} {node1} {node2} {node3}\n')
    f.write('$EndElements\n')


def _write22(f, V, E, B, names):
    """Write MSH format 2.2, for tools that do not read 4.1."""
    nB = len(B)
    surfacePhys = nB + 1

    f.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')

    f.write('$PhysicalNames\n')
    f.write(f'{nB + 1}\n')
    for i, name in enumerate(names):
        f.write(f'1 {i + 1} "{name}"\n')
    f.write(f'2 {surfacePhys} "{INTERIOR_NAME}"\n')
    f.write('$EndPhysicalNames\n')

    f.write('$Nodes\n')
    f.write(f'{len(V)}\n')
    for n, (x, y) in enumerate(V, start=1):
        f.write(f'{n} {x} {y} 0\n')
    f.write('$EndNodes\n')

    # 2.2 carries the tags on each element: physical group then geometric
    # entity, which here are numbered alike.
    numElements = sum(len(bGroup) for bGroup in B) + len(E)
    f.write('$Elements\n')
    f.write(f'{numElements}\n')
    tag = 0
    for i, bGroup in enumerate(B):
        for node1, node2 in bGroup:
            tag += 1
            f.write(f'{tag} {LINE2} 2 {i + 1} {i + 1} {node1} {node2}\n')
    for node1, node2, node3 in E:
        tag += 1
        f.write(f'{tag} {TRI3} 2 {surfacePhys} 1 {node1} {node2} {node3}\n')
    f.write('$EndElements\n')


WRITERS = {'4.1': _write41, '2.2': _write22}


def gri2msh(fnameInput, fnameOutput, version='4.1'):
    """Convert the .gri at `fnameInput` to a Gmsh .msh at `fnameOutput`."""
    if version not in WRITERS:
        raise ValueError(f'unsupported .msh version {version!r}, '
                         f'expected one of {sorted(WRITERS)}')

    mesh = readgri(fnameInput)
    V, E, B = mesh['V'], mesh['E'], mesh['B']
    names = _groupNames(B, mesh['Bname'])

    # .gri and .msh are both 1-based, so the connectivity copies over directly;
    # a bad index would otherwise surface only as a silent Gmsh failure.
    numNodes = len(V)
    for group in list(B) + [E]:
        if len(group) and (group.min() < 1 or group.max() > numNodes):
            raise ValueError(f'{fnameInput} references a node outside '
                             f'1..{numNodes}')

    directory = os.path.dirname(fnameOutput)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(fnameOutput, 'w') as f:
        WRITERS[version](f, V, E, B, names)

    return {'nodes': numNodes, 'elements': len(E), 'groups': names}


def defaultOutput(fnameInput):
    """Output path for an input .gri, mirroring the gri/ and msh/ layout."""
    stem = os.path.splitext(os.path.basename(fnameInput))[0]
    return os.path.join('msh', f'{stem}.msh')


def convert(fnameInput, fnameOutput, version, force):
    """Convert one mesh, reporting what happened. Returns True if written."""
    # msh/all.msh is the committed Gmsh output that gri/all.gri came from, so
    # clobbering it by default would quietly destroy the source of the pipeline.
    if os.path.exists(fnameOutput) and not force:
        print(f'{fnameInput} -> {fnameOutput}: exists, skipped (--force to '
              f'overwrite)')
        return False

    info = gri2msh(fnameInput, fnameOutput, version=version)
    print(f"{fnameInput} -> {fnameOutput} (format {version}): "
          f"{info['nodes']} nodes, {info['elements']} elements, "
          f"boundary groups {', '.join(info['groups'])}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Convert a .gri mesh to a Gmsh .msh file. With no '
                    'arguments, converts every gri/*.gri into msh/.')
    parser.add_argument('input', nargs='?',
                        help='input .gri file (default: every gri/*.gri)')
    parser.add_argument('output', nargs='?',
                        help='output .msh file (default: msh/<name>.msh)')
    parser.add_argument('--version', default='4.1', choices=sorted(WRITERS),
                        help='.msh format version to write (default: 4.1)')
    parser.add_argument('--force', action='store_true',
                        help='overwrite outputs that already exist')
    args = parser.parse_args()

    if args.input is not None:
        output = args.output or defaultOutput(args.input)
        if not convert(args.input, output, args.version, args.force):
            raise SystemExit(1)
        return

    # bare `python gri2msh.py`, matching how the other scripts in this project
    # run: batch the whole gri/ directory, leaving existing .msh files alone.
    inputs = sorted(glob.glob(os.path.join('gri', '*.gri')))
    if not inputs:
        parser.error('no gri/*.gri found; run from the repository root or '
                     'name an input file')

    written = sum(convert(f, defaultOutput(f), args.version, args.force)
                  for f in inputs)
    print(f'\n{written} of {len(inputs)} converted into msh/')


if __name__ == "__main__":
    main()
