import numpy as np

from txt2geo import txt2geo
from msh2gri import msh2gri


def main():
    """Rebuild the coarse mesh from the airfoil point files.

    Step two runs outside this script: open geo/all.geo in Gmsh and mesh it to
    produce msh/all.msh, then re-run to convert it to gri/all.gri.
    """
    maintxt = np.loadtxt('geometries/main.txt')
    flap = np.loadtxt('geometries/flap.txt')
    slat = np.loadtxt('geometries/slat.txt')

    # convert geometries .txt to .geo
    txt2geo(maintxt, flap, slat)
    # after generating the mesh file from Gmsh, convert the .msh to .gri
    msh2gri('msh/all.msh', 'gri/all.gri')


if __name__ == "__main__":
    main()
