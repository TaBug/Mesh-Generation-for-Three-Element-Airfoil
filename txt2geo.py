import numpy as np


def txt2geo(maintxt, flap, slat):
    # Open the output .geo file
    with open("geo/all.geo", "w") as f:
        # Write the node coordinates to the .geo file
        for i, line in enumerate(maintxt):
            x = maintxt[i][0]
            y = maintxt[i][1]
            f.write(f"Point({i + 1}) = {{{x}, {y}, 0}};\n")
        for i, line in enumerate(flap):
            index = i + len(maintxt)
            x = flap[i][0]
            y = flap[i][1]
            f.write(f"Point({index + 1}) = {{{x}, {y}, 0}};\n")
        for i, line in enumerate(slat):
            index = i + len(maintxt) + len(flap)
            x = slat[i][0]
            y = slat[i][1]
            f.write(f"Point({index + 1}) = {{{x}, {y}, 0}};\n")

        # Write the instructions for connecting the nodes to form a triangular mesh
        counter = 1
        resFactor = 5
        for i in range(1, len(maintxt) - resFactor, resFactor):
            f.write(f"Line({counter}) = {{{i}, {i + resFactor}}};\n")
            counter += 1
            i -= 1
        f.write(f"Line({counter}) = {{{len(maintxt) - resFactor}, {1}}};\n")
        counter += 1
        for i in range(1, len(flap) - resFactor, resFactor):
            index = i + len(maintxt)
            f.write(f"Line({counter}) = {{{index}, {index + resFactor}}};\n")
            counter += 1
            i -= 1
        f.write(f"Line({counter}) = {{{len(maintxt) + len(flap) - resFactor}, {len(maintxt) + 1}}};\n")
        counter += 1
        for i in range(1, len(slat) - resFactor, resFactor):
            index = i + len(maintxt) + len(flap)
            f.write(f"Line({counter}) = {{{index}, {index + resFactor}}};\n")
            counter += 1
            i -= 1
        f.write(
            f"Line({counter}) = {{{len(maintxt) + len(flap) + len(slat) - resFactor}, {len(maintxt) + len(slat) + 1}}};\n")

        boxSize = 100
        Nnodes = len(maintxt) + len(flap) + len(slat)
        f.write(f"Point({Nnodes + 1}) = {{{-boxSize}, {-boxSize}, 0}};\n")
        f.write(f"Point({Nnodes + 2}) = {{{boxSize}, {-boxSize}, 0}};\n")
        f.write(f"Point({Nnodes + 3}) = {{{boxSize}, {boxSize}, 0}};\n")
        f.write(f"Point({Nnodes + 4}) = {{{-boxSize}, {boxSize}, 0}};\n")

        f.write(f"Line({counter + 1}) = {{{Nnodes + 1}, {Nnodes + 2}}};\n")
        f.write(f"Line({counter + 2}) = {{{Nnodes + 2}, {Nnodes + 3}}};\n")
        f.write(f"Line({counter + 3}) = {{{Nnodes + 3}, {Nnodes + 4}}};\n")
        f.write(f"Line({counter + 4}) = {{{Nnodes + 4}, {Nnodes + 1}}};\n")
        # Create a physical group for the elements of the mesh
        f.write(f"Line Loop(1) = {{1:{counter}}};\n")
        f.write(f"Line Loop(2) = {{{counter + 1}:{counter + 4}}};\n")
        f.write("Plane Surface(1) = {1, 2};\n")

        f.write(f"Mesh.MeshSizeFactor  = {1.5};\n")
        f.write(f"Mesh.Algorithm  = 5;\n")  # 2D mesh algorithm
        f.write("Mesh 2;")

        f.close()


def main():
    maintxt = np.loadtxt('geometries/main.txt')
    slat = np.loadtxt('geometries/slat.txt')
    flap = np.loadtxt('geometries/flap.txt')
    txt2geo(maintxt, flap, slat)


if __name__ == "__main__":
    main()
