# AE 623 Project 1 — Mesh Generation

Unstructured triangular mesh generation around the three-element airfoil, with
cubic-spline boundary representation, local refinement with smoothing, uniform
refinement, and the face-normal verification test.

## Requirements

Python 3.10 with `numpy`, `scipy`, `sympy`, and `matplotlib`. Mesh generation
also needs [Gmsh](https://gmsh.info/), run once by hand (see step 2 below).

```
pip install numpy scipy sympy matplotlib
```

Run every script from the repository root — all paths are relative to it.

## Layout

| Path | Contents |
| --- | --- |
| `geometries/` | Given airfoil point files (`main.txt`, `slat.txt`, `flap.txt`) and `test.txt` |
| `geo/`, `msh/` | Gmsh input (`all.geo`) and output (`all.msh`) |
| `gri/` | All meshes in `.gri` format |
| `spline/` | Spline fitting code and the fitted splines (`spline_*.npy`) |
| `matrices/` | `I2E`, `B2E`, `In`, `Bn`, `Area` printouts |
| `figures/` | Report figures |

## Pipeline

Each step depends on the previous one.

**1. Geometry → Gmsh input**

```
python main.py
```

Writes `geo/all.geo` from the airfoil point files, then converts `msh/all.msh`
to `gri/all.gri`. On a first run, `msh/all.msh` must already exist — see step 2.

**2. Mesh in Gmsh** *(manual, already done; `msh/all.msh` is committed)*

Open `geo/all.geo` in Gmsh, mesh it 2D, and save as `msh/all.msh` in format 4.1.
Then re-run `python main.py` to regenerate `gri/all.gri`.

**3. Fit the boundary splines**

```
python -m spline.spline
```

Fits an arclength-parameterised cubic spline through each airfoil's points and
writes `spline/spline_main.npy`, `spline_slat.npy`, `spline_flap.npy`. Both
refinement scripts read these to project new boundary nodes onto the true
geometry, so this must run before them.

**4. Local refinement** *(Task 4)*

```
python refinement_local.py
```

Refines around the main trailing edge `(1, 0)` and the leading edge `(0, 0)`,
radius `0.1`, writing the refined and smoothed mesh at each stage. Final mesh is
`gri/smoothed_local_all.gri`.

**5. Uniform refinement** *(Task 5)*

```
python refinement_uniform.py
```

Bisects every edge, splitting each element into four, three times over. Writes
`gri/refinement_uniform_all_{8k,32k,128k}.gri`.

**6. Matrices and verification** *(Tasks 2 and 3)*

```
python matrices_generator.py     # matrices for gri/test.gri -> matrices/
python mesh_verification.py      # verification table for every mesh
```

**7. Figures**

```
python plotgri.py
```

`plotmesh(mesh, fname)` saves to `fname`, or displays interactively when passed
an empty `fname`. Pass `showNodes=True` to overlay node markers.

## Results

| Mesh | File | Elements | max \|E_e\| |
| --- | --- | ---: | ---: |
| Test | `gri/test.gri` | 2 | 0.000e+00 |
| Coarse | `gri/all.gri` | 1288 | 5.024e-15 |
| Locally refined | `gri/smoothed_local_all.gri` | 2149 | 5.024e-15 |
| Uniform ×1 | `gri/refinement_uniform_all_8k.gri` | 8596 | 3.972e-15 |
| Uniform ×2 | `gri/refinement_uniform_all_32k.gri` | 34384 | 1.776e-15 |
| Uniform ×3 | `gri/refinement_uniform_all_128k.gri` | 137536 | 8.951e-16 |

`E_e = sum_i n_ei^outward * l_ei` is zero to machine precision on every element,
as required.

## Conventions

**Boundary groups** are 1-based and ordered as written by `msh2gri.py`:

| Group | Name | Geometry |
| --- | --- | --- |
| 1 | `main` | Main element, x ∈ [0, 1] |
| 2 | `slat` | Slat, x ∈ [−0.192, −0.026] |
| 3 | `flap` | Flap, x ∈ [1.039, 1.267] |
| 4–7 | `bot`, `right`, `top`, `left` | Farfield box, [−100, 100]² |

Groups 1–3 are curved and have splines; `spline.AIRFOIL_GROUPS` is the single
source of truth for which those are. `msh2gri.py` labels the airfoil groups by
matching the mean x of each group's nodes against the geometry files rather than
by assuming an emission order, so the labels stay correct if Gmsh reorders the
curves.

**Local faces** are 1-based, with face *i* opposite local node *i*
(`matrices_generator.LOCAL_FACE_NODES`). Node ordering is counterclockwise, so
face normals `(dy, -dx)` point out of the element.

**Complexity.** All mesh operations are O(N). Element edges are hashed on their
sorted node pair, and new nodes during refinement are keyed on the edge they
split, which both keeps the mesh conforming and avoids any search over existing
nodes. Measured timings scale ~4× per 4× elements.

## Source files

| File | Purpose |
| --- | --- |
| `main.py` | Drives geometry → `.geo` → `.gri` |
| `txt2geo.py` | Airfoil points → Gmsh `.geo` |
| `msh2gri.py` | Gmsh `.msh` → `.gri` |
| `matrices_generator.py` | `.gri` I/O, `I2E`/`B2E`/`In`/`Bn`/`Area`, verification |
| `mesh_verification.py` | Verification table across all meshes |
| `refinement_local.py` | Flagged-region refinement with smoothing |
| `refinement_uniform.py` | Uniform edge-bisection refinement |
| `spline/spline.py` | Cubic spline fitting and boundary projection |
| `plotgri.py` | Mesh plotting |
