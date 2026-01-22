# Origami Crease Pattern Dataset
The accompanying h5df file comprises the dataset of 8 million crease patterns used for the 
training of a VAE with normalizing flows as described in the paper:
"Inverse Design of Origami for Trajectory Following", Journal of Mechanisms and Robotics, 2025.
 
#### Authors
- Nicolas Hochuli (a), [nhochuli@ethz.ch](nhochuli@ethz.ch)
- Tino Stankovic (a,1), [tinos@ethz.ch](tinos@ethz.ch)

(a) Engineering Design and Computing Laboratory, Department of Mechanical and Process
    Engineering, ETH Zurich, Tannenstrasse 3, 8092 Zurich, Switzerland \
(1) To whom correspondence should be addressed

## Usage
The h5df file `dataset.hdf5` can be opened in Python using the pip-installable `h5py` package.
The h5df format is a widely used format for storing large data in single files. Basic usage
is described in [docs.h5py.org](https://docs.h5py.org/en/stable/quick.html#quick)

The file `dataset.hdf5` holds the connectivity information (`edge_lists`, shape: `[Ex2]`, E=Number of edges)
and geometric vertex positions (`vertex_coords`, shape: `[Nx2]`, N=Number of vertices) for all 8 million
planar crease patterns in two separate groups.

### File Structure
The hierarchical file structure in `dataset.hdf5` is organized as described here.
For efficiency reasons, the two groups `edge_lists` and  `vertex_coords` are split into 400 subgroups each,
where each subgroup holds 20000 datasets (i.e. samples / crease patterns). The tree structure is defined as:

```bash
dataset.hdf5        
├── metadata (attributes)
├── edge_lists/     ← Connectivity information (edge lists) of all 8 million crease patterns, in 400 subgroups.
│   ├── 0/          ← In a nested group, holds 20000 samples (as so-called datasets)
│   ├── 1/          ← In a nested group, holds 20000 samples (as so-called datasets)
│   ├── ...
│   └── 399/
└── vertex_coords/  ← Vertex position information (2D) of all 8 million crease patterns.
    ├── 0/
    ├── 1/
    ├── ...
    └── 399/
```

### Example
To load the dataset into memory using Python, printing the metadata, retrieving the geometry of a single crease pattern 
and generating a visualization, run this code:

```python
import h5py
import matplotlib.pyplot as plt

h5 = h5py.File("dataset.hdf5", "r")

# Example: Get creasepattern 18340 from subgroup 20
edge_list = h5["edge_lists/20/18340"][:,:]  # Numpy array (Ex2)
vertex_coords = h5["vertex_coords/20/18340"][:,:]  # Numpy array (Nx2)

# Print all metadata attributes 
print(h5["metadata"].attrs["Title"])
print(h5["metadata"].attrs["Journal"])
print(h5["metadata"].attrs["Authors"])
print(h5["metadata"].attrs["Year"])  
print(h5["metadata"].attrs["Size"]) 
print(h5["metadata"].attrs["Version"])

### Create plot
plt.figure(figsize=(8, 6))
for edge in edge_list:
    p1 = vertex_coords[edge[0]]
    p2 = vertex_coords[edge[1]]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=1)

plt.scatter(vertex_coords[:, 0], vertex_coords[:, 1], color="blue", zorder=3)
for idx, (x, y) in enumerate(vertex_coords):
    plt.text(x, y, str(idx), fontsize=8, ha="right", va="bottom")

plt.axis("equal")
plt.title("Crease Pattern Visualization")
plt.grid(True)
plt.tight_layout()
plt.show()
```
