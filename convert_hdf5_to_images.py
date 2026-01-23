import h5py
import os
import matplotlib.pyplot as plt

HDF5_PATH = r"C:\Users\Sarika Kannan\satellite_origami\Satellite_Origami\CreasePatternDataset\CreasePatternDataset\dataset.hdf5"
OUT_DIR = "./data/processed_images"
os.makedirs(OUT_DIR, exist_ok=True)

h5 = h5py.File(HDF5_PATH, 'r')

print("✅ HDF5 loaded")
print("Top-level keys:", list(h5.keys()))

MAX_IMAGES = 2000
count = 0

for group_id in h5["edge_lists"].keys():
    for sample_id in h5[f"edge_lists/{group_id}"].keys():

        if count >= MAX_IMAGES:
            break

        try:
            vertices = h5[f"vertex_coords/{group_id}/{sample_id}"][:]
            edges = h5[f"edge_lists/{group_id}/{sample_id}"][:]

            plt.figure(figsize=(1, 1))
            for e in edges:
                v1, v2 = int(e[0]), int(e[1])
                x1, y1 = vertices[v1]
                x2, y2 = vertices[v2]
                plt.plot([x1, x2], [y1, y2], 'k', linewidth=1)

            plt.axis('off')
            plt.savefig(
                os.path.join(OUT_DIR, f"{count}.png"),
                dpi=64,
                bbox_inches='tight',
                pad_inches=0
            )
            plt.close()

            if count % 100 == 0:
                print(f"Generated {count} images")

            count += 1

        except Exception:
            continue

    if count >= MAX_IMAGES:
        break

print("🎉 DONE — Images generated:", count)
