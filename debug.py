import os

BASE_DIR = r"C:\Users\Sarika Kannan\satellite_origami\Satellite_Origami"

print("🔍 Walking directory tree...\n")

found = False
for root, dirs, files in os.walk(BASE_DIR):
    if "dataset.hdf5" in files:
        print("✅ FOUND dataset.hdf5 at:")
        print(os.path.join(root, "dataset.hdf5"))
        found = True

if not found:
    print("❌ dataset.hdf5 NOT FOUND anywhere under:")
    print(BASE_DIR)
