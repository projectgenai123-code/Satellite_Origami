import shutil
import random
from pathlib import Path

# ==============================
# SETTINGS
# ==============================
SPLIT_RATIO = 0.8   # 80% train, 20% validation
RANDOM_SEED = 42
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]

random.seed(RANDOM_SEED)

# ==============================
# PATHS (YOUR EXACT PATH)
# ==============================
SOURCE_DIR = Path(
    r"C:/Users/Sarika Kannan/satellite_origami/Satellite_Origami/data/processed_images"
)

DEST_DIR = Path(
    r"C:/Users/Sarika Kannan/satellite_origami/Satellite_Origami/data/processed_images_split"
)

TRAIN_DIR = DEST_DIR / "train"
VAL_DIR = DEST_DIR / "val"

# ==============================
# CREATE OUTPUT FOLDERS
# ==============================
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# COLLECT IMAGES
# ==============================
images = [
    img for img in SOURCE_DIR.iterdir()
    if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Found {len(images)} images")

if len(images) == 0:
    raise RuntimeError("❌ No images found in source directory")

# ==============================
# SPLIT
# ==============================
random.shuffle(images)

split_idx = int(len(images) * SPLIT_RATIO)
train_images = images[:split_idx]
val_images = images[split_idx:]

# ==============================
# COPY FILES
# ==============================
for img in train_images:
    shutil.copy2(img, TRAIN_DIR / img.name)

for img in val_images:
    shutil.copy2(img, VAL_DIR / img.name)

print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print("\n✅ Dataset split completed successfully!")
