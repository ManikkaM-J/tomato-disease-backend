"""
build_dataset.py
-----------------
STEP 1: Build Validator_Dataset from scratch.

This script ONLY builds the dataset. It does NOT train.
After running this, check the dataset, then run:
    python train_stage1_validator.py

DATASET STRUCTURE CREATED:
---------------------------
Validator_Dataset/
    tomato_leaf/   <- 2000 real tomato leaf images
    non_tomato/    <- 2000 mixed non-tomato images:
                      800 plant leaves (pepper, potato, corn, grape...)
                      400 animals
                      400 food/fruits
                      400 synthetic (sky, soil, objects, random)

SOURCES:
--------
Tomato images   -> PlantVillageDataset/PlantVillage/
Plant leaves    -> kaggle_downloads/plantvillage dataset/color/
Animals         -> kaggle_downloads/raw-img/
Food            -> kaggle_downloads/ (fruits dataset)
Synthetic       -> Generated automatically if needed

HOW TO RUN:
-----------
1. Activate venv:   venv/Scripts/activate
2. Run:             python build_dataset.py
3. Check dataset:   ls Validator_Dataset/tomato_leaf
4. Then train:      python train_stage1_validator.py
"""

import os
import sys
import shutil
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PIL import Image

print(f"Project root: {PROJECT_ROOT}")

# ── Configuration ─────────────────────────────────────────────────────────────

VALIDATOR_DATASET_DIR = os.path.join(PROJECT_ROOT, "Validator_Dataset")
DOWNLOAD_DIR          = os.path.join(PROJECT_ROOT, "kaggle_downloads")

# Tomato source — your PlantVillage dataset
TOMATO_SOURCE = r"D:\PROJECT\tomato_leaf_backend_v2\PlantVillageDataset\PlantVillage"

# Images per class
TOMATO_COUNT      = 2000
NON_TOMATO_COUNT  = 2000

# Distribution of non-tomato images
PLANT_LEAVES_COUNT = 800   # other plant leaves (most important!)
ANIMALS_COUNT      = 400   # animals
FOOD_COUNT         = 400   # food/fruits
SYNTHETIC_COUNT    = 400   # generated synthetic images

IMAGE_SIZE = (224, 224)

# PlantVillage color folder
PLANTVILLAGE_COLOR = os.path.join(
    DOWNLOAD_DIR, "plantvillage dataset", "color"
)

# Non-tomato plant folders from PlantVillage
NON_TOMATO_PLANT_FOLDERS = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Blueberry___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
]

# Kaggle datasets to download
KAGGLE_DATASETS = [
    "abdallahalidev/plantvillage-dataset",
    "alessiocorrado99/animals10",
    "moltean/fruits",
]


# ── Download ──────────────────────────────────────────────────────────────────

def download_dataset(dataset: str) -> bool:
    print(f"\n[Download] Downloading: {dataset}")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset,
             "--path", DOWNLOAD_DIR, "--unzip"],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode == 0:
            print(f"[Download] Done: {dataset}")
            return True
        else:
            print(f"[Download] Failed: {result.stderr[:150]}")
            return False
    except Exception as e:
        print(f"[Download] Error: {e}")
        return False


def download_all():
    print("\n" + "=" * 60)
    print("  STEP 1: Downloading Datasets from Kaggle")
    print("=" * 60)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for ds in KAGGLE_DATASETS:
        try:
            download_dataset(ds)
        except Exception as e:
            print(f"[Warning] {ds}: {e}")


# ── Image utilities ───────────────────────────────────────────────────────────

def copy_images(src_dir, dst_dir, max_images, label, start_idx=0):
    os.makedirs(dst_dir, exist_ok=True)
    count      = 0
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    for root, _, files in os.walk(src_dir):
        for fname in files:
            if count >= max_images:
                return count
            if os.path.splitext(fname)[1] not in extensions:
                continue
            src  = os.path.join(root, fname)
            dst  = os.path.join(dst_dir, f"{label}_{start_idx+count:05d}.jpg")
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                img.save(dst, "JPEG", quality=92)
                count += 1
            except Exception:
                continue
    return count


def copy_from_folders(base_dir, folder_names, dst_dir, max_images, label, start_idx=0):
    os.makedirs(dst_dir, exist_ok=True)
    total      = 0
    per_folder = max(1, max_images // max(len(folder_names), 1))
    for folder in folder_names:
        if total >= max_images:
            break
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            continue
        to_copy = min(per_folder, max_images - total)
        copied  = copy_images(
            src_dir    = path,
            dst_dir    = dst_dir,
            max_images = to_copy,
            label      = f"{label}_{folder[:8].replace(' ','_')}",
            start_idx  = start_idx + total,
        )
        total += copied
    return total


def generate_synthetic(dst_dir, count, start_idx):
    os.makedirs(dst_dir, exist_ok=True)
    generated = 0
    for i in range(count):
        np.random.seed(i + start_idx + 77777)
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        p   = i % 10
        if p == 0:
            img[:,:,0] = np.random.randint(100,180,(224,224))
            img[:,:,1] = np.random.randint(150,220,(224,224))
            img[:,:,2] = np.random.randint(200,255,(224,224))
        elif p == 1:
            img[:,:,0] = np.random.randint(120,200,(224,224))
            img[:,:,1] = np.random.randint(70,130,(224,224))
            img[:,:,2] = np.random.randint(20,70,(224,224))
        elif p == 2:
            v = np.random.randint(120,220,(224,224))
            img[:,:,0] = v; img[:,:,1] = v; img[:,:,2] = v
        elif p == 3:
            img[:,:,0] = np.random.randint(180,255,(224,224))
            img[:,:,1] = np.random.randint(30,100,(224,224))
            img[:,:,2] = np.random.randint(10,60,(224,224))
        elif p == 4:
            img[:,:,0] = np.random.randint(0,50,(224,224))
            img[:,:,1] = np.random.randint(0,50,(224,224))
            img[:,:,2] = np.random.randint(0,50,(224,224))
        elif p == 5:
            img[:,:,0] = np.random.randint(80,140,(224,224))
            img[:,:,1] = np.random.randint(20,70,(224,224))
            img[:,:,2] = np.random.randint(100,180,(224,224))
        elif p == 6:
            img[:,:,0] = np.random.randint(200,255,(224,224))
            img[:,:,1] = np.random.randint(180,240,(224,224))
            img[:,:,2] = np.random.randint(20,80,(224,224))
        elif p == 7:
            v = np.random.randint(200,255,(224,224))
            img[:,:,0] = v; img[:,:,1] = v; img[:,:,2] = v
        elif p == 8:
            img[:,:,0] = np.random.randint(0,255,(224,224))
            img[:,:,1] = np.random.randint(0,255,(224,224))
            img[:,:,2] = np.random.randint(0,255,(224,224))
        elif p == 9:
            img[:,:,0] = np.random.randint(10,80,(224,224))
            img[:,:,1] = np.random.randint(150,220,(224,224))
            img[:,:,2] = np.random.randint(150,220,(224,224))
        Image.fromarray(img).save(
            os.path.join(dst_dir, f"synthetic_{start_idx+i:05d}.jpg"),
            quality=85,
        )
        generated += 1
    return generated


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset():
    print("\n" + "=" * 60)
    print("  STEP 2: Building Validator_Dataset from Scratch")
    print("=" * 60)

    # Remove old dataset
    if os.path.exists(VALIDATOR_DATASET_DIR):
        shutil.rmtree(VALIDATOR_DATASET_DIR)
        print("[Dataset] Removed old Validator_Dataset.")

    tomato_dst     = os.path.join(VALIDATOR_DATASET_DIR, "tomato_leaf")
    non_tomato_dst = os.path.join(VALIDATOR_DATASET_DIR, "non_tomato")
    non_count      = 0

    # ── TOMATO IMAGES ─────────────────────────────────────────────────────
    print(f"\n[Tomato] Copying {TOMATO_COUNT} tomato leaf images...")
    if os.path.exists(TOMATO_SOURCE):
        tomato_count = copy_images(
            src_dir    = TOMATO_SOURCE,
            dst_dir    = tomato_dst,
            max_images = TOMATO_COUNT,
            label      = "tomato",
        )
    else:
        print(f"[Warning] Tomato source not found: {TOMATO_SOURCE}")
        pv_color = PLANTVILLAGE_COLOR
        if os.path.exists(pv_color):
            tomato_folders = [
                f for f in os.listdir(pv_color)
                if f.startswith("Tomato") and
                os.path.isdir(os.path.join(pv_color, f))
            ]
            tomato_count = copy_from_folders(
                base_dir     = pv_color,
                folder_names = tomato_folders,
                dst_dir      = tomato_dst,
                max_images   = TOMATO_COUNT,
                label        = "tomato",
            )
        else:
            tomato_count = 0
    print(f"[Tomato] Collected: {tomato_count} images")

    # ── NON-TOMATO: Plant leaves ───────────────────────────────────────────
    print(f"\n[Non-Tomato] Copying plant leaf images ({PLANT_LEAVES_COUNT})...")
    print("  Plants: Pepper, Potato, Corn, Grape, Apple,")
    print("          Strawberry, Cherry, Peach, Blueberry...")
    if os.path.exists(PLANTVILLAGE_COLOR):
        pv_count = copy_from_folders(
            base_dir     = PLANTVILLAGE_COLOR,
            folder_names = NON_TOMATO_PLANT_FOLDERS,
            dst_dir      = non_tomato_dst,
            max_images   = PLANT_LEAVES_COUNT,
            label        = "leaf",
            start_idx    = non_count,
        )
        non_count += pv_count
        print(f"[Non-Tomato] Plant leaves collected: {pv_count}")
    else:
        print(f"[Warning] PlantVillage color not found: {PLANTVILLAGE_COLOR}")

    # ── NON-TOMATO: Animals ────────────────────────────────────────────────
    animals_dir = os.path.join(DOWNLOAD_DIR, "raw-img")
    if os.path.exists(animals_dir) and non_count < NON_TOMATO_COUNT:
        print(f"\n[Non-Tomato] Copying animal images ({ANIMALS_COUNT})...")
        animal_count = copy_images(
            src_dir    = animals_dir,
            dst_dir    = non_tomato_dst,
            max_images = min(ANIMALS_COUNT, NON_TOMATO_COUNT - non_count),
            label      = "animal",
            start_idx  = non_count,
        )
        non_count += animal_count
        print(f"[Non-Tomato] Animals collected: {animal_count}")

    # ── NON-TOMATO: Food/Fruits ────────────────────────────────────────────
    fruits_dir = os.path.join(DOWNLOAD_DIR, "fruits-360")
    if not os.path.exists(fruits_dir):
        fruits_dir = os.path.join(DOWNLOAD_DIR, "Training")
    if not os.path.exists(fruits_dir):
        # Search for any fruits folder
        for item in os.listdir(DOWNLOAD_DIR):
            candidate = os.path.join(DOWNLOAD_DIR, item)
            if os.path.isdir(candidate) and "fruit" in item.lower():
                fruits_dir = candidate
                break

    if os.path.exists(fruits_dir) and non_count < NON_TOMATO_COUNT:
        print(f"\n[Non-Tomato] Copying food/fruit images ({FOOD_COUNT})...")
        food_count = copy_images(
            src_dir    = fruits_dir,
            dst_dir    = non_tomato_dst,
            max_images = min(FOOD_COUNT, NON_TOMATO_COUNT - non_count),
            label      = "food",
            start_idx  = non_count,
        )
        non_count += food_count
        print(f"[Non-Tomato] Food images collected: {food_count}")
    else:
        print(f"\n[Non-Tomato] Food dataset not found — will use synthetic.")

    # ── NON-TOMATO: Synthetic fill ─────────────────────────────────────────
    if non_count < NON_TOMATO_COUNT:
        remaining = NON_TOMATO_COUNT - non_count
        print(f"\n[Non-Tomato] Generating {remaining} synthetic images...")
        synth = generate_synthetic(non_tomato_dst, remaining, non_count)
        non_count += synth
        print(f"[Non-Tomato] Synthetic generated: {synth}")

    # ── Final Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"  tomato_leaf : {tomato_count} images")
    print(f"  non_tomato  : {non_count} images")
    print(f"  Total       : {tomato_count + non_count} images")
    print(f"\n  Saved to: {VALIDATOR_DATASET_DIR}")

    if tomato_count < 100:
        print(f"\n[WARNING] Only {tomato_count} tomato images!")
        print(f"Check your PlantVillage source: {TOMATO_SOURCE}")

    print("\n" + "=" * 60)
    print("  NEXT STEP")
    print("=" * 60)
    print("  Check the dataset visually:")
    print(f"  Open: {VALIDATOR_DATASET_DIR}")
    print()
    print("  Then run training:")
    print("  python train_stage1_validator.py")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Validator Dataset Builder")
    print("  Building dataset for Stage 1 Tomato Validator")
    print("=" * 60)

    # Download datasets from Kaggle
    download_all()

    # Build the dataset
    build_dataset()


if __name__ == "__main__":
    main()
