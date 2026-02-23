# Script will filter the empty tiles from the dataset and save the filtered dataset in the filtered_dataset folder with 20% background images

import json
import random
import os

def filter_empty_tiles(coco_path, images_dir, output_json, empty_ratio=0.2):
    print("Loading COCO...")
    
    with open(coco_path) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]

    annotated_ids = set(ann["image_id"] for ann in annotations)

    annotated_images = []
    empty_images = []

    for img in images:
        if img["id"] in annotated_ids:
            annotated_images.append(img)
        else:
            empty_images.append(img)

    print(f"✅ Annotated tiles: {len(annotated_images)}")
    print(f"✅ Empty tiles BEFORE filtering: {len(empty_images)}")

    # -------- Keep controlled empty ratio ----------
    keep_empty = int(len(annotated_images) * empty_ratio)

    if len(empty_images) > keep_empty:
        empty_images = random.sample(empty_images, keep_empty)

    final_images = annotated_images + empty_images
    final_ids = set(img["id"] for img in final_images)

    final_annotations = [
        ann for ann in annotations
        if ann["image_id"] in final_ids
    ]

    coco["images"] = final_images
    coco["annotations"] = final_annotations

    with open(output_json, "w") as f:
        json.dump(coco, f)

    print("✅ Filtered COCO saved!")

    # =====================================================
    # SAFE IMAGE DELETION
    # =====================================================

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    keep_filenames = set(img["file_name"] for img in final_images)

    deleted = 0
    skipped = 0

    for file in os.listdir(images_dir):

        file_path = os.path.join(images_dir, file)

        # ✅ NEVER touch directories
        if os.path.isdir(file_path):
            skipped += 1
            continue

        ext = os.path.splitext(file)[1].lower()

        # ✅ Skip non-image files
        if ext not in IMAGE_EXTENSIONS:
            skipped += 1
            continue

        # Delete only unused images
        if file not in keep_filenames:
            os.remove(file_path)
            deleted += 1

    print(f"🔥 Deleted UNUSED images: {deleted}")
    print(f"✅ Skipped non-image files/folders: {skipped}")
    print("✅ Dataset is now CLEAN and SAFE.")


# ================= RUN ================= #

filter_empty_tiles(
    coco_path=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\train\train_sliced.json_coco.json",
    images_dir=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\train",
    output_json=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\train\train_filtered.json",
    empty_ratio=0.2
)

filter_empty_tiles(
    coco_path=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\valid\valid_sliced.json_coco.json",
    images_dir=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\valid",
    output_json=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\valid\valid_filtered.json",
    empty_ratio=0.2
)
