##script will slice the dataset into 800x800 patches with 15% overlap and save the sliced dataset in the sliced_dataset folder
from sahi.slicing import slice_coco

# ================= TRAIN =================
slice_coco(
    coco_annotation_file_path=r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\train\_annotations.coco.json",
    image_dir=r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\train",
    output_coco_annotation_file_name="train_sliced.json",
    output_dir=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\train",
    slice_height=800,
    slice_width=800,
    overlap_height_ratio=0.15,
    overlap_width_ratio=0.15,
    min_area_ratio=0.1   # VERY important → removes tiny broken annotations
)

# ================= VALID =================
slice_coco(
    coco_annotation_file_path=r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\valid\_annotations.coco.json",
    image_dir=r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\valid",
    output_coco_annotation_file_name="valid_sliced.json",
    output_dir=r"D:\Sakshi\chattishgarh\hehe\sliced_dataset\valid",
    slice_height=800,
    slice_width=800,
    overlap_height_ratio=0.15,
    overlap_width_ratio=0.15,
    min_area_ratio=0.1
)

print("✅ Slicing completed successfully!")
