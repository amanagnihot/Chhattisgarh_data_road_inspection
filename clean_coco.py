##script will clean the coco annotations by removing the annotations with area less than 1 pixel 
import json

def clean_coco(input_path, output_path):
    with open(input_path) as f:
        coco = json.load(f)

    cleaned_annotations = []
    removed = 0

    for ann in coco["annotations"]:
        if ann.get("area", 0) > 1:   # remove zero or near-zero area
            cleaned_annotations.append(ann)
        else:
            removed += 1

    coco["annotations"] = cleaned_annotations

    with open(output_path, "w") as f:
        json.dump(coco, f)

    print(f"✅ Removed {removed} bad annotations")


# CLEAN TRAIN
clean_coco(
    r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\train\_annotations.coco.json",
    r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\train\_annotations.coco.json"
)

# CLEAN VALID
clean_coco(
    r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\valid\_annotations.coco.json",
    r"D:\Sakshi\chattishgarh\hehe\Road_inspection-1\valid\_annotations.coco.json"
)
