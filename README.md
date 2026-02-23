## 0-115 km Chattishgarh Data 
### Input Data - https://drive.google.com/drive/folders/1HGz8EK5aTGpkk5cscqrNbeHumkQw3AiH?usp=drive_link
### Output processed_1 videos - https://drive.google.com/drive/folders/1pOVpe2RFDQu17JV-nPdBcStd4TZ5TeaR?usp=drive_link
### Output processed_2 videos - https://drive.google.com/drive/folders/1dLFEMcj79v2i9SKQgH7glYlJpfdT36hK?usp=drive_link

## NH-10 Data
### Input Data - https://drive.google.com/drive/folders/1P_6wn8k7hlxBLZt4P_8Hd9ZYbcPwOS6m?usp=drive_link
### Output processed videos - 


# How to Create Tile Dataset

This guide explains the complete workflow to prepare a tiled dataset (COCO format) for training detection models like RF-DETR and YOLO.

---

## 📥 Step 1: Download Dataset from Roboflow

Use the following script to download the dataset in COCO format:

```bash
python download_dataset.py
```

This will download the dataset from Roboflow in COCO format.

---

## 🧹 Step 2: Clean Small Annotations

Remove invalid annotations (area < 1 pixel):

```bash
python clean_coco.py
```

This script removes annotations with area less than 1 pixel, ensuring extremely small or corrupted bounding boxes are filtered out.

---

## ✂️ Step 3: Slice Dataset into Tiles (800x800)

Convert the dataset into tiles using SAHI:

```bash
python slice_dataset.py
```

This script:
- Converts images into **800 × 800** tiles
- Uses **15% overlap**
- Uses **SAHI** slicing method

> Tiling helps improve detection of small objects.

---

## 🗑️ Step 4: Remove Empty Tiles

Delete tiles that do not contain any annotations:

```bash
python filter_empty_tile.py
```

This removes unnecessary images and keeps only annotated tiles for training.

---

## ✅ Dataset is Ready

Your dataset is now prepared in **COCO format** and ready for training models that support COCO datasets.

---

## 🚀 Training Models

### 🔹 Train RF-DETR Model

To train the RF-DETR model:

```bash
python rfdetr_train.py
```

This will train the model using the prepared tiled COCO dataset.

---

### 🟡 Using This Dataset for YOLO Training (Easy Method)

YOLO typically requires YOLO-format annotations. Follow these steps:

1. Upload the sliced dataset (images + COCO JSON file) to Roboflow.
2. Verify that images and annotations are visible after upload.
3. Create a new dataset version.
4. Download the dataset in **YOLO format**.
5. Use the downloaded dataset to train YOLO models.
