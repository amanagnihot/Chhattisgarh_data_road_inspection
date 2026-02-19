import torch
import cv2
import supervision as sv
from rfdetr import RFDETRSegMedium
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Architecture params — confirmed from inspect_checkpoint.py output:
#   refpoint_embed.weight : [2600, 4]         => base num_queries = 200 (2600/group_detr=13)
#   position_embeddings   : [1, 1297, 384]    => num_patches = 1296
#   patch_projection      : [384, 3, 12, 12]  => patch_size  = 12
#   image_size = sqrt(1296) * 12 = 36 * 12   => 432 (confirmed by args.resolution=432)
# ─────────────────────────────────────────────────────────────────────────────
NUM_QUERIES = 200
IMAGE_SIZE  = 432

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = r"/media/user/New Volume1/Sakshi/chattishgarh/new_output/rfdetr_medium/checkpoint_best_total.pth"
INPUT_VIDEO     = r"/media/user/New Volume1/Sakshi/chattishgarh/DJI_0051.MP4"
OUTPUT_VIDEO    = r"out_video_51.mp4"

# ─────────────────────────────────────────────────────────────────────────────
# Output resolution — resize 4K frames to this before inference
# Reduces mask memory ~9x vs 4K, no accuracy loss (model runs at 432px anyway)
# ─────────────────────────────────────────────────────────────────────────────
OUT_WIDTH  = 1280
OUT_HEIGHT = 720

# ─────────────────────────────────────────────────────────────────────────────
# Classes — 28 total, taken directly from args.class_names in checkpoint
# ─────────────────────────────────────────────────────────────────────────────
import json

# === Load classes from training COCO JSON ===
COCO_JSON_PATH = r"/media/user/New Volume1/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"  # <-- USE TRAIN JSON

with open(COCO_JSON_PATH, "r") as f:
    coco_data = json.load(f)

# Sort categories by ID to match model training order
# categories = sorted(coco_data["categories"], key=lambda x: x["id"])
# CLASS_NAMES = [cat["name"] for cat in categories]
# Sort categories by ID to match model training order
categories = sorted(coco_data["categories"], key=lambda x: x["id"])

# COCO IDs are 1-based, RF-DETR predicts 0-based
# Subtract 1 from each ID to get correct 0-indexed position
CLASS_NAMES = [""] * len(categories)
for cat in categories:
    CLASS_NAMES[cat["id"] - 1] = cat["name"]

print("Loaded Classes:")
for i, name in enumerate(CLASS_NAMES):
    print(f"{i}: {name}")
# CLASS_NAMES = [
#     "Corrugations_and_shoving",   # 0
#     "Cracking",                   # 1
#     "Edge_breaking",              # 2
#     "Edge_drop",                  # 3
#     "Embankment slope",           # 4
#     "MBCB_defect",                # 5
#     "MBCB_missing",               # 6
#     "Patch",                      # 7
#     "Ravelling",                  # 8
#     "Scaling",                    # 9
#     "Wear",                       # 10
#     "bump",                       # 11
#     "cattle",                     # 12
#     "depression",                 # 13
#     "doubt",                      # 14
#     "guard_post",                 # 15
#     "honeycomb",                  # 16
#     "kerb_damage",                # 17
#     "km_stone",                   # 18
#     "overhead_sign_board",        # 19
#     "parallel_crack",             # 20
#     "pothole",                    # 21
#     "sign_board",                 # 22
#     "strip_seal_expansion_join",  # 23
#     "tyre_marks",                 # 24
#     "vegetation_on_road",         # 25
#     "water_mark",                 # 26
#     "white_mark",                 # 27
# ]

# ─────────────────────────────────────────────────────────────────────────────
# Colors — hex, index-aligned with CLASS_NAMES above
# ─────────────────────────────────────────────────────────────────────────────
HEX_COLORS = [
    "#00A5FF",  # 0:  Corrugations_and_shoving
    "#008CFF",  # 1:  Cracking
    "#FF00FF",  # 2:  Edge_breaking
    "#8000FF",  # 3:  Edge_drop
    "#00FF80",  # 4:  Embankment slope
    "#FF0080",  # 5:  MBCB_defect
    "#B400B4",  # 6:  MBCB_missing
    "#0000FF",  # 7:  Patch
    "#02D32E",  # 8:  Ravelling
    "#FF8000",  # 9:  Scaling
    "#00FFFF",  # 10: Wear
    "#FFFF00",  # 11: bump
    "#FF4040",  # 12: cattle
    "#4040FF",  # 13: depression
    "#A0A0A0",  # 14: doubt
    "#006400",  # 15: guard_post
    "#D2691E",  # 16: honeycomb
    "#008080",  # 17: kerb_damage
    "#FFD700",  # 18: km_stone
    "#873CBE",  # 19: overhead_sign_board
    "#00C8C8",  # 20: parallel_crack
    "#00008C",  # 21: pothole
    "#32CD32",  # 22: sign_board
    "#FF69B4",  # 23: strip_seal_expansion_join
    "#696969",  # 24: tyre_marks
    "#228B22",  # 25: vegetation_on_road
    "#ADD8E6",  # 26: water_mark
    "#F5F5F5",  # 27: white_mark
]

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_THRESHOLD = 0.35
NMS_THRESHOLD    = 0.50
MASK_OPACITY     = 0.40   # 0.0 = transparent, 1.0 = fully opaque
BATCH_SIZE       = 4
IGNORED_CLASSES  = {"white_mark", "water_mark", "doubt"}

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device : {device}")
print(f"NUM_QUERIES  : {NUM_QUERIES}")
print(f"IMAGE_SIZE   : {IMAGE_SIZE}")
print(f"Output res   : {OUT_WIDTH}x{OUT_HEIGHT}")
print(f"Total classes: {len(CLASS_NAMES)}")

model = RFDETRSegMedium(
    pretrain_weights=CHECKPOINT_PATH,
    num_queries=NUM_QUERIES,
    resolution=IMAGE_SIZE,
)
print("Model loaded successfully.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Video I/O
# Input: read original 4K frames
# Output: write at OUT_WIDTH x OUT_HEIGHT (1280x720)
# ─────────────────────────────────────────────────────────────────────────────
cap    = cv2.VideoCapture(INPUT_VIDEO)
fps    = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (OUT_WIDTH, OUT_HEIGHT))

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Input video  : {orig_width}x{orig_height} @ {fps} fps")
print(f"Total frames : {total_frames}")
print(f"Output       : {OUTPUT_VIDEO}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Supervision annotators — built for 1280x720
# ─────────────────────────────────────────────────────────────────────────────
text_scale = sv.calculate_optimal_text_scale(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
thickness  = sv.calculate_optimal_line_thickness(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))

color_palette   = sv.ColorPalette.from_hex(HEX_COLORS)

mask_annotator  = sv.MaskAnnotator(
    color=color_palette,
    opacity=MASK_OPACITY,
)
bbox_annotator  = sv.BoxAnnotator(
    color=color_palette,
    thickness=thickness,
)
label_annotator = sv.LabelAnnotator(
    color=color_palette,
    text_color=sv.Color.WHITE,
    text_scale=text_scale,
    text_thickness=max(1, thickness - 1),
)

# ─────────────────────────────────────────────────────────────────────────────
# Main processing loop
# ─────────────────────────────────────────────────────────────────────────────
frame_count = 0
frame_batch = []

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # ── Resize 4K → 1280x720 before adding to batch ──────────────────────
        frame_resized = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
        frame_batch.append(frame_resized)

    # Process when batch is full OR video ended with leftover frames
    if len(frame_batch) == BATCH_SIZE or (not ret and len(frame_batch) > 0):

        # ── Batch inference ───────────────────────────────────────────────────
        batch_detections = model.predict(frame_batch, threshold=GLOBAL_THRESHOLD)

        for i, detections in enumerate(batch_detections):

            # ── NMS ───────────────────────────────────────────────────────────
            detections = detections.with_nms(threshold=NMS_THRESHOLD)

            # ── Filter: ignored classes + confidence + invalid class_id ─────────
            if len(detections) > 0:
                indices_to_keep = []

                for idx in range(len(detections)):
                    class_id   = detections.class_id[idx]
                    confidence = detections.confidence[idx]

                    # Skip any class_id out of range
                    if class_id < 0 or class_id >= len(CLASS_NAMES):
                        continue

                    class_name = CLASS_NAMES[class_id]

                    if class_name in IGNORED_CLASSES:
                        continue
                    if confidence < GLOBAL_THRESHOLD:
                        continue

                    indices_to_keep.append(idx)

                keep_mask  = np.array(
                    [idx in indices_to_keep for idx in range(len(detections))]
                )
                detections = detections[keep_mask]

            # ── Build labels (class name + confidence) ────────────────────────
            detections_labels = [
                f"{CLASS_NAMES[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(
                    detections.class_id, detections.confidence
                )
                if 0 <= class_id < len(CLASS_NAMES)  # safety guard
            ]

            # ── Annotate: mask → bbox → label (order matters) ─────────────────
            annotated = frame_batch[i].copy()

            annotated = mask_annotator.annotate(
                scene=annotated,
                detections=detections,
            )
            annotated = bbox_annotator.annotate(
                scene=annotated,
                detections=detections,
            )
            annotated = label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=detections_labels,
            )

            writer.write(annotated)

            frame_count += 1
            if frame_count % 30 == 0:
                pct = (frame_count / total_frames) * 100
                print(f"Processed {frame_count}/{total_frames} frames ({pct:.1f}%)")

        frame_batch = []

    if not ret:
        break

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
cap.release()
writer.release()
print(f"\nDone! Output saved to : {OUTPUT_VIDEO}")
print(f"Total frames processed: {frame_count}")