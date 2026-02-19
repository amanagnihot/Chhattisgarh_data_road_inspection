import torch
import cv2
import supervision as sv
from rfdetr import RFDETRSegMedium
import numpy as np
import json

# ─────────────────────────────────────────────────────────────────────────────
# Architecture params
# ─────────────────────────────────────────────────────────────────────────────
NUM_QUERIES = 200
IMAGE_SIZE  = 432

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = r"/media/user/New Volume1/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
INPUT_VIDEO     = r"/media/user/New Volume1/Sakshi/chattishgarh/DJI_20260110101923_0001_D.MP4"
OUTPUT_VIDEO    = r"out_video_001_100_epoch.mp4"
COCO_JSON_PATH  = r"/media/user/New Volume1/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"

# ─────────────────────────────────────────────────────────────────────────────
# Output resolution — resize 4K frames to this before inference
# ─────────────────────────────────────────────────────────────────────────────
OUT_WIDTH  = 1280
OUT_HEIGHT = 720

# ─────────────────────────────────────────────────────────────────────────────
# Load classes from COCO JSON - FIXED VERSION
# ─────────────────────────────────────────────────────────────────────────────
with open(COCO_JSON_PATH, "r") as f:
    coco_data = json.load(f)

# ✅ FIXED: No ID subtraction - COCO IDs start at 0
categories = sorted(coco_data["categories"], key=lambda x: x["id"])
CLASS_NAMES = [cat["name"] for cat in categories]

print("=" * 60)
print("LOADED CLASSES (FIXED)")
print("=" * 60)
for i, name in enumerate(CLASS_NAMES):
    print(f"{i:2d}: {name}")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# Colors — hex, index-aligned with CLASS_NAMES
# ─────────────────────────────────────────────────────────────────────────────
HEX_COLORS = [
    "#FF0000",  # 0:  pothole (RED for visibility)
    "#00A5FF",  # 1:  Corrugations_and_shoving
    "#008CFF",  # 2:  Cracking
    "#FF00FF",  # 3:  Edge_breaking
    "#8000FF",  # 4:  Edge_drop
    "#00FF80",  # 5:  Embankment slope
    "#FF0080",  # 6:  MBCB_defect
    "#B400B4",  # 7:  MBCB_missing
    "#0000FF",  # 8:  Patch (BLUE)
    "#02D32E",  # 9:  Ravelling
    "#FF8000",  # 10: Scaling
    "#00FFFF",  # 11: Wear
    "#FFFF00",  # 12: bump
    "#FF4040",  # 13: cattle
    "#4040FF",  # 14: depression
    "#A0A0A0",  # 15: doubt
    "#006400",  # 16: guard_post
    "#D2691E",  # 17: honeycomb
    "#008080",  # 18: kerb_damage
    "#FFD700",  # 19: km_stone
    "#873CBE",  # 20: overhead_sign_board
    "#00C8C8",  # 21: parallel_crack
    "#FF1493",  # 22: pothole (duplicate - different color)
    "#32CD32",  # 23: sign_board
    "#FF69B4",  # 24: strip_seal_expansion_join
    "#696969",  # 25: tyre_marks
    "#228B22",  # 26: vegetation_on_road
    "#ADD8E6",  # 27: water_mark
    "#F5F5F5",  # 28: white_mark
]

# Ensure we have enough colors
while len(HEX_COLORS) < len(CLASS_NAMES):
    HEX_COLORS.append("#FFFFFF")  # Add white as fallback

# ─────────────────────────────────────────────────────────────────────────────
# Settings - UPDATED THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_THRESHOLD = 0.25   # ✅ Lowered for better recall
NMS_THRESHOLD    = 0.65   # ✅ Less aggressive NMS
MASK_OPACITY     = 0.15
BATCH_SIZE       = 4
IGNORED_CLASSES  = {"white_mark", "water_mark", "doubt","bump","guard_post","overhead_sign_board","kerb_damage"}  # Example: ignore these classes during annotation

# Optional: Class-specific thresholds
CLASS_THRESHOLDS = {
    "Patch": 0.40,           # ✅ Even lower for patches
    "pothole": 0.25,
    # Add more if needed
}

# ─────────────────────────────────────────────────────────────────────────────
# Load model - FIXED PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device : {device}")
print(f"NUM_QUERIES  : {NUM_QUERIES}")
print(f"IMAGE_SIZE   : {IMAGE_SIZE}")
print(f"Output res   : {OUT_WIDTH}x{OUT_HEIGHT}")
print(f"Total classes: {len(CLASS_NAMES)}\n")

model = RFDETRSegMedium(
    pretrain_weights=CHECKPOINT_PATH,
    num_queries=NUM_QUERIES,
    image_size=IMAGE_SIZE,        # ✅ Added
    max_image_size=IMAGE_SIZE,    # ✅ Added
)
print("✅ Model loaded successfully.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Verify checkpoint classes (optional but recommended)
# ─────────────────────────────────────────────────────────────────────────────
try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if 'args' in checkpoint:
        args = checkpoint['args']
        if hasattr(args, 'class_names'):
            trained_classes = args.class_names
            if trained_classes != CLASS_NAMES:
                print("⚠️  WARNING: Class mismatch detected!")
                print(f"   Trained on: {trained_classes[:5]}...")
                print(f"   Loading:    {CLASS_NAMES[:5]}...")
        if hasattr(args, 'resolution'):
            print(f"✅ Checkpoint resolution: {args.resolution}")
except Exception as e:
    print(f"ℹ️  Could not verify checkpoint metadata: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Video I/O
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
# Supervision annotators
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

print("Processing video...")
print("-" * 60)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Resize 4K → 1280x720 before adding to batch
        frame_resized = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
        frame_batch.append(frame_resized)

    # Process when batch is full OR video ended with leftover frames
    if len(frame_batch) == BATCH_SIZE or (not ret and len(frame_batch) > 0):

        # ── Batch inference ───────────────────────────────────────────────────
        batch_detections = model.predict(frame_batch, threshold=GLOBAL_THRESHOLD)

        for i, detections in enumerate(batch_detections):

            # ── NMS (less aggressive now) ─────────────────────────────────────
            detections = detections.with_nms(threshold=NMS_THRESHOLD)

            # ── Filter: ignored classes + confidence + invalid class_id ───────
            if len(detections) > 0:
                indices_to_keep = []

                for idx in range(len(detections)):
                    class_id   = detections.class_id[idx]
                    confidence = detections.confidence[idx]

                    # ✅ Safety: Skip invalid class_id
                    if class_id < 0 or class_id >= len(CLASS_NAMES):
                        print(f"⚠️  Warning: Invalid class_id {class_id} detected!")
                        continue

                    class_name = CLASS_NAMES[class_id]

                    # Skip ignored classes
                    if class_name in IGNORED_CLASSES:
                        continue

                    # ✅ Use class-specific threshold if available
                    threshold_to_use = CLASS_THRESHOLDS.get(class_name, GLOBAL_THRESHOLD)
                    if confidence < threshold_to_use:
                        continue

                    indices_to_keep.append(idx)

                # Apply filter mask
                keep_mask  = np.array(
                    [idx in indices_to_keep for idx in range(len(detections))]
                )
                detections = detections[keep_mask]

            # ── Build labels (class name + confidence) ────────────────────────
            detections_labels = []
            for class_id, confidence in zip(detections.class_id, detections.confidence):
                if 0 <= class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                    detections_labels.append(f"{class_name} {confidence:.2f}")
                else:
                    detections_labels.append(f"Unknown {confidence:.2f}")

            # ── Annotate: mask → bbox → label ─────────────────────────────────
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

print("-" * 60)
print(f"\n✅ Done! Output saved to : {OUTPUT_VIDEO}")
print(f"Total frames processed: {frame_count}")
print("\n" + "=" * 60)
print("KEY FIXES APPLIED:")
print("=" * 60)
print("✅ 1. Fixed class ID mapping (no more off-by-one error)")
print("✅ 2. Lowered confidence threshold: 0.35 → 0.25")
print("✅ 3. Relaxed NMS threshold: 0.50 → 0.65")
print("✅ 4. Added class-specific thresholds (Patch: 0.20)")
print("✅ 5. Added image_size parameters to model")
print("=" * 60)
