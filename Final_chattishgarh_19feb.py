# ============================
# ROAD ASSESSMENT SCRIPT - RF-DETR SEG MEDIUM
# Inference: Direct batch inference on resized frames (NO SAHI/slicing)
# ============================

import os
import cv2
import math
import numpy as np
import re
import torch
import supervision as sv
import json
import threading
import queue
from datetime import datetime
from PIL import Image as PILImage
import glob
from concurrent.futures import ThreadPoolExecutor
from rfdetr import RFDETRSegMedium

# ============================
# CONFIGURATION - UPDATE THESE
# ============================

SINGLE_VIDEO_MODE = True
MULTI_VIDEO_MODE  = False
FOLDER_MODE       = False

VIDEO_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260206174112_0754_D.MP4"
SRT_PATH   = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260206174112_0754_D.SRT"

VIDEO_SRT_PAIRS = [
    {
        "video": "/content/drive/MyDrive/videos/DJI_0001.MP4",
        "srt":   "/content/drive/MyDrive/videos/DJI_0001.SRT"
    }
]

INPUT_FOLDER = "/content/drive/MyDrive/videos"

SEGMENTATION_MODEL_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
OUTPUT_BASE_DIR         = "/media/user/New Volume/Sakshi/chattishgarh/final_output"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ============================
# PROCESSING SETTINGS
# ============================

PROCESS_EVERY_N_FRAMES = 1
BATCH_SIZE             = 1
MASK_OPACITY           = 0.20

# Output resolution — resize frames to this before inference
OUT_WIDTH  = 1280
OUT_HEIGHT = 720

# Architecture params
NUM_QUERIES = 200
IMAGE_SIZE  = 432

# ─── Class-Specific Confidence Thresholds ───
CLASS_THRESHOLDS = {
    # "Patch":                      0.50,
    # "pothole":                    0.35,
    # "Cracking":                   0.30,
    # "Ravelling":                  0.50,
    # "Edge_breaking":              0.35,
    # "Edge_drop":                  0.35,
    # "MBCB_defect":                0.30,
    # "MBCB_missing":               0.30,
    # "Corrugations_and_shoving":   0.30,
    # "Scaling":                    0.30,
    # "Wear":                       0.40,
    # "honeycomb":                  0.30,
    # "depression":                 0.30,
    # "Embankment slope":           0.30,
    # "strip_seal_expansion_join":  0.30,
    # "tyre_marks":                 0.35,
    # "vegetation_on_road":         0.30,
    # "km_stone":                   0.40,
    # "cattle":                     0.45,
}

GLOBAL_THRESHOLD = 0.30
NMS_THRESHOLD    = 0.50

# ─── Ignored Classes — NOT annotated, NOT saved, completely skipped ───
IGNORED_CLASSES = {
    "white_mark", "water_mark", "doubt", "bump",
    "guard_post", "overhead_sign_board", "sign_board", "kerb_damage"
}

IOU_THRESHOLD = 0.30    # tracker
MAX_DISTANCE  = 50
MAX_LOST      = 30

FRAME_BUFFER_SIZE   = 16
SAVE_WORKER_THREADS = 4

# ============================
# CLASS NAMES & COLORS
# ============================

COCO_JSON_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"

with open(COCO_JSON_PATH, "r") as f:
    coco_data = json.load(f)

# Sort categories by ID to match model training order
categories = sorted(coco_data["categories"], key=lambda x: x["id"])
CLASS_NAMES = [cat["name"] for cat in categories]

print("Loaded Classes:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i:2d}: {name}")

# ─── Index-aligned HEX colors — must match CLASS_NAMES order from COCO JSON ───
# Adjust this list if your CLASS_NAMES order differs
HEX_COLORS = [
    "#FF0000",   #  Corrugations_and_shoving
    "#008CFF",   #  Cracking
    "#FF00FF",   #  Edge_breaking
    "#8000FF",   #  Edge_drop
    "#00FF80",   #  Embankment slope
    "#FF0080",   #  MBCB_defect
    "#B400B4",   #  MBCB_missing
    "#0000FF",   #  Patch
    "#02D32E",   #  Ravelling
    "#FF8000",   #  Scaling
    "#00FFFF",   #  Wear
    "#9E4292",   #  bump
    "#FF4040",   #  cattle
    "#4040FF",   #  depression
    "#A0A0A0",   #  doubt
    "#006400",   #  guard_post
    "#D2691E",   #  honeycomb
    "#008080",   #  kerb_damage
    "#52240E",   #  km_stone
    "#873CBE",   #  overhead_sign_board
    "#00C8C8",   #  parallel_crack
    "#0000C8",   #  pothole
    "#32CD32",   #  sign_board
    "#FF69B4",   #  strip_seal_expansion_join
    "#696969",   #  tyre_marks
    "#228B22",   #  vegetation_on_road
    "#ADD8E6",   #  water_mark
    "#F5F5F5",   #  white_mark
]

# Pad with fallback whites if COCO has more classes than colors defined
while len(HEX_COLORS) < len(CLASS_NAMES):
    HEX_COLORS.append("#FFFFFF")

# Build a name→BGR color dict for tracker drawing (used for mask/polygon overlay)
def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip("#")
    r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
    return (b, g, r)

color_map = {
    CLASS_NAMES[i]: hex_to_bgr(HEX_COLORS[i])
    for i in range(len(CLASS_NAMES))
}

# ============================
# GPU UTILITIES
# ============================

def check_gpu():
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"   VRAM: {total_mem:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        return device
    else:
        print("⚠️  No GPU found — running on CPU (slow)")
        return 'cpu'


def print_gpu_memory():
    if torch.cuda.is_available():
        used   = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        print(f"   🖥️  GPU Memory: {used:.2f}GB used / {cached:.2f}GB cached")


def warmup_model(model, device):
    print("🔥 Warming up GPU (RF-DETR)...")
    dummy = [np.random.randint(0, 255, (OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)]
    with torch.no_grad():
        model.predict(dummy, threshold=GLOBAL_THRESHOLD)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    print("✅ GPU warmed up")


# ============================
# THREADED FRAME READER
# ============================

class ThreadedVideoReader:
    def __init__(self, video_path, skip_n=PROCESS_EVERY_N_FRAMES,
                 buffer_size=FRAME_BUFFER_SIZE):
        self.cap     = cv2.VideoCapture(video_path)
        self.skip_n  = skip_n
        self.buffer  = queue.Queue(maxsize=buffer_size)
        self.stopped = False

        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        local_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.buffer.put(None)   # sentinel
                break
            if local_index % self.skip_n == 0:
                self.buffer.put((local_index, frame))
            local_index += 1

    def read(self):
        return self.buffer.get()

    def release(self):
        self.stopped = True
        self.cap.release()


# ============================
# ASYNC IMAGE SAVER
# ============================

class AsyncImageSaver:
    def __init__(self, num_workers=SAVE_WORKER_THREADS):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures  = []

    def save(self, path, image):
        self.futures.append(self.executor.submit(cv2.imwrite, path, image))

    def wait_all(self):
        for f in self.futures:
            f.result()
        self.futures.clear()

    def shutdown(self):
        self.wait_all()
        self.executor.shutdown(wait=True)


# ============================
# SRT PARSING
# ============================

def parse_srt(srt_path):
    """
    Parse DJI SRT files. Handles two known formats:

    Format A (older DJI):
        4
        00:00:00,049 --> 00:00:00,066
        <font size="28">FrameCnt: 4, DiffTime: 17ms
        2026-01-10 10:19:23.934
        [latitude: 20.665134] [longitude: 81.484726] [abs_alt: 309.551] ...

    Format B (newer DJI):
        8718
        00:04:50,838 --> 00:04:50,871
        <font size="36">SrtCnt : 8718, DiffTime : 33ms
        2026-01-09 15:31:21,948,194
        [latitude: 20.536977] [longitude: 80.962568] [altitude: 52.800000] ...
    """
    srt_data = []
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Strip HTML tags (<font ...>, </font>) globally so regexes work cleanly
    content = re.sub(r'<[^>]+>', '', content)

    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        if not block.strip():
            continue
        lines = [l.strip() for l in block.strip().split('\n') if l.strip()]
        if len(lines) < 3:
            continue
        try:
            # ── Frame number ──────────────────────────────────────────────
            # Format A first line: plain integer  →  "4"
            # Format B first line: plain integer  →  "8718"
            frame_num = int(lines[0])

            # ── Timecode from SRT arrow line ──────────────────────────────
            tc = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
            if not tc:
                continue
            h, m, s, ms = map(int, tc.groups())
            timestamp_ms = (h * 3600 + m * 60 + s) * 1000 + ms

            # ── Join remaining lines as meta block ────────────────────────
            meta = ' '.join(lines[2:])

            # ── Absolute timestamp ─────────────────────────────────────────
            # Format A:  2026-01-10 10:19:23.934        (dot before ms)
            # Format B:  2026-01-09 15:31:21,948,194    (comma before ms,
            #                                            extra sub-ms part)
            absolute_timestamp = None

            # Try Format A first: YYYY-MM-DD HH:MM:SS.mmm
            ts_a = re.search(
                r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\.(\d+)',
                meta
            )
            if ts_a:
                ts_str = f"{ts_a.group(1)} {ts_a.group(2)}.{ts_a.group(3)[:6]}"
                absolute_timestamp = datetime.strptime(ts_str,
                                                       '%Y-%m-%d %H:%M:%S.%f')
            else:
                # Try Format B: YYYY-MM-DD HH:MM:SS,mmm,xxx
                ts_b = re.search(
                    r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}),(\d+)',
                    meta
                )
                if ts_b:
                    ms_part = ts_b.group(3)[:6].ljust(6, '0')  # pad to microseconds
                    ts_str  = f"{ts_b.group(1)} {ts_b.group(2)}.{ms_part}"
                    absolute_timestamp = datetime.strptime(ts_str,
                                                           '%Y-%m-%d %H:%M:%S.%f')

            if absolute_timestamp is None:
                continue

            # ── GPS coordinates ───────────────────────────────────────────
            # Both formats use [latitude: X] [longitude: X] with optional spaces
            lat = re.search(r'\[latitude\s*:\s*([-\d.]+)\]',  meta)
            lon = re.search(r'\[longitude\s*:\s*([-\d.]+)\]', meta)
            if not (lat and lon):
                continue

            # ── Altitude: abs_alt (Format A) or altitude (Format B) ───────
            alt = re.search(
                r'\[(?:abs_alt|altitude)\s*:\s*([-\d.]+)', meta
            )

            srt_data.append({
                'frame_number':       frame_num,
                'timestamp_ms':       timestamp_ms,
                'absolute_timestamp': absolute_timestamp,
                'latitude':           float(lat.group(1)),
                'longitude':          float(lon.group(1)),
                'altitude':           float(alt.group(1)) if alt else 50.0,
            })

        except Exception:
            continue

    if not srt_data:
        print("❌ No valid GPS data in SRT file!")
        print("   Tip: check that the SRT has [latitude:] and [longitude:] fields")
        return None

    print(f"✅ Parsed {len(srt_data)} SRT entries")
    return srt_data


def haversine_distance(lat1, lon1, lat2, lon2):
    R  = 6371000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_cumulative_chainage(srt_data, starting_chainage_m=0):
    cum = starting_chainage_m
    for i, entry in enumerate(srt_data):
        if i == 0:
            entry['cumulative_chainage_m'] = starting_chainage_m
        else:
            prev = srt_data[i - 1]
            cum += haversine_distance(
                prev['latitude'], prev['longitude'],
                entry['latitude'], entry['longitude']
            )
            entry['cumulative_chainage_m'] = cum
    print(f"✅ Chainage: {starting_chainage_m:.1f}m → {cum:.1f}m "
          f"({(cum - starting_chainage_m)/1000:.3f} km)")
    return cum


def get_srt_data_for_frame(frame_index, srt_data):
    for entry in srt_data:
        if entry['frame_number'] == frame_index:
            return entry
    if srt_data:
        return min(srt_data, key=lambda x: abs(x['frame_number'] - frame_index))
    return None


# ============================
# TRACKER
# ============================

class SegmentationTracker:
    def __init__(self, iou_threshold=0.3, max_distance=50, max_lost=30):
        self.next_id       = 1
        self.tracks        = {}
        self.iou_threshold = iou_threshold
        self.max_distance  = max_distance
        self.max_lost      = max_lost

    def _mask_iou(self, m1, m2):
        if m1 is None or m2 is None:
            return 0.0
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return inter / union if union > 0 else 0.0

    def _centroid(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def update(self, detections):
        for t in self.tracks.values():
            t['lost'] += 1

        assigned       = set()
        updated_tracks = []

        for tid, t in list(self.tracks.items()):
            best_i, best_score = -1, 0
            for i, det in enumerate(detections):
                if i in assigned or t.get('class') != det.get('class'):
                    continue
                iou  = self._mask_iou(t.get('mask'), det.get('mask'))
                dist = math.hypot(
                    t['centroid'][0] - self._centroid(det['bbox'])[0],
                    t['centroid'][1] - self._centroid(det['bbox'])[1]
                )
                if iou > self.iou_threshold or dist < self.max_distance:
                    score = iou - (dist / self.max_distance) * 0.5
                    if score > best_score:
                        best_score, best_i = score, i

            if best_i >= 0:
                det = detections[best_i]
                self.tracks[tid].update({
                    'bbox':     det['bbox'],
                    'centroid': self._centroid(det['bbox']),
                    'mask':     det['mask'],
                    'lost':     0,
                    'class':    det['class'],
                })
                assigned.add(best_i)
                updated_tracks.append({
                    'track_id': tid,
                    'bbox':     det['bbox'],
                    'mask':     det['mask'],
                    'class':    det['class'],
                    'polygon':  det['polygon'],
                })
            elif t['lost'] > self.max_lost:
                del self.tracks[tid]

        for i, det in enumerate(detections):
            if i in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'bbox':     det['bbox'],
                'centroid': self._centroid(det['bbox']),
                'mask':     det['mask'],
                'class':    det['class'],
                'lost':     0,
            }
            updated_tracks.append({
                'track_id': tid,
                'bbox':     det['bbox'],
                'mask':     det['mask'],
                'class':    det['class'],
                'polygon':  det['polygon'],
            })

        return updated_tracks


# ============================
# INFERENCE HELPERS
# ============================

def filter_detections(detections_sv):
    """
    Apply per-class confidence thresholds and remove IGNORED_CLASSES.
    Returns filtered supervision Detections object.
    Ignored classes are completely dropped — not annotated, not saved.
    """
    if len(detections_sv) == 0:
        return detections_sv

    indices_to_keep = []
    for idx in range(len(detections_sv)):
        class_id   = int(detections_sv.class_id[idx])
        confidence = float(detections_sv.confidence[idx])

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            print(f"⚠️  Warning: invalid class_id {class_id} — skipped")
            continue

        class_name = CLASS_NAMES[class_id]

        # ── Drop ignored classes completely ──
        if class_name in IGNORED_CLASSES:
            continue

        # ── Apply per-class or global threshold ──
        threshold = CLASS_THRESHOLDS.get(class_name, GLOBAL_THRESHOLD)
        if confidence < threshold:
            continue

        indices_to_keep.append(idx)

    if not indices_to_keep:
        # Return empty detections with correct structure
        return detections_sv[np.array([], dtype=int)]

    keep_mask = np.zeros(len(detections_sv), dtype=bool)
    keep_mask[indices_to_keep] = True
    return detections_sv[keep_mask]


def sv_detections_to_list(detections_sv, frame_h, frame_w):
    """
    Convert supervision Detections to list of dicts expected by tracker.
    Each dict: { bbox, mask, polygon, class, confidence }
    """
    result = []
    if len(detections_sv) == 0:
        return result

    for idx in range(len(detections_sv)):
        class_id   = int(detections_sv.class_id[idx])
        confidence = float(detections_sv.confidence[idx])

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            continue

        class_name = CLASS_NAMES[class_id]

        if class_name in IGNORED_CLASSES:
            continue

        x1, y1, x2, y2 = detections_sv.xyxy[idx].tolist()

        # ── Extract mask ──
        mask    = None
        polygon = None
        if detections_sv.mask is not None:
            raw_mask = detections_sv.mask[idx]
            if raw_mask.shape != (frame_h, frame_w):
                raw_mask = cv2.resize(
                    raw_mask.astype(np.uint8),
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            mask = raw_mask

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                polygon = contours[0].squeeze().tolist()
                if polygon and isinstance(polygon[0], (int, float)):
                    polygon = [polygon]
                if polygon and len(polygon) < 3:
                    polygon = None

        result.append({
            'bbox':       [x1, y1, x2, y2],
            'mask':       mask,
            'polygon':    polygon,
            'class':      class_name,
            'confidence': confidence,
        })

    return result


# ============================
# VISUALIZATION
# ============================

def draw_segmentation_overlay(frame, tracked_objects):
    """Draw polygon fills + outlines + track ID labels for tracked objects."""
    overlay = frame.copy()
    for det in tracked_objects:
        cls_name = det['class']

        # ── Safety: never draw ignored classes ──
        if cls_name in IGNORED_CLASSES:
            continue

        polygon  = det['polygon']
        track_id = det['track_id']
        color    = color_map.get(cls_name, (200, 200, 200))

        if polygon is not None and len(polygon) > 0:
            pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(frame, [pts], True, color, 2)

        x1, y1 = int(det['bbox'][0]), int(det['bbox'][1])
        cv2.putText(frame, f"{cls_name} ID:{track_id}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    alpha = MASK_OPACITY
    return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)


def draw_legend(frame, width, height):
    """Draw color legend for non-ignored classes only."""
    # Only show classes that are NOT in IGNORED_CLASSES
    visible_classes = [
        (name, col) for name, col in color_map.items()
        if name not in IGNORED_CLASSES
    ]

    scale     = max(0.6, min(width / 1920.0, 1.2))
    lx        = int(0.02 * width)
    ly        = int(0.04 * height)
    line_h    = int(22 * scale)
    line_len  = int(35 * scale)
    leg_w     = int(300 * scale)
    txt_scale = 0.6 * scale
    txt_thick = max(1, int(1.5 * scale))

    cv2.rectangle(
        frame,
        (lx - 12, ly - 12),
        (lx + leg_w, ly + line_h * len(visible_classes) + 12),
        (0, 0, 0),
        max(2, int(2 * scale))
    )

    for idx, (cls_name, color) in enumerate(visible_classes):
        y = ly + idx * line_h
        cv2.line(frame, (lx, y + 10), (lx + line_len, y + 10), color, 2)
        cv2.putText(frame, cls_name,
                    (lx + line_len + 12, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    txt_scale, (255, 255, 255), txt_thick)


# ============================
# FOLDER SCANNING
# ============================

def find_video_srt_pairs(folder_path):
    print(f"\n🔍 Scanning folder: {folder_path}")
    video_files = []
    for ext in ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']:
        video_files.extend(glob.glob(os.path.join(folder_path, ext)))
    srt_files = (glob.glob(os.path.join(folder_path, '*.srt')) +
                 glob.glob(os.path.join(folder_path, '*.SRT')))

    print(f"   Found {len(video_files)} video(s), {len(srt_files)} SRT(s)")
    pairs = []
    for vp in sorted(video_files):
        base = os.path.splitext(os.path.basename(vp))[0]
        srt  = next((s for s in srt_files
                     if os.path.splitext(os.path.basename(s))[0] == base), None)
        if srt:
            pairs.append({"video": vp, "srt": srt})
            print(f"   ✅ {os.path.basename(vp)} ↔ {os.path.basename(srt)}")
        else:
            print(f"   ⚠️  No SRT: {os.path.basename(vp)}")
    return pairs


# ============================
# MAIN PROCESSING FUNCTION
# ============================

def process_video(video_path, srt_path, rfdetr_model,
                  output_dir, device, starting_chainage_m=0):
    import time

    video_name     = os.path.basename(video_path)
    srt_name       = os.path.basename(srt_path)
    video_basename = os.path.splitext(video_name)[0]

    print(f"\n{'='*70}")
    print(f"Processing: {video_name}")
    print(f"{'='*70}")

    srt_data = parse_srt(srt_path)
    if srt_data is None:
        return None, starting_chainage_m
    ending_chainage = calculate_cumulative_chainage(srt_data, starting_chainage_m)

    print("🎥 Opening video (threaded reader)...")
    reader = ThreadedVideoReader(video_path, skip_n=PROCESS_EVERY_N_FRAMES)
    orig_width    = reader.width
    orig_height   = reader.height
    fps           = reader.fps
    total_frames  = reader.total_frames

    print(f"   Original : {orig_width}x{orig_height} @ {fps:.1f}fps | {total_frames} total frames")
    print(f"   Resized  : {OUT_WIDTH}x{OUT_HEIGHT} for inference")
    print(f"   Processing every {PROCESS_EVERY_N_FRAMES} frames → "
          f"~{total_frames // PROCESS_EVERY_N_FRAMES} frames to process")
    print(f"   Batch size: {BATCH_SIZE}")
    print_gpu_memory()

    # ── Output directories ─────────────────────────────────────────────────
    video_out_dir = os.path.join(output_dir, video_basename)
    defects_dir   = os.path.join(video_out_dir, "defect_images")
    frames_dir    = os.path.join(defects_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Only create subdirs for non-ignored classes
    for cls in color_map:
        if cls not in IGNORED_CLASSES:
            os.makedirs(os.path.join(defects_dir, cls), exist_ok=True)

    output_video_path = os.path.join(video_out_dir, f"{video_basename}_output.mp4")
    output_fps        = max(1.0, fps / PROCESS_EVERY_N_FRAMES)
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        output_fps,
        (OUT_WIDTH, OUT_HEIGHT)
    )

    tracker         = SegmentationTracker(IOU_THRESHOLD, MAX_DISTANCE, MAX_LOST)
    image_saver     = AsyncImageSaver(num_workers=SAVE_WORKER_THREADS)
    track_history   = {}
    all_detections  = []
    det_id_counter  = 1
    processed_frames = 0
    start_time       = time.time()

    # ── Supervision annotators (same as Script 2) ──────────────────────────
    color_palette   = sv.ColorPalette.from_hex(HEX_COLORS)
    text_scale      = sv.calculate_optimal_text_scale(
                          resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
    line_thickness  = sv.calculate_optimal_line_thickness(
                          resolution_wh=(OUT_WIDTH, OUT_HEIGHT))

    mask_annotator  = sv.MaskAnnotator(
        color=color_palette,
        opacity=MASK_OPACITY,
    )
    bbox_annotator  = sv.BoxAnnotator(
        color=color_palette,
        thickness=line_thickness,
    )
    label_annotator = sv.LabelAnnotator(
        color=color_palette,
        text_color=sv.Color.WHITE,
        text_scale=text_scale,
        text_thickness=max(1, line_thickness - 1),
    )

    # ── Batch accumulation state ───────────────────────────────────────────
    frame_batch        = []   # resized frames
    frame_meta_batch   = []   # (frame_index, srt_entry) per frame in batch

    def process_batch(f_batch, m_batch):
        """
        Run inference on a collected batch.
        Returns:
          per_frame_dets — tracker-compatible det dicts per frame
          per_frame_sv   — filtered sv.Detections per frame (for sv annotators)
        """
        nonlocal processed_frames

        raw = rfdetr_model.predict(f_batch, threshold=GLOBAL_THRESHOLD)

        # model.predict() returns a bare sv.Detections for batch=1,
        # or a list/tuple for batch>1. Normalise to list.
        if isinstance(raw, (list, tuple)):
            batch_sv_dets = list(raw)
        else:
            batch_sv_dets = [raw]

        per_frame_dets = []
        per_frame_sv   = []
        for sv_dets in batch_sv_dets:
            sv_dets = sv_dets.with_nms(threshold=NMS_THRESHOLD)
            sv_dets = filter_detections(sv_dets)
            per_frame_sv.append(sv_dets)
            det_list = sv_detections_to_list(sv_dets, OUT_HEIGHT, OUT_WIDTH)
            per_frame_dets.append(det_list)

        return per_frame_dets, per_frame_sv

    print(f"\n🚀 Starting inference...")

    def flush_batch():
        """Process whatever is in frame_batch right now."""
        nonlocal frame_batch, frame_meta_batch, \
                 track_history, all_detections, det_id_counter, processed_frames

        if not frame_batch:
            return

        per_frame_dets, per_frame_sv = process_batch(frame_batch, frame_meta_batch)

        for i, (frame_resized, (frame_index, srt_entry)) in \
                enumerate(zip(frame_batch, frame_meta_batch)):

            processed_frames += 1
            chainage_m = srt_entry['cumulative_chainage_m']

            # ── Track ────────────────────────────────────────────────────
            tracked_objects = tracker.update(per_frame_dets[i])

            # ── Annotate with supervision (mask → bbox → label) ──────────
            sv_dets = per_frame_sv[i]

            # Build labels: "class_name confidence"
            detections_labels = []
            for class_id, confidence in zip(sv_dets.class_id, sv_dets.confidence):
                if 0 <= class_id < len(CLASS_NAMES):
                    detections_labels.append(
                        f"{CLASS_NAMES[class_id]} {confidence:.2f}"
                    )
                else:
                    detections_labels.append(f"Unknown {confidence:.2f}")

            annotated = frame_resized.copy()
            annotated = mask_annotator.annotate(
                scene=annotated, detections=sv_dets)
            annotated = bbox_annotator.annotate(
                scene=annotated, detections=sv_dets)
            annotated = label_annotator.annotate(
                scene=annotated, detections=sv_dets,
                labels=detections_labels)
            draw_legend(annotated, OUT_WIDTH, OUT_HEIGHT)

            # ── Update track history ──────────────────────────────────────
            for tr in tracked_objects:
                tid      = tr['track_id']
                bbox     = tr['bbox']
                cls_name = tr['class']
                polygon  = tr['polygon']

                # Extra safety — never track ignored classes
                if cls_name in IGNORED_CLASSES:
                    continue

                if tid not in track_history:
                    track_history[tid] = {
                        'type':            cls_name,
                        'first_frame':     frame_index + 1,
                        'last_frame':      frame_index + 1,
                        'first_chainage':  chainage_m,
                        'last_chainage':   chainage_m,
                        'first_timestamp': srt_entry['absolute_timestamp'],
                        'last_timestamp':  srt_entry['absolute_timestamp'],
                        'gps_lat':         srt_entry['latitude'],
                        'gps_lon':         srt_entry['longitude'],
                        'best_snapshot':   None,
                    }

                track_history[tid]['last_frame']     = frame_index + 1
                track_history[tid]['last_chainage']  = chainage_m
                track_history[tid]['last_timestamp'] = srt_entry['absolute_timestamp']

                if track_history[tid]['best_snapshot'] is None:
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = frame_resized[
                        max(0, y1):min(OUT_HEIGHT, y2),
                        max(0, x1):min(OUT_WIDTH,  x2)
                    ].copy()
                    track_history[tid]['best_snapshot'] = {
                        'frame_idx':  frame_index + 1,
                        'crop':       crop,
                        'full_frame': annotated.copy(),
                        'polygon':    polygon,
                    }

            out.write(annotated)

            # ── Finalise lost tracks ────────────────────────────────────
            active_tids = {t['track_id'] for t in tracked_objects}
            for tid in list(track_history.keys()):
                if tid not in active_tids and tid not in tracker.tracks:
                    track = track_history.pop(tid)

                    # Never save ignored class detections
                    if track['type'] in IGNORED_CLASSES:
                        continue

                    if track['best_snapshot'] is None:
                        continue

                    dtype      = track['type']
                    crop_path  = os.path.join(
                        defects_dir, dtype,
                        f"{dtype}_crop_{det_id_counter}.jpg"
                    )
                    frame_path = os.path.join(
                        frames_dir,
                        f"{dtype}_frame_{det_id_counter}.jpg"
                    )
                    image_saver.save(crop_path,  track['best_snapshot']['crop'])
                    image_saver.save(frame_path, track['best_snapshot']['full_frame'])

                    all_detections.append({
                        'id':               det_id_counter,
                        'track_id':         tid,
                        'defect_type':      dtype,
                        'video_name':       video_name,
                        'srt_name':         srt_name,
                        'frame_start':      track['first_frame'],
                        'frame_end':        track['last_frame'],
                        'timestamp_start':  track['first_timestamp'].isoformat(),
                        'timestamp_end':    track['last_timestamp'].isoformat(),
                        'chainage_start_m': track['first_chainage'],
                        'chainage_end_m':   track['last_chainage'],
                        'chainage_avg_m':   (track['first_chainage'] +
                                             track['last_chainage']) / 2,
                        'gps': {
                            'latitude':  track['gps_lat'],
                            'longitude': track['gps_lon'],
                        },
                        'polygon': track['best_snapshot']['polygon'],
                        'images': {
                            'crop':  os.path.relpath(crop_path,  video_out_dir),
                            'frame': os.path.relpath(frame_path, video_out_dir),
                        },
                    })
                    det_id_counter += 1

            # ── Progress ────────────────────────────────────────────────
            if processed_frames % 50 == 0:
                elapsed    = time.time() - start_time
                fps_actual = processed_frames / elapsed
                remaining  = (total_frames // PROCESS_EVERY_N_FRAMES) - processed_frames
                eta_min    = (remaining / fps_actual / 60) if fps_actual > 0 else 0
                progress   = (frame_index * 100) // total_frames
                print(f"   [{progress:3d}%] Frame {frame_index}/{total_frames} | "
                      f"Speed: {fps_actual:.1f} fps | ETA: {eta_min:.1f} min | "
                      f"Detections: {len(all_detections)}")
                print_gpu_memory()

        frame_batch.clear()
        frame_meta_batch.clear()

    # ── Main read loop ─────────────────────────────────────────────────────
    while True:
        item = reader.read()
        if item is None:
            # Video ended — flush remaining frames in batch
            flush_batch()
            break

        frame_index, frame = item

        srt_entry = get_srt_data_for_frame(frame_index + 1, srt_data)
        if srt_entry is None:
            continue

        # Resize frame for inference
        frame_resized = cv2.resize(
            frame, (OUT_WIDTH, OUT_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )

        frame_batch.append(frame_resized)
        frame_meta_batch.append((frame_index, srt_entry))

        # Process when batch is full
        if len(frame_batch) >= BATCH_SIZE:
            flush_batch()

    # ── Flush remaining active tracks at video end ─────────────────────────
    for tid, track in track_history.items():
        # Never save ignored class detections
        if track['type'] in IGNORED_CLASSES:
            continue
        if track['best_snapshot'] is None:
            continue

        dtype      = track['type']
        crop_path  = os.path.join(
            defects_dir, dtype,
            f"{dtype}_crop_{det_id_counter}.jpg"
        )
        frame_path = os.path.join(
            frames_dir,
            f"{dtype}_frame_{det_id_counter}.jpg"
        )
        image_saver.save(crop_path,  track['best_snapshot']['crop'])
        image_saver.save(frame_path, track['best_snapshot']['full_frame'])

        all_detections.append({
            'id':               det_id_counter,
            'track_id':         tid,
            'defect_type':      dtype,
            'video_name':       video_name,
            'srt_name':         srt_name,
            'frame_start':      track['first_frame'],
            'frame_end':        track['last_frame'],
            'timestamp_start':  track['first_timestamp'].isoformat(),
            'timestamp_end':    track['last_timestamp'].isoformat(),
            'chainage_start_m': track['first_chainage'],
            'chainage_end_m':   track['last_chainage'],
            'chainage_avg_m':   (track['first_chainage'] +
                                  track['last_chainage']) / 2,
            'gps': {
                'latitude':  track['gps_lat'],
                'longitude': track['gps_lon'],
            },
            'polygon': track['best_snapshot']['polygon'],
            'images': {
                'crop':  os.path.relpath(crop_path,  video_out_dir),
                'frame': os.path.relpath(frame_path, video_out_dir),
            },
        })
        det_id_counter += 1

    print("💾 Flushing image save queue...")
    image_saver.shutdown()
    reader.release()
    out.release()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Summary & JSON ─────────────────────────────────────────────────────
    summary = {}
    for d in all_detections:
        summary.setdefault(d['defect_type'], {'count': 0})['count'] += 1

    detection_data = {
        'video_name':          video_name,
        'srt_name':            srt_name,
        'processing_date':     datetime.now().isoformat(),
        'model':               'RFDETRSegMedium',
        'total_frames':        total_frames,
        'processed_frames':    processed_frames,
        'starting_chainage_m': starting_chainage_m,
        'ending_chainage_m':   ending_chainage,
        'total_detections':    len(all_detections),
        'summary':             summary,
        'detections':          all_detections,
    }

    json_path = os.path.join(video_out_dir, f"{video_basename}_detections.json")
    with open(json_path, 'w') as f:
        json.dump(detection_data, f, indent=2)

    elapsed_total = time.time() - start_time
    print(f"\n✅ Done! {elapsed_total/60:.1f} min | {len(all_detections)} detections")
    print(f"   📹 {output_video_path}")
    print(f"   📊 {json_path}")
    return json_path, ending_chainage


# ============================
# MAIN
# ============================

def main():
    print("\n" + "="*70)
    print("ROAD ASSESSMENT — RFDETRSegMedium | Batch Inference (No SAHI)")
    print("="*70)

    device = check_gpu()

    active_modes = sum([SINGLE_VIDEO_MODE, MULTI_VIDEO_MODE, FOLDER_MODE])
    if active_modes != 1:
        print("❌ Set exactly ONE of SINGLE_VIDEO_MODE / MULTI_VIDEO_MODE / FOLDER_MODE to True")
        return

    pairs = []
    if SINGLE_VIDEO_MODE:
        pairs = [{"video": VIDEO_PATH, "srt": SRT_PATH}]
    elif MULTI_VIDEO_MODE:
        pairs = [p for p in VIDEO_SRT_PAIRS
                 if os.path.exists(p['video']) and os.path.exists(p['srt'])]
    elif FOLDER_MODE:
        pairs = find_video_srt_pairs(INPUT_FOLDER)

    if not pairs:
        print("❌ No valid video-SRT pairs found!")
        return

    if not os.path.exists(SEGMENTATION_MODEL_PATH):
        print(f"❌ Checkpoint not found: {SEGMENTATION_MODEL_PATH}")
        return

    rfdetr_device = "cuda" if device == "cuda" else "cpu"
    print(f"\n🤖 Loading RFDETRSegMedium...")
    print(f"   Checkpoint  : {SEGMENTATION_MODEL_PATH}")
    print(f"   Device      : {rfdetr_device}")
    print(f"   Image size  : {IMAGE_SIZE}")
    print(f"   Output res  : {OUT_WIDTH}x{OUT_HEIGHT}")
    print(f"   Classes     : {len(CLASS_NAMES)}")
    print(f"   Ignored     : {IGNORED_CLASSES}")

    rfdetr_model = RFDETRSegMedium(
        pretrain_weights=SEGMENTATION_MODEL_PATH,
        num_queries=NUM_QUERIES,
        device=rfdetr_device,
        image_size=IMAGE_SIZE,
        max_image_size=IMAGE_SIZE,
    )
    print("✅ Model loaded")

    try:
        rfdetr_model.optimize_for_inference()
        print("✅ optimize_for_inference() enabled (FP16 + compile)")
    except Exception as e:
        print(f"⚠️  optimize_for_inference() skipped: {e}")

    warmup_model(rfdetr_model, device)

    print(f"\n{'='*70}")
    print(f"PROCESSING {len(pairs)} VIDEO(S)")
    print(f"{'='*70}")

    all_json_files      = []
    cumulative_chainage = 0

    for idx, pair in enumerate(pairs, 1):
        print(f"\n[VIDEO {idx}/{len(pairs)}]")
        json_path, ending_chainage = process_video(
            pair['video'], pair['srt'],
            rfdetr_model, OUTPUT_BASE_DIR, device,
            starting_chainage_m=cumulative_chainage
        )
        if json_path:
            all_json_files.append(json_path)
            cumulative_chainage = ending_chainage

    print(f"\n{'='*70}")
    print(f"✅ ALL DONE!")
    print(f"   Videos processed : {len(all_json_files)}")
    print(f"   Total chainage   : {cumulative_chainage/1000:.2f} km")
    print(f"   Output           : {OUTPUT_BASE_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()