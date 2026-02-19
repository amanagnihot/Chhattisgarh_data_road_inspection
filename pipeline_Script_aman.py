# ============================================================================================
# RF-DETR ROAD DEFECT DETECTION - MAXIMUM SPEED OPTIMIZED
# ============================================================================================
# Speed Optimizations:
#   ✅ Aggressive frame skipping (every 5 frames)
#   ✅ Large batch size (16 frames)
#   ✅ FP16 inference
#   ✅ Reduced buffer size (20 frames)
#   ✅ Faster confirmation (2 hits)
#   ✅ Threaded prefetching (64 frames ahead)
#   ✅ Async image saving (8 workers)
#   ✅ Frequent GPU cache clearing
#   ✅ Diagnostic timing output
# ============================================================================================

import os
import cv2
import math
import numpy as np
import re
import torch
import json
import threading
import queue
from datetime import datetime
from rfdetr import RFDETRSegMedium
import supervision as sv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from collections import deque
import time

# ============================================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================================
VIDEO_PATH      = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.MP4"
SRT_PATH        = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.SRT"
CHECKPOINT_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
COCO_JSON_PATH  = r"/media/user/New Volume/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"
OUTPUT_BASE_DIR = r"/media/user/New Volume/Sakshi/chattishgarh/output_rfdetr_speed_optimized"

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ============================================================================================
# MODEL SETTINGS
# ============================================================================================

NUM_QUERIES  = 200
IMAGE_SIZE   = 432
OUT_WIDTH    = 1280
OUT_HEIGHT   = 720

# ============================================================================================
# SPEED-OPTIMIZED SETTINGS
# ============================================================================================

PROCESS_EVERY_N_FRAMES = 5   # ✅ Process every 5th frame (5x faster)
BATCH_SIZE             = 16  # ✅ Large batch for GPU efficiency
MASK_OPACITY           = 0.15

# ─── Class-Specific Confidence Thresholds ───
CLASS_THRESHOLDS = {
    "Patch":                      0.50,
    "pothole":                    0.35,
    "Cracking":                   0.30,
    "Ravelling":                  0.50,
    "Edge_breaking":              0.35,
    "Edge_drop":                  0.35,
    "MBCB_defect":                0.30,
    "MBCB_missing":               0.30,
    "Corrugations_and_shoving":   0.30,
    "Scaling":                    0.30,
    "Wear":                       0.40,
    "honeycomb":                  0.30,
    "depression":                 0.30,
    "Embankment slope":           0.30,
    "strip_seal_expansion_join":  0.30,
    "tyre_marks":                 0.35,
    "vegetation_on_road":         0.30,
    "km_stone":                   0.40,
    "cattle":                     0.45,
}

GLOBAL_THRESHOLD   = 0.30
NMS_THRESHOLD      = 0.65

# ─── Ignored Classes ───
IGNORED_CLASSES = {
    "white_mark", "water_mark", "doubt", "bump",
    "guard_post", "overhead_sign_board", "sign_board", "kerb_damage"
}

# ─── Fast Tracking Parameters ───
IOU_THRESHOLD     = 0.3
MAX_DISTANCE      = 50
MAX_LOST          = 15       # ✅ Reduced from 30 (faster finalization)
CONFIRMATION_HITS = 2        # ✅ Reduced from 3 (faster confirmation)

# ─── Reduced Buffer Size ───
BUFFER_SIZE = 20             # ✅ Smaller buffer (was 37)

# ─── Special Logic ───
POTHOLE_PATCHING_IOU_THRESHOLD = 0.90

# ─── Aggressive Threading ───
PREFETCH_QUEUE_SIZE  = 64    # ✅ More prefetch (was 32)
SAVE_WORKER_THREADS  = 8     # ✅ More writers (was 4)

# ─── Report Image Sizes ───
CROP_SIZE  = (400, 400)
FRAME_SIZE = (800, 600)

# ─── GPU Cache Clear Interval ───
GPU_CLEAR_EVERY = 100        # ✅ Clear more often (was 200)

# ─── Diagnostic Mode ───
ENABLE_TIMING_DIAGNOSTICS = True  # Set to False to disable timing output

# ============================================================================================
# LOAD CLASS NAMES
# ============================================================================================

print("=" * 80)
print("LOADING CONFIGURATION (SPEED-OPTIMIZED)")
print("=" * 80)

with open(COCO_JSON_PATH, "r") as f:
    coco_data = json.load(f)

categories  = sorted(coco_data["categories"], key=lambda x: x["id"])
CLASS_NAMES = [cat["name"] for cat in categories]

print(f"\n✅ Loaded {len(CLASS_NAMES)} classes")
print(f"⚡ SPEED MODE: Skip={PROCESS_EVERY_N_FRAMES}, Batch={BATCH_SIZE}, Buffer={BUFFER_SIZE}")

# ============================================================================================
# COLOR PALETTE
# ============================================================================================

HEX_COLORS = [
    "#FF0000", "#00A5FF", "#008CFF", "#FF00FF", "#8000FF",
    "#00FF80", "#FF0080", "#B400B4", "#0000FF", "#02D32E",
    "#FF8000", "#00FFFF", "#FFFF00", "#FF4040", "#4040FF",
    "#A0A0A0", "#006400", "#D2691E", "#008080", "#FFD700",
    "#873CBE", "#00C8C8", "#FF1493", "#32CD32", "#FF69B4",
    "#696969", "#228B22", "#ADD8E6", "#F5F5F5",
]
while len(HEX_COLORS) < len(CLASS_NAMES):
    HEX_COLORS.append("#FFFFFF")

# ============================================================================================
# GPU UTILITIES
# ============================================================================================

def check_gpu():
    if torch.cuda.is_available():
        device    = 'cuda:0'
        gpu_name  = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU: {gpu_name}")
        print(f"   VRAM: {total_mem:.1f} GB")
        print(f"   CUDA: {torch.version.cuda}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark        = True
        torch.backends.cudnn.enabled          = True
        return device
    else:
        print("\n⚠️  No GPU found — running on CPU (slow)")
        return 'cpu'

def print_gpu_memory():
    if torch.cuda.is_available():
        used   = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved()   / 1e9
        print(f"   🖥️  GPU Memory: {used:.2f}GB used / {cached:.2f}GB cached")

# ============================================================================================
# ASYNC IMAGE SAVER
# ============================================================================================

class AsyncImageSaver:
    def __init__(self, num_workers=SAVE_WORKER_THREADS):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.futures  = []

    def save(self, path, image):
        future = self.executor.submit(cv2.imwrite, path, image)
        self.futures.append(future)

    def wait_all(self):
        for f in self.futures:
            f.result()
        self.futures.clear()

    def shutdown(self):
        self.wait_all()
        self.executor.shutdown(wait=True)

# ============================================================================================
# THREADED FRAME PREFETCHER
# ============================================================================================

class FramePrefetcher:
    def __init__(self, video_path, out_w, out_h, queue_size=PREFETCH_QUEUE_SIZE):
        self.cap     = cv2.VideoCapture(video_path)
        self.out_w   = out_w
        self.out_h   = out_h
        self.q       = queue.Queue(maxsize=queue_size)
        self.thread  = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            resized = cv2.resize(frame, (self.out_w, self.out_h),
                                 interpolation=cv2.INTER_LINEAR)
            self.q.put((idx, resized))
            idx += 1
        self.cap.release()

    def get(self):
        return self.q.get()

def get_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total_frames, orig_w, orig_h

# ============================================================================================
# SRT PARSING & GPS UTILITIES
# ============================================================================================

def parse_srt(srt_path):
    srt_data = []
    if not os.path.exists(srt_path):
        print(f"❌ SRT file not found: {srt_path}")
        return None

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            frame_num = int(lines[0].strip())
            tc_match = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
            if not tc_match:
                continue
            h, m, s, ms = map(int, tc_match.groups())
            timestamp_ms = (h * 3600 + m * 60 + s) * 1000 + ms
            meta = ' '.join(lines[2:])

            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', meta)
            if not ts_match:
                continue
            abs_ts = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S.%f')

            lat_m = re.search(r'\[latitude:\s*([-\d.]+)\]', meta)
            lon_m = re.search(r'\[longitude:\s*([-\d.]+)\]', meta)
            alt_m = re.search(r'\[(?:abs_alt|altitude):\s*([-\d.]+)', meta)

            if not (lat_m and lon_m):
                continue

            srt_data.append({
                'frame_number': frame_num,
                'timestamp_ms': timestamp_ms,
                'absolute_timestamp': abs_ts,
                'latitude': float(lat_m.group(1)),
                'longitude': float(lon_m.group(1)),
                'altitude': float(alt_m.group(1)) if alt_m else 50.0
            })
        except Exception:
            continue

    if not srt_data:
        print("❌ No GPS data in SRT!")
        return None

    print(f"✅ Parsed {len(srt_data)} SRT entries")
    return srt_data

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_cumulative_chainage(srt_data, start_m=0):
    dist = start_m
    for i, entry in enumerate(srt_data):
        if i == 0:
            entry['cumulative_chainage_m'] = start_m
        else:
            prev = srt_data[i-1]
            dist += haversine_distance(prev['latitude'], prev['longitude'],
                                       entry['latitude'], entry['longitude'])
            entry['cumulative_chainage_m'] = dist
    total_km = (dist - start_m) / 1000
    print(f"✅ Chainage: {start_m:.1f}m → {dist:.1f}m ({total_km:.3f} km)")
    return dist

def get_srt_for_frame(frame_idx, srt_data):
    for e in srt_data:
        if e['frame_number'] == frame_idx:
            return e
    if srt_data:
        return min(srt_data, key=lambda x: abs(x['frame_number'] - frame_idx))
    return None

# ============================================================================================
# FAST TRACKER
# ============================================================================================

class ImprovedSegmentationTracker:
    def __init__(self):
        self.next_internal_id = 1
        self.next_display_id  = 1
        self.active_tracks    = {}
        self.confirmed_tracks = {}
        self.internal_to_display = {}

    def _mask_iou(self, m1, m2):
        if m1 is None or m2 is None:
            return 0.0
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return inter / union if union > 0 else 0.0

    def _bbox_iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _centroid(self, bbox):
        return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

    def _dist(self, p1, p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def update(self, detections, frame_index):
        for t in list(self.active_tracks.values()) + list(self.confirmed_tracks.values()):
            t['lost'] += 1

        newly_confirmed = []
        all_tracked = []

        for det in detections:
            best_id = None
            best_score = 0

            for pool in [self.confirmed_tracks, self.active_tracks]:
                for tid, track in pool.items():
                    if track['class'] != det['class']:
                        continue
                    iou = (self._mask_iou(det.get('mask'), track.get('mask'))
                           if det.get('mask') is not None and track.get('mask') is not None
                           else self._bbox_iou(det['bbox'], track['bbox']))
                    cd = self._dist(self._centroid(det['bbox']),
                                    self._centroid(track['bbox']))
                    if iou > IOU_THRESHOLD or cd < MAX_DISTANCE:
                        score = iou - (cd / MAX_DISTANCE) * 0.3
                        if score > best_score:
                            best_score = score
                            best_id = tid

            if best_id is not None:
                pool = (self.confirmed_tracks if best_id in self.confirmed_tracks
                        else self.active_tracks)
                track = pool[best_id]
                track.update({
                    'bbox': det['bbox'],
                    'mask': det.get('mask'),
                    'polygon': det.get('polygon'),
                    'confidence': det['confidence'],
                    'last_frame': frame_index,
                    'lost': 0
                })
                track['confidences'].append(det['confidence'])
                track['hits'] += 1

                if best_id in self.active_tracks and track['hits'] >= CONFIRMATION_HITS:
                    self.confirmed_tracks[best_id] = track
                    del self.active_tracks[best_id]
                    self.internal_to_display[best_id] = self.next_display_id
                    track['display_id'] = self.next_display_id
                    self.next_display_id += 1
                    newly_confirmed.append({
                        'internal_id': best_id,
                        'display_id': track['display_id'],
                        'bbox': track['bbox'],
                        'mask': track['mask'],
                        'polygon': track['polygon'],
                        'class': track['class'],
                        'confidence': track['confidence']
                    })

                if best_id in self.confirmed_tracks:
                    all_tracked.append({
                        'internal_id': best_id,
                        'display_id': track.get('display_id', -1),
                        'bbox': track['bbox'],
                        'mask': track['mask'],
                        'polygon': track['polygon'],
                        'class': track['class'],
                        'confidence': track['confidence']
                    })
            else:
                nid = self.next_internal_id
                self.next_internal_id += 1
                self.active_tracks[nid] = {
                    'bbox': det['bbox'],
                    'mask': det.get('mask'),
                    'polygon': det.get('polygon'),
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'confidences': [det['confidence']],
                    'hits': 1,
                    'lost': 0,
                    'first_frame': frame_index,
                    'last_frame': frame_index
                }

        for tid in list(self.active_tracks.keys()):
            if self.active_tracks[tid]['lost'] > MAX_LOST:
                del self.active_tracks[tid]
        for tid in list(self.confirmed_tracks.keys()):
            if self.confirmed_tracks[tid]['lost'] > MAX_LOST:
                del self.confirmed_tracks[tid]

        return all_tracked, newly_confirmed

# ============================================================================================
# UTILITIES
# ============================================================================================

def calculate_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def resize_to_fixed_size(image, target_size):
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    yo = (th-nh)//2; xo = (tw-nw)//2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas

# ============================================================================================
# ROLLING FRAME BUFFER
# ============================================================================================

class RollingFrameBuffer:
    def __init__(self, maxsize=BUFFER_SIZE):
        self.maxsize = maxsize
        self.buffer = deque()

    def push(self, frame_idx, frame, tracked_objects):
        self.buffer.append({
            'idx': frame_idx,
            'frame': frame,
            'tracked': tracked_objects
        })
        flushed = []
        while len(self.buffer) > self.maxsize:
            flushed.append(self.buffer.popleft())
        return flushed

    def flush_all(self):
        flushed = list(self.buffer)
        self.buffer.clear()
        return flushed

    def retroactive_confirm(self, internal_id, display_id):
        for item in self.buffer:
            for tr in item['tracked']:
                if tr['internal_id'] == internal_id:
                    tr['display_id'] = display_id

# ============================================================================================
# ANNOTATION
# ============================================================================================

def annotate_frame(frame, tracked_objects, mask_annotator, bbox_annotator, 
                   label_annotator, color_palette, active_classes):
    confirmed = [t for t in tracked_objects if t.get('display_id', -1) > 0]
    if not confirmed:
        return frame

    xyxy = np.array([t['bbox'] for t in confirmed])
    class_ids = np.array([CLASS_NAMES.index(t['class']) for t in confirmed])
    confidences = np.array([t['confidence'] for t in confirmed])

    masks = None
    if confirmed[0].get('mask') is not None:
        try:
            masks = np.array([t['mask'] for t in confirmed])
        except Exception:
            masks = None

    sv_det = sv.Detections(xyxy=xyxy, class_id=class_ids,
                           confidence=confidences, mask=masks)

    labels = [f"{t['display_id']}-{t['class']} {t['confidence']:.2f}"
              for t in confirmed]

    annotated = frame.copy()
    if masks is not None:
        annotated = mask_annotator.annotate(scene=annotated, detections=sv_det)
    annotated = bbox_annotator.annotate(scene=annotated, detections=sv_det)
    annotated = label_annotator.annotate(scene=annotated, detections=sv_det, labels=labels)

    if active_classes:
        draw_legend(annotated, OUT_WIDTH, OUT_HEIGHT, active_classes)

    return annotated

def draw_legend(frame, width, height, active_classes):
    scale = max(0.6, min(width/1920.0, 1.2))
    lx, ly = int(0.02*width), int(0.04*height)
    ts = 0.6 * scale
    tt = max(1, int(1.5*scale))
    lh = int(22*scale)
    ll = int(35*scale)
    lw = int(300*scale)

    cv2.rectangle(frame, (lx-12, ly-12),
                  (lx+lw, ly+lh*len(active_classes)+12),
                  (0, 0, 0), max(2, int(2*scale)))

    for idx, (cls_name, color_hex) in enumerate(active_classes.items()):
        y = ly + idx*lh
        hex_ = color_hex.lstrip('#')
        r,g,b = tuple(int(hex_[i:i+2], 16) for i in (0,2,4))
        cv2.line(frame, (lx, y+10), (lx+ll, y+10), (b,g,r), 2)
        cv2.putText(frame, cls_name, (lx+ll+12, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, ts, (255,255,255), tt)

# ============================================================================================
# WORD REPORT
# ============================================================================================

def generate_word_report(confirmed_detections, output_dir):
    doc = Document()
    doc.add_heading('Road Defects Detection Report - RF-DETR', 0)

    sorted_dets = sorted(confirmed_detections, key=lambda x: x['display_id'])

    by_class = {}
    for det in sorted_dets:
        by_class.setdefault(det['defect_type'], []).append(det)

    doc.add_heading('Summary', 1)
    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = 'Light Grid Accent 1'
    tbl.rows[0].cells[0].text = 'Defect Type'
    tbl.rows[0].cells[1].text = 'Count'
    for cls in sorted(by_class):
        r = tbl.add_row().cells
        r[0].text = cls
        r[1].text = str(len(by_class[cls]))

    doc.add_paragraph('')

    for cls in sorted(by_class):
        dets = by_class[cls]
        doc.add_heading(f'{cls.upper()} DETECTIONS', 1)
        doc.add_paragraph(f'Total: {len(dets)}')
        doc.add_paragraph('')

        tbl = doc.add_table(rows=1, cols=4)
        tbl.style = 'Light Grid Accent 1'
        hdr = tbl.rows[0].cells
        hdr[0].text = 'ID'; hdr[1].text = 'Chainage (m)'
        hdr[2].text = 'Crop'; hdr[3].text = 'Frame'

        for det in dets:
            row = tbl.add_row().cells
            row[0].text = f"{det['display_id']}-{cls}"
            row[1].text = f"{det['chainage_avg_m']:.1f}"
            for col_idx, img_key, w_inches in [(2,'crop',1.5),(3,'frame',2.5)]:
                p = det['images'][img_key]
                if os.path.exists(p):
                    try:
                        para = row[col_idx].paragraphs[0]
                        para.add_run().add_picture(p, width=Inches(w_inches))
                        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except Exception:
                        row[col_idx].text = str(det['display_id'])

        doc.add_page_break()

    rpath = os.path.join(output_dir, "Road_Defects_Report.docx")
    doc.save(rpath)
    print(f"\n📄 Word report: {rpath}")

# ============================================================================================
# MAIN PROCESSING - SPEED OPTIMIZED
# ============================================================================================

def process_video(video_path, srt_path, model, output_dir, device, starting_chainage_m=0):
    
    video_name = os.path.basename(video_path)
    srt_name = os.path.basename(srt_path)
    video_base = os.path.splitext(video_name)[0]

    print(f"\n{'='*80}")
    print(f"PROCESSING: {video_name}")
    print(f"{'='*80}")

    # Parse SRT
    srt_data = parse_srt(srt_path)
    if srt_data is None:
        return None, starting_chainage_m
    ending_chainage = calculate_cumulative_chainage(srt_data, starting_chainage_m)

    # Video properties
    fps, total_frames, orig_w, orig_h = get_video_props(video_path)
    print(f"\n🎥 Video: {orig_w}×{orig_h} @ {fps:.1f}fps  |  {total_frames} frames")
    print(f"   Processing: {OUT_WIDTH}×{OUT_HEIGHT}  every {PROCESS_EVERY_N_FRAMES} frame(s)")
    print(f"   Expected frames to process: ~{total_frames // PROCESS_EVERY_N_FRAMES}")
    print_gpu_memory()

    # Output paths
    video_out_dir = os.path.join(output_dir, video_base)
    defects_dir = os.path.join(video_out_dir, "defect_images")
    frames_dir = os.path.join(defects_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for cls in CLASS_NAMES:
        if cls not in IGNORED_CLASSES:
            os.makedirs(os.path.join(defects_dir, cls), exist_ok=True)

    out_video_path = os.path.join(video_out_dir, f"{video_base}_output.mp4")
    out_fps = fps / PROCESS_EVERY_N_FRAMES

    # Supervision annotators
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
    thickness = sv.calculate_optimal_line_thickness(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
    color_palette = sv.ColorPalette.from_hex(HEX_COLORS)

    mask_annotator = sv.MaskAnnotator(color=color_palette, opacity=MASK_OPACITY)
    bbox_annotator = sv.BoxAnnotator(color=color_palette, thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        color=color_palette, text_color=sv.Color.WHITE,
        text_scale=text_scale, text_thickness=max(1, thickness-1)
    )

    # Initialize objects
    tracker = ImprovedSegmentationTracker()
    image_saver = AsyncImageSaver()
    frame_buffer = RollingFrameBuffer(maxsize=BUFFER_SIZE)
    video_writer = cv2.VideoWriter(out_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   out_fps, (OUT_WIDTH, OUT_HEIGHT))

    confirmed_detections = {}
    active_classes = {}
    start_time = time.time()

    # Prefetcher
    prefetcher = FramePrefetcher(video_path, OUT_WIDTH, OUT_HEIGHT)

    batch_frames = []
    batch_indices = []
    all_frame_idx = 0
    processed_count = 0

    # Timing accumulators
    total_inf_time = 0
    total_proc_time = 0
    total_write_time = 0
    timing_samples = 0

    print(f"\n{'='*80}")
    print("SPEED-OPTIMIZED PROCESSING")
    print(f"{'='*80}")

    def flush_and_write(flushed_items):
        t_w_start = time.time()
        for item in flushed_items:
            annotated = annotate_frame(
                item['frame'], item['tracked'],
                mask_annotator, bbox_annotator, label_annotator,
                color_palette, active_classes
            )
            video_writer.write(annotated)
        return time.time() - t_w_start

    def run_inference_batch():
        nonlocal processed_count, total_inf_time, total_proc_time, total_write_time, timing_samples

        if not batch_frames:
            return

        t_batch_start = time.time()

        # ── FP16 inference with timing ──
        t_inf_start = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
                batch_detections = model.predict(batch_frames, threshold=GLOBAL_THRESHOLD)
        t_inf_end = time.time()
        inf_time = t_inf_end - t_inf_start

        # ── Processing loop with timing ──
        t_proc_start = time.time()
        write_time_accum = 0

        for b_idx, detections in enumerate(batch_detections):
            frame_idx = batch_indices[b_idx]
            frame = batch_frames[b_idx]
            srt_entry = get_srt_for_frame(frame_idx + 1, srt_data)
            if srt_entry is None:
                flushed = frame_buffer.push(frame_idx, frame, [])
                write_time_accum += flush_and_write(flushed)
                continue

            chainage_m = srt_entry['cumulative_chainage_m']

            # NMS
            detections = detections.with_nms(threshold=NMS_THRESHOLD)

            # Filter
            frame_dets = []
            if len(detections) > 0:
                for idx in range(len(detections)):
                    cid = detections.class_id[idx]
                    conf = detections.confidence[idx]
                    if cid < 0 or cid >= len(CLASS_NAMES):
                        continue
                    cname = CLASS_NAMES[cid]
                    if cname in IGNORED_CLASSES:
                        continue
                    if conf < CLASS_THRESHOLDS.get(cname, GLOBAL_THRESHOLD):
                        continue

                    bbox = detections.xyxy[idx]
                    mask, polygon = None, None
                    if detections.mask is not None:
                        mask = detections.mask[idx].astype(np.uint8)
                        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            polygon = contours[0].squeeze().tolist()
                            if isinstance(polygon[0], (int, float)):
                                polygon = [polygon]

                    frame_dets.append({
                        'bbox': bbox, 'mask': mask, 'polygon': polygon,
                        'class': cname, 'confidence': float(conf)
                    })

            # Pothole-Patch overlap removal
            pot_idx = [i for i,d in enumerate(frame_dets) if d['class'] == 'pothole']
            pat_idx = [i for i,d in enumerate(frame_dets) if d['class'] == 'Patch']
            remove = set()
            for pi in pot_idx:
                for pai in pat_idx:
                    if calculate_iou(frame_dets[pi]['bbox'],
                                     frame_dets[pai]['bbox']) >= POTHOLE_PATCHING_IOU_THRESHOLD:
                        remove.add(pi)
                        break
            frame_dets = [d for i,d in enumerate(frame_dets) if i not in remove]

            # Track
            tracked_objects, newly_confirmed = tracker.update(frame_dets, frame_idx)

            # Update active classes
            for tr in tracked_objects:
                cname = tr['class']
                if cname not in active_classes:
                    active_classes[cname] = HEX_COLORS[CLASS_NAMES.index(cname)]

            # Retroactive confirmation
            for confirmed in newly_confirmed:
                iid = confirmed['internal_id']
                did = confirmed['display_id']
                frame_buffer.retroactive_confirm(iid, did)

                # Save images
                x1,y1,x2,y2 = map(int, confirmed['bbox'])
                crop = frame[max(0,y1-30):min(OUT_HEIGHT,y2+30),
                             max(0,x1-30):min(OUT_WIDTH, x2+30)]
                crop_resized = resize_to_fixed_size(crop, CROP_SIZE)
                frame_resized = resize_to_fixed_size(frame, FRAME_SIZE)

                cname = confirmed['class']
                crop_path = os.path.join(defects_dir, cname, f"{did}-{cname}.jpg")
                frm_path = os.path.join(frames_dir, f"{did}-{cname}_frame.jpg")
                image_saver.save(crop_path, crop_resized)
                image_saver.save(frm_path, frame_resized)

                # Store metadata
                track = tracker.confirmed_tracks[iid]
                confirmed_detections[iid] = {
                    'display_id': did,
                    'internal_id': iid,
                    'defect_type': cname,
                    'video_name': video_name,
                    'srt_name': srt_name,
                    'frame_start': track['first_frame'],
                    'frame_end': track['last_frame'],
                    'timestamp_start': srt_entry['absolute_timestamp'].isoformat(),
                    'timestamp_end': srt_entry['absolute_timestamp'].isoformat(),
                    'chainage_start_m': chainage_m,
                    'chainage_end_m': chainage_m,
                    'chainage_avg_m': chainage_m,
                    'confidence_avg': float(np.mean(track['confidences'])),
                    'gps': {
                        'latitude': srt_entry['latitude'],
                        'longitude': srt_entry['longitude']
                    },
                    'polygon': confirmed['polygon'],
                    'images': {'crop': crop_path, 'frame': frm_path}
                }

            # Push to buffer and flush
            flushed = frame_buffer.push(frame_idx, frame, tracked_objects)
            write_time_accum += flush_and_write(flushed)

            processed_count += 1

        t_proc_end = time.time()
        proc_time = t_proc_end - t_proc_start - write_time_accum

        # Accumulate timing
        total_inf_time += inf_time
        total_proc_time += proc_time
        total_write_time += write_time_accum
        timing_samples += 1

        # Periodic GPU cache clear
        if processed_count % GPU_CLEAR_EVERY == 0 and device != 'cpu':
            torch.cuda.empty_cache()

        batch_frames.clear()
        batch_indices.clear()

        # Progress with diagnostics
        elapsed = time.time() - start_time
        fps_act = processed_count / elapsed if elapsed > 0 else 0
        
        if processed_count % 50 == 0:
            progress_pct = (all_frame_idx * 100) // total_frames if total_frames > 0 else 0
            
            print(f"\n   [{progress_pct:3d}%] Frame {all_frame_idx}/{total_frames}")
            print(f"   Processing: {fps_act:.1f} fps | Confirmed: {len(confirmed_detections)}")
            
            if ENABLE_TIMING_DIAGNOSTICS and timing_samples > 0:
                avg_inf = (total_inf_time / timing_samples) * 1000
                avg_proc = (total_proc_time / timing_samples) * 1000
                avg_write = (total_write_time / timing_samples) * 1000
                print(f"   ⏱️  Avg Timing: Inference={avg_inf:.1f}ms | "
                      f"Processing={avg_proc:.1f}ms | Writing={avg_write:.1f}ms")
            
            print_gpu_memory()

    # Main loop
    while True:
        item = prefetcher.get()

        if item is None:
            run_inference_batch()
            write_time = flush_and_write(frame_buffer.flush_all())
            total_write_time += write_time
            break

        frame_idx, frame = item
        all_frame_idx = frame_idx

        if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
            batch_frames.append(frame)
            batch_indices.append(frame_idx)

        if len(batch_frames) == BATCH_SIZE:
            run_inference_batch()

    # Cleanup
    video_writer.release()
    print("\n💾 Waiting for image saves...")
    image_saver.shutdown()

    if device != 'cpu':
        torch.cuda.empty_cache()

    # Update final metadata
    for iid, det in confirmed_detections.items():
        if iid in tracker.confirmed_tracks:
            track = tracker.confirmed_tracks[iid]
            last_srt = get_srt_for_frame(track['last_frame'] + 1, srt_data)
            if last_srt:
                det['frame_end'] = track['last_frame']
                det['timestamp_end'] = last_srt['absolute_timestamp'].isoformat()
                det['chainage_end_m'] = last_srt['cumulative_chainage_m']
                det['chainage_avg_m'] = (det['chainage_start_m'] + det['chainage_end_m']) / 2

    all_detections = list(confirmed_detections.values())

    # Save JSON
    detection_data = {
        'video_name': video_name,
        'srt_name': srt_name,
        'processing_date': datetime.now().isoformat(),
        'model': 'RF-DETR-Seg-Medium-SpeedOptimized',
        'total_frames': total_frames,
        'processed_frames': processed_count,
        'starting_chainage_m': starting_chainage_m,
        'ending_chainage_m': ending_chainage,
        'total_distance_km': (ending_chainage - starting_chainage_m) / 1000,
        'total_detections': len(all_detections),
        'detections': all_detections,
        'summary': {}
    }

    for det in all_detections:
        dtype = det['defect_type']
        entry = detection_data['summary'].setdefault(
            dtype, {'count': 0, 'avg_confidence': []})
        entry['count'] += 1
        entry['avg_confidence'].append(det['confidence_avg'])

    for dtype in detection_data['summary']:
        confs = detection_data['summary'][dtype]['avg_confidence']
        detection_data['summary'][dtype]['avg_confidence'] = float(np.mean(confs))

    json_path = os.path.join(video_out_dir, f"{video_base}_detections.json")
    with open(json_path, 'w') as f:
        json.dump(detection_data, f, indent=2)

    # Generate Word report
    print("\n📄 Generating Word report...")
    generate_word_report(all_detections, video_out_dir)

    elapsed_total = time.time() - start_time
    actual_fps = processed_count / elapsed_total if elapsed_total > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"   Time:        {elapsed_total/60:.1f} min ({elapsed_total:.1f}s)")
    print(f"   Throughput:  {actual_fps:.1f} fps")
    print(f"   Detections:  {len(all_detections)}")
    print(f"   📹 Video:    {out_video_path}")
    print(f"   📊 JSON:     {json_path}")
    print(f"   🖼️  Images:   {defects_dir}")
    
    print(f"\n   Defect Summary:")
    for dtype, stats in detection_data['summary'].items():
        print(f"      {dtype}: {stats['count']} (avg conf: {stats['avg_confidence']:.2f})")

    return json_path, ending_chainage

# ============================================================================================
# MAIN
# ============================================================================================

def main():
    print("\n" + "="*80)
    print("RF-DETR ROAD DEFECT DETECTION - MAXIMUM SPEED")
    print("="*80)
    print(f"⚡ Frame Skip: {PROCESS_EVERY_N_FRAMES}x")
    print(f"⚡ Batch Size: {BATCH_SIZE}")
    print(f"⚡ Buffer: {BUFFER_SIZE} frames")
    print(f"⚡ Confirmation: {CONFIRMATION_HITS} hits")

    device = check_gpu()

    for label, path in [("Video", VIDEO_PATH), ("SRT", SRT_PATH),
                         ("Checkpoint", CHECKPOINT_PATH), ("COCO JSON", COCO_JSON_PATH)]:
        if not os.path.exists(path):
            print(f"❌ {label} not found: {path}")
            return

    print(f"\n{'='*80}\nLOADING RF-DETR MODEL\n{'='*80}")
    model = RFDETRSegMedium(
        pretrain_weights=CHECKPOINT_PATH,
        num_queries=NUM_QUERIES,
        image_size=IMAGE_SIZE,
        max_image_size=IMAGE_SIZE,
    )
    print("✅ Model loaded")
    print_gpu_memory()

    # Warmup
    print("\n🔥 Warming up GPU...")
    dummy = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
            _ = model.predict([dummy], threshold=0.5)
    if device != 'cpu':
        torch.cuda.synchronize()
    print("✅ GPU warmed up")
    print_gpu_memory()

    process_video(
        VIDEO_PATH, SRT_PATH, model,
        OUTPUT_BASE_DIR, device,
        starting_chainage_m=0
    )

    print(f"\n{'='*80}")
    print("✅ ALL DONE!")
    print(f"Output: {OUTPUT_BASE_DIR}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()





















# # # ============================================================================================
# # # RF-DETR ROAD DEFECT DETECTION WITH GPS TRACKING & IMPROVED TRACKER
# # # ============================================================================================
# # # Features:
# # #   - RF-DETR segmentation model
# # #   - 5-hit confirmation system (prevents false positives)
# # #   - Sequential display IDs (clean numbering: 1, 2, 3...)
# # #   - Mask IOU tracking (more accurate than bbox only)
# # #   - SRT parsing for GPS coordinates & timestamps
# # #   - Chainage calculation (distance along road)
# # #   - Class-specific confidence thresholds
# # #   - Ignored classes filtering
# # #   - Pothole-Patching overlap removal
# # #   - GPU-optimized (FP16, threaded I/O, batch processing)
# # #   - Two-pass video processing (track first, annotate with confirmed IDs only)
# # #   - Organized output: JSON + Word report + cropped images + annotated video
# # # ============================================================================================

# # import os
# # import cv2
# # import math
# # import numpy as np
# # import re
# # import torch
# # import json
# # import threading
# # import queue
# # from datetime import datetime
# # from rfdetr import RFDETRSegMedium
# # import supervision as sv
# # from concurrent.futures import ThreadPoolExecutor
# # from pathlib import Path
# # from docx import Document
# # from docx.shared import Inches
# # from docx.enum.text import WD_ALIGN_PARAGRAPH

# # # ============================================================================================
# # # CONFIGURATION - UPDATE THESE PATHS
# # # ============================================================================================

# # # ─── Input Files ───
# # VIDEO_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.MP4"
# # SRT_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.SRT"
# # CHECKPOINT_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
# # COCO_JSON_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"

# # # ─── Output Directory ───
# # OUTPUT_BASE_DIR = r"/media/user/New Volume/Sakshi/chattishgarh/output_rfdetr_gps_final"
                    
# # os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# # # ============================================================================================
# # # RF-DETR MODEL SETTINGS
# # # ============================================================================================

# # NUM_QUERIES = 200
# # IMAGE_SIZE = 432
# # OUT_WIDTH = 1280      # Resize 4K → 1280×720 for processing
# # OUT_HEIGHT = 720

# # # ============================================================================================
# # # PROCESSING SETTINGS
# # # ============================================================================================

# # PROCESS_EVERY_N_FRAMES = 1   # Process every 5th frame (30fps → 6fps effective)
# # BATCH_SIZE = 4               # RF-DETR batch inference
# # MASK_OPACITY = 0.15

# # # ─── Class-Specific Confidence Thresholds ───
# # CLASS_THRESHOLDS = {
# #     "Patch": 0.50,
# #     "pothole": 0.35,
# #     "Cracking": 0.30,
# #     "Ravelling": 0.50,
# #     "Edge_breaking": 0.35,
# #     "Edge_drop": 0.35,
# #     "MBCB_defect": 0.30,
# #     "MBCB_missing": 0.30,
# #     "Corrugations_and_shoving": 0.30,
# #     "Scaling": 0.30,
# #     "Wear": 0.40,
# #     # "Edge_drop": 0.30,
# #     "honeycomb": 0.30,
# #     "depression": 0.30,
# #     "Embankment slope": 0.30,
# #     "strip_seal_expansion_join": 0.30,
# #     "tyre_marks": 0.35,
# #     "vegetation_on_road": 0.30,
# #     "km_stone": 0.40,
# #     "cattle": 0.45,
# # }

# # GLOBAL_THRESHOLD = 0.30
# # NMS_THRESHOLD = 0.65

# # # ─── Ignored Classes (won't be tracked or saved) ───
# # IGNORED_CLASSES = {
# #     "white_mark",
# #     "water_mark",
# #     "doubt",
# #     "bump",
# #     "guard_post",
# #     "overhead_sign_board",
# #     "sign_board",
# #     "kerb_damage"
# # }

# # # ─── Tracking Parameters ───
# # IOU_THRESHOLD = 0.3
# # MAX_DISTANCE = 50           # pixels
# # MAX_LOST = 30               # frames before finalizing a track
# # CONFIRMATION_HITS = 5       # ✅ NEW: Detections needed before confirming object

# # # ─── Special Logic ───
# # POTHOLE_PATCHING_IOU_THRESHOLD = 0.90  # If pothole overlaps >90% with patch → remove pothole

# # # ─── Threading ───
# # FRAME_BUFFER_SIZE = 16
# # SAVE_WORKER_THREADS = 4

# # # ─── Report Image Sizes ───
# # CROP_SIZE = (400, 400)
# # FRAME_SIZE = (800, 600)

# # # ============================================================================================
# # # LOAD CLASS NAMES FROM COCO JSON
# # # ============================================================================================

# # print("=" * 80)
# # print("LOADING CONFIGURATION")
# # print("=" * 80)

# # with open(COCO_JSON_PATH, "r") as f:
# #     coco_data = json.load(f)

# # categories = sorted(coco_data["categories"], key=lambda x: x["id"])
# # CLASS_NAMES = [cat["name"] for cat in categories]

# # print(f"\n✅ Loaded {len(CLASS_NAMES)} classes from COCO JSON:")
# # for i, name in enumerate(CLASS_NAMES):
# #     ignored_marker = " [IGNORED]" if name in IGNORED_CLASSES else ""
# #     threshold = CLASS_THRESHOLDS.get(name, GLOBAL_THRESHOLD)
# #     print(f"   {i:2d}: {name:30s} (conf ≥ {threshold:.2f}){ignored_marker}")

# # # ============================================================================================
# # # COLOR PALETTE
# # # ============================================================================================

# # HEX_COLORS = [
# #     "#FF0000",  # 0:  pothole (RED)
# #     "#00A5FF",  # 1:  Corrugations_and_shoving
# #     "#008CFF",  # 2:  Cracking
# #     "#FF00FF",  # 3:  Edge_breaking
# #     "#8000FF",  # 4:  Edge_drop
# #     "#00FF80",  # 5:  Embankment slope
# #     "#FF0080",  # 6:  MBCB_defect
# #     "#B400B4",  # 7:  MBCB_missing
# #     "#0000FF",  # 8:  Patch (BLUE)
# #     "#02D32E",  # 9:  Ravelling
# #     "#FF8000",  # 10: Scaling
# #     "#00FFFF",  # 11: Wear
# #     "#FFFF00",  # 12: bump
# #     "#FF4040",  # 13: cattle
# #     "#4040FF",  # 14: depression
# #     "#A0A0A0",  # 15: doubt
# #     "#006400",  # 16: guard_post
# #     "#D2691E",  # 17: honeycomb
# #     "#008080",  # 18: kerb_damage
# #     "#FFD700",  # 19: km_stone
# #     "#873CBE",  # 20: overhead_sign_board
# #     "#00C8C8",  # 21: parallel_crack
# #     "#FF1493",  # 22: pothole (duplicate)
# #     "#32CD32",  # 23: sign_board
# #     "#FF69B4",  # 24: strip_seal_expansion_join
# #     "#696969",  # 25: tyre_marks
# #     "#228B22",  # 26: vegetation_on_road
# #     "#ADD8E6",  # 27: water_mark
# #     "#F5F5F5",  # 28: white_mark
# # ]

# # while len(HEX_COLORS) < len(CLASS_NAMES):
# #     HEX_COLORS.append("#FFFFFF")

# # # ============================================================================================
# # # GPU UTILITIES
# # # ============================================================================================

# # def check_gpu():
# #     if torch.cuda.is_available():
# #         device = 'cuda:0'
# #         gpu_name = torch.cuda.get_device_name(0)
# #         total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
# #         print(f"\n✅ GPU: {gpu_name}")
# #         print(f"   VRAM: {total_mem:.1f} GB")
# #         print(f"   CUDA Version: {torch.version.cuda}")
        
# #         torch.backends.cuda.matmul.allow_tf32 = True
# #         torch.backends.cudnn.benchmark = True
# #         torch.backends.cudnn.enabled = True
# #         return device
# #     else:
# #         print("\n⚠️  WARNING: No GPU found! Running on CPU (will be very slow)")
# #         return 'cpu'

# # def print_gpu_memory():
# #     if torch.cuda.is_available():
# #         used = torch.cuda.memory_allocated() / 1e9
# #         cached = torch.cuda.memory_reserved() / 1e9
# #         print(f"   🖥️  GPU Memory: {used:.2f}GB used / {cached:.2f}GB cached")

# # # ============================================================================================
# # # ASYNC IMAGE SAVER
# # # ============================================================================================

# # class AsyncImageSaver:
# #     def __init__(self, num_workers=SAVE_WORKER_THREADS):
# #         self.executor = ThreadPoolExecutor(max_workers=num_workers)
# #         self.futures = []

# #     def save(self, path, image):
# #         future = self.executor.submit(cv2.imwrite, path, image)
# #         self.futures.append(future)

# #     def wait_all(self):
# #         for f in self.futures:
# #             f.result()
# #         self.futures.clear()

# #     def shutdown(self):
# #         self.wait_all()
# #         self.executor.shutdown(wait=True)

# # # ============================================================================================
# # # SRT PARSING & GPS UTILITIES
# # # ============================================================================================

# # def parse_srt(srt_path):
# #     srt_data = []
    
# #     if not os.path.exists(srt_path):
# #         print(f"❌ ERROR: SRT file not found: {srt_path}")
# #         return None
    
# #     with open(srt_path, 'r', encoding='utf-8') as f:
# #         content = f.read()
    
# #     blocks = re.split(r'\n\n+', content.strip())
    
# #     for block in blocks:
# #         if not block.strip():
# #             continue
        
# #         lines = block.strip().split('\n')
# #         if len(lines) < 3:
# #             continue
        
# #         try:
# #             frame_num = int(lines[0].strip())
            
# #             timecode_match = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
# #             if not timecode_match:
# #                 continue
# #             hours, minutes, seconds, milliseconds = map(int, timecode_match.groups())
# #             timestamp_ms = (hours * 3600 + minutes * 60 + seconds) * 1000 + milliseconds
            
# #             metadata_text = ' '.join(lines[2:])
            
# #             timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', metadata_text)
# #             if not timestamp_match:
# #                 continue
# #             timestamp_str = timestamp_match.group(1)
# #             absolute_timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
            
# #             lat_match = re.search(r'\[latitude:\s*([-\d.]+)\]', metadata_text)
# #             lon_match = re.search(r'\[longitude:\s*([-\d.]+)\]', metadata_text)
# #             alt_match = re.search(r'\[(?:abs_alt|altitude):\s*([-\d.]+)', metadata_text)
            
# #             if not (lat_match and lon_match):
# #                 continue
            
# #             latitude = float(lat_match.group(1))
# #             longitude = float(lon_match.group(1))
# #             altitude = float(alt_match.group(1)) if alt_match else 50.0
            
# #             srt_data.append({
# #                 'frame_number': frame_num,
# #                 'timestamp_ms': timestamp_ms,
# #                 'absolute_timestamp': absolute_timestamp,
# #                 'latitude': latitude,
# #                 'longitude': longitude,
# #                 'altitude': altitude
# #             })
        
# #         except Exception:
# #             continue
    
# #     if not srt_data:
# #         print("❌ ERROR: No valid GPS data extracted from SRT file!")
# #         return None
    
# #     print(f"✅ Parsed {len(srt_data)} SRT entries with GPS data")
# #     return srt_data

# # def haversine_distance(lat1, lon1, lat2, lon2):
# #     R = 6371000
# #     phi1, phi2 = math.radians(lat1), math.radians(lat2)
# #     delta_phi = math.radians(lat2 - lat1)
# #     delta_lambda = math.radians(lon2 - lon1)
    
# #     a = math.sin(delta_phi / 2) ** 2 + \
# #         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
# #     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
# #     return R * c

# # def calculate_cumulative_chainage(srt_data, starting_chainage_m=0):
# #     cumulative_distance = starting_chainage_m
    
# #     for i in range(len(srt_data)):
# #         if i == 0:
# #             srt_data[i]['cumulative_chainage_m'] = starting_chainage_m
# #         else:
# #             prev, curr = srt_data[i - 1], srt_data[i]
# #             dist = haversine_distance(prev['latitude'], prev['longitude'],
# #                                      curr['latitude'], curr['longitude'])
# #             cumulative_distance += dist
# #             srt_data[i]['cumulative_chainage_m'] = cumulative_distance
    
# #     total_distance_km = (cumulative_distance - starting_chainage_m) / 1000
# #     print(f"✅ Chainage: {starting_chainage_m:.1f}m → {cumulative_distance:.1f}m ({total_distance_km:.3f} km)")
    
# #     return cumulative_distance

# # def get_srt_data_for_frame(frame_index, srt_data):
# #     for entry in srt_data:
# #         if entry['frame_number'] == frame_index:
# #             return entry
    
# #     if srt_data:
# #         return min(srt_data, key=lambda x: abs(x['frame_number'] - frame_index))
    
# #     return None

# # # ============================================================================================
# # # IMPROVED SEGMENTATION TRACKER (BEST OF BOTH WORLDS)
# # # ============================================================================================

# # class ImprovedSegmentationTracker:
# #     """
# #     Combines:
# #     - 5-hit confirmation system (prevents false positives)
# #     - Mask IOU tracking (more accurate)
# #     - Sequential display IDs (clean numbering)
# #     """
    
# #     def __init__(self, iou_threshold=IOU_THRESHOLD, max_distance=MAX_DISTANCE, 
# #                  max_lost=MAX_LOST, confirmation_hits=CONFIRMATION_HITS):
# #         self.next_internal_id = 1
# #         self.next_display_id = 1
        
# #         self.active_tracks = {}      # Unconfirmed (< 5 hits)
# #         self.confirmed_tracks = {}   # Confirmed (≥ 5 hits)
        
# #         self.internal_to_display = {}
        
# #         self.iou_threshold = iou_threshold
# #         self.max_distance = max_distance
# #         self.max_lost = max_lost
# #         self.confirmation_hits = confirmation_hits

# #     def calculate_mask_iou(self, mask1, mask2):
# #         if mask1 is None or mask2 is None:
# #             return 0.0
# #         intersection = np.logical_and(mask1, mask2).sum()
# #         union = np.logical_or(mask1, mask2).sum()
# #         return intersection / union if union > 0 else 0.0

# #     def calculate_bbox_iou(self, box1, box2):
# #         x1 = max(box1[0], box2[0])
# #         y1 = max(box1[1], box2[1])
# #         x2 = min(box1[2], box2[2])
# #         y2 = min(box1[3], box2[3])
        
# #         intersection = max(0, x2 - x1) * max(0, y2 - y1)
# #         area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
# #         area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
# #         union = area1 + area2 - intersection
        
# #         return intersection / union if union > 0 else 0.0

# #     def centroid(self, bbox):
# #         return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

# #     def distance(self, point1, point2):
# #         return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# #     def update(self, detections, frame_index):
# #         """
# #         Update tracking with new detections
# #         Returns: (all_tracked_confirmed, newly_confirmed)
# #         """
        
# #         # Increment lost counter
# #         for track in list(self.active_tracks.values()) + list(self.confirmed_tracks.values()):
# #             track['lost'] += 1
        
# #         newly_confirmed = []
# #         all_tracked = []
# #         assigned = set()
        
# #         # Match detections to existing tracks
# #         for det in detections:
# #             best_match_id = None
# #             best_score = 0
            
# #             # Check confirmed first, then active
# #             for pool in [self.confirmed_tracks, self.active_tracks]:
# #                 for track_id, track in pool.items():
# #                     if track['class'] != det['class']:
# #                         continue
                    
# #                     # Mask IOU (preferred) or bbox IOU
# #                     if det.get('mask') is not None and track.get('mask') is not None:
# #                         iou = self.calculate_mask_iou(det['mask'], track['mask'])
# #                     else:
# #                         iou = self.calculate_bbox_iou(det['bbox'], track['bbox'])
                    
# #                     cent_dist = self.distance(
# #                         self.centroid(det['bbox']),
# #                         self.centroid(track['bbox'])
# #                     )
                    
# #                     if iou > self.iou_threshold or cent_dist < self.max_distance:
# #                         score = iou - (cent_dist / self.max_distance) * 0.3
# #                         if score > best_score:
# #                             best_score = score
# #                             best_match_id = track_id
            
# #             # Update existing or create new
# #             if best_match_id is not None:
# #                 if best_match_id in self.confirmed_tracks:
# #                     track = self.confirmed_tracks[best_match_id]
# #                 else:
# #                     track = self.active_tracks[best_match_id]
                
# #                 track['bbox'] = det['bbox']
# #                 track['mask'] = det.get('mask')
# #                 track['polygon'] = det.get('polygon')
# #                 track['confidence'] = det['confidence']
# #                 track['confidences'].append(det['confidence'])
# #                 track['hits'] += 1
# #                 track['lost'] = 0
# #                 track['last_frame'] = frame_index
                
# #                 assigned.add(best_match_id)
                
# #                 # ✅ CHECK FOR CONFIRMATION (5 hits)
# #                 if best_match_id in self.active_tracks and track['hits'] >= self.confirmation_hits:
# #                     self.confirmed_tracks[best_match_id] = track
# #                     del self.active_tracks[best_match_id]
                    
# #                     # Assign sequential display ID
# #                     self.internal_to_display[best_match_id] = self.next_display_id
# #                     track['display_id'] = self.next_display_id
# #                     self.next_display_id += 1
                    
# #                     newly_confirmed.append({
# #                         'internal_id': best_match_id,
# #                         'display_id': track['display_id'],
# #                         'bbox': track['bbox'],
# #                         'mask': track['mask'],
# #                         'polygon': track['polygon'],
# #                         'class': track['class'],
# #                         'confidence': track['confidence']
# #                     })
                
# #                 # Add to tracked (if confirmed)
# #                 if best_match_id in self.confirmed_tracks:
# #                     all_tracked.append({
# #                         'internal_id': best_match_id,
# #                         'display_id': track.get('display_id', -1),
# #                         'bbox': track['bbox'],
# #                         'mask': track['mask'],
# #                         'polygon': track['polygon'],
# #                         'class': track['class'],
# #                         'confidence': track['confidence']
# #                     })
            
# #             else:
# #                 # Create new active track
# #                 new_id = self.next_internal_id
# #                 self.next_internal_id += 1
                
# #                 self.active_tracks[new_id] = {
# #                     'bbox': det['bbox'],
# #                     'mask': det.get('mask'),
# #                     'polygon': det.get('polygon'),
# #                     'class': det['class'],
# #                     'confidence': det['confidence'],
# #                     'confidences': [det['confidence']],
# #                     'hits': 1,
# #                     'lost': 0,
# #                     'first_frame': frame_index,
# #                     'last_frame': frame_index
# #                 }
        
# #         # Remove tracks lost for too long
# #         for track_id in list(self.active_tracks.keys()):
# #             if self.active_tracks[track_id]['lost'] > self.max_lost:
# #                 del self.active_tracks[track_id]
        
# #         for track_id in list(self.confirmed_tracks.keys()):
# #             if self.confirmed_tracks[track_id]['lost'] > self.max_lost:
# #                 del self.confirmed_tracks[track_id]
        
# #         return all_tracked, newly_confirmed

# # # ============================================================================================
# # # UTILITY FUNCTIONS
# # # ============================================================================================

# # def calculate_iou(box1, box2):
# #     x1 = max(box1[0], box2[0])
# #     y1 = max(box1[1], box2[1])
# #     x2 = min(box1[2], box2[2])
# #     y2 = min(box1[3], box2[3])
    
# #     intersection = max(0, x2 - x1) * max(0, y2 - y1)
# #     area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
# #     area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
# #     union = area1 + area2 - intersection
    
# #     return intersection / union if union > 0 else 0.0

# # def resize_to_fixed_size(image, target_size):
# #     """Resize with padding for report images"""
# #     target_w, target_h = target_size
# #     h, w = image.shape[:2]
    
# #     scale = min(target_w / w, target_h / h)
# #     new_w = int(w * scale)
# #     new_h = int(h * scale)
    
# #     resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
# #     canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
# #     y_offset = (target_h - new_h) // 2
# #     x_offset = (target_w - new_w) // 2
# #     canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
# #     return canvas

# # # ============================================================================================
# # # WORD REPORT GENERATION
# # # ============================================================================================

# # def generate_word_report(confirmed_detections, output_dir):
# #     """Generate Word document report with images"""
# #     doc = Document()
# #     doc.add_heading('Road Defects Detection Report - RF-DETR', 0)
    
# #     # Sort by display_id
# #     sorted_detections = sorted(confirmed_detections, key=lambda x: x['display_id'])
    
# #     # Group by class
# #     by_class = {}
# #     for det in sorted_detections:
# #         cls = det['defect_type']
# #         if cls not in by_class:
# #             by_class[cls] = []
# #         by_class[cls].append(det)
    
# #     # Summary section
# #     doc.add_heading('Summary', 1)
# #     summary_table = doc.add_table(rows=1, cols=2)
# #     summary_table.style = 'Light Grid Accent 1'
# #     hdr_cells = summary_table.rows[0].cells
# #     hdr_cells[0].text = 'Defect Type'
# #     hdr_cells[1].text = 'Count'
    
# #     for cls in sorted(by_class.keys()):
# #         row = summary_table.add_row().cells
# #         row[0].text = cls
# #         row[1].text = str(len(by_class[cls]))
    
# #     doc.add_paragraph('')
    
# #     # Detailed sections
# #     for cls in sorted(by_class.keys()):
# #         detections = by_class[cls]
        
# #         doc.add_heading(f'{cls.upper()} DETECTIONS', 1)
# #         doc.add_paragraph(f'Total: {len(detections)}')
# #         doc.add_paragraph('')
        
# #         table = doc.add_table(rows=1, cols=4)
# #         table.style = 'Light Grid Accent 1'
        
# #         hdr = table.rows[0].cells
# #         hdr[0].text = 'ID'
# #         hdr[1].text = 'Chainage (m)'
# #         hdr[2].text = 'Crop'
# #         hdr[3].text = 'Frame'
        
# #         for det in detections:
# #             row = table.add_row().cells
            
# #             row[0].text = f"{det['display_id']}-{cls}"
# #             row[1].text = f"{det['chainage_avg_m']:.1f}"
            
# #             crop_path = det['images']['crop']
# #             if os.path.exists(crop_path):
# #                 try:
# #                     p = row[2].paragraphs[0]
# #                     r = p.add_run()
# #                     r.add_picture(crop_path, width=Inches(1.5))
# #                     p.alignment = WD_ALIGN_PARAGRAPH.CENTER
# #                 except:
# #                     row[2].text = f"ID {det['display_id']}"
            
# #             frame_path = det['images']['frame']
# #             if os.path.exists(frame_path):
# #                 try:
# #                     p = row[3].paragraphs[0]
# #                     r = p.add_run()
# #                     r.add_picture(frame_path, width=Inches(2.5))
# #                     p.alignment = WD_ALIGN_PARAGRAPH.CENTER
# #                 except:
# #                     row[3].text = f"Frame {det['display_id']}"
        
# #         doc.add_page_break()
    
# #     report_path = os.path.join(output_dir, "Road_Defects_Report.docx")
# #     doc.save(report_path)
# #     print(f"\n📄 Word Report: {report_path}")

# # # ============================================================================================
# # # VISUALIZATION
# # # ============================================================================================

# # def draw_legend(frame, width, height, active_classes):
# #     scale_factor = max(0.6, min(width / 1920.0, 1.2))
# #     legend_x = int(0.02 * width)
# #     legend_y = int(0.04 * height)
# #     text_scale = 0.6 * scale_factor
# #     text_thickness = max(1, int(1.5 * scale_factor))
# #     line_height = int(22 * scale_factor)
# #     line_len = int(35 * scale_factor)
# #     legend_width = int(300 * scale_factor)
    
# #     cv2.rectangle(frame,
# #                  (legend_x - 12, legend_y - 12),
# #                  (legend_x + legend_width, legend_y + line_height * len(active_classes) + 12),
# #                  (0, 0, 0), max(2, int(2 * scale_factor)))
    
# #     for idx, (cls_name, color_hex) in enumerate(active_classes.items()):
# #         y = legend_y + idx * line_height
        
# #         color_hex = color_hex.lstrip('#')
# #         r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
# #         color_bgr = (b, g, r)
        
# #         cv2.line(frame, (legend_x, y + 10), (legend_x + line_len, y + 10), color_bgr, 2)
# #         cv2.putText(frame, cls_name, (legend_x + line_len + 12, y + 12),
# #                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)

# # # ============================================================================================
# # # MAIN PROCESSING FUNCTION
# # # ============================================================================================

# # def process_video_with_rfdetr(video_path, srt_path, model, output_dir, device, starting_chainage_m=0):
    
# #     video_name = os.path.basename(video_path)
# #     srt_name = os.path.basename(srt_path)
# #     video_basename = os.path.splitext(video_name)[0]
    
# #     print(f"\n{'='*80}")
# #     print(f"PROCESSING: {video_name}")
# #     print(f"{'='*80}")
    
# #     # Parse SRT
# #     srt_data = parse_srt(srt_path)
# #     if srt_data is None:
# #         return None, starting_chainage_m
    
# #     ending_chainage = calculate_cumulative_chainage(srt_data, starting_chainage_m)
    
# #     # Open video
# #     cap = cv2.VideoCapture(video_path)
# #     fps = cap.get(cv2.CAP_PROP_FPS)
# #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
# #     print(f"\n🎥 Video Info:")
# #     print(f"   Original: {orig_width}×{orig_height} @ {fps:.1f}fps")
# #     print(f"   Processing: {OUT_WIDTH}×{OUT_HEIGHT}")
# #     print(f"   Total frames: {total_frames}")
# #     print(f"   Processing every {PROCESS_EVERY_N_FRAMES} frames")
# #     print_gpu_memory()
    
# #     # Setup output
# #     video_output_dir = os.path.join(output_dir, video_basename)
# #     defects_dir = os.path.join(video_output_dir, "defect_images")
# #     frames_dir = os.path.join(defects_dir, "frames")
# #     os.makedirs(frames_dir, exist_ok=True)
    
# #     for cls in CLASS_NAMES:
# #         if cls not in IGNORED_CLASSES:
# #             os.makedirs(os.path.join(defects_dir, cls), exist_ok=True)
    
# #     output_video_path = os.path.join(video_output_dir, f"{video_basename}_output.mp4")
    
# #     # Initialize
# #     tracker = ImprovedSegmentationTracker()
# #     image_saver = AsyncImageSaver()
    
# #     confirmed_detections = {}
# #     all_detections = []
    
# #     # Store all frames for second pass
# #     all_frames = []
# #     frame_detections_map = {}  # frame_idx → list of tracked objects with display_ids
    
# #     # Supervision annotators
# #     text_scale = sv.calculate_optimal_text_scale(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
# #     thickness = sv.calculate_optimal_line_thickness(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
# #     color_palette = sv.ColorPalette.from_hex(HEX_COLORS)
    
# #     mask_annotator = sv.MaskAnnotator(color=color_palette, opacity=MASK_OPACITY)
# #     bbox_annotator = sv.BoxAnnotator(color=color_palette, thickness=thickness)
# #     label_annotator = sv.LabelAnnotator(
# #         color=color_palette,
# #         text_color=sv.Color.WHITE,
# #         text_scale=text_scale,
# #         text_thickness=max(1, thickness - 1)
# #     )
    
# #     active_classes = {}
    
# #     print(f"\n{'='*80}")
# #     print("PASS 1: TRACKING & DETECTION")
# #     print(f"{'='*80}")
    
# #     import time
# #     start_time = time.time()
    
# #     frame_count = 0
# #     frame_batch = []
# #     frame_indices = []
    
# #     # PASS 1: Track and detect
# #     while True:
# #         ret, frame = cap.read()
        
# #         if ret:
# #             frame_resized = cv2.resize(frame, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
# #             all_frames.append(frame_resized)
            
# #             if frame_count % PROCESS_EVERY_N_FRAMES == 0:
# #                 frame_batch.append(frame_resized)
# #                 frame_indices.append(frame_count)
        
# #         frame_count += 1
        
# #         if len(frame_batch) == BATCH_SIZE or (not ret and len(frame_batch) > 0):
            
# #             # RF-DETR inference
# #             with torch.no_grad():
# #                 batch_detections = model.predict(frame_batch, threshold=GLOBAL_THRESHOLD)
            
# #             for batch_idx, detections in enumerate(batch_detections):
# #                 frame_idx = frame_indices[batch_idx]
                
# #                 srt_entry = get_srt_data_for_frame(frame_idx + 1, srt_data)
# #                 if srt_entry is None:
# #                     continue
                
# #                 chainage_m = srt_entry['cumulative_chainage_m']
                
# #                 # NMS
# #                 detections = detections.with_nms(threshold=NMS_THRESHOLD)
                
# #                 # Filter detections
# #                 frame_detections = []
                
# #                 if len(detections) > 0:
# #                     for idx in range(len(detections)):
# #                         class_id = detections.class_id[idx]
# #                         confidence = detections.confidence[idx]
                        
# #                         if class_id < 0 or class_id >= len(CLASS_NAMES):
# #                             continue
                        
# #                         class_name = CLASS_NAMES[class_id]
                        
# #                         if class_name in IGNORED_CLASSES:
# #                             continue
                        
# #                         threshold_to_use = CLASS_THRESHOLDS.get(class_name, GLOBAL_THRESHOLD)
# #                         if confidence < threshold_to_use:
# #                             continue
                        
# #                         bbox = detections.xyxy[idx]
                        
# #                         mask = None
# #                         polygon = None
# #                         if detections.mask is not None:
# #                             mask = detections.mask[idx].astype(np.uint8)
# #                             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #                             if contours:
# #                                 polygon = contours[0].squeeze().tolist()
# #                                 if isinstance(polygon[0], (int, float)):
# #                                     polygon = [polygon]
                        
# #                         frame_detections.append({
# #                             'bbox': bbox,
# #                             'mask': mask,
# #                             'polygon': polygon,
# #                             'class': class_name,
# #                             'confidence': float(confidence)
# #                         })
                
# #                 # ✅ POTHOLE-PATCHING OVERLAP REMOVAL
# #                 pothole_indices = []
# #                 patching_indices = []
# #                 for i, det in enumerate(frame_detections):
# #                     if det['class'] == 'pothole':
# #                         pothole_indices.append(i)
# #                     elif det['class'] == 'Patch':
# #                         patching_indices.append(i)
                
# #                 remove_indices = set()
# #                 for pot_i in pothole_indices:
# #                     for pat_i in patching_indices:
# #                         iou = calculate_iou(frame_detections[pot_i]['bbox'], 
# #                                           frame_detections[pat_i]['bbox'])
# #                         if iou >= POTHOLE_PATCHING_IOU_THRESHOLD:
# #                             remove_indices.add(pot_i)
# #                             break
                
# #                 frame_detections = [det for i, det in enumerate(frame_detections) 
# #                                    if i not in remove_indices]
                
# #                 # ✅ TRACKING
# #                 tracked_objects, newly_confirmed = tracker.update(frame_detections, frame_idx)
                
# #                 # Update active classes
# #                 for tr in tracked_objects:
# #                     cls_name = tr['class']
# #                     cls_idx = CLASS_NAMES.index(cls_name)
# #                     if cls_name not in active_classes:
# #                         active_classes[cls_name] = HEX_COLORS[cls_idx]
                
# #                 # Store for second pass
# #                 frame_detections_map[frame_idx] = tracked_objects
                
# #                 # ✅ SAVE NEWLY CONFIRMED OBJECTS
# #                 for confirmed in newly_confirmed:
# #                     internal_id = confirmed['internal_id']
# #                     display_id = confirmed['display_id']
# #                     cls_name = confirmed['class']
                    
# #                     # Save images (fixed size with padding)
# #                     x1, y1, x2, y2 = map(int, confirmed['bbox'])
# #                     x1_crop = max(0, x1 - 30)
# #                     y1_crop = max(0, y1 - 30)
# #                     x2_crop = min(OUT_WIDTH, x2 + 30)
# #                     y2_crop = min(OUT_HEIGHT, y2 + 30)
                    
# #                     cropped = frame_batch[batch_idx][y1_crop:y2_crop, x1_crop:x2_crop]
# #                     cropped_resized = resize_to_fixed_size(cropped, CROP_SIZE)
                    
# #                     crop_path = os.path.join(defects_dir, cls_name, f"{display_id}-{cls_name}.jpg")
# #                     image_saver.save(crop_path, cropped_resized)
                    
# #                     frame_resized = resize_to_fixed_size(frame_batch[batch_idx], FRAME_SIZE)
# #                     frame_path = os.path.join(frames_dir, f"{display_id}-{cls_name}_frame.jpg")
# #                     image_saver.save(frame_path, frame_resized)
                    
# #                     # Store metadata
# #                     track = tracker.confirmed_tracks[internal_id]
                    
# #                     confirmed_detections[internal_id] = {
# #                         'display_id': display_id,
# #                         'internal_id': internal_id,
# #                         'defect_type': cls_name,
# #                         'video_name': video_name,
# #                         'srt_name': srt_name,
# #                         'frame_start': track['first_frame'],
# #                         'frame_end': track['last_frame'],
# #                         'timestamp_start': srt_entry['absolute_timestamp'].isoformat(),
# #                         'timestamp_end': srt_entry['absolute_timestamp'].isoformat(),
# #                         'chainage_start_m': chainage_m,
# #                         'chainage_end_m': chainage_m,
# #                         'chainage_avg_m': chainage_m,
# #                         'confidence_avg': float(np.mean(track['confidences'])),
# #                         'gps': {
# #                             'latitude': srt_entry['latitude'],
# #                             'longitude': srt_entry['longitude']
# #                         },
# #                         'polygon': confirmed['polygon'],
# #                         'images': {
# #                             'crop': crop_path,
# #                             'frame': frame_path
# #                         }
# #                     }
            
# #             frame_batch = []
# #             frame_indices = []
            
# #             if len(confirmed_detections) > 0 and len(confirmed_detections) % 10 == 0:
# #                 elapsed = time.time() - start_time
# #                 fps_actual = frame_count / elapsed
# #                 print(f"   Frames: {frame_count}/{total_frames} | "
# #                       f"Speed: {fps_actual:.1f} fps | "
# #                       f"Confirmed: {len(confirmed_detections)}")
        
# #         if not ret:
# #             break
    
# #     cap.release()
    
# #     # Flush image saves
# #     print("\n💾 Saving images...")
# #     image_saver.shutdown()
    
# #     # Update final metadata
# #     for internal_id, det in confirmed_detections.items():
# #         if internal_id in tracker.confirmed_tracks:
# #             track = tracker.confirmed_tracks[internal_id]
            
# #             last_srt = get_srt_data_for_frame(track['last_frame'] + 1, srt_data)
# #             if last_srt:
# #                 det['frame_end'] = track['last_frame']
# #                 det['timestamp_end'] = last_srt['absolute_timestamp'].isoformat()
# #                 det['chainage_end_m'] = last_srt['cumulative_chainage_m']
# #                 det['chainage_avg_m'] = (det['chainage_start_m'] + det['chainage_end_m']) / 2
    
# #     # Convert to list
# #     all_detections = list(confirmed_detections.values())
    
# #     print(f"\n{'='*80}")
# #     print("PASS 2: VIDEO ANNOTATION")
# #     print(f"{'='*80}")
    
# #     # Video writer
# #     output_fps = fps / PROCESS_EVERY_N_FRAMES
# #     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
# #                          output_fps, (OUT_WIDTH, OUT_HEIGHT))
    
# #     # Annotate all frames with confirmed IDs only
# #     for frame_idx, frame in enumerate(all_frames):
# #         if frame_idx in frame_detections_map:
# #             tracked = frame_detections_map[frame_idx]
            
# #             if len(tracked) > 0:
# #                 xyxy = np.array([t['bbox'] for t in tracked])
# #                 class_ids = np.array([CLASS_NAMES.index(t['class']) for t in tracked])
# #                 confidences = np.array([t['confidence'] for t in tracked])
                
# #                 masks = None
# #                 if tracked[0].get('mask') is not None:
# #                     masks = np.array([t['mask'] for t in tracked])
                
# #                 sv_detections = sv.Detections(
# #                     xyxy=xyxy,
# #                     class_id=class_ids,
# #                     confidence=confidences,
# #                     mask=masks
# #                 )
                
# #                 labels = [
# #                     f"{t['display_id']}-{t['class']} {t['confidence']:.2f}"
# #                     for t in tracked
# #                 ]
                
# #                 annotated = frame.copy()
                
# #                 if masks is not None:
# #                     annotated = mask_annotator.annotate(scene=annotated, detections=sv_detections)
                
# #                 annotated = bbox_annotator.annotate(scene=annotated, detections=sv_detections)
# #                 annotated = label_annotator.annotate(scene=annotated, detections=sv_detections, labels=labels)
                
# #                 if active_classes:
# #                     draw_legend(annotated, OUT_WIDTH, OUT_HEIGHT, active_classes)
                
# #                 frame = annotated
        
# #         out.write(frame)
        
# #         if (frame_idx + 1) % 300 == 0:
# #             print(f"   Writing: {frame_idx + 1}/{len(all_frames)} frames")
    
# #     out.release()
    
# #     if torch.cuda.is_available():
# #         torch.cuda.empty_cache()
    
# #     # Save JSON
# #     detection_data = {
# #         'video_name': video_name,
# #         'srt_name': srt_name,
# #         'processing_date': datetime.now().isoformat(),
# #         'model': 'RF-DETR-Seg-Medium',
# #         'total_frames': total_frames,
# #         'starting_chainage_m': starting_chainage_m,
# #         'ending_chainage_m': ending_chainage,
# #         'total_distance_km': (ending_chainage - starting_chainage_m) / 1000,
# #         'total_detections': len(all_detections),
# #         'detections': all_detections,
# #         'summary': {}
# #     }
    
# #     for det in all_detections:
# #         dtype = det['defect_type']
# #         if dtype not in detection_data['summary']:
# #             detection_data['summary'][dtype] = {'count': 0, 'avg_confidence': []}
# #         detection_data['summary'][dtype]['count'] += 1
# #         detection_data['summary'][dtype]['avg_confidence'].append(det['confidence_avg'])
    
# #     for dtype in detection_data['summary']:
# #         confs = detection_data['summary'][dtype]['avg_confidence']
# #         detection_data['summary'][dtype]['avg_confidence'] = float(np.mean(confs))
    
# #     json_path = os.path.join(video_output_dir, f"{video_basename}_detections.json")
# #     with open(json_path, 'w') as f:
# #         json.dump(detection_data, f, indent=2)
    
# #     # Generate Word report
# #     print("\n📄 Generating Word report...")
# #     generate_word_report(all_detections, video_output_dir)
    
# #     elapsed_total = time.time() - start_time
# #     print(f"\n{'='*80}")
# #     print(f"✅ PROCESSING COMPLETE!")
# #     print(f"{'='*80}")
# #     print(f"   Time: {elapsed_total/60:.1f} min")
# #     print(f"   Detections: {len(all_detections)}")
# #     print(f"   📹 Video: {output_video_path}")
# #     print(f"   📊 JSON: {json_path}")
# #     print(f"   🖼️  Images: {defects_dir}")
    
# #     print(f"\n   Defect Summary:")
# #     for dtype, stats in detection_data['summary'].items():
# #         print(f"      {dtype}: {stats['count']} (avg conf: {stats['avg_confidence']:.2f})")
    
# #     return json_path, ending_chainage

# # # ============================================================================================
# # # MAIN
# # # ============================================================================================

# # def main():
# #     print("\n" + "="*80)
# #     print("RF-DETR ROAD DEFECT DETECTION - FINAL VERSION")
# #     print("="*80)
    
# #     device = check_gpu()
    
# #     # Verify paths
# #     if not os.path.exists(VIDEO_PATH):
# #         print(f"❌ ERROR: Video not found: {VIDEO_PATH}")
# #         return
    
# #     if not os.path.exists(SRT_PATH):
# #         print(f"❌ ERROR: SRT file not found: {SRT_PATH}")
# #         return
    
# #     if not os.path.exists(CHECKPOINT_PATH):
# #         print(f"❌ ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
# #         return
    
# #     # Load model
# #     print(f"\n{'='*80}")
# #     print("LOADING RF-DETR MODEL")
# #     print(f"{'='*80}")
    
# #     model = RFDETRSegMedium(
# #         pretrain_weights=CHECKPOINT_PATH,
# #         num_queries=NUM_QUERIES,
# #         image_size=IMAGE_SIZE,
# #         max_image_size=IMAGE_SIZE,
# #     )
    
# #     print("✅ RF-DETR model loaded")
# #     print_gpu_memory()
    
# #     # Warmup
# #     print("\n🔥 Warming up GPU...")
# #     dummy = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
# #     with torch.no_grad():
# #         _ = model.predict([dummy], threshold=0.5)
# #     if device != 'cpu':
# #         torch.cuda.synchronize()
# #     print("✅ GPU warmed up")
    
# #     # Process
# #     json_path, _ = process_video_with_rfdetr(
# #         VIDEO_PATH, SRT_PATH, model,
# #         OUTPUT_BASE_DIR, device,
# #         starting_chainage_m=0
# #     )
    
# #     print(f"\n{'='*80}")
# #     print("✅ ALL DONE!")
# #     print(f"{'='*80}")
# #     print(f"Output: {OUTPUT_BASE_DIR}")
# #     print(f"{'='*80}\n")

# # if __name__ == "__main__":
# #     main()



# # ============================================================================================
# # RF-DETR ROAD DEFECT DETECTION - OPTIMIZED VERSION
# # ============================================================================================
# # Optimizations over original:
# #   ✅ Rolling frame buffer (replaces two-pass) → constant ~200MB RAM vs 4-5GB
# #   ✅ FP16 inference → ~2x GPU speedup
# #   ✅ Threaded frame prefetching → GPU never waits for CPU
# #   ✅ Streaming video writer → frames written as they exit buffer
# #   ✅ Periodic GPU cache clearing → no OOM crashes
# #   ✅ Mask stored as polygon points only → saves RAM
# #   ✅ Single pass → entire second loop eliminated
# # ============================================================================================

# import os
# import cv2
# import math
# import numpy as np
# import re
# import torch
# import json
# import threading
# import queue
# from datetime import datetime
# from rfdetr import RFDETRSegMedium
# import supervision as sv
# from concurrent.futures import ThreadPoolExecutor
# from pathlib import Path
# from docx import Document
# from docx.shared import Inches
# from docx.enum.text import WD_ALIGN_PARAGRAPH
# from collections import deque

# # ============================================================================================
# # CONFIGURATION - UPDATE THESE PATHS
# # ============================================================================================
# VIDEO_PATH      = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.MP4"
# SRT_PATH        = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.SRT"
# # VIDEO_PATH      = r"/run/user/1000/gvfs/google-drive:host=gmail.com,user=novametrics.processing/GVfsSharedWithMe/1HGz8EK5aTGpkk5cscqrNbeHumkQw3AiH/1Nid1NUkO1DcfTVu5oANAC6VfLIm4GwBC/1BopTWABa4V6I4SraRaSbY1yU3mVzlYSx/DJI_20260110101923_0001_D.MP4"
# # SRT_PATH        = r"/run/user/1000/gvfs/google-drive:host=gmail.com,user=novametrics.processing/GVfsSharedWithMe/1HGz8EK5aTGpkk5cscqrNbeHumkQw3AiH/1Nid1NUkO1DcfTVu5oANAC6VfLIm4GwBC/1BopTWABa4V6I4SraRaSbY1yU3mVzlYSx/DJI_20260110101923_0001_D.SRT"
# CHECKPOINT_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
# COCO_JSON_PATH  = r"/media/user/New Volume/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"
# OUTPUT_BASE_DIR = r"/media/user/New Volume/Sakshi/chattishgarh/output_rfdetr_gps_final"

# os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# # ============================================================================================
# # MODEL SETTINGS
# # ============================================================================================

# NUM_QUERIES  = 200
# IMAGE_SIZE   = 432
# OUT_WIDTH    = 1280
# OUT_HEIGHT   = 720

# # ============================================================================================
# # PROCESSING SETTINGS
# # ============================================================================================

# PROCESS_EVERY_N_FRAMES = 1
# BATCH_SIZE             = 4
# MASK_OPACITY           = 0.15

# # ─── Class-Specific Confidence Thresholds ───
# CLASS_THRESHOLDS = {
#     "Patch":                      0.50,
#     "pothole":                    0.35,
#     "Cracking":                   0.30,
#     "Ravelling":                  0.50,
#     "Edge_breaking":              0.35,
#     "Edge_drop":                  0.35,
#     "MBCB_defect":                0.30,
#     "MBCB_missing":               0.30,
#     "Corrugations_and_shoving":   0.30,
#     "Scaling":                    0.30,
#     "Wear":                       0.40,
#     "honeycomb":                  0.30,
#     "depression":                 0.30,
#     "Embankment slope":           0.30,
#     "strip_seal_expansion_join":  0.30,
#     "tyre_marks":                 0.35,
#     "vegetation_on_road":         0.30,
#     "km_stone":                   0.40,
#     "cattle":                     0.45,
# }

# GLOBAL_THRESHOLD   = 0.30
# NMS_THRESHOLD      = 0.65

# # ─── Ignored Classes ───
# IGNORED_CLASSES = {
#     "white_mark", "water_mark", "doubt", "bump",
#     "guard_post", "overhead_sign_board", "sign_board", "kerb_damage"
# }

# # ─── Tracking Parameters ───
# IOU_THRESHOLD   = 0.3
# MAX_DISTANCE    = 50
# MAX_LOST        = 30
# CONFIRMATION_HITS = 3

# # ─── Rolling Buffer Size ───
# # Must be >= MAX_LOST so we can retroactively annotate buffered frames
# BUFFER_SIZE = MAX_LOST + CONFIRMATION_HITS + 2   # ~37 frames, ~200MB max

# # ─── Special Logic ───
# POTHOLE_PATCHING_IOU_THRESHOLD = 0.90

# # ─── Threading ───
# PREFETCH_QUEUE_SIZE  = 32   # frames prefetched ahead
# SAVE_WORKER_THREADS  = 4

# # ─── Report Image Sizes ───
# CROP_SIZE  = (400, 400)
# FRAME_SIZE = (800, 600)

# # ─── GPU Cache Clear Interval ───
# GPU_CLEAR_EVERY = 200   # clear every N processed frames

# # ============================================================================================
# # LOAD CLASS NAMES
# # ============================================================================================

# print("=" * 80)
# print("LOADING CONFIGURATION")
# print("=" * 80)

# with open(COCO_JSON_PATH, "r") as f:
#     coco_data = json.load(f)

# categories  = sorted(coco_data["categories"], key=lambda x: x["id"])
# CLASS_NAMES = [cat["name"] for cat in categories]

# print(f"\n✅ Loaded {len(CLASS_NAMES)} classes from COCO JSON:")
# for i, name in enumerate(CLASS_NAMES):
#     ignored_marker = " [IGNORED]" if name in IGNORED_CLASSES else ""
#     threshold = CLASS_THRESHOLDS.get(name, GLOBAL_THRESHOLD)
#     print(f"   {i:2d}: {name:30s} (conf ≥ {threshold:.2f}){ignored_marker}")

# # ============================================================================================
# # COLOR PALETTE
# # ============================================================================================

# HEX_COLORS = [
#     "#FF0000", "#00A5FF", "#008CFF", "#FF00FF", "#8000FF",
#     "#00FF80", "#FF0080", "#B400B4", "#0000FF", "#02D32E",
#     "#FF8000", "#00FFFF", "#FFFF00", "#FF4040", "#4040FF",
#     "#A0A0A0", "#006400", "#D2691E", "#008080", "#FFD700",
#     "#873CBE", "#00C8C8", "#FF1493", "#32CD32", "#FF69B4",
#     "#696969", "#228B22", "#ADD8E6", "#F5F5F5",
# ]
# while len(HEX_COLORS) < len(CLASS_NAMES):
#     HEX_COLORS.append("#FFFFFF")

# # ============================================================================================
# # GPU UTILITIES
# # ============================================================================================

# def check_gpu():
#     if torch.cuda.is_available():
#         device    = 'cuda:0'
#         gpu_name  = torch.cuda.get_device_name(0)
#         total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
#         print(f"\n✅ GPU: {gpu_name}")
#         print(f"   VRAM: {total_mem:.1f} GB")
#         print(f"   CUDA: {torch.version.cuda}")
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.benchmark        = True
#         torch.backends.cudnn.enabled          = True
#         return device
#     else:
#         print("\n⚠️  No GPU found — running on CPU (slow)")
#         return 'cpu'

# def print_gpu_memory():
#     if torch.cuda.is_available():
#         used   = torch.cuda.memory_allocated() / 1e9
#         cached = torch.cuda.memory_reserved()   / 1e9
#         print(f"   🖥️  GPU Memory: {used:.2f}GB used / {cached:.2f}GB cached")

# # ============================================================================================
# # ASYNC IMAGE SAVER
# # ============================================================================================

# class AsyncImageSaver:
#     def __init__(self, num_workers=SAVE_WORKER_THREADS):
#         self.executor = ThreadPoolExecutor(max_workers=num_workers)
#         self.futures  = []

#     def save(self, path, image):
#         future = self.executor.submit(cv2.imwrite, path, image)
#         self.futures.append(future)

#     def wait_all(self):
#         for f in self.futures:
#             f.result()
#         self.futures.clear()

#     def shutdown(self):
#         self.wait_all()
#         self.executor.shutdown(wait=True)

# # ============================================================================================
# # THREADED FRAME PREFETCHER
# # ============================================================================================

# class FramePrefetcher:
#     """
#     Reads frames from disk on a background thread so GPU never waits for I/O.
#     Puts (frame_index, resized_frame) into a queue.
#     Sentinel value None signals end of video.
#     """
#     def __init__(self, video_path, out_w, out_h, queue_size=PREFETCH_QUEUE_SIZE):
#         self.cap     = cv2.VideoCapture(video_path)
#         self.out_w   = out_w
#         self.out_h   = out_h
#         self.q       = queue.Queue(maxsize=queue_size)
#         self.thread  = threading.Thread(target=self._reader, daemon=True)
#         self.thread.start()

#     def _reader(self):
#         idx = 0
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.q.put(None)   # sentinel
#                 break
#             resized = cv2.resize(frame, (self.out_w, self.out_h),
#                                  interpolation=cv2.INTER_LINEAR)
#             self.q.put((idx, resized))
#             idx += 1
#         self.cap.release()

#     def get(self):
#         return self.q.get()

#     @property
#     def fps(self):
#         return self.cap.get(cv2.CAP_PROP_FPS) if self.cap.isOpened() else 30.0

#     @property
#     def total_frames(self):
#         return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.isOpened() else 0


# def get_video_props(video_path):
#     """Read video properties before starting prefetcher."""
#     cap = cv2.VideoCapture(video_path)
#     fps          = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     cap.release()
#     return fps, total_frames, orig_w, orig_h

# # ============================================================================================
# # SRT PARSING & GPS UTILITIES
# # ============================================================================================

# def parse_srt(srt_path):
#     srt_data = []
#     if not os.path.exists(srt_path):
#         print(f"❌ SRT file not found: {srt_path}")
#         return None

#     with open(srt_path, 'r', encoding='utf-8') as f:
#         content = f.read()

#     blocks = re.split(r'\n\n+', content.strip())
#     for block in blocks:
#         if not block.strip():
#             continue
#         lines = block.strip().split('\n')
#         if len(lines) < 3:
#             continue
#         try:
#             frame_num       = int(lines[0].strip())
#             tc_match        = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
#             if not tc_match:
#                 continue
#             h, m, s, ms     = map(int, tc_match.groups())
#             timestamp_ms    = (h * 3600 + m * 60 + s) * 1000 + ms
#             meta            = ' '.join(lines[2:])

#             ts_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', meta)
#             if not ts_match:
#                 continue
#             abs_ts = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S.%f')

#             lat_m = re.search(r'\[latitude:\s*([-\d.]+)\]', meta)
#             lon_m = re.search(r'\[longitude:\s*([-\d.]+)\]', meta)
#             alt_m = re.search(r'\[(?:abs_alt|altitude):\s*([-\d.]+)', meta)

#             if not (lat_m and lon_m):
#                 continue

#             srt_data.append({
#                 'frame_number':      frame_num,
#                 'timestamp_ms':      timestamp_ms,
#                 'absolute_timestamp': abs_ts,
#                 'latitude':          float(lat_m.group(1)),
#                 'longitude':         float(lon_m.group(1)),
#                 'altitude':          float(alt_m.group(1)) if alt_m else 50.0
#             })
#         except Exception:
#             continue

#     if not srt_data:
#         print("❌ No GPS data in SRT!")
#         return None

#     print(f"✅ Parsed {len(srt_data)} SRT entries")
#     return srt_data


# def haversine_distance(lat1, lon1, lat2, lon2):
#     R   = 6371000
#     p1, p2 = math.radians(lat1), math.radians(lat2)
#     dp  = math.radians(lat2 - lat1)
#     dl  = math.radians(lon2 - lon1)
#     a   = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
#     return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# def calculate_cumulative_chainage(srt_data, start_m=0):
#     dist = start_m
#     for i, entry in enumerate(srt_data):
#         if i == 0:
#             entry['cumulative_chainage_m'] = start_m
#         else:
#             prev = srt_data[i-1]
#             dist += haversine_distance(prev['latitude'], prev['longitude'],
#                                        entry['latitude'], entry['longitude'])
#             entry['cumulative_chainage_m'] = dist
#     total_km = (dist - start_m) / 1000
#     print(f"✅ Chainage: {start_m:.1f}m → {dist:.1f}m ({total_km:.3f} km)")
#     return dist


# def get_srt_for_frame(frame_idx, srt_data):
#     for e in srt_data:
#         if e['frame_number'] == frame_idx:
#             return e
#     if srt_data:
#         return min(srt_data, key=lambda x: abs(x['frame_number'] - frame_idx))
#     return None

# # ============================================================================================
# # TRACKER (unchanged logic, same 5-hit confirmation)
# # ============================================================================================

# class ImprovedSegmentationTracker:
#     def __init__(self):
#         self.next_internal_id = 1
#         self.next_display_id  = 1
#         self.active_tracks    = {}
#         self.confirmed_tracks = {}
#         self.internal_to_display = {}

#     def _mask_iou(self, m1, m2):
#         if m1 is None or m2 is None:
#             return 0.0
#         inter = np.logical_and(m1, m2).sum()
#         union = np.logical_or(m1,  m2).sum()
#         return inter / union if union > 0 else 0.0

#     def _bbox_iou(self, b1, b2):
#         x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
#         x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
#         inter = max(0, x2-x1) * max(0, y2-y1)
#         a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
#         a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
#         union = a1 + a2 - inter
#         return inter / union if union > 0 else 0.0

#     def _centroid(self, bbox):
#         return ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)

#     def _dist(self, p1, p2):
#         return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

#     def update(self, detections, frame_index):
#         for t in list(self.active_tracks.values()) + list(self.confirmed_tracks.values()):
#             t['lost'] += 1

#         newly_confirmed = []
#         all_tracked     = []

#         for det in detections:
#             best_id    = None
#             best_score = 0

#             for pool in [self.confirmed_tracks, self.active_tracks]:
#                 for tid, track in pool.items():
#                     if track['class'] != det['class']:
#                         continue
#                     iou = (self._mask_iou(det.get('mask'), track.get('mask'))
#                            if det.get('mask') is not None and track.get('mask') is not None
#                            else self._bbox_iou(det['bbox'], track['bbox']))
#                     cd  = self._dist(self._centroid(det['bbox']),
#                                      self._centroid(track['bbox']))
#                     if iou > IOU_THRESHOLD or cd < MAX_DISTANCE:
#                         score = iou - (cd / MAX_DISTANCE) * 0.3
#                         if score > best_score:
#                             best_score = score
#                             best_id    = tid

#             if best_id is not None:
#                 pool  = (self.confirmed_tracks if best_id in self.confirmed_tracks
#                          else self.active_tracks)
#                 track = pool[best_id]
#                 track.update({
#                     'bbox':       det['bbox'],
#                     'mask':       det.get('mask'),
#                     'polygon':    det.get('polygon'),
#                     'confidence': det['confidence'],
#                     'last_frame': frame_index,
#                     'lost':       0
#                 })
#                 track['confidences'].append(det['confidence'])
#                 track['hits'] += 1

#                 # Promote active → confirmed on 5th hit
#                 if best_id in self.active_tracks and track['hits'] >= CONFIRMATION_HITS:
#                     self.confirmed_tracks[best_id] = track
#                     del self.active_tracks[best_id]
#                     self.internal_to_display[best_id] = self.next_display_id
#                     track['display_id'] = self.next_display_id
#                     self.next_display_id += 1
#                     newly_confirmed.append({
#                         'internal_id': best_id,
#                         'display_id':  track['display_id'],
#                         'bbox':        track['bbox'],
#                         'mask':        track['mask'],
#                         'polygon':     track['polygon'],
#                         'class':       track['class'],
#                         'confidence':  track['confidence']
#                     })

#                 if best_id in self.confirmed_tracks:
#                     all_tracked.append({
#                         'internal_id': best_id,
#                         'display_id':  track.get('display_id', -1),
#                         'bbox':        track['bbox'],
#                         'mask':        track['mask'],
#                         'polygon':     track['polygon'],
#                         'class':       track['class'],
#                         'confidence':  track['confidence']
#                     })
#             else:
#                 nid = self.next_internal_id
#                 self.next_internal_id += 1
#                 self.active_tracks[nid] = {
#                     'bbox':        det['bbox'],
#                     'mask':        det.get('mask'),
#                     'polygon':     det.get('polygon'),
#                     'class':       det['class'],
#                     'confidence':  det['confidence'],
#                     'confidences': [det['confidence']],
#                     'hits':        1,
#                     'lost':        0,
#                     'first_frame': frame_index,
#                     'last_frame':  frame_index
#                 }

#         # Prune lost tracks
#         for tid in list(self.active_tracks.keys()):
#             if self.active_tracks[tid]['lost'] > MAX_LOST:
#                 del self.active_tracks[tid]
#         for tid in list(self.confirmed_tracks.keys()):
#             if self.confirmed_tracks[tid]['lost'] > MAX_LOST:
#                 del self.confirmed_tracks[tid]

#         return all_tracked, newly_confirmed

# # ============================================================================================
# # UTILITIES
# # ============================================================================================

# def calculate_iou(b1, b2):
#     x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
#     x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
#     inter = max(0, x2-x1) * max(0, y2-y1)
#     a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
#     a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
#     union = a1 + a2 - inter
#     return inter / union if union > 0 else 0.0


# def resize_to_fixed_size(image, target_size):
#     tw, th = target_size
#     h, w   = image.shape[:2]
#     scale  = min(tw/w, th/h)
#     nw, nh = int(w*scale), int(h*scale)
#     resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
#     canvas  = np.zeros((th, tw, 3), dtype=np.uint8)
#     yo = (th-nh)//2; xo = (tw-nw)//2
#     canvas[yo:yo+nh, xo:xo+nw] = resized
#     return canvas

# # ============================================================================================
# # ROLLING FRAME BUFFER
# # ============================================================================================

# class RollingFrameBuffer:
#     """
#     Stores last BUFFER_SIZE (frame_index, frame, tracked_objects) tuples.
#     When a detection is confirmed, we retroactively attach its display_id
#     to all buffered frames that contain the same internal_id.
#     Frames are flushed (written to video) when they fall out of the buffer.
#     """
#     def __init__(self, maxsize=BUFFER_SIZE):
#         self.maxsize = maxsize
#         self.buffer  = deque()   # each item: {'idx', 'frame', 'tracked': [...]}

#     def push(self, frame_idx, frame, tracked_objects):
#         """Add a new frame. Returns list of frames ready to be written (flushed)."""
#         self.buffer.append({
#             'idx':     frame_idx,
#             'frame':   frame,
#             'tracked': tracked_objects   # list of track dicts
#         })
#         flushed = []
#         while len(self.buffer) > self.maxsize:
#             flushed.append(self.buffer.popleft())
#         return flushed

#     def flush_all(self):
#         """Drain remaining frames at end of video."""
#         flushed = list(self.buffer)
#         self.buffer.clear()
#         return flushed

#     def retroactive_confirm(self, internal_id, display_id):
#         """
#         Once a track is confirmed, stamp its display_id on all buffered frames
#         that reference this internal_id. This is what makes single-pass work:
#         frames that were buffered before confirmation now get the correct label.
#         """
#         for item in self.buffer:
#             for tr in item['tracked']:
#                 if tr['internal_id'] == internal_id:
#                     tr['display_id'] = display_id

# # ============================================================================================
# # ANNOTATION HELPER
# # ============================================================================================

# def annotate_frame(frame, tracked_objects,
#                    mask_annotator, bbox_annotator, label_annotator,
#                    color_palette, active_classes):
#     """Annotate a single frame. Only draws confirmed detections (display_id > 0)."""
#     confirmed = [t for t in tracked_objects if t.get('display_id', -1) > 0]
#     if not confirmed:
#         return frame

#     xyxy        = np.array([t['bbox'] for t in confirmed])
#     class_ids   = np.array([CLASS_NAMES.index(t['class']) for t in confirmed])
#     confidences = np.array([t['confidence'] for t in confirmed])

#     masks = None
#     if confirmed[0].get('mask') is not None:
#         try:
#             masks = np.array([t['mask'] for t in confirmed])
#         except Exception:
#             masks = None

#     sv_det = sv.Detections(xyxy=xyxy, class_id=class_ids,
#                            confidence=confidences, mask=masks)

#     labels = [f"{t['display_id']}-{t['class']} {t['confidence']:.2f}"
#               for t in confirmed]

#     annotated = frame.copy()
#     if masks is not None:
#         annotated = mask_annotator.annotate(scene=annotated, detections=sv_det)
#     annotated = bbox_annotator.annotate(scene=annotated, detections=sv_det)
#     annotated = label_annotator.annotate(scene=annotated, detections=sv_det, labels=labels)

#     if active_classes:
#         draw_legend(annotated, OUT_WIDTH, OUT_HEIGHT, active_classes)

#     return annotated

# # ============================================================================================
# # LEGEND
# # ============================================================================================

# def draw_legend(frame, width, height, active_classes):
#     scale      = max(0.6, min(width/1920.0, 1.2))
#     lx, ly     = int(0.02*width), int(0.04*height)
#     ts         = 0.6 * scale
#     tt         = max(1, int(1.5*scale))
#     lh         = int(22*scale)
#     ll         = int(35*scale)
#     lw         = int(300*scale)

#     cv2.rectangle(frame, (lx-12, ly-12),
#                   (lx+lw, ly+lh*len(active_classes)+12),
#                   (0, 0, 0), max(2, int(2*scale)))

#     for idx, (cls_name, color_hex) in enumerate(active_classes.items()):
#         y   = ly + idx*lh
#         hex_ = color_hex.lstrip('#')
#         r,g,b = tuple(int(hex_[i:i+2], 16) for i in (0,2,4))
#         cv2.line(frame, (lx, y+10), (lx+ll, y+10), (b,g,r), 2)
#         cv2.putText(frame, cls_name, (lx+ll+12, y+12),
#                     cv2.FONT_HERSHEY_SIMPLEX, ts, (255,255,255), tt)

# # ============================================================================================
# # WORD REPORT
# # ============================================================================================

# def generate_word_report(confirmed_detections, output_dir):
#     doc  = Document()
#     doc.add_heading('Road Defects Detection Report - RF-DETR', 0)

#     sorted_dets = sorted(confirmed_detections, key=lambda x: x['display_id'])

#     by_class = {}
#     for det in sorted_dets:
#         by_class.setdefault(det['defect_type'], []).append(det)

#     doc.add_heading('Summary', 1)
#     tbl = doc.add_table(rows=1, cols=2)
#     tbl.style = 'Light Grid Accent 1'
#     tbl.rows[0].cells[0].text = 'Defect Type'
#     tbl.rows[0].cells[1].text = 'Count'
#     for cls in sorted(by_class):
#         r = tbl.add_row().cells
#         r[0].text = cls
#         r[1].text = str(len(by_class[cls]))

#     doc.add_paragraph('')

#     for cls in sorted(by_class):
#         dets = by_class[cls]
#         doc.add_heading(f'{cls.upper()} DETECTIONS', 1)
#         doc.add_paragraph(f'Total: {len(dets)}')
#         doc.add_paragraph('')

#         tbl = doc.add_table(rows=1, cols=4)
#         tbl.style = 'Light Grid Accent 1'
#         hdr = tbl.rows[0].cells
#         hdr[0].text = 'ID'; hdr[1].text = 'Chainage (m)'
#         hdr[2].text = 'Crop'; hdr[3].text = 'Frame'

#         for det in dets:
#             row = tbl.add_row().cells
#             row[0].text = f"{det['display_id']}-{cls}"
#             row[1].text = f"{det['chainage_avg_m']:.1f}"
#             for col_idx, img_key, w_inches in [(2,'crop',1.5),(3,'frame',2.5)]:
#                 p = os.path.join(det['images'][img_key])
#                 if os.path.exists(p):
#                     try:
#                         para = row[col_idx].paragraphs[0]
#                         para.add_run().add_picture(p, width=Inches(w_inches))
#                         para.alignment = WD_ALIGN_PARAGRAPH.CENTER
#                     except Exception:
#                         row[col_idx].text = str(det['display_id'])

#         doc.add_page_break()

#     rpath = os.path.join(output_dir, "Road_Defects_Report.docx")
#     doc.save(rpath)
#     print(f"\n📄 Word report: {rpath}")

# # ============================================================================================
# # MAIN PROCESSING  — SINGLE PASS WITH ROLLING BUFFER
# # ============================================================================================

# def process_video(video_path, srt_path, model, output_dir, device, starting_chainage_m=0):
#     import time

#     video_name    = os.path.basename(video_path)
#     srt_name      = os.path.basename(srt_path)
#     video_base    = os.path.splitext(video_name)[0]

#     print(f"\n{'='*80}")
#     print(f"PROCESSING: {video_name}")
#     print(f"{'='*80}")

#     # ── Parse SRT ──────────────────────────────────────────────────────────────
#     srt_data = parse_srt(srt_path)
#     if srt_data is None:
#         return None, starting_chainage_m
#     ending_chainage = calculate_cumulative_chainage(srt_data, starting_chainage_m)

#     # ── Video properties ───────────────────────────────────────────────────────
#     fps, total_frames, orig_w, orig_h = get_video_props(video_path)
#     print(f"\n🎥 Video: {orig_w}×{orig_h} @ {fps:.1f}fps  |  {total_frames} frames")
#     print(f"   Processing: {OUT_WIDTH}×{OUT_HEIGHT}  every {PROCESS_EVERY_N_FRAMES} frame(s)")
#     print_gpu_memory()

#     # ── Output paths ───────────────────────────────────────────────────────────
#     video_out_dir = os.path.join(output_dir, video_base)
#     defects_dir   = os.path.join(video_out_dir, "defect_images")
#     frames_dir    = os.path.join(defects_dir,   "frames")
#     os.makedirs(frames_dir, exist_ok=True)
#     for cls in CLASS_NAMES:
#         if cls not in IGNORED_CLASSES:
#             os.makedirs(os.path.join(defects_dir, cls), exist_ok=True)

#     out_video_path = os.path.join(video_out_dir, f"{video_base}_output.mp4")
#     out_fps        = fps / PROCESS_EVERY_N_FRAMES

#     # ── Supervision annotators ─────────────────────────────────────────────────
#     text_scale    = sv.calculate_optimal_text_scale(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
#     thickness     = sv.calculate_optimal_line_thickness(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
#     color_palette = sv.ColorPalette.from_hex(HEX_COLORS)

#     mask_annotator  = sv.MaskAnnotator(color=color_palette, opacity=MASK_OPACITY)
#     bbox_annotator  = sv.BoxAnnotator(color=color_palette, thickness=thickness)
#     label_annotator = sv.LabelAnnotator(
#         color=color_palette, text_color=sv.Color.WHITE,
#         text_scale=text_scale, text_thickness=max(1, thickness-1)
#     )

#     # ── Objects ────────────────────────────────────────────────────────────────
#     tracker          = ImprovedSegmentationTracker()
#     image_saver      = AsyncImageSaver()
#     frame_buffer     = RollingFrameBuffer(maxsize=BUFFER_SIZE)
#     video_writer     = cv2.VideoWriter(out_video_path,
#                                        cv2.VideoWriter_fourcc(*'mp4v'),
#                                        out_fps, (OUT_WIDTH, OUT_HEIGHT))

#     confirmed_detections = {}
#     active_classes       = {}
#     start_time           = time.time()

#     # ── Prefetcher ─────────────────────────────────────────────────────────────
#     prefetcher = FramePrefetcher(video_path, OUT_WIDTH, OUT_HEIGHT)

#     batch_frames   = []   # frames for inference
#     batch_indices  = []   # their frame indices
#     all_frame_idx  = 0    # total frames seen (including skipped)
#     processed_count = 0   # frames actually inferred

#     print(f"\n{'='*80}")
#     print("SINGLE-PASS PROCESSING (rolling buffer)")
#     print(f"{'='*80}")

#     def flush_and_write(flushed_items):
#         """Annotate and write a list of flushed frame items."""
#         for item in flushed_items:
#             annotated = annotate_frame(
#                 item['frame'], item['tracked'],
#                 mask_annotator, bbox_annotator, label_annotator,
#                 color_palette, active_classes
#             )
#             video_writer.write(annotated)

#     def run_inference_batch():
#         nonlocal processed_count

#         if not batch_frames:
#             return

#         # ── FP16 inference ──────────────────────────────────────────────────
#         with torch.no_grad():
#             with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
#                 batch_detections = model.predict(batch_frames, threshold=GLOBAL_THRESHOLD)

#         for b_idx, detections in enumerate(batch_detections):
#             frame_idx  = batch_indices[b_idx]
#             frame      = batch_frames[b_idx]
#             srt_entry  = get_srt_for_frame(frame_idx + 1, srt_data)
#             if srt_entry is None:
#                 # Still push frame to buffer (no GPS), tracked = []
#                 flushed = frame_buffer.push(frame_idx, frame, [])
#                 flush_and_write(flushed)
#                 continue

#             chainage_m = srt_entry['cumulative_chainage_m']

#             # NMS
#             detections = detections.with_nms(threshold=NMS_THRESHOLD)

#             # Filter
#             frame_dets = []
#             if len(detections) > 0:
#                 for idx in range(len(detections)):
#                     cid   = detections.class_id[idx]
#                     conf  = detections.confidence[idx]
#                     if cid < 0 or cid >= len(CLASS_NAMES):
#                         continue
#                     cname = CLASS_NAMES[cid]
#                     if cname in IGNORED_CLASSES:
#                         continue
#                     if conf < CLASS_THRESHOLDS.get(cname, GLOBAL_THRESHOLD):
#                         continue

#                     bbox = detections.xyxy[idx]
#                     mask, polygon = None, None
#                     if detections.mask is not None:
#                         mask = detections.mask[idx].astype(np.uint8)
#                         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
#                                                        cv2.CHAIN_APPROX_SIMPLE)
#                         if contours:
#                             polygon = contours[0].squeeze().tolist()
#                             if isinstance(polygon[0], (int, float)):
#                                 polygon = [polygon]

#                     frame_dets.append({
#                         'bbox': bbox, 'mask': mask, 'polygon': polygon,
#                         'class': cname, 'confidence': float(conf)
#                     })

#             # Pothole-Patch overlap removal
#             pot_idx  = [i for i,d in enumerate(frame_dets) if d['class'] == 'pothole']
#             pat_idx  = [i for i,d in enumerate(frame_dets) if d['class'] == 'Patch']
#             remove   = set()
#             for pi in pot_idx:
#                 for pai in pat_idx:
#                     if calculate_iou(frame_dets[pi]['bbox'],
#                                      frame_dets[pai]['bbox']) >= POTHOLE_PATCHING_IOU_THRESHOLD:
#                         remove.add(pi)
#                         break
#             frame_dets = [d for i,d in enumerate(frame_dets) if i not in remove]

#             # Track
#             tracked_objects, newly_confirmed = tracker.update(frame_dets, frame_idx)

#             # Update active class legend
#             for tr in tracked_objects:
#                 cname = tr['class']
#                 if cname not in active_classes:
#                     active_classes[cname] = HEX_COLORS[CLASS_NAMES.index(cname)]

#             # ── Retroactive confirmation ────────────────────────────────────
#             for confirmed in newly_confirmed:
#                 iid = confirmed['internal_id']
#                 did = confirmed['display_id']

#                 # Stamp display_id on all buffered frames referencing this track
#                 frame_buffer.retroactive_confirm(iid, did)

#                 # Save crop + frame image
#                 x1,y1,x2,y2 = map(int, confirmed['bbox'])
#                 crop = frame[max(0,y1-30):min(OUT_HEIGHT,y2+30),
#                              max(0,x1-30):min(OUT_WIDTH, x2+30)]
#                 crop_resized  = resize_to_fixed_size(crop, CROP_SIZE)
#                 frame_resized = resize_to_fixed_size(frame, FRAME_SIZE)

#                 cname     = confirmed['class']
#                 crop_path = os.path.join(defects_dir, cname, f"{did}-{cname}.jpg")
#                 frm_path  = os.path.join(frames_dir,  f"{did}-{cname}_frame.jpg")
#                 image_saver.save(crop_path, crop_resized)
#                 image_saver.save(frm_path,  frame_resized)

#                 # Store metadata
#                 track = tracker.confirmed_tracks[iid]
#                 confirmed_detections[iid] = {
#                     'display_id':       did,
#                     'internal_id':      iid,
#                     'defect_type':      cname,
#                     'video_name':       video_name,
#                     'srt_name':         srt_name,
#                     'frame_start':      track['first_frame'],
#                     'frame_end':        track['last_frame'],
#                     'timestamp_start':  srt_entry['absolute_timestamp'].isoformat(),
#                     'timestamp_end':    srt_entry['absolute_timestamp'].isoformat(),
#                     'chainage_start_m': chainage_m,
#                     'chainage_end_m':   chainage_m,
#                     'chainage_avg_m':   chainage_m,
#                     'confidence_avg':   float(np.mean(track['confidences'])),
#                     'gps': {
#                         'latitude':  srt_entry['latitude'],
#                         'longitude': srt_entry['longitude']
#                     },
#                     'polygon': confirmed['polygon'],
#                     'images': {'crop': crop_path, 'frame': frm_path}
#                 }

#             # ── Push to rolling buffer, flush old frames to video ──────────
#             flushed = frame_buffer.push(frame_idx, frame, tracked_objects)
#             flush_and_write(flushed)

#             processed_count += 1

#         # Periodic GPU cache clear
#         if processed_count % GPU_CLEAR_EVERY == 0 and device != 'cpu':
#             torch.cuda.empty_cache()

#         batch_frames.clear()
#         batch_indices.clear()

#         # Progress log
#         elapsed = time.time() - start_time
#         fps_act = processed_count / elapsed if elapsed > 0 else 0
#         if processed_count % 50 == 0:
#             print(f"   Frame {all_frame_idx}/{total_frames} | "
#                   f"{fps_act:.1f} fps | "
#                   f"Confirmed: {len(confirmed_detections)} | "
#                   f"Buffer: {len(frame_buffer.buffer)}")

#     # ── Main loop ──────────────────────────────────────────────────────────────
#     while True:
#         item = prefetcher.get()

#         if item is None:
#             # End of video — flush batch then buffer
#             run_inference_batch()
#             flush_and_write(frame_buffer.flush_all())
#             break

#         frame_idx, frame = item
#         all_frame_idx    = frame_idx

#         if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
#             batch_frames.append(frame)
#             batch_indices.append(frame_idx)

#         if len(batch_frames) == BATCH_SIZE:
#             run_inference_batch()

#     # ── Finish ─────────────────────────────────────────────────────────────────
#     video_writer.release()
#     print("\n💾 Waiting for image saves...")
#     image_saver.shutdown()

#     if device != 'cpu':
#         torch.cuda.empty_cache()

#     # Update final chainage metadata
#     for iid, det in confirmed_detections.items():
#         if iid in tracker.confirmed_tracks:
#             track    = tracker.confirmed_tracks[iid]
#             last_srt = get_srt_for_frame(track['last_frame'] + 1, srt_data)
#             if last_srt:
#                 det['frame_end']       = track['last_frame']
#                 det['timestamp_end']   = last_srt['absolute_timestamp'].isoformat()
#                 det['chainage_end_m']  = last_srt['cumulative_chainage_m']
#                 det['chainage_avg_m']  = (det['chainage_start_m'] +
#                                           det['chainage_end_m']) / 2

#     all_detections = list(confirmed_detections.values())

#     # Save JSON
#     detection_data = {
#         'video_name':        video_name,
#         'srt_name':          srt_name,
#         'processing_date':   datetime.now().isoformat(),
#         'model':             'RF-DETR-Seg-Medium',
#         'total_frames':      total_frames,
#         'starting_chainage_m': starting_chainage_m,
#         'ending_chainage_m': ending_chainage,
#         'total_distance_km': (ending_chainage - starting_chainage_m) / 1000,
#         'total_detections':  len(all_detections),
#         'detections':        all_detections,
#         'summary':           {}
#     }

#     for det in all_detections:
#         dtype = det['defect_type']
#         entry = detection_data['summary'].setdefault(
#             dtype, {'count': 0, 'avg_confidence': []})
#         entry['count'] += 1
#         entry['avg_confidence'].append(det['confidence_avg'])

#     for dtype in detection_data['summary']:
#         confs = detection_data['summary'][dtype]['avg_confidence']
#         detection_data['summary'][dtype]['avg_confidence'] = float(np.mean(confs))

#     json_path = os.path.join(video_out_dir, f"{video_base}_detections.json")
#     with open(json_path, 'w') as f:
#         json.dump(detection_data, f, indent=2)

#     print("\n📄 Generating Word report...")
#     generate_word_report(all_detections, video_out_dir)

#     elapsed_total = time.time() - start_time
#     print(f"\n{'='*80}")
#     print(f"✅ DONE in {elapsed_total/60:.1f} min")
#     print(f"   Detections : {len(all_detections)}")
#     print(f"   📹 Video   : {out_video_path}")
#     print(f"   📊 JSON    : {json_path}")
#     print(f"   🖼️  Images  : {defects_dir}")
#     print(f"\n   Defect Summary:")
#     for dtype, stats in detection_data['summary'].items():
#         print(f"      {dtype}: {stats['count']} (avg conf: {stats['avg_confidence']:.2f})")

#     return json_path, ending_chainage

# # ============================================================================================
# # MAIN
# # ============================================================================================

# def main():
#     print("\n" + "="*80)
#     print("RF-DETR ROAD DEFECT DETECTION — OPTIMIZED (SINGLE-PASS + ROLLING BUFFER)")
#     print("="*80)

#     device = check_gpu()

#     for label, path in [("Video", VIDEO_PATH), ("SRT", SRT_PATH),
#                          ("Checkpoint", CHECKPOINT_PATH), ("COCO JSON", COCO_JSON_PATH)]:
#         if not os.path.exists(path):
#             print(f"❌ {label} not found: {path}")
#             return

#     print(f"\n{'='*80}\nLOADING RF-DETR MODEL\n{'='*80}")
#     model = RFDETRSegMedium(
#         pretrain_weights=CHECKPOINT_PATH,
#         num_queries=NUM_QUERIES,
#         image_size=IMAGE_SIZE,
#         max_image_size=IMAGE_SIZE,
#     )
#     print("✅ Model loaded")
#     print_gpu_memory()

#     # Warmup
#     print("\n🔥 Warming up GPU...")
#     dummy = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
#     with torch.no_grad():
#         with torch.cuda.amp.autocast(enabled=(device != 'cpu')):
#             _ = model.predict([dummy], threshold=0.5)
#     if device != 'cpu':
#         torch.cuda.synchronize()
#     print("✅ GPU warmed up")
#     print_gpu_memory()

#     process_video(
#         VIDEO_PATH, SRT_PATH, model,
#         OUTPUT_BASE_DIR, device,
#         starting_chainage_m=0
#     )

#     print(f"\n{'='*80}")
#     print("✅ ALL DONE!")
#     print(f"Output: {OUTPUT_BASE_DIR}")
#     print(f"{'='*80}\n")

# if __name__ == "__main__":
#     main()