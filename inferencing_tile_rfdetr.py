# ## ============================
# # ROAD ASSESSMENT SCRIPT - RF-DETR SEG MEDIUM + CUSTOM SLICING
# # Inference backbone: RFDETRSegMedium (no SAHI wrapper — direct inference)
# # Changes from v1:
# #   1. Fixed class name alignment using index-aligned COCO JSON approach
# #   2. Batch tile inference (all tiles per frame in single model.predict() call)
# #   3. Output video resized to 1920x1080 (configurable)
# #   4. Class-specific confidence thresholds
# # ============================
# import os
# import cv2
# import math
# import numpy as np
# import re
# import torch
# import torchvision
# import json
# import threading
# import queue
# from datetime import datetime
# from PIL import Image as PILImage
# import glob
# from concurrent.futures import ThreadPoolExecutor
# from rfdetr import RFDETRSegMedium

# # ============================
# # CONFIGURATION - UPDATE THESE
# # ============================
# SINGLE_VIDEO_MODE = True
# MULTI_VIDEO_MODE  = False
# FOLDER_MODE       = False

# VIDEO_PATH = "/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.MP4"
# SRT_PATH   = "/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.SRT"

# VIDEO_SRT_PAIRS = [
#     {
#         "video": "/content/drive/MyDrive/videos/DJI_0001.MP4",
#         "srt":   "/content/drive/MyDrive/videos/DJI_0001.SRT"
#     }
# ]

# INPUT_FOLDER          = "/content/drive/MyDrive/videos"
# SEGMENTATION_MODEL_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
# OUTPUT_BASE_DIR       = "/media/user/New Volume/Sakshi/chattishgarh/output"
# COCO_JSON_PATH        = r"/media/user/New Volume/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"

# os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# # ============================
# # PROCESSING SETTINGS
# # ============================
# PROCESS_EVERY_N_FRAMES = 5

# # Tile size — must match image_size=800 used during training
# SLICE_HEIGHT          = 800
# SLICE_WIDTH           = 800
# OVERLAP_HEIGHT_RATIO  = 0.10
# OVERLAP_WIDTH_RATIO   = 0.10

# # Output video resolution (set to None to keep original resolution)
# OUTPUT_WIDTH  = 1920
# OUTPUT_HEIGHT = 1080

# NMS_IOU_THRESHOLD   = 0.50
# IOU_THRESHOLD       = 0.30    # tracker
# MAX_DISTANCE        = 50
# MAX_LOST            = 30
# FRAME_BUFFER_SIZE   = 16
# SAVE_WORKER_THREADS = 4

# # ============================
# # CLASS-SPECIFIC THRESHOLDS
# # ============================
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
#     "parallel_crack":             0.30,
# }
# GLOBAL_THRESHOLD = 0.30

# # ============================
# # IGNORED CLASSES
# # completely skipped — not annotated, not saved
# # ============================
# IGNORED_CLASSES = {
#     "white_mark", "water_mark", "doubt", "bump",
#     "guard_post", "overhead_sign_board", "sign_board", "kerb_damage"
# }

# # ============================
# # CLASS NAMES & COLORS
# # index-aligned from COCO JSON — fixes class name misalignment
# # ============================
# with open(COCO_JSON_PATH, "r") as f:
#     coco_data = json.load(f)

# # Sort categories by ID to match model's 0-indexed output order
# categories  = sorted(coco_data["categories"], key=lambda x: x["id"])
# CLASS_NAMES = [cat["name"] for cat in categories]

# print("Loaded Classes:")
# for i, name in enumerate(CLASS_NAMES):
#     print(f"  {i:2d}: {name}")

# # Index-aligned HEX colors — position must match CLASS_NAMES order from COCO JSON
# HEX_COLORS = [
#     "#FF0000",   #  Corrugations_and_shoving
#     "#008CFF",   #  Cracking
#     "#FF00FF",   #  Edge_breaking
#     "#8000FF",   #  Edge_drop
#     "#00FF80",   #  Embankment slope
#     "#FF0080",   #  MBCB_defect
#     "#B400B4",   #  MBCB_missing
#     "#0000FF",   #  Patch
#     "#02D32E",   #  Ravelling
#     "#FF8000",   #  Scaling
#     "#00FFFF",   #  Wear
#     "#9E4292",   #  bump
#     "#FF4040",   #  cattle
#     "#4040FF",   #  depression
#     "#A0A0A0",   #  doubt
#     "#006400",   #  guard_post
#     "#D2691E",   #  honeycomb
#     "#008080",   #  kerb_damage
#     "#52240E",   #  km_stone
#     "#873CBE",   #  overhead_sign_board
#     "#00C8C8",   #  parallel_crack
#     "#0000C8",   #  pothole
#     "#32CD32",   #  sign_board
#     "#FF69B4",   #  strip_seal_expansion_join
#     "#696969",   #  tyre_marks
#     "#228B22",   #  vegetation_on_road
#     "#ADD8E6",   #  water_mark
#     "#F5F5F5",   #  white_mark
# ]

# # Pad with fallback white if COCO has more classes than colors defined
# while len(HEX_COLORS) < len(CLASS_NAMES):
#     HEX_COLORS.append("#FFFFFF")

# def hex_to_bgr(hex_str):
#     hex_str = hex_str.lstrip("#")
#     r = int(hex_str[0:2], 16)
#     g = int(hex_str[2:4], 16)
#     b = int(hex_str[4:6], 16)
#     return (b, g, r)

# # name → BGR color dict (index-aligned)
# color_map = {
#     CLASS_NAMES[i]: hex_to_bgr(HEX_COLORS[i])
#     for i in range(len(CLASS_NAMES))
# }

# # ============================
# # GPU UTILITIES
# # ============================
# def check_gpu():
#     if torch.cuda.is_available():
#         device    = 'cuda'
#         gpu_name  = torch.cuda.get_device_name(0)
#         total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
#         print(f"✅ GPU: {gpu_name}")
#         print(f"   VRAM: {total_mem:.1f} GB")
#         torch.backends.cudnn.benchmark = True
#         torch.backends.cudnn.enabled   = True
#         return device
#     else:
#         print("⚠️ No GPU found — running on CPU (slow)")
#         return 'cpu'

# def print_gpu_memory():
#     if torch.cuda.is_available():
#         used   = torch.cuda.memory_allocated() / 1e9
#         cached = torch.cuda.memory_reserved()   / 1e9
#         print(f"   🖥️ GPU Memory: {used:.2f}GB used / {cached:.2f}GB cached")

# def warmup_model(model, device):
#     print("🔥 Warming up GPU (RF-DETR)...")
#     # Single tile warmup — batch size is dynamic at inference time
#     # so we just warm up the GPU pipeline with one tile
#     dummy = [np.random.randint(0, 255, (SLICE_HEIGHT, SLICE_WIDTH, 3), dtype=np.uint8)]
#     with torch.no_grad():
#         model.predict(dummy, threshold=GLOBAL_THRESHOLD)
#     if device != 'cpu' and torch.cuda.is_available():
#         torch.cuda.synchronize()
#     print("✅ GPU warmed up")

# # ============================
# # BATCH SLICED INFERENCE
# # All tiles from a single frame are collected first,
# # then passed to model.predict() in ONE call — faster GPU utilisation
# # ============================
# def sliced_predict_batch(frame_bgr, model, slice_h, slice_w,
#                          overlap_h, overlap_w):
#     img_h, img_w = frame_bgr.shape[:2]
#     step_h = max(1, int(slice_h * (1 - overlap_h)))
#     step_w = max(1, int(slice_w * (1 - overlap_w)))

#     tiles        = []   # list of cropped numpy arrays
#     tile_origins = []   # (x1, y1) offset for each tile

#     # ── Collect all tiles ────────────────────────────────────────────
#     y = 0
#     while True:
#         y2 = min(y + slice_h, img_h)
#         y1 = max(0, y2 - slice_h)
#         x = 0
#         while True:
#             x2 = min(x + slice_w, img_w)
#             x1 = max(0, x2 - slice_w)
#             tile = frame_bgr[y1:y2, x1:x2]
#             tiles.append(tile)
#             tile_origins.append((x1, y1))
#             if x2 >= img_w:
#                 break
#             x += step_w
#         if y2 >= img_h:
#             break
#         y += step_h

#     if not tiles:
#         return []

#     # ── Single batched model call ────────────────────────────────────
#     with torch.no_grad():
#         batch_dets = model.predict(tiles, threshold=GLOBAL_THRESHOLD)

#     # batch_dets is a list — one Detections object per tile
#     # (if rfdetr returns a single merged object for a batch,
#     #  handle both cases below)
#     if not isinstance(batch_dets, (list, tuple)):
#         batch_dets = [batch_dets]

#     # ── Unpack detections & project back to full-frame coords ────────
#     all_boxes   = []
#     all_scores  = []
#     all_classes = []
#     all_masks   = []

#     for tile_idx, dets in enumerate(batch_dets):
#         if dets is None or len(dets) == 0:
#             continue

#         x1_off, y1_off = tile_origins[tile_idx]
#         tile            = tiles[tile_idx]
#         tile_h, tile_w  = tile.shape[:2]

#         for i in range(len(dets)):
#             class_id   = int(dets.class_id[i])
#             confidence = float(dets.confidence[i])

#             # Safety guard
#             if class_id >= len(CLASS_NAMES):
#                 continue

#             class_name = CLASS_NAMES[class_id]

#             # Skip ignored classes entirely
#             if class_name in IGNORED_CLASSES:
#                 continue

#             # Apply per-class threshold (fall back to global)
#             threshold = CLASS_THRESHOLDS.get(class_name, GLOBAL_THRESHOLD)
#             if confidence < threshold:
#                 continue

#             bx1, by1, bx2, by2 = dets.xyxy[i]

#             # Shift bbox to full-frame coordinates
#             all_boxes.append([
#                 float(bx1) + x1_off,
#                 float(by1) + y1_off,
#                 float(bx2) + x1_off,
#                 float(by2) + y1_off,
#             ])
#             all_scores.append(confidence)
#             all_classes.append(class_id)

#             # Handle segmentation mask
#             if hasattr(dets, "mask") and dets.mask is not None:
#                 tile_mask = dets.mask[i].astype(np.uint8)
#                 if tile_mask.shape != (tile_h, tile_w):
#                     tile_mask = cv2.resize(
#                         tile_mask, (tile_w, tile_h),
#                         interpolation=cv2.INTER_NEAREST
#                     )
#                 full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
#                 full_mask[y1_off:y1_off + tile_h,
#                           x1_off:x1_off + tile_w] = tile_mask
#                 all_masks.append(full_mask.astype(bool))
#             else:
#                 all_masks.append(None)

#     if not all_boxes:
#         return []

#     # ── Global NMS across all tiles ──────────────────────────────────
#     boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
#     scores_t = torch.tensor(all_scores, dtype=torch.float32)
#     keep     = torchvision.ops.nms(boxes_t, scores_t, NMS_IOU_THRESHOLD).tolist()

#     results = []
#     for idx in keep:
#         class_id   = all_classes[idx]
#         if class_id >= len(CLASS_NAMES):
#             continue
#         mask    = all_masks[idx]
#         polygon = None

#         if mask is not None:
#             contours, _ = cv2.findContours(
#                 mask.astype(np.uint8),
#                 cv2.RETR_EXTERNAL,
#                 cv2.CHAIN_APPROX_SIMPLE
#             )
#             if contours:
#                 polygon = contours[0].squeeze().tolist()
#                 if polygon and isinstance(polygon[0], (int, float)):
#                     polygon = [polygon]
#                 if polygon and len(polygon) < 3:
#                     polygon = None

#         results.append({
#             'bbox':       all_boxes[idx],
#             'mask':       mask,
#             'polygon':    polygon,
#             'class':      CLASS_NAMES[class_id],
#             'confidence': all_scores[idx],
#         })

#     return results

# # ============================
# # THREADED FRAME READER
# # ============================
# class ThreadedVideoReader:
#     def __init__(self, video_path, skip_n=PROCESS_EVERY_N_FRAMES,
#                  buffer_size=FRAME_BUFFER_SIZE):
#         self.cap         = cv2.VideoCapture(video_path)
#         self.skip_n      = skip_n
#         self.buffer      = queue.Queue(maxsize=buffer_size)
#         self.stopped     = False
#         self.width       = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         self.height      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
#         self.total_frames= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.thread      = threading.Thread(target=self._read_loop, daemon=True)
#         self.thread.start()

#     def _read_loop(self):
#         local_index = 0
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 self.buffer.put(None)
#                 break
#             if local_index % self.skip_n == 0:
#                 self.buffer.put((local_index, frame))
#             local_index += 1

#     def read(self):
#         return self.buffer.get()

#     def release(self):
#         self.stopped = True
#         self.cap.release()

# # ============================
# # ASYNC IMAGE SAVER
# # ============================
# class AsyncImageSaver:
#     def __init__(self, num_workers=SAVE_WORKER_THREADS):
#         self.executor = ThreadPoolExecutor(max_workers=num_workers)
#         self.futures  = []

#     def save(self, path, image):
#         self.futures.append(self.executor.submit(cv2.imwrite, path, image))

#     def wait_all(self):
#         for f in self.futures:
#             f.result()
#         self.futures.clear()

#     def shutdown(self):
#         self.wait_all()
#         self.executor.shutdown(wait=True)

# # ============================
# # SRT PARSING
# # ============================
# def parse_srt(srt_path):
#     srt_data = []
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
#             frame_num = int(lines[0].strip())
#             tc = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
#             if not tc:
#                 continue
#             h, m, s, ms  = map(int, tc.groups())
#             timestamp_ms = (h * 3600 + m * 60 + s) * 1000 + ms
#             meta         = ' '.join(lines[2:])

#             ts_match = re.search(
#                 r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', meta)
#             if not ts_match:
#                 continue
#             absolute_timestamp = datetime.strptime(
#                 ts_match.group(1), '%Y-%m-%d %H:%M:%S.%f')

#             lat = re.search(r'\[latitude:\s*([-\d.]+)\]',  meta)
#             lon = re.search(r'\[longitude:\s*([-\d.]+)\]', meta)
#             alt = re.search(r'\[(?:abs_alt|altitude):\s*([-\d.]+)', meta)

#             if not (lat and lon):
#                 continue

#             srt_data.append({
#                 'frame_number':      frame_num,
#                 'timestamp_ms':      timestamp_ms,
#                 'absolute_timestamp':absolute_timestamp,
#                 'latitude':          float(lat.group(1)),
#                 'longitude':         float(lon.group(1)),
#                 'altitude':          float(alt.group(1)) if alt else 50.0,
#             })
#         except Exception:
#             continue

#     if not srt_data:
#         print("❌ No valid GPS data in SRT file!")
#         return None
#     print(f"✅ Parsed {len(srt_data)} SRT entries")
#     return srt_data

# def haversine_distance(lat1, lon1, lat2, lon2):
#     R  = 6371000
#     p1 = math.radians(lat1)
#     p2 = math.radians(lat2)
#     dp = math.radians(lat2 - lat1)
#     dl = math.radians(lon2 - lon1)
#     a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
#     return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# def calculate_cumulative_chainage(srt_data, starting_chainage_m=0):
#     cum = starting_chainage_m
#     for i, entry in enumerate(srt_data):
#         if i == 0:
#             entry['cumulative_chainage_m'] = starting_chainage_m
#         else:
#             prev = srt_data[i - 1]
#             cum += haversine_distance(
#                 prev['latitude'], prev['longitude'],
#                 entry['latitude'], entry['longitude']
#             )
#             entry['cumulative_chainage_m'] = cum
#     print(f"✅ Chainage: {starting_chainage_m:.1f}m → {cum:.1f}m "
#           f"({(cum - starting_chainage_m)/1000:.3f} km)")
#     return cum

# def get_srt_data_for_frame(frame_index, srt_data):
#     for entry in srt_data:
#         if entry['frame_number'] == frame_index:
#             return entry
#     if srt_data:
#         return min(srt_data, key=lambda x: abs(x['frame_number'] - frame_index))
#     return None

# # ============================
# # TRACKER
# # ============================
# class SegmentationTracker:
#     def __init__(self, iou_threshold=0.3, max_distance=50, max_lost=30):
#         self.next_id       = 1
#         self.tracks        = {}
#         self.iou_threshold = iou_threshold
#         self.max_distance  = max_distance
#         self.max_lost      = max_lost

#     def _mask_iou(self, m1, m2):
#         if m1 is None or m2 is None:
#             return 0.0
#         inter = np.logical_and(m1, m2).sum()
#         union = np.logical_or(m1, m2).sum()
#         return inter / union if union > 0 else 0.0

#     def _centroid(self, bbox):
#         return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

#     def update(self, detections):
#         for t in self.tracks.values():
#             t['lost'] += 1

#         assigned      = set()
#         updated_tracks = []

#         for tid, t in list(self.tracks.items()):
#             best_i, best_score = -1, 0
#             for i, det in enumerate(detections):
#                 if i in assigned or t.get('class') != det.get('class'):
#                     continue
#                 iou  = self._mask_iou(t.get('mask'), det.get('mask'))
#                 dist = math.hypot(
#                     t['centroid'][0] - self._centroid(det['bbox'])[0],
#                     t['centroid'][1] - self._centroid(det['bbox'])[1]
#                 )
#                 if iou > self.iou_threshold or dist < self.max_distance:
#                     score = iou - (dist / self.max_distance) * 0.5
#                     if score > best_score:
#                         best_score, best_i = score, i

#             if best_i >= 0:
#                 det = detections[best_i]
#                 self.tracks[tid].update({
#                     'bbox':     det['bbox'],
#                     'centroid': self._centroid(det['bbox']),
#                     'mask':     det['mask'],
#                     'lost':     0,
#                     'class':    det['class'],
#                 })
#                 assigned.add(best_i)
#                 updated_tracks.append({
#                     'track_id': tid,
#                     'bbox':     det['bbox'],
#                     'mask':     det['mask'],
#                     'class':    det['class'],
#                     'polygon':  det['polygon'],
#                 })
#             elif t['lost'] > self.max_lost:
#                 del self.tracks[tid]

#         for i, det in enumerate(detections):
#             if i in assigned:
#                 continue
#             tid = self.next_id
#             self.next_id += 1
#             self.tracks[tid] = {
#                 'bbox':     det['bbox'],
#                 'centroid': self._centroid(det['bbox']),
#                 'mask':     det['mask'],
#                 'class':    det['class'],
#                 'lost':     0,
#             }
#             updated_tracks.append({
#                 'track_id': tid,
#                 'bbox':     det['bbox'],
#                 'mask':     det['mask'],
#                 'class':    det['class'],
#                 'polygon':  det['polygon'],
#             })

#         return updated_tracks

# # ============================
# # VISUALIZATION
# # ============================
# def draw_segmentation_overlay(frame, detections):
#     overlay = frame.copy()
#     for det in detections:
#         cls_name  = det['class']
#         polygon   = det['polygon']
#         track_id  = det['track_id']
#         color     = color_map.get(cls_name, (200, 200, 200))

#         if polygon is not None and len(polygon) > 0:
#             pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
#             cv2.fillPoly(overlay, [pts], color)
#             cv2.polylines(frame, [pts], True, color, 2)

#         x1, y1 = int(det['bbox'][0]), int(det['bbox'][1])
#         cv2.putText(frame, f"{cls_name} ID:{track_id}",
#                     (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

#     return cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

# def draw_legend(frame, width, height):
#     # Only draw legend for non-ignored classes
#     visible_classes = {
#         name: color for name, color in color_map.items()
#         if name not in IGNORED_CLASSES
#     }
#     scale    = max(0.6, min(width / 1920.0, 1.2))
#     lx       = int(0.02 * width)
#     ly       = int(0.04 * height)
#     line_h   = int(22 * scale)
#     line_len = int(35 * scale)
#     leg_w    = int(300 * scale)
#     txt_scale= 0.6 * scale
#     txt_thick= max(1, int(1.5 * scale))

#     cv2.rectangle(
#         frame,
#         (lx - 12, ly - 12),
#         (lx + leg_w, ly + line_h * len(visible_classes) + 12),
#         (0, 0, 0),
#         max(2, int(2 * scale))
#     )
#     for idx, (cls_name, color) in enumerate(visible_classes.items()):
#         y = ly + idx * line_h
#         cv2.line(frame, (lx, y + 10), (lx + line_len, y + 10), color, 2)
#         cv2.putText(frame, cls_name,
#                     (lx + line_len + 12, y + 12),
#                     cv2.FONT_HERSHEY_SIMPLEX, txt_scale,
#                     (255, 255, 255), txt_thick)

# # ============================
# # FOLDER SCANNING
# # ============================
# def find_video_srt_pairs(folder_path):
#     print(f"\n🔍 Scanning folder: {folder_path}")
#     video_files = []
#     for ext in ['*.mp4', '*.MP4', '*.mov', '*.MOV', '*.avi', '*.AVI']:
#         video_files.extend(glob.glob(os.path.join(folder_path, ext)))

#     srt_files = (glob.glob(os.path.join(folder_path, '*.srt')) +
#                  glob.glob(os.path.join(folder_path, '*.SRT')))

#     print(f"  Found {len(video_files)} video(s), {len(srt_files)} SRT(s)")
#     pairs = []
#     for vp in sorted(video_files):
#         base = os.path.splitext(os.path.basename(vp))[0]
#         srt  = next(
#             (s for s in srt_files
#              if os.path.splitext(os.path.basename(s))[0] == base),
#             None
#         )
#         if srt:
#             pairs.append({"video": vp, "srt": srt})
#             print(f"  ✅ {os.path.basename(vp)} ↔ {os.path.basename(srt)}")
#         else:
#             print(f"  ⚠️ No SRT: {os.path.basename(vp)}")
#     return pairs

# # ============================
# # MAIN PROCESSING FUNCTION
# # ============================
# def process_video(video_path, srt_path, rfdetr_model, output_dir,
#                   device, starting_chainage_m=0):
#     import time

#     video_name     = os.path.basename(video_path)
#     srt_name       = os.path.basename(srt_path)
#     video_basename = os.path.splitext(video_name)[0]

#     print(f"\n{'='*70}")
#     print(f"Processing: {video_name}")
#     print(f"{'='*70}")

#     srt_data = parse_srt(srt_path)
#     if srt_data is None:
#         return None, starting_chainage_m

#     ending_chainage = calculate_cumulative_chainage(srt_data, starting_chainage_m)

#     print("🎥 Opening video (threaded reader)...")
#     reader       = ThreadedVideoReader(video_path, skip_n=PROCESS_EVERY_N_FRAMES)
#     width        = reader.width
#     height       = reader.height
#     fps          = reader.fps
#     total_frames = reader.total_frames

#     # ── Output resolution ────────────────────────────────────────────
#     if OUTPUT_WIDTH and OUTPUT_HEIGHT:
#         out_w = OUTPUT_WIDTH
#         out_h = OUTPUT_HEIGHT
#     else:
#         out_w = width
#         out_h = height

#     scale_x = out_w / width
#     scale_y = out_h / height

#     print(f"  Source  : {width}x{height} @ {fps:.1f}fps | {total_frames} total frames")
#     print(f"  Output  : {out_w}x{out_h}")
#     print(f"  Processing every {PROCESS_EVERY_N_FRAMES} frames → "
#           f"~{total_frames // PROCESS_EVERY_N_FRAMES} frames")
#     print(f"  Tile size: {SLICE_WIDTH}x{SLICE_HEIGHT} | "
#           f"Overlap: {int(OVERLAP_WIDTH_RATIO*100)}%")
#     print_gpu_memory()

#     # ── Output directories ───────────────────────────────────────────
#     video_out_dir = os.path.join(output_dir, video_basename)
#     defects_dir   = os.path.join(video_out_dir, "defect_images")
#     frames_dir    = os.path.join(defects_dir,   "frames")
#     os.makedirs(frames_dir, exist_ok=True)
#     for cls in CLASS_NAMES:
#         if cls not in IGNORED_CLASSES:
#             os.makedirs(os.path.join(defects_dir, cls), exist_ok=True)

#     output_video_path = os.path.join(
#         video_out_dir, f"{video_basename}_output.mp4")
#     output_fps = max(1.0, fps / PROCESS_EVERY_N_FRAMES)
#     out = cv2.VideoWriter(
#         output_video_path,
#         cv2.VideoWriter_fourcc(*'mp4v'),
#         output_fps,
#         (out_w, out_h)
#     )

#     tracker       = SegmentationTracker(IOU_THRESHOLD, MAX_DISTANCE, MAX_LOST)
#     image_saver   = AsyncImageSaver(num_workers=SAVE_WORKER_THREADS)
#     track_history = {}
#     all_detections= []
#     det_id_counter= 1
#     processed_frames = 0
#     start_time    = time.time()

#     print(f"\n🚀 Starting inference (batched tile processing)...")

#     while True:
#         item = reader.read()
#         if item is None:
#             break

#         frame_index, frame = item
#         processed_frames += 1

#         srt_entry = get_srt_data_for_frame(frame_index + 1, srt_data)
#         if srt_entry is None:
#             continue

#         chainage_m = srt_entry['cumulative_chainage_m']

#         # ── Batched sliced RF-DETR inference ─────────────────────────
#         frame_detections = sliced_predict_batch(
#             frame, rfdetr_model,
#             slice_h=SLICE_HEIGHT, slice_w=SLICE_WIDTH,
#             overlap_h=OVERLAP_HEIGHT_RATIO, overlap_w=OVERLAP_WIDTH_RATIO
#         )

#         # ── Track ─────────────────────────────────────────────────────
#         tracked_objects = tracker.update(frame_detections)

#         # ── Draw on full-resolution frame ────────────────────────────
#         annotated = draw_segmentation_overlay(frame.copy(), tracked_objects)
#         draw_legend(annotated, width, height)

#         # ── Resize annotated frame to output resolution ───────────────
#         if (out_w, out_h) != (width, height):
#             annotated_out = cv2.resize(
#                 annotated, (out_w, out_h),
#                 interpolation=cv2.INTER_LINEAR
#             )
#         else:
#             annotated_out = annotated

#         # ── Update track history ──────────────────────────────────────
#         for tr in tracked_objects:
#             tid      = tr['track_id']
#             bbox     = tr['bbox']
#             cls_name = tr['class']
#             polygon  = tr['polygon']

#             if tid not in track_history:
#                 track_history[tid] = {
#                     'type':            cls_name,
#                     'first_frame':     frame_index + 1,
#                     'last_frame':      frame_index + 1,
#                     'first_chainage':  chainage_m,
#                     'last_chainage':   chainage_m,
#                     'first_timestamp': srt_entry['absolute_timestamp'],
#                     'last_timestamp':  srt_entry['absolute_timestamp'],
#                     'gps_lat':         srt_entry['latitude'],
#                     'gps_lon':         srt_entry['longitude'],
#                     'best_snapshot':   None,
#                 }

#             track_history[tid]['last_frame']     = frame_index + 1
#             track_history[tid]['last_chainage']  = chainage_m
#             track_history[tid]['last_timestamp'] = srt_entry['absolute_timestamp']

#             if track_history[tid]['best_snapshot'] is None:
#                 x1, y1, x2, y2 = map(int, bbox)
#                 crop = frame[
#                     max(0, y1):min(height, y2),
#                     max(0, x1):min(width,  x2)
#                 ].copy()
#                 track_history[tid]['best_snapshot'] = {
#                     'frame_idx':  frame_index + 1,
#                     'crop':       crop,
#                     'full_frame': annotated_out.copy(),  # save at output res
#                     'polygon':    polygon,
#                 }

#         out.write(annotated_out)

#         # ── Finalise lost tracks ──────────────────────────────────────
#         active_tids = {t['track_id'] for t in tracked_objects}
#         for tid in list(track_history.keys()):
#             if tid not in active_tids and tid not in tracker.tracks:
#                 track = track_history.pop(tid)
#                 if track['best_snapshot'] is None:
#                     continue
#                 dtype      = track['type']
#                 crop_path  = os.path.join(
#                     defects_dir, dtype,
#                     f"{dtype}_crop_{det_id_counter}.jpg")
#                 frame_path = os.path.join(
#                     frames_dir,
#                     f"{dtype}_frame_{det_id_counter}.jpg")
#                 image_saver.save(crop_path,  track['best_snapshot']['crop'])
#                 image_saver.save(frame_path, track['best_snapshot']['full_frame'])

#                 all_detections.append({
#                     'id':               det_id_counter,
#                     'track_id':         tid,
#                     'defect_type':      dtype,
#                     'video_name':       video_name,
#                     'srt_name':         srt_name,
#                     'frame_start':      track['first_frame'],
#                     'frame_end':        track['last_frame'],
#                     'timestamp_start':  track['first_timestamp'].isoformat(),
#                     'timestamp_end':    track['last_timestamp'].isoformat(),
#                     'chainage_start_m': track['first_chainage'],
#                     'chainage_end_m':   track['last_chainage'],
#                     'chainage_avg_m':   (track['first_chainage'] +
#                                          track['last_chainage']) / 2,
#                     'gps': {
#                         'latitude':  track['gps_lat'],
#                         'longitude': track['gps_lon'],
#                     },
#                     'polygon': track['best_snapshot']['polygon'],
#                     'images': {
#                         'crop':  os.path.relpath(crop_path,  video_out_dir),
#                         'frame': os.path.relpath(frame_path, video_out_dir),
#                     },
#                 })
#                 det_id_counter += 1

#         # ── Progress ──────────────────────────────────────────────────
#         if processed_frames % 50 == 0:
#             elapsed    = time.time() - start_time
#             fps_actual = processed_frames / elapsed
#             remaining  = (total_frames // PROCESS_EVERY_N_FRAMES) - processed_frames
#             eta_min    = (remaining / fps_actual / 60) if fps_actual > 0 else 0
#             progress   = (frame_index * 100) // total_frames
#             print(f"  [{progress:3d}%] Frame {frame_index}/{total_frames} | "
#                   f"Speed: {fps_actual:.1f} fps | ETA: {eta_min:.1f} min | "
#                   f"Detections: {len(all_detections)}")
#             print_gpu_memory()

#     # ── Flush remaining active tracks ────────────────────────────────
#     for tid, track in track_history.items():
#         if track['best_snapshot'] is None:
#             continue
#         dtype      = track['type']
#         crop_path  = os.path.join(
#             defects_dir, dtype,
#             f"{dtype}_crop_{det_id_counter}.jpg")
#         frame_path = os.path.join(
#             frames_dir,
#             f"{dtype}_frame_{det_id_counter}.jpg")
#         image_saver.save(crop_path,  track['best_snapshot']['crop'])
#         image_saver.save(frame_path, track['best_snapshot']['full_frame'])

#         all_detections.append({
#             'id':               det_id_counter,
#             'track_id':         tid,
#             'defect_type':      dtype,
#             'video_name':       video_name,
#             'srt_name':         srt_name,
#             'frame_start':      track['first_frame'],
#             'frame_end':        track['last_frame'],
#             'timestamp_start':  track['first_timestamp'].isoformat(),
#             'timestamp_end':    track['last_timestamp'].isoformat(),
#             'chainage_start_m': track['first_chainage'],
#             'chainage_end_m':   track['last_chainage'],
#             'chainage_avg_m':   (track['first_chainage'] +
#                                   track['last_chainage']) / 2,
#             'gps': {
#                 'latitude':  track['gps_lat'],
#                 'longitude': track['gps_lon'],
#             },
#             'polygon': track['best_snapshot']['polygon'],
#             'images': {
#                 'crop':  os.path.relpath(crop_path,  video_out_dir),
#                 'frame': os.path.relpath(frame_path, video_out_dir),
#             },
#         })
#         det_id_counter += 1

#     print("💾 Flushing image save queue...")
#     image_saver.shutdown()
#     reader.release()
#     out.release()

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     # ── Summary & JSON ───────────────────────────────────────────────
#     summary = {}
#     for d in all_detections:
#         summary.setdefault(d['defect_type'], {'count': 0})['count'] += 1

#     detection_data = {
#         'video_name':        video_name,
#         'srt_name':          srt_name,
#         'processing_date':   datetime.now().isoformat(),
#         'model':             'RFDETRSegMedium',
#         'total_frames':      total_frames,
#         'processed_frames':  processed_frames,
#         'output_resolution': f"{out_w}x{out_h}",
#         'starting_chainage_m': starting_chainage_m,
#         'ending_chainage_m':   ending_chainage,
#         'total_detections':  len(all_detections),
#         'summary':           summary,
#         'detections':        all_detections,
#     }

#     json_path = os.path.join(
#         video_out_dir, f"{video_basename}_detections.json")
#     with open(json_path, 'w') as f:
#         json.dump(detection_data, f, indent=2)

#     elapsed_total = time.time() - start_time
#     print(f"\n✅ Done! {elapsed_total/60:.1f} min | {len(all_detections)} detections")
#     print(f"   📹 {output_video_path}")
#     print(f"   📊 {json_path}")

#     return json_path, ending_chainage

# # ============================
# # MAIN
# # ============================
# def main():
#     print("\n" + "="*70)
#     print("ROAD ASSESSMENT — RFDETRSegMedium + Batched Sliced Inference")
#     print("="*70)

#     device = check_gpu()

#     active_modes = sum([SINGLE_VIDEO_MODE, MULTI_VIDEO_MODE, FOLDER_MODE])
#     if active_modes != 1:
#         print("❌ Set exactly ONE of SINGLE_VIDEO_MODE / MULTI_VIDEO_MODE / FOLDER_MODE to True")
#         return

#     pairs = []
#     if SINGLE_VIDEO_MODE:
#         pairs = [{"video": VIDEO_PATH, "srt": SRT_PATH}]
#     elif MULTI_VIDEO_MODE:
#         pairs = [p for p in VIDEO_SRT_PAIRS
#                  if os.path.exists(p['video']) and os.path.exists(p['srt'])]
#     elif FOLDER_MODE:
#         pairs = find_video_srt_pairs(INPUT_FOLDER)

#     if not pairs:
#         print("❌ No valid video-SRT pairs found!")
#         return

#     if not os.path.exists(SEGMENTATION_MODEL_PATH):
#         print(f"❌ Checkpoint not found: {SEGMENTATION_MODEL_PATH}")
#         return

#     # ── Load RF-DETR ─────────────────────────────────────────────────
#     rfdetr_device = "cuda" if device == "cuda" else "cpu"
#     print(f"\n🤖 Loading RFDETRSegMedium...")
#     print(f"   Checkpoint : {SEGMENTATION_MODEL_PATH}")
#     print(f"   Device     : {rfdetr_device}")
#     print(f"   Tile size  : {SLICE_WIDTH}x{SLICE_HEIGHT}")
#     print(f"   Classes    : {len(CLASS_NAMES)}")
#     print(f"   Output res : {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")

#     rfdetr_model = RFDETRSegMedium(
#         pretrain_weights=SEGMENTATION_MODEL_PATH,
#         device=rfdetr_device,
#         image_size=SLICE_WIDTH,
#         max_image_size=SLICE_WIDTH
#     )

#     # optimize_for_inference() is intentionally skipped —
#     # it locks the model to a fixed batch size at compile time,
#     # but our tile count per frame is dynamic (varies with video resolution).
#     # Batching all tiles per frame in a single predict() call already
#     # gives significant GPU speedup without needing FP16 compile.

#     warmup_model(rfdetr_model, device)

#     # ── Process videos ───────────────────────────────────────────────
#     print(f"\n{'='*70}")
#     print(f"PROCESSING {len(pairs)} VIDEO(S)")
#     print(f"{'='*70}")

#     all_json_files      = []
#     cumulative_chainage = 0

#     for idx, pair in enumerate(pairs, 1):
#         print(f"\n[VIDEO {idx}/{len(pairs)}]")
#         json_path, ending_chainage = process_video(
#             pair['video'], pair['srt'],
#             rfdetr_model, OUTPUT_BASE_DIR,
#             device,
#             starting_chainage_m=cumulative_chainage
#         )
#         if json_path:
#             all_json_files.append(json_path)
#         cumulative_chainage = ending_chainage

#     print(f"\n{'='*70}")
#     print(f"✅ ALL DONE!")
#     print(f"   Videos processed : {len(all_json_files)}")
#     print(f"   Total chainage   : {cumulative_chainage/1000:.2f} km")
#     print(f"   Output           : {OUTPUT_BASE_DIR}")
#     print(f"{'='*70}\n")

# if __name__ == "__main__":
#     main()








# ============================
# ROAD ASSESSMENT SCRIPT - RF-DETR SEG MEDIUM + CUSTOM SLICING
# Inference backbone: RFDETRSegMedium (no SAHI wrapper — direct inference)
# Changes from v1:
#   1. Fixed class name alignment using index-aligned COCO JSON approach
#   2. Batch tile inference (all tiles per frame in single model.predict() call)
#   3. Output video resized to 1920x1080 (configurable)
#   4. Class-specific confidence thresholds
#   5. Supervision-based visualization (better text rendering and mask opacity)
# ============================
import os
import cv2
import math
import numpy as np
import re
import torch
import torchvision
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

VIDEO_PATH = "/media/user/New Volume/Sakshi/chattishgarh/DJI_20260206133817_0731_D.MP4"
SRT_PATH   = "/media/user/New Volume/Sakshi/chattishgarh/DJI_20260206133817_0731_D.SRT"

VIDEO_SRT_PAIRS = [
    {
        "video": "/content/drive/MyDrive/videos/DJI_0001.MP4",
        "srt":   "/content/drive/MyDrive/videos/DJI_0001.SRT"
    }
]

INPUT_FOLDER          = "/content/drive/MyDrive/videos"
SEGMENTATION_MODEL_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
OUTPUT_BASE_DIR       = "/media/user/New Volume/Sakshi/chattishgarh/output"
COCO_JSON_PATH        = r"/media/user/New Volume/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ============================
# PROCESSING SETTINGS
# ============================
PROCESS_EVERY_N_FRAMES = 5

# Tile size — must match image_size=800 used during training
SLICE_HEIGHT          = 800
SLICE_WIDTH           = 800
OVERLAP_HEIGHT_RATIO  = 0.10
OVERLAP_WIDTH_RATIO   = 0.10

# Output video resolution (set to None to keep original resolution)
OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080

# Mask opacity for segmentation overlay (0.0 = transparent, 1.0 = opaque)
MASK_OPACITY = 0.35

NMS_IOU_THRESHOLD   = 0.50
IOU_THRESHOLD       = 0.30    # tracker
MAX_DISTANCE        = 50
MAX_LOST            = 30
FRAME_BUFFER_SIZE   = 16
SAVE_WORKER_THREADS = 4

# ============================
# CLASS-SPECIFIC THRESHOLDS
# ============================
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
    "parallel_crack":             0.30,
}
GLOBAL_THRESHOLD = 0.30

# ============================
# IGNORED CLASSES
# completely skipped — not annotated, not saved
# ============================
IGNORED_CLASSES = {
    "white_mark", "water_mark", "doubt", "bump",
    "guard_post", "overhead_sign_board", "sign_board", "kerb_damage"
}

# ============================
# CLASS NAMES & COLORS
# index-aligned from COCO JSON — fixes class name misalignment
# ============================
with open(COCO_JSON_PATH, "r") as f:
    coco_data = json.load(f)

# Sort categories by ID to match model's 0-indexed output order
categories  = sorted(coco_data["categories"], key=lambda x: x["id"])
CLASS_NAMES = [cat["name"] for cat in categories]

print("Loaded Classes:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i:2d}: {name}")

# Index-aligned HEX colors — position must match CLASS_NAMES order from COCO JSON
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

# Pad with fallback white if COCO has more classes than colors defined
while len(HEX_COLORS) < len(CLASS_NAMES):
    HEX_COLORS.append("#FFFFFF")

def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)

# name → BGR color dict (index-aligned)
color_map = {
    CLASS_NAMES[i]: hex_to_bgr(HEX_COLORS[i])
    for i in range(len(CLASS_NAMES))
}

# ============================
# GPU UTILITIES
# ============================
def check_gpu():
    if torch.cuda.is_available():
        device    = 'cuda'
        gpu_name  = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"   VRAM: {total_mem:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled   = True
        return device
    else:
        print("⚠️ No GPU found — running on CPU (slow)")
        return 'cpu'

def print_gpu_memory():
    if torch.cuda.is_available():
        used   = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved()   / 1e9
        print(f"   🖥️ GPU Memory: {used:.2f}GB used / {cached:.2f}GB cached")

def warmup_model(model, device):
    print("🔥 Warming up GPU (RF-DETR)...")
    # Single tile warmup — batch size is dynamic at inference time
    # so we just warm up the GPU pipeline with one tile
    dummy = [np.random.randint(0, 255, (SLICE_HEIGHT, SLICE_WIDTH, 3), dtype=np.uint8)]
    with torch.no_grad():
        model.predict(dummy, threshold=GLOBAL_THRESHOLD)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.synchronize()
    print("✅ GPU warmed up")

# ============================
# BATCH SLICED INFERENCE
# All tiles from a single frame are collected first,
# then passed to model.predict() in ONE call — faster GPU utilisation
# ============================
def sliced_predict_batch(frame_bgr, model, slice_h, slice_w,
                         overlap_h, overlap_w):
    img_h, img_w = frame_bgr.shape[:2]
    step_h = max(1, int(slice_h * (1 - overlap_h)))
    step_w = max(1, int(slice_w * (1 - overlap_w)))

    tiles        = []   # list of cropped numpy arrays
    tile_origins = []   # (x1, y1) offset for each tile

    # ── Collect all tiles ────────────────────────────────────────────
    y = 0
    while True:
        y2 = min(y + slice_h, img_h)
        y1 = max(0, y2 - slice_h)
        x = 0
        while True:
            x2 = min(x + slice_w, img_w)
            x1 = max(0, x2 - slice_w)
            tile = frame_bgr[y1:y2, x1:x2]
            tiles.append(tile)
            tile_origins.append((x1, y1))
            if x2 >= img_w:
                break
            x += step_w
        if y2 >= img_h:
            break
        y += step_h

    if not tiles:
        return []

    # ── Single batched model call ────────────────────────────────────
    with torch.no_grad():
        batch_dets = model.predict(tiles, threshold=GLOBAL_THRESHOLD)

    # batch_dets is a list — one Detections object per tile
    # (if rfdetr returns a single merged object for a batch,
    #  handle both cases below)
    if not isinstance(batch_dets, (list, tuple)):
        batch_dets = [batch_dets]

    # ── Unpack detections & project back to full-frame coords ────────
    all_boxes   = []
    all_scores  = []
    all_classes = []
    all_masks   = []

    for tile_idx, dets in enumerate(batch_dets):
        if dets is None or len(dets) == 0:
            continue

        x1_off, y1_off = tile_origins[tile_idx]
        tile            = tiles[tile_idx]
        tile_h, tile_w  = tile.shape[:2]

        for i in range(len(dets)):
            class_id   = int(dets.class_id[i])
            confidence = float(dets.confidence[i])

            # Safety guard
            if class_id >= len(CLASS_NAMES):
                continue

            class_name = CLASS_NAMES[class_id]

            # Skip ignored classes entirely
            if class_name in IGNORED_CLASSES:
                continue

            # Apply per-class threshold (fall back to global)
            threshold = CLASS_THRESHOLDS.get(class_name, GLOBAL_THRESHOLD)
            if confidence < threshold:
                continue

            bx1, by1, bx2, by2 = dets.xyxy[i]

            # Shift bbox to full-frame coordinates
            all_boxes.append([
                float(bx1) + x1_off,
                float(by1) + y1_off,
                float(bx2) + x1_off,
                float(by2) + y1_off,
            ])
            all_scores.append(confidence)
            all_classes.append(class_id)

            # Handle segmentation mask
            if hasattr(dets, "mask") and dets.mask is not None:
                tile_mask = dets.mask[i].astype(np.uint8)
                if tile_mask.shape != (tile_h, tile_w):
                    tile_mask = cv2.resize(
                        tile_mask, (tile_w, tile_h),
                        interpolation=cv2.INTER_NEAREST
                    )
                full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                full_mask[y1_off:y1_off + tile_h,
                          x1_off:x1_off + tile_w] = tile_mask
                all_masks.append(full_mask.astype(bool))
            else:
                all_masks.append(None)

    if not all_boxes:
        return []

    # ── Global NMS across all tiles ──────────────────────────────────
    boxes_t  = torch.tensor(all_boxes,  dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep     = torchvision.ops.nms(boxes_t, scores_t, NMS_IOU_THRESHOLD).tolist()

    results = []
    for idx in keep:
        class_id   = all_classes[idx]
        if class_id >= len(CLASS_NAMES):
            continue
        mask    = all_masks[idx]
        polygon = None

        if mask is not None:
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

        results.append({
            'bbox':       all_boxes[idx],
            'mask':       mask,
            'polygon':    polygon,
            'class':      CLASS_NAMES[class_id],
            'confidence': all_scores[idx],
        })

    return results

# ============================
# THREADED FRAME READER
# ============================
class ThreadedVideoReader:
    def __init__(self, video_path, skip_n=PROCESS_EVERY_N_FRAMES,
                 buffer_size=FRAME_BUFFER_SIZE):
        self.cap         = cv2.VideoCapture(video_path)
        self.skip_n      = skip_n
        self.buffer      = queue.Queue(maxsize=buffer_size)
        self.stopped     = False
        self.width       = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height      = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps         = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames= int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.thread      = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        local_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.buffer.put(None)
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
    srt_data = []
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
            tc = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
            if not tc:
                continue
            h, m, s, ms  = map(int, tc.groups())
            timestamp_ms = (h * 3600 + m * 60 + s) * 1000 + ms
            meta         = ' '.join(lines[2:])

            ts_match = re.search(
                r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', meta)
            if not ts_match:
                continue
            absolute_timestamp = datetime.strptime(
                ts_match.group(1), '%Y-%m-%d %H:%M:%S.%f')

            lat = re.search(r'\[latitude:\s*([-\d.]+)\]',  meta)
            lon = re.search(r'\[longitude:\s*([-\d.]+)\]', meta)
            alt = re.search(r'\[(?:abs_alt|altitude):\s*([-\d.]+)', meta)

            if not (lat and lon):
                continue

            srt_data.append({
                'frame_number':      frame_num,
                'timestamp_ms':      timestamp_ms,
                'absolute_timestamp':absolute_timestamp,
                'latitude':          float(lat.group(1)),
                'longitude':         float(lon.group(1)),
                'altitude':          float(alt.group(1)) if alt else 50.0,
            })
        except Exception:
            continue

    if not srt_data:
        print("❌ No valid GPS data in SRT file!")
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

        assigned      = set()
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
# VISUALIZATION
# ============================
def tracked_objects_to_sv_detections(tracked_objects, frame_h, frame_w):
    """
    Convert tracked objects (dicts) to supervision Detections format.
    Returns sv.Detections and labels list.
    """
    if not tracked_objects:
        return sv.Detections.empty(), []
    
    xyxy_list = []
    mask_list = []
    confidence_list = []
    class_id_list = []
    track_id_list = []
    labels = []
    
    for obj in tracked_objects:
        cls_name = obj['class']
        
        # Skip ignored classes
        if cls_name in IGNORED_CLASSES:
            continue
            
        # Get class_id from CLASS_NAMES
        try:
            class_id = CLASS_NAMES.index(cls_name)
        except ValueError:
            continue
            
        xyxy_list.append(obj['bbox'])
        
        # Handle mask
        mask = obj.get('mask')
        if mask is not None:
            if mask.shape != (frame_h, frame_w):
                mask = cv2.resize(
                    mask.astype(np.uint8), 
                    (frame_w, frame_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            mask_list.append(mask)
        else:
            # Create empty mask if none exists
            mask_list.append(np.zeros((frame_h, frame_w), dtype=bool))
        
        confidence_list.append(obj.get('confidence', 1.0))
        class_id_list.append(class_id)
        track_id_list.append(obj['track_id'])
        
        # Create label with class name and track ID
        labels.append(f"{cls_name} ID:{obj['track_id']}")
    
    if not xyxy_list:
        return sv.Detections.empty(), []
    
    # Create supervision Detections object
    detections = sv.Detections(
        xyxy=np.array(xyxy_list, dtype=np.float32),
        mask=np.array(mask_list, dtype=bool) if mask_list else None,
        confidence=np.array(confidence_list, dtype=np.float32),
        class_id=np.array(class_id_list, dtype=int),
        tracker_id=np.array(track_id_list, dtype=int)
    )
    
    return detections, labels


def draw_legend(frame, width, height):
    # Only draw legend for non-ignored classes
    visible_classes = {
        name: color for name, color in color_map.items()
        if name not in IGNORED_CLASSES
    }
    scale    = max(0.6, min(width / 1920.0, 1.2))
    lx       = int(0.02 * width)
    ly       = int(0.04 * height)
    line_h   = int(22 * scale)
    line_len = int(35 * scale)
    leg_w    = int(300 * scale)
    txt_scale= 0.6 * scale
    txt_thick= max(1, int(1.5 * scale))

    cv2.rectangle(
        frame,
        (lx - 12, ly - 12),
        (lx + leg_w, ly + line_h * len(visible_classes) + 12),
        (0, 0, 0),
        max(2, int(2 * scale))
    )
    for idx, (cls_name, color) in enumerate(visible_classes.items()):
        y = ly + idx * line_h
        cv2.line(frame, (lx, y + 10), (lx + line_len, y + 10), color, 2)
        cv2.putText(frame, cls_name,
                    (lx + line_len + 12, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, txt_scale,
                    (255, 255, 255), txt_thick)
    scale    = max(0.6, min(width / 1920.0, 1.2))
    lx       = int(0.02 * width)
    ly       = int(0.04 * height)
    line_h   = int(22 * scale)
    line_len = int(35 * scale)
    leg_w    = int(300 * scale)
    txt_scale= 0.6 * scale
    txt_thick= max(1, int(1.5 * scale))

    cv2.rectangle(
        frame,
        (lx - 12, ly - 12),
        (lx + leg_w, ly + line_h * len(visible_classes) + 12),
        (0, 0, 0),
        max(2, int(2 * scale))
    )
    for idx, (cls_name, color) in enumerate(visible_classes.items()):
        y = ly + idx * line_h
        cv2.line(frame, (lx, y + 10), (lx + line_len, y + 10), color, 2)
        cv2.putText(frame, cls_name,
                    (lx + line_len + 12, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, txt_scale,
                    (255, 255, 255), txt_thick)

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

    print(f"  Found {len(video_files)} video(s), {len(srt_files)} SRT(s)")
    pairs = []
    for vp in sorted(video_files):
        base = os.path.splitext(os.path.basename(vp))[0]
        srt  = next(
            (s for s in srt_files
             if os.path.splitext(os.path.basename(s))[0] == base),
            None
        )
        if srt:
            pairs.append({"video": vp, "srt": srt})
            print(f"  ✅ {os.path.basename(vp)} ↔ {os.path.basename(srt)}")
        else:
            print(f"  ⚠️ No SRT: {os.path.basename(vp)}")
    return pairs

# ============================
# MAIN PROCESSING FUNCTION
# ============================
def process_video(video_path, srt_path, rfdetr_model, output_dir,
                  device, starting_chainage_m=0):
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
    reader       = ThreadedVideoReader(video_path, skip_n=PROCESS_EVERY_N_FRAMES)
    width        = reader.width
    height       = reader.height
    fps          = reader.fps
    total_frames = reader.total_frames

    # ── Output resolution ────────────────────────────────────────────
    if OUTPUT_WIDTH and OUTPUT_HEIGHT:
        out_w = OUTPUT_WIDTH
        out_h = OUTPUT_HEIGHT
    else:
        out_w = width
        out_h = height

    scale_x = out_w / width
    scale_y = out_h / height

    print(f"  Source  : {width}x{height} @ {fps:.1f}fps | {total_frames} total frames")
    print(f"  Output  : {out_w}x{out_h}")
    print(f"  Processing every {PROCESS_EVERY_N_FRAMES} frames → "
          f"~{total_frames // PROCESS_EVERY_N_FRAMES} frames")
    print(f"  Tile size: {SLICE_WIDTH}x{SLICE_HEIGHT} | "
          f"Overlap: {int(OVERLAP_WIDTH_RATIO*100)}%")
    print_gpu_memory()

    # ── Output directories ───────────────────────────────────────────
    video_out_dir = os.path.join(output_dir, video_basename)
    defects_dir   = os.path.join(video_out_dir, "defect_images")
    frames_dir    = os.path.join(defects_dir,   "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for cls in CLASS_NAMES:
        if cls not in IGNORED_CLASSES:
            os.makedirs(os.path.join(defects_dir, cls), exist_ok=True)

    output_video_path = os.path.join(
        video_out_dir, f"{video_basename}_output.mp4")
    output_fps = max(1.0, fps / PROCESS_EVERY_N_FRAMES)
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        output_fps,
        (out_w, out_h)
    )

    tracker       = SegmentationTracker(IOU_THRESHOLD, MAX_DISTANCE, MAX_LOST)
    image_saver   = AsyncImageSaver(num_workers=SAVE_WORKER_THREADS)
    track_history = {}
    all_detections= []
    det_id_counter= 1
    processed_frames = 0
    start_time    = time.time()

    # ── Setup supervision annotators ─────────────────────────────────────
    color_palette   = sv.ColorPalette.from_hex(HEX_COLORS)
    text_scale      = sv.calculate_optimal_text_scale(
                          resolution_wh=(out_w, out_h))
    line_thickness  = sv.calculate_optimal_line_thickness(
                          resolution_wh=(out_w, out_h))

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

    print(f"\n🚀 Starting inference (batched tile processing)...")

    while True:
        item = reader.read()
        if item is None:
            break

        frame_index, frame = item
        processed_frames += 1

        srt_entry = get_srt_data_for_frame(frame_index + 1, srt_data)
        if srt_entry is None:
            continue

        chainage_m = srt_entry['cumulative_chainage_m']

        # ── Batched sliced RF-DETR inference ─────────────────────────
        frame_detections = sliced_predict_batch(
            frame, rfdetr_model,
            slice_h=SLICE_HEIGHT, slice_w=SLICE_WIDTH,
            overlap_h=OVERLAP_HEIGHT_RATIO, overlap_w=OVERLAP_WIDTH_RATIO
        )

        # ── Track ─────────────────────────────────────────────────────
        tracked_objects = tracker.update(frame_detections)

        # ── Convert tracked objects to supervision format ─────────────
        sv_detections, labels = tracked_objects_to_sv_detections(
            tracked_objects, height, width
        )

        # ── Draw with supervision annotators (mask → bbox → label) ───
        annotated = frame.copy()
        annotated = mask_annotator.annotate(
            scene=annotated, detections=sv_detections
        )
        annotated = bbox_annotator.annotate(
            scene=annotated, detections=sv_detections
        )
        annotated = label_annotator.annotate(
            scene=annotated, detections=sv_detections, labels=labels
        )
        draw_legend(annotated, width, height)

        # ── Resize annotated frame to output resolution ───────────────
        if (out_w, out_h) != (width, height):
            annotated_out = cv2.resize(
                annotated, (out_w, out_h),
                interpolation=cv2.INTER_LINEAR
            )
        else:
            annotated_out = annotated

        # ── Update track history ──────────────────────────────────────
        for tr in tracked_objects:
            tid      = tr['track_id']
            bbox     = tr['bbox']
            cls_name = tr['class']
            polygon  = tr['polygon']

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
                crop = frame[
                    max(0, y1):min(height, y2),
                    max(0, x1):min(width,  x2)
                ].copy()
                track_history[tid]['best_snapshot'] = {
                    'frame_idx':  frame_index + 1,
                    'crop':       crop,
                    'full_frame': annotated_out.copy(),  # save at output res
                    'polygon':    polygon,
                }

        out.write(annotated_out)

        # ── Finalise lost tracks ──────────────────────────────────────
        active_tids = {t['track_id'] for t in tracked_objects}
        for tid in list(track_history.keys()):
            if tid not in active_tids and tid not in tracker.tracks:
                track = track_history.pop(tid)
                if track['best_snapshot'] is None:
                    continue
                dtype      = track['type']
                crop_path  = os.path.join(
                    defects_dir, dtype,
                    f"{dtype}_crop_{det_id_counter}.jpg")
                frame_path = os.path.join(
                    frames_dir,
                    f"{dtype}_frame_{det_id_counter}.jpg")
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

        # ── Progress ──────────────────────────────────────────────────
        if processed_frames % 50 == 0:
            elapsed    = time.time() - start_time
            fps_actual = processed_frames / elapsed
            remaining  = (total_frames // PROCESS_EVERY_N_FRAMES) - processed_frames
            eta_min    = (remaining / fps_actual / 60) if fps_actual > 0 else 0
            progress   = (frame_index * 100) // total_frames
            print(f"  [{progress:3d}%] Frame {frame_index}/{total_frames} | "
                  f"Speed: {fps_actual:.1f} fps | ETA: {eta_min:.1f} min | "
                  f"Detections: {len(all_detections)}")
            print_gpu_memory()

    # ── Flush remaining active tracks ────────────────────────────────
    for tid, track in track_history.items():
        if track['best_snapshot'] is None:
            continue
        dtype      = track['type']
        crop_path  = os.path.join(
            defects_dir, dtype,
            f"{dtype}_crop_{det_id_counter}.jpg")
        frame_path = os.path.join(
            frames_dir,
            f"{dtype}_frame_{det_id_counter}.jpg")
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

    # ── Summary & JSON ───────────────────────────────────────────────
    summary = {}
    for d in all_detections:
        summary.setdefault(d['defect_type'], {'count': 0})['count'] += 1

    detection_data = {
        'video_name':        video_name,
        'srt_name':          srt_name,
        'processing_date':   datetime.now().isoformat(),
        'model':             'RFDETRSegMedium',
        'total_frames':      total_frames,
        'processed_frames':  processed_frames,
        'output_resolution': f"{out_w}x{out_h}",
        'starting_chainage_m': starting_chainage_m,
        'ending_chainage_m':   ending_chainage,
        'total_detections':  len(all_detections),
        'summary':           summary,
        'detections':        all_detections,
    }

    json_path = os.path.join(
        video_out_dir, f"{video_basename}_detections.json")
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
    print("ROAD ASSESSMENT — RFDETRSegMedium + Batched Sliced Inference")
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

    # ── Load RF-DETR ─────────────────────────────────────────────────
    rfdetr_device = "cuda" if device == "cuda" else "cpu"
    print(f"\n🤖 Loading RFDETRSegMedium...")
    print(f"   Checkpoint : {SEGMENTATION_MODEL_PATH}")
    print(f"   Device     : {rfdetr_device}")
    print(f"   Tile size  : {SLICE_WIDTH}x{SLICE_HEIGHT}")
    print(f"   Classes    : {len(CLASS_NAMES)}")
    print(f"   Output res : {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}")

    rfdetr_model = RFDETRSegMedium(
        pretrain_weights=SEGMENTATION_MODEL_PATH,
        device=rfdetr_device,
        image_size=SLICE_WIDTH,
        max_image_size=SLICE_WIDTH
    )

    # optimize_for_inference() is intentionally skipped —
    # it locks the model to a fixed batch size at compile time,
    # but our tile count per frame is dynamic (varies with video resolution).
    # Batching all tiles per frame in a single predict() call already
    # gives significant GPU speedup without needing FP16 compile.

    warmup_model(rfdetr_model, device)

    # ── Process videos ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PROCESSING {len(pairs)} VIDEO(S)")
    print(f"{'='*70}")

    all_json_files      = []
    cumulative_chainage = 0

    for idx, pair in enumerate(pairs, 1):
        print(f"\n[VIDEO {idx}/{len(pairs)}]")
        json_path, ending_chainage = process_video(
            pair['video'], pair['srt'],
            rfdetr_model, OUTPUT_BASE_DIR,
            device,
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