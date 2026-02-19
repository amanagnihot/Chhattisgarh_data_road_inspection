# ============================================================================================
# RF-DETR ROAD DEFECT DETECTION - ULTRA-FAST (NO MASK IOU TRACKER)
# ============================================================================================
# Maximum Speed Optimizations:
#   ✅ BBOX-only tracker (NO expensive mask IOU calculations)
#   ✅ Aggressive frame skipping (every 5 frames)
#   ✅ Large batch size (16 frames)
#   ✅ FP16 inference
#   ✅ Minimal buffer (15 frames)
#   ✅ Fast confirmation (2 hits)
#   ✅ Threaded prefetching
#   ✅ Async image saving
#   Expected: 5-10x faster than mask-based tracker!
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
# CONFIGURATION
# ============================================================================================
VIDEO_PATH      = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.MP4"
SRT_PATH        = r"/media/user/New Volume/Sakshi/chattishgarh/DJI_20260110101923_0001_D.SRT"
CHECKPOINT_PATH = r"/media/user/New Volume/Sakshi/chattishgarh/new_output/rfdetr_medium_100/checkpoint_best_total.pth"
COCO_JSON_PATH  = r"/media/user/New Volume/Sakshi/chattishgarh/Road_inspection-8/train/_annotations.coco.json"
OUTPUT_BASE_DIR = r"/media/user/New Volume/Sakshi/chattishgarh/output_ultrafast"

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# ============================================================================================
# ULTRA-FAST SETTINGS
# ============================================================================================

NUM_QUERIES  = 200
IMAGE_SIZE   = 432
OUT_WIDTH    = 1280
OUT_HEIGHT   = 720

PROCESS_EVERY_N_FRAMES = 5
BATCH_SIZE             = 16
MASK_OPACITY           = 0.15

CLASS_THRESHOLDS = {
    "Patch": 0.50, "pothole": 0.35, "Cracking": 0.30, "Ravelling": 0.50,
    "Edge_breaking": 0.35, "Edge_drop": 0.35, "MBCB_defect": 0.30,
    "MBCB_missing": 0.30, "Corrugations_and_shoving": 0.30, "Scaling": 0.30,
    "Wear": 0.40, "honeycomb": 0.30, "depression": 0.30,
    "Embankment slope": 0.30, "strip_seal_expansion_join": 0.30,
    "tyre_marks": 0.35, "vegetation_on_road": 0.30, "km_stone": 0.40, "cattle": 0.45,
}

GLOBAL_THRESHOLD = 0.30
NMS_THRESHOLD = 0.65

IGNORED_CLASSES = {
    "white_mark", "water_mark", "doubt", "bump",
    "guard_post", "overhead_sign_board", "sign_board", "kerb_damage"
}

# ── Ultra-fast tracking ──
IOU_THRESHOLD = 0.3
MAX_DISTANCE = 50
MAX_LOST = 15
CONFIRMATION_HITS = 2
BUFFER_SIZE = 15  # Minimal buffer

POTHOLE_PATCHING_IOU_THRESHOLD = 0.90

PREFETCH_QUEUE_SIZE = 64
SAVE_WORKER_THREADS = 8
GPU_CLEAR_EVERY = 100
CROP_SIZE = (400, 400)
FRAME_SIZE = (800, 600)

ENABLE_TIMING = True

# ============================================================================================
# LOAD CLASSES
# ============================================================================================

print("=" * 80)
print("ULTRA-FAST MODE (BBOX-ONLY TRACKER)")
print("=" * 80)

with open(COCO_JSON_PATH, "r") as f:
    coco_data = json.load(f)

categories = sorted(coco_data["categories"], key=lambda x: x["id"])
CLASS_NAMES = [cat["name"] for cat in categories]

print(f"✅ {len(CLASS_NAMES)} classes | Skip={PROCESS_EVERY_N_FRAMES} | Batch={BATCH_SIZE}")

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
# GPU
# ============================================================================================

def check_gpu():
    if torch.cuda.is_available():
        device = 'cuda:0'
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✅ GPU: {gpu_name} | {total_mem:.1f}GB VRAM")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return device
    else:
        print("\n⚠️ No GPU")
        return 'cpu'

def print_gpu_mem():
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1e9
        print(f"   GPU: {used:.2f}GB")

# ============================================================================================
# ASYNC SAVER
# ============================================================================================

class AsyncImageSaver:
    def __init__(self, workers=SAVE_WORKER_THREADS):
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.futures = []

    def save(self, path, image):
        self.futures.append(self.executor.submit(cv2.imwrite, path, image))

    def wait_all(self):
        for f in self.futures:
            f.result()
        self.futures.clear()

    def shutdown(self):
        self.wait_all()
        self.executor.shutdown(wait=True)

# ============================================================================================
# THREADED PREFETCHER
# ============================================================================================

class FramePrefetcher:
    def __init__(self, video_path, w, h, qsize=PREFETCH_QUEUE_SIZE):
        self.cap = cv2.VideoCapture(video_path)
        self.w, self.h = w, h
        self.q = queue.Queue(maxsize=qsize)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.q.put(None)
                break
            resized = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            self.q.put((idx, resized))
            idx += 1
        self.cap.release()

    def get(self):
        return self.q.get()

def get_video_props(vpath):
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, total, w, h

# ============================================================================================
# SRT & GPS
# ============================================================================================

def parse_srt(path):
    data = []
    if not os.path.exists(path):
        print(f"❌ No SRT: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = re.split(r'\n\n+', content.strip())
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        try:
            fnum = int(lines[0].strip())
            tc = re.search(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', lines[1])
            if not tc:
                continue
            h, m, s, ms = map(int, tc.groups())
            ts_ms = (h*3600 + m*60 + s)*1000 + ms
            meta = ' '.join(lines[2:])
            ts_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{3})', meta)
            if not ts_match:
                continue
            abs_ts = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S.%f')
            lat = re.search(r'\[latitude:\s*([-\d.]+)\]', meta)
            lon = re.search(r'\[longitude:\s*([-\d.]+)\]', meta)
            alt = re.search(r'\[(?:abs_alt|altitude):\s*([-\d.]+)', meta)
            if not (lat and lon):
                continue
            data.append({
                'frame_number': fnum,
                'timestamp_ms': ts_ms,
                'absolute_timestamp': abs_ts,
                'latitude': float(lat.group(1)),
                'longitude': float(lon.group(1)),
                'altitude': float(alt.group(1)) if alt else 50.0
            })
        except:
            continue
    if not data:
        print("❌ No GPS data")
        return None
    print(f"✅ {len(data)} SRT entries")
    return data

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2-lat1)
    dl = math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calc_chainage(data, start=0):
    dist = start
    for i, e in enumerate(data):
        if i == 0:
            e['cumulative_chainage_m'] = start
        else:
            dist += haversine(data[i-1]['latitude'], data[i-1]['longitude'],
                             e['latitude'], e['longitude'])
            e['cumulative_chainage_m'] = dist
    km = (dist - start) / 1000
    print(f"✅ Chainage: {start:.1f}m → {dist:.1f}m ({km:.3f}km)")
    return dist

def get_srt(fidx, data):
    for e in data:
        if e['frame_number'] == fidx:
            return e
    if data:
        return min(data, key=lambda x: abs(x['frame_number'] - fidx))
    return None

# ============================================================================================
# ULTRA-FAST BBOX-ONLY TRACKER (NO MASK IOU!)
# ============================================================================================

class FastBBoxTracker:
    """Lightning-fast tracker using ONLY bounding boxes. No mask IOU!"""
    
    def __init__(self):
        self.next_iid = 1
        self.next_did = 1
        self.active = {}
        self.confirmed = {}
        self.iid_to_did = {}

    def _iou(self, b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2])
        y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0:
            return 0.0
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0

    def _dist(self, b1, b2):
        c1x = (b1[0]+b1[2])/2; c1y = (b1[1]+b1[3])/2
        c2x = (b2[0]+b2[2])/2; c2y = (b2[1]+b2[3])/2
        return ((c1x-c2x)**2 + (c1y-c2y)**2)**0.5

    def update(self, dets, fidx):
        for t in list(self.active.values()) + list(self.confirmed.values()):
            t['lost'] += 1

        new_confirmed = []
        tracked = []
        matched = set()

        for det in dets:
            best_tid = None
            best_score = 0

            for pool in [self.confirmed, self.active]:
                for tid, trk in pool.items():
                    if trk['class'] != det['class']:
                        continue
                    
                    iou = self._iou(det['bbox'], trk['bbox'])
                    dist = self._dist(det['bbox'], trk['bbox'])
                    
                    if iou > IOU_THRESHOLD or dist < MAX_DISTANCE:
                        score = iou + (1.0 - min(dist/MAX_DISTANCE, 1.0))*0.5
                        if score > best_score:
                            best_score = score
                            best_tid = tid

            if best_tid:
                pool = self.confirmed if best_tid in self.confirmed else self.active
                trk = pool[best_tid]
                trk.update({
                    'bbox': det['bbox'], 'mask': det.get('mask'),
                    'polygon': det.get('polygon'), 'confidence': det['confidence'],
                    'last_frame': fidx, 'lost': 0
                })
                trk['hits'] += 1
                trk['confidences'].append(det['confidence'])
                matched.add(best_tid)

                if best_tid in self.active and trk['hits'] >= CONFIRMATION_HITS:
                    self.confirmed[best_tid] = trk
                    del self.active[best_tid]
                    self.iid_to_did[best_tid] = self.next_did
                    trk['display_id'] = self.next_did
                    self.next_did += 1
                    new_confirmed.append({
                        'internal_id': best_tid, 'display_id': trk['display_id'],
                        'bbox': trk['bbox'], 'mask': trk['mask'],
                        'polygon': trk['polygon'], 'class': trk['class'],
                        'confidence': trk['confidence']
                    })

                if best_tid in self.confirmed:
                    tracked.append({
                        'internal_id': best_tid, 'display_id': trk.get('display_id', -1),
                        'bbox': trk['bbox'], 'mask': trk['mask'],
                        'polygon': trk['polygon'], 'class': trk['class'],
                        'confidence': trk['confidence']
                    })
            else:
                nid = self.next_iid
                self.next_iid += 1
                self.active[nid] = {
                    'bbox': det['bbox'], 'mask': det.get('mask'),
                    'polygon': det.get('polygon'), 'class': det['class'],
                    'confidence': det['confidence'], 'confidences': [det['confidence']],
                    'hits': 1, 'lost': 0, 'first_frame': fidx, 'last_frame': fidx
                }

        for tid in list(self.active.keys()):
            if self.active[tid]['lost'] > MAX_LOST:
                del self.active[tid]
        for tid in list(self.confirmed.keys()):
            if self.confirmed[tid]['lost'] > MAX_LOST:
                del self.confirmed[tid]

        return tracked, new_confirmed

# ============================================================================================
# UTILS
# ============================================================================================

def calc_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter) if (a1+a2-inter) > 0 else 0.0

def resize_fixed(img, tsize):
    tw, th = tsize
    h, w = img.shape[:2]
    scale = min(tw/w, th/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    yo = (th-nh)//2; xo = (tw-nw)//2
    canvas[yo:yo+nh, xo:xo+nw] = resized
    return canvas

# ============================================================================================
# ROLLING BUFFER
# ============================================================================================

class RollingBuffer:
    def __init__(self, maxsize=BUFFER_SIZE):
        self.buf = deque(maxlen=maxsize)

    def push(self, fidx, frame, tracked):
        self.buf.append({'idx': fidx, 'frame': frame, 'tracked': tracked})
        return []  # No flushing, just accumulate

    def flush_all(self):
        flushed = list(self.buf)
        self.buf.clear()
        return flushed

    def retroactive(self, iid, did):
        for item in self.buf:
            for tr in item['tracked']:
                if tr['internal_id'] == iid:
                    tr['display_id'] = did

# ============================================================================================
# ANNOTATION
# ============================================================================================

def annotate(frame, tracked, m_ann, b_ann, l_ann, palette, classes):
    confirmed = [t for t in tracked if t.get('display_id', -1) > 0]
    if not confirmed:
        return frame

    xyxy = np.array([t['bbox'] for t in confirmed])
    cids = np.array([CLASS_NAMES.index(t['class']) for t in confirmed])
    confs = np.array([t['confidence'] for t in confirmed])

    masks = None
    if confirmed[0].get('mask') is not None:
        try:
            masks = np.array([t['mask'] for t in confirmed])
        except:
            pass

    det = sv.Detections(xyxy=xyxy, class_id=cids, confidence=confs, mask=masks)
    labels = [f"{t['display_id']}-{t['class']} {t['confidence']:.2f}" for t in confirmed]

    anno = frame.copy()
    if masks is not None:
        anno = m_ann.annotate(scene=anno, detections=det)
    anno = b_ann.annotate(scene=anno, detections=det)
    anno = l_ann.annotate(scene=anno, detections=det, labels=labels)

    if classes:
        draw_legend(anno, OUT_WIDTH, OUT_HEIGHT, classes)
    return anno

def draw_legend(frame, w, h, classes):
    s = max(0.6, min(w/1920, 1.2))
    lx, ly = int(0.02*w), int(0.04*h)
    ts = 0.6*s; tt = max(1, int(1.5*s))
    lh = int(22*s); ll = int(35*s); lw = int(300*s)
    cv2.rectangle(frame, (lx-12, ly-12), (lx+lw, ly+lh*len(classes)+12), (0,0,0), max(2, int(2*s)))
    for idx, (cn, ch) in enumerate(classes.items()):
        y = ly + idx*lh
        ch = ch.lstrip('#')
        r,g,b = tuple(int(ch[i:i+2], 16) for i in (0,2,4))
        cv2.line(frame, (lx, y+10), (lx+ll, y+10), (b,g,r), 2)
        cv2.putText(frame, cn, (lx+ll+12, y+12), cv2.FONT_HERSHEY_SIMPLEX, ts, (255,255,255), tt)

# ============================================================================================
# WORD REPORT
# ============================================================================================

def gen_report(dets, odir):
    doc = Document()
    doc.add_heading('Road Defects - RF-DETR (Ultra-Fast)', 0)
    sorted_dets = sorted(dets, key=lambda x: x['display_id'])
    by_cls = {}
    for d in sorted_dets:
        by_cls.setdefault(d['defect_type'], []).append(d)

    doc.add_heading('Summary', 1)
    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = 'Light Grid Accent 1'
    tbl.rows[0].cells[0].text = 'Type'
    tbl.rows[0].cells[1].text = 'Count'
    for c in sorted(by_cls):
        r = tbl.add_row().cells
        r[0].text = c
        r[1].text = str(len(by_cls[c]))
    doc.add_paragraph('')

    for c in sorted(by_cls):
        ds = by_cls[c]
        doc.add_heading(f'{c.upper()}', 1)
        doc.add_paragraph(f'Total: {len(ds)}')
        doc.add_paragraph('')
        tbl = doc.add_table(rows=1, cols=4)
        tbl.style = 'Light Grid Accent 1'
        h = tbl.rows[0].cells
        h[0].text = 'ID'; h[1].text = 'Chainage'; h[2].text = 'Crop'; h[3].text = 'Frame'
        for d in ds:
            r = tbl.add_row().cells
            r[0].text = f"{d['display_id']}-{c}"
            r[1].text = f"{d['chainage_avg_m']:.1f}"
            for ci, k, w in [(2,'crop',1.5), (3,'frame',2.5)]:
                p = d['images'][k]
                if os.path.exists(p):
                    try:
                        para = r[ci].paragraphs[0]
                        para.add_run().add_picture(p, width=Inches(w))
                        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    except:
                        r[ci].text = str(d['display_id'])
        doc.add_page_break()
    rp = os.path.join(odir, "Report.docx")
    doc.save(rp)
    print(f"\n📄 {rp}")

# ============================================================================================
# MAIN PROCESS
# ============================================================================================

def process(vpath, spath, model, odir, dev, start_m=0):
    vname = os.path.basename(vpath)
    sname = os.path.basename(spath)
    vbase = os.path.splitext(vname)[0]

    print(f"\n{'='*80}\n{vname}\n{'='*80}")

    sdata = parse_srt(spath)
    if not sdata:
        return None, start_m
    end_m = calc_chainage(sdata, start_m)

    fps, total, ow, oh = get_video_props(vpath)
    print(f"🎥 {ow}×{oh} @ {fps:.1f}fps | {total} frames")
    print(f"   Process: {OUT_WIDTH}×{OUT_HEIGHT} every {PROCESS_EVERY_N_FRAMES}")
    print_gpu_mem()

    vodir = os.path.join(odir, vbase)
    ddir = os.path.join(vodir, "defects")
    fdir = os.path.join(ddir, "frames")
    os.makedirs(fdir, exist_ok=True)
    for c in CLASS_NAMES:
        if c not in IGNORED_CLASSES:
            os.makedirs(os.path.join(ddir, c), exist_ok=True)

    ovpath = os.path.join(vodir, f"{vbase}_out.mp4")
    ofps = fps / PROCESS_EVERY_N_FRAMES

    ts = sv.calculate_optimal_text_scale(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
    tk = sv.calculate_optimal_line_thickness(resolution_wh=(OUT_WIDTH, OUT_HEIGHT))
    pal = sv.ColorPalette.from_hex(HEX_COLORS)
    m_ann = sv.MaskAnnotator(color=pal, opacity=MASK_OPACITY)
    b_ann = sv.BoxAnnotator(color=pal, thickness=tk)
    l_ann = sv.LabelAnnotator(color=pal, text_color=sv.Color.WHITE, text_scale=ts, text_thickness=max(1, tk-1))

    tracker = FastBBoxTracker()
    saver = AsyncImageSaver()
    buf = RollingBuffer()
    writer = cv2.VideoWriter(ovpath, cv2.VideoWriter_fourcc(*'mp4v'), ofps, (OUT_WIDTH, OUT_HEIGHT))

    confirmed = {}
    active_cls = {}
    t0 = time.time()

    prefetch = FramePrefetcher(vpath, OUT_WIDTH, OUT_HEIGHT)
    batch_f, batch_i = [], []
    all_idx = 0
    proc_cnt = 0

    t_inf, t_proc, t_write = 0, 0, 0
    samples = 0

    print(f"\n{'='*80}\nULTRA-FAST PROCESSING\n{'='*80}")

    def write_buf(items):
        tw = time.time()
        for it in items:
            anno = annotate(it['frame'], it['tracked'], m_ann, b_ann, l_ann, pal, active_cls)
            writer.write(anno)
        return time.time() - tw

    def run_batch():
        nonlocal proc_cnt, t_inf, t_proc, t_write, samples
        if not batch_f:
            return

        ti0 = time.time()
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(dev != 'cpu')):
                batch_dets = model.predict(batch_f, threshold=GLOBAL_THRESHOLD)
        ti1 = time.time()

        tp0 = time.time()
        tw_acc = 0

        for bi, dets in enumerate(batch_dets):
            fidx = batch_i[bi]
            frame = batch_f[bi]
            se = get_srt(fidx+1, sdata)
            if not se:
                buf.push(fidx, frame, [])
                continue

            ch = se['cumulative_chainage_m']
            dets = dets.with_nms(threshold=NMS_THRESHOLD)

            fdets = []
            if len(dets) > 0:
                for idx in range(len(dets)):
                    cid = dets.class_id[idx]
                    conf = dets.confidence[idx]
                    if cid < 0 or cid >= len(CLASS_NAMES):
                        continue
                    cn = CLASS_NAMES[cid]
                    if cn in IGNORED_CLASSES:
                        continue
                    if conf < CLASS_THRESHOLDS.get(cn, GLOBAL_THRESHOLD):
                        continue

                    bbox = dets.xyxy[idx]
                    mask, poly = None, None
                    if dets.mask is not None:
                        mask = dets.mask[idx].astype(np.uint8)
                        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if conts:
                            poly = conts[0].squeeze().tolist()
                            if isinstance(poly[0], (int, float)):
                                poly = [poly]
                    fdets.append({'bbox': bbox, 'mask': mask, 'polygon': poly, 'class': cn, 'confidence': float(conf)})

            # Pothole-patch removal
            pots = [i for i,d in enumerate(fdets) if d['class']=='pothole']
            pats = [i for i,d in enumerate(fdets) if d['class']=='Patch']
            rem = set()
            for pi in pots:
                for pai in pats:
                    if calc_iou(fdets[pi]['bbox'], fdets[pai]['bbox']) >= POTHOLE_PATCHING_IOU_THRESHOLD:
                        rem.add(pi)
                        break
            fdets = [d for i,d in enumerate(fdets) if i not in rem]

            tracked, new_conf = tracker.update(fdets, fidx)

            for tr in tracked:
                cn = tr['class']
                if cn not in active_cls:
                    active_cls[cn] = HEX_COLORS[CLASS_NAMES.index(cn)]

            for nc in new_conf:
                iid = nc['internal_id']
                did = nc['display_id']
                buf.retroactive(iid, did)

                x1,y1,x2,y2 = map(int, nc['bbox'])
                crop = frame[max(0,y1-30):min(OUT_HEIGHT,y2+30), max(0,x1-30):min(OUT_WIDTH,x2+30)]
                crop_r = resize_fixed(crop, CROP_SIZE)
                frame_r = resize_fixed(frame, FRAME_SIZE)

                cn = nc['class']
                cp = os.path.join(ddir, cn, f"{did}-{cn}.jpg")
                fp = os.path.join(fdir, f"{did}-{cn}_f.jpg")
                saver.save(cp, crop_r)
                saver.save(fp, frame_r)

                trk = tracker.confirmed[iid]
                confirmed[iid] = {
                    'display_id': did, 'internal_id': iid, 'defect_type': cn,
                    'video_name': vname, 'srt_name': sname,
                    'frame_start': trk['first_frame'], 'frame_end': trk['last_frame'],
                    'timestamp_start': se['absolute_timestamp'].isoformat(),
                    'timestamp_end': se['absolute_timestamp'].isoformat(),
                    'chainage_start_m': ch, 'chainage_end_m': ch, 'chainage_avg_m': ch,
                    'confidence_avg': float(np.mean(trk['confidences'])),
                    'gps': {'latitude': se['latitude'], 'longitude': se['longitude']},
                    'polygon': nc['polygon'],
                    'images': {'crop': cp, 'frame': fp}
                }

            buf.push(fidx, frame, tracked)
            proc_cnt += 1

        tp1 = time.time()
        t_inf += (ti1-ti0)
        t_proc += (tp1-tp0-tw_acc)
        samples += 1

        if proc_cnt % GPU_CLEAR_EVERY == 0 and dev != 'cpu':
            torch.cuda.empty_cache()

        batch_f.clear()
        batch_i.clear()

        elapsed = time.time() - t0
        fps_act = proc_cnt / elapsed if elapsed > 0 else 0

        if proc_cnt % 50 == 0:
            pct = (all_idx*100)//total if total > 0 else 0
            print(f"\n   [{pct:3d}%] {all_idx}/{total} | {fps_act:.1f}fps | Confirmed:{len(confirmed)}")
            if ENABLE_TIMING and samples > 0:
                print(f"   ⏱️  Inf:{(t_inf/samples)*1000:.0f}ms | Proc:{(t_proc/samples)*1000:.0f}ms")
            print_gpu_mem()

    while True:
        item = prefetch.get()
        if item is None:
            run_batch()
            tw = write_buf(buf.flush_all())
            t_write += tw
            break

        fidx, frame = item
        all_idx = fidx

        if fidx % PROCESS_EVERY_N_FRAMES == 0:
            batch_f.append(frame)
            batch_i.append(fidx)

        if len(batch_f) == BATCH_SIZE:
            run_batch()

    # Write buffered frames
    tw = write_buf(buf.flush_all())
    t_write += tw

    writer.release()
    print("\n💾 Saving images...")
    saver.shutdown()

    if dev != 'cpu':
        torch.cuda.empty_cache()

    # Update final metadata
    for iid, det in confirmed.items():
        if iid in tracker.confirmed:
            trk = tracker.confirmed[iid]
            lse = get_srt(trk['last_frame']+1, sdata)
            if lse:
                det['frame_end'] = trk['last_frame']
                det['timestamp_end'] = lse['absolute_timestamp'].isoformat()
                det['chainage_end_m'] = lse['cumulative_chainage_m']
                det['chainage_avg_m'] = (det['chainage_start_m'] + det['chainage_end_m'])/2

    all_dets = list(confirmed.values())

    jdata = {
        'video_name': vname, 'srt_name': sname,
        'date': datetime.now().isoformat(),
        'model': 'RF-DETR-UltraFast',
        'total_frames': total, 'processed': proc_cnt,
        'start_m': start_m, 'end_m': end_m,
        'distance_km': (end_m-start_m)/1000,
        'total_detections': len(all_dets),
        'detections': all_dets, 'summary': {}
    }

    for d in all_dets:
        dt = d['defect_type']
        e = jdata['summary'].setdefault(dt, {'count': 0, 'avg_conf': []})
        e['count'] += 1
        e['avg_conf'].append(d['confidence_avg'])

    for dt in jdata['summary']:
        jdata['summary'][dt]['avg_conf'] = float(np.mean(jdata['summary'][dt]['avg_conf']))

    jp = os.path.join(vodir, f"{vbase}.json")
    with open(jp, 'w') as f:
        json.dump(jdata, f, indent=2)

    print("\n📄 Generating report...")
    gen_report(all_dets, vodir)

    elapsed = time.time() - t0
    fps_act = proc_cnt / elapsed if elapsed > 0 else 0

    print(f"\n{'='*80}")
    print(f"✅ DONE in {elapsed/60:.1f}min ({elapsed:.1f}s)")
    print(f"   Speed: {fps_act:.1f}fps")
    print(f"   Detections: {len(all_dets)}")
    print(f"   📹 {ovpath}")
    print(f"   📊 {jp}")

    print(f"\n   Summary:")
    for dt, st in jdata['summary'].items():
        print(f"      {dt}: {st['count']} (conf:{st['avg_conf']:.2f})")

    return jp, end_m

# ============================================================================================
# MAIN
# ============================================================================================

def main():
    print("\n"+"="*80)
    print("RF-DETR ULTRA-FAST (BBOX-ONLY TRACKER)")
    print("="*80)
    print(f"⚡ Skip:{PROCESS_EVERY_N_FRAMES} | Batch:{BATCH_SIZE} | Buffer:{BUFFER_SIZE}")

    dev = check_gpu()

    for lbl, p in [("Video", VIDEO_PATH), ("SRT", SRT_PATH), ("Checkpoint", CHECKPOINT_PATH), ("JSON", COCO_JSON_PATH)]:
        if not os.path.exists(p):
            print(f"❌ {lbl}: {p}")
            return

    print(f"\n{'='*80}\nLOADING MODEL\n{'='*80}")
    mdl = RFDETRSegMedium(pretrain_weights=CHECKPOINT_PATH, num_queries=NUM_QUERIES,
                          image_size=IMAGE_SIZE, max_image_size=IMAGE_SIZE)
    print("✅ Model loaded")
    print_gpu_mem()

    print("\n🔥 Warmup...")
    dummy = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=np.uint8)
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(dev != 'cpu')):
            _ = mdl.predict([dummy], threshold=0.5)
    if dev != 'cpu':
        torch.cuda.synchronize()
    print("✅ Ready")
    print_gpu_mem()

    process(VIDEO_PATH, SRT_PATH, mdl, OUTPUT_BASE_DIR, dev, 0)

    print(f"\n{'='*80}\n✅ COMPLETE\n{OUTPUT_BASE_DIR}\n{'='*80}\n")

if __name__ == "__main__":
    main()