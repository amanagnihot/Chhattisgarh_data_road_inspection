# ============================
# ROAD ASSESSMENT - INDIVIDUAL DEFECT REPORTS v8.0 KML-BASED
#
# APPROACH:
#   - KML file = GROUND TRUTH for chainage
#   - Each detection matched to nearest KML chainage point
#   - NO dependency on video order, folder names, or SRT
#   - Simple, accurate, reliable
#
# CHAINAGE CALCULATION:
#   For each detection:
#     1. Get detection GPS (lat, lon) from JSON
#     2. Find nearest KML chainage point
#     3. Assign that KML chainage to detection
#     4. Done!
#
# ADVANTAGES:
#   ✅ Single source of truth (KML file)
#   ✅ Works regardless of video order
#   ✅ Works even if road curves/loops
#   ✅ No cumulative errors
#   ✅ Super simple and accurate
#
# FEATURES:
#   - Correct image aspect ratio (no stretching)
#   - Parallel image loading + compression
#   - GPU duplicate removal
# ============================

import os
import re
import json
import math
import glob
import shutil
import time
from datetime import datetime
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================
# CONFIGURATION
# ============================

GDRIVE_MOUNT      = "/mnt/gdrive"

# Output base (contains chainage folders)
ALL_CHAINAGE_BASE = "/mnt/gdrive/chattishgrah_road_defects_everything_ex/All_Chattisgarh_Output"

# Chainage folders (order doesn't matter for KML approach)
CHAINAGE_FOLDERS  = [
    "0.00 to 30.700",
    "Ch.30.600 to 71.800",
    "Ch. 70.00 to 115.00",
]

# KML chainage reference file (GROUND TRUTH)
KML_REFERENCE_FILE = "/media/user/New Volume/Sakshi/chattishgarh/0-115 Kms Chainage Points 1.kml"

# Output reports
OUTPUT_REPORT_DIR = "/mnt/gdrive/chattishgrah_road_defects_everything_ex/All_Chattisgarh_Output/Reports_KML_v8"

# ── Image quality ────────────────────────────────────────────────────────────
CROP_IMAGE_QUALITY    = 55
CROP_IMAGE_MAX_WIDTH  = 320
CROP_IMAGE_MAX_HEIGHT = 240

FRAME_IMAGE_QUALITY   = 50
FRAME_IMAGE_MAX_WIDTH = 500
FRAME_IMAGE_MAX_HEIGHT= 320

# ── Duplicate detection ──────────────────────────────────────────────────────
IMAGE_SIMILARITY_THRESHOLD = 0.92
GPS_PROXIMITY_METERS       = 3
COMPARE_SIZE               = (128, 128)

# ── Performance ──────────────────────────────────────────────────────────────
IMAGE_LOAD_THREADS = 16
COMPRESS_THREADS   = 16
LOCAL_CACHE_DIR    = "/tmp/road_report_cache_v8"

# ============================
# GPU SETUP
# ============================

def setup_device():
    if torch.cuda.is_available():
        device   = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        device = torch.device('cpu')
        print("⚠️  No GPU — using CPU")
    return device

DEVICE = setup_device()

# ============================
# GPS UTILITIES
# ============================

def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two GPS points."""
    R    = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ============================
# KML PARSING (GROUND TRUTH CHAINAGE)
# ============================

def load_kml_chainage_reference(kml_path):
    """
    Load KML chainage reference file.
    Returns dict: {chainage_m: (lat, lon), ...}
    
    KML format:
      <SimpleData name="distance">0.000000000000000</SimpleData>
      <coordinates>81.4847176000001,20.66513936</coordinates>
    """
    print(f"\n📍 Loading KML chainage reference...")
    print(f"   File: {kml_path}")
    
    if not os.path.exists(kml_path):
        print(f"   ❌ KML file not found!")
        return None
    
    try:
        with open(kml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse all placemarks
        chainage_points = {}
        
        # Find all <Placemark> sections
        placemarks = re.findall(
            r'<Placemark>.*?</Placemark>',
            content,
            re.DOTALL
        )
        
        for pm in placemarks:
            # Extract distance (chainage in meters)
            dist_match = re.search(
                r'<SimpleData name="distance">([\d.]+)</SimpleData>',
                pm
            )
            
            # Extract coordinates (lon,lat format in KML)
            coord_match = re.search(
                r'<coordinates>([\d.-]+),([\d.-]+)</coordinates>',
                pm
            )
            
            if dist_match and coord_match:
                chainage_m = float(dist_match.group(1))
                lon        = float(coord_match.group(1))
                lat        = float(coord_match.group(2))
                
                chainage_points[chainage_m] = (lat, lon)
        
        if not chainage_points:
            print(f"   ❌ No chainage points found in KML!")
            return None
        
        # Sort by chainage
        chainage_points = dict(sorted(chainage_points.items()))
        
        min_ch = min(chainage_points.keys()) / 1000.0
        max_ch = max(chainage_points.keys()) / 1000.0
        
        print(f"   ✅ Loaded {len(chainage_points)} chainage points")
        print(f"   📏 Range: Ch. {min_ch:.3f} → {max_ch:.3f} km")
        
        if len(chainage_points) > 1:
            intervals = [list(chainage_points.keys())[i+1] - list(chainage_points.keys())[i] 
                        for i in range(min(5, len(chainage_points)-1))]
            avg_interval = sum(intervals) / len(intervals)
            print(f"   📍 Interval: ~{avg_interval:.0f} m")
        
        return chainage_points
        
    except Exception as e:
        print(f"   ❌ Error parsing KML: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_nearest_kml_chainage(detection_gps, kml_points):
    """
    Find nearest KML chainage point to detection GPS.
    Returns (chainage_m, distance_m).
    """
    det_lat, det_lon = detection_gps
    
    min_dist = float('inf')
    nearest_ch = None
    
    for chainage_m, (kml_lat, kml_lon) in kml_points.items():
        dist = haversine(det_lat, det_lon, kml_lat, kml_lon)
        if dist < min_dist:
            min_dist = dist
            nearest_ch = chainage_m
    
    return nearest_ch, min_dist

# ============================
# MOUNT VERIFICATION
# ============================

def verify_mount():
    print(f"\n🔍 Verifying Google Drive mount...")

    if not os.path.exists(GDRIVE_MOUNT) or not os.listdir(GDRIVE_MOUNT):
        print(f"❌ Drive not mounted at {GDRIVE_MOUNT}")
        return False

    print(f"✅ Drive mounted ({len(os.listdir(GDRIVE_MOUNT))} items)")

    for path, label in [
        (ALL_CHAINAGE_BASE,   "Output base"),
        (KML_REFERENCE_FILE,  "KML file  "),
    ]:
        if os.path.exists(path):
            print(f"✅ {label}: {path}")
        else:
            print(f"⚠️  {label}: {path}")

    return True

# ============================
# LOAD VIDEOS (ORDER DOESN'T MATTER)
# ============================

def load_all_videos(chainage_folders):
    """
    Load all videos from all folders.
    Order doesn't matter for KML approach.
    """
    print(f"\n📂 Loading videos from all folders...")

    all_videos = []

    for folder_name in chainage_folders:
        folder_path = os.path.join(ALL_CHAINAGE_BASE, folder_name)
        if not os.path.exists(folder_path):
            print(f"   ⚠️  Not found: {folder_name}")
            continue

        # Get video folders
        video_dirs = sorted([
            d for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ])

        print(f"   📁 '{folder_name}': {len(video_dirs)} videos")

        for video_dir in video_dirs:
            json_path = os.path.join(
                folder_path, video_dir, f"{video_dir}_detections.json"
            )
            
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    video_name = data.get('video_name', video_dir + '.MP4')
                    
                    all_videos.append({
                        'video_name':  video_name,
                        'folder_name': folder_name,
                        'json_path':   json_path,
                        'json_data':   data,
                    })
                except Exception as e:
                    print(f"      ❌ Failed: {video_dir}: {e}")

    print(f"\n   ✅ Total videos loaded: {len(all_videos)}")
    return all_videos

# ============================
# BUILD DETECTIONS WITH KML CHAINAGE
# ============================

def build_detections_from_kml(all_videos, kml_points):
    """
    Build detections with global chainage from KML reference.
    
    For each detection:
      - Use detection GPS coordinates
      - Find nearest KML chainage point
      - Assign that chainage to detection
    """
    print(f"\n🔗 Assigning chainage from KML reference...")
    
    all_detections = []
    total_raw      = 0
    max_distance   = 0.0
    total_distance = 0.0
    
    for v in all_videos:
        vname      = v['video_name']
        json_data  = v['json_data']
        video_dir  = os.path.dirname(v['json_path'])
        folder     = v['folder_name']

        for det in json_data.get('detections', []):
            # Get detection GPS
            det_gps = (
                det['gps']['latitude'],
                det['gps']['longitude']
            )
            
            # Find nearest KML chainage point
            nearest_ch, dist = find_nearest_kml_chainage(det_gps, kml_points)
            
            # Assign KML chainage as global chainage
            det['global_chainage_m'] = nearest_ch
            det['kml_distance_m']    = dist  # Distance from KML point
            
            # Track max distance for verification
            max_distance    = max(max_distance, dist)
            total_distance += dist
            
            det['images']['crop']  = os.path.join(
                video_dir, det['images']['crop']
            )
            det['images']['frame'] = os.path.join(
                video_dir, det['images']['frame']
            )
            det['video_name']      = vname
            det['chainage_folder'] = folder
            
            all_detections.append(det)
            total_raw += 1

    # Sort by global chainage (KML-based)
    all_detections.sort(key=lambda x: x['global_chainage_m'])

    avg_dist = total_distance / total_raw if total_raw > 0 else 0.0

    print(f"   ✅ Total detections : {total_raw}")
    print(f"   ✅ Sorted by KML chainage")
    print(f"\n   📊 KML Matching Quality:")
    print(f"      Average distance from KML: {avg_dist:.1f} m")
    print(f"      Max distance from KML    : {max_distance:.1f} m")
    
    if max_distance > 500:
        print(f"      ⚠️  Some detections >500m from KML points")
    else:
        print(f"      ✅ All detections close to KML reference")
    
    return all_detections, total_raw

# ============================
# IMAGE LOADING
# ============================

def load_image_tensor(image_path):
    try:
        if not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img    = cv2.resize(img, COMPARE_SIZE)
        tensor = torch.from_numpy(img).float() / 255.0
        return tensor.to(DEVICE)
    except Exception:
        return None


def load_images_parallel(detections):
    print(f"   ⚡ Loading {len(detections)} images "
          f"({IMAGE_LOAD_THREADS} threads)...")
    tensors      = {}
    failed_loads = 0

    with ThreadPoolExecutor(max_workers=IMAGE_LOAD_THREADS) as executor:
        futures = {
            executor.submit(
                lambda args: (args[0], load_image_tensor(args[1])),
                (i, det['images']['crop'])
            ): i
            for i, det in enumerate(detections)
        }
        done = 0
        for future in as_completed(futures):
            try:
                idx, tensor = future.result()
                tensors[idx] = tensor
                if tensor is None:
                    failed_loads += 1
            except Exception:
                failed_loads += 1
            done += 1
            if done % 300 == 0:
                print(f"   📷 {done}/{len(detections)} "
                      f"({failed_loads} failed)...")

    print(f"   ✅ {len(detections)-failed_loads} OK, "
          f"{failed_loads} failed")
    return tensors

# ============================
# GPU DUPLICATE REMOVAL
# ============================

def remove_duplicates_gpu(detections):
    print(f"\n🔍 Removing duplicates (GPU)...")
    print(f"   Total         : {len(detections)}")
    print(f"   GPS threshold : {GPS_PROXIMITY_METERS}m")
    print(f"   Similarity    : {IMAGE_SIMILARITY_THRESHOLD:.0%}")

    tensors            = load_images_parallel(detections)
    unique_indices     = []
    duplicates_removed = 0

    for i, det in enumerate(detections):
        is_dup = False
        for j in unique_indices:
            u = detections[j]

            if det['defect_type'] != u['defect_type']:
                continue

            gps_dist = haversine(
                det['gps']['latitude'],  det['gps']['longitude'],
                u['gps']['latitude'],    u['gps']['longitude']
            )
            if gps_dist > GPS_PROXIMITY_METERS:
                continue

            t1_img = tensors.get(i)
            t2_img = tensors.get(j)
            if t1_img is not None and t2_img is not None:
                sim = F.cosine_similarity(
                    t1_img.flatten().unsqueeze(0),
                    t2_img.flatten().unsqueeze(0)
                ).item()
                sim = (sim + 1.0) / 2.0
            else:
                sim = 0.0

            if sim >= IMAGE_SIMILARITY_THRESHOLD:
                is_dup = True
                duplicates_removed += 1
                break

        if not is_dup:
            unique_indices.append(i)

        if (i + 1) % 300 == 0:
            print(f"   Checked {i+1}/{len(detections)} | "
                  f"Dupes: {duplicates_removed}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    unique = [detections[i] for i in unique_indices]
    print(f"   ✅ Removed : {duplicates_removed}")
    print(f"   ✅ Unique  : {len(unique)}")
    return unique

# ============================
# IMAGE COMPRESSION
# ============================

def compress_image_maintain_ratio(image_path, max_width, max_height, quality):
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
        cache_key       = image_path.replace("/", "_").replace(" ", "_")[-150:]
        compressed_path = os.path.join(LOCAL_CACHE_DIR, cache_key)

        if os.path.exists(compressed_path):
            return compressed_path

        img = Image.open(image_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        img.thumbnail((max_width, max_height), Image.LANCZOS)
        img.save(compressed_path, 'JPEG', quality=quality, optimize=True)
        return compressed_path

    except Exception:
        return image_path


def precompress_all_images(detections):
    print(f"\n⚡ Pre-compressing {len(detections)*2} images "
          f"({COMPRESS_THREADS} threads)...")

    tasks = []
    for det in detections:
        tasks.append((det['images']['crop'],
                      CROP_IMAGE_MAX_WIDTH, CROP_IMAGE_MAX_HEIGHT,
                      CROP_IMAGE_QUALITY))
        tasks.append((det['images']['frame'],
                      FRAME_IMAGE_MAX_WIDTH, FRAME_IMAGE_MAX_HEIGHT,
                      FRAME_IMAGE_QUALITY))

    total  = len(tasks)
    done   = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=COMPRESS_THREADS) as executor:
        futures = {
            executor.submit(
                lambda a: compress_image_maintain_ratio(*a), t
            ): t for t in tasks
        }
        for future in as_completed(futures):
            try:
                if future.result() is None:
                    failed += 1
            except Exception:
                failed += 1
            done += 1
            if done % 500 == 0:
                print(f"   🗜️  {done}/{total} ({failed} failed)...")

    print(f"   ✅ Done: {total-failed} OK, {failed} failed")


def safe_add_image(cell, image_path, max_width_in, max_height_in,
                   max_px_w, max_px_h, quality):
    try:
        if not image_path or not os.path.exists(image_path):
            cell.text = "N/A"
            return False

        compressed = compress_image_maintain_ratio(
            image_path, max_px_w, max_px_h, quality
        )
        use_path = compressed if (compressed and os.path.exists(compressed)) \
                   else image_path

        with Image.open(use_path) as img:
            img_w, img_h = img.size

        if img_w == 0 or img_h == 0:
            cell.text = "N/A"
            return False

        ratio   = img_h / img_w
        final_w = min(max_width_in, max_height_in / ratio)
        final_h = final_w * ratio
        if final_h > max_height_in:
            final_h = max_height_in
            final_w = final_h / ratio

        run = cell.paragraphs[0].add_run()
        run.add_picture(use_path,
                        width=Inches(final_w),
                        height=Inches(final_h))
        return True

    except Exception:
        cell.text = "Img error"
        return False

# ============================
# GPS HYPERLINK
# ============================

def add_hyperlink(paragraph, url, text, color="0563C1"):
    part      = paragraph.part
    r_id      = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True
    )
    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)
    run  = OxmlElement('w:r')
    rPr  = OxmlElement('w:rPr')
    c    = OxmlElement('w:color')
    c.set(qn('w:val'), color)
    rPr.append(c)
    u = OxmlElement('w:u')
    u.set(qn('w:val'), 'single')
    rPr.append(u)
    run.append(rPr)
    run.text = text
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_gps_link_to_cell(cell, lat, lon):
    cell.text = ""
    add_hyperlink(
        cell.paragraphs[0],
        f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
        f"Lat: {lat:.6f}, Long: {lon:.6f}"
    )

# ============================
# REPORT BUILDER
# ============================

def generate_individual_report(dtype, detections, total_road_m,
                                output_dir, timestamp, chainage_folders):
    doc      = Document()
    total_km = total_road_m / 1000.0

    # Title page
    title           = doc.add_heading('ROAD ASSESSMENT REPORT', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()
    for text in [
        f"Defect Type: {dtype}",
        f"Date: {datetime.now().strftime('%d %B %Y')}",
        f"Total Road Distance: {total_km:.3f} km",
        f"Chainage: Ch. 0+000 to Ch. {total_km:.3f} km",
    ]:
        p = doc.add_paragraph(text)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # Executive Summary
    doc.add_heading('Executive Summary', 1)
    doc.add_paragraph(f'• Defect Type          : {dtype}')
    doc.add_paragraph(f'• Total Detections     : {len(detections)}')
    doc.add_paragraph(f'• Total Road Distance  : {total_km:.3f} km')
    doc.add_paragraph(
        f'• Chainage Range       : Ch. 0+000 to Ch. {total_km:.3f} km'
    )
    doc.add_paragraph(
        f'• Chainage Method      : KML Reference (Ground Truth)'
    )
    doc.add_paragraph(
        f'• Report Generated     : {datetime.now().strftime("%d-%b-%Y %H:%M")}'
    )
    doc.add_page_break()

    # Detections table
    doc.add_heading(f'{dtype.upper()} DETECTIONS', 1)
    doc.add_paragraph(f'Total: {len(detections)}')
    doc.add_paragraph()

    table       = doc.add_table(rows=1, cols=6)
    table.style = 'Light Grid Accent 1'
    hdr         = table.rows[0].cells
    hdr[0].text = 'S.No'
    hdr[1].text = 'Chainage'
    hdr[2].text = 'Defect Crop'
    hdr[3].text = 'Full Frame'
    hdr[4].text = 'GPS Location'
    hdr[5].text = 'Video'

    for idx, det in enumerate(detections, 1):
        row         = table.add_row().cells
        ch_km       = det['global_chainage_m'] / 1000.0
        row[0].text = str(idx)
        row[1].text = f"Ch. {ch_km:.3f} km"

        safe_add_image(row[2], det['images']['crop'],
                       1.2, 1.0,
                       CROP_IMAGE_MAX_WIDTH, CROP_IMAGE_MAX_HEIGHT,
                       CROP_IMAGE_QUALITY)

        safe_add_image(row[3], det['images']['frame'],
                       1.8, 1.2,
                       FRAME_IMAGE_MAX_WIDTH, FRAME_IMAGE_MAX_HEIGHT,
                       FRAME_IMAGE_QUALITY)

        add_gps_link_to_cell(row[4],
                              det['gps']['latitude'],
                              det['gps']['longitude'])
        row[5].text = det.get('video_name', 'N/A')

        if idx % 50 == 0:
            print(f"      [{idx}/{len(detections)}] rows written...")

    safe_dtype  = dtype.replace(" ", "_").replace("/", "_")
    report_path = os.path.join(
        output_dir, f"{safe_dtype}_Report_{timestamp}.docx"
    )
    print(f"      💾 Saving: {os.path.basename(report_path)}")
    doc.save(report_path)
    return report_path

# ============================
# MAIN
# ============================

def main():
    print("\n" + "="*80)
    print("ROAD ASSESSMENT — INDIVIDUAL DEFECT REPORTS v8.0 KML-BASED")
    print("  ✅ KML chainage reference (GROUND TRUTH)")
    print("  ✅ Each detection matched to nearest KML point")
    print("  ✅ No dependency on video order or SRT")
    print("  ✅ Correct image aspect ratio + GPU duplicate removal")
    print("="*80)

    t0 = time.time()

    # Step 1: Verify mount
    if not verify_mount():
        print("\n⚠️  Warning: Mount issues, but continuing...")

    # Step 2: Load KML chainage reference (CRITICAL)
    kml_points = load_kml_chainage_reference(KML_REFERENCE_FILE)
    if not kml_points:
        print("\n❌ STOPPING: Cannot proceed without KML reference")
        return

    # Step 3: Create output dirs
    os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
    print(f"\n✅ Output : {OUTPUT_REPORT_DIR}")
    print(f"✅ Cache  : {LOCAL_CACHE_DIR}")

    # Step 4: Load videos (order doesn't matter for KML approach)
    all_videos = load_all_videos(CHAINAGE_FOLDERS)
    if not all_videos:
        print("❌ No videos found!")
        return

    # Step 5: Build detections with KML chainage
    all_detections, total_raw = build_detections_from_kml(
        all_videos, kml_points
    )
    
    # Get total road distance from KML
    total_road_m = max(kml_points.keys())

    print(f"\n⏱️  Pre-processing: {(time.time()-t0)/60:.1f} min")

    # Step 6: Remove duplicates (GPU)
    all_detections = remove_duplicates_gpu(all_detections)

    # Step 7: Group by defect type
    print(f"\n📑 Grouping by defect type...")
    grouped = {}
    for det in all_detections:
        grouped.setdefault(det['defect_type'], []).append(det)

    total_unique = sum(len(v) for v in grouped.values())

    print(f"\n📋 Final Summary:")
    print(f"   {'Defect Type':<35} {'Count':>6}")
    print(f"   {'-'*42}")
    for dtype, dets in sorted(grouped.items(), key=lambda x: -len(x[1])):
        print(f"   {dtype:<35} {len(dets):>6}")
    print(f"   {'-'*42}")
    print(f"   {'TOTAL':<35} {total_unique:>6}")
    print(f"   {'Duplicates removed':<35} {total_raw-total_unique:>6}")
    print(f"   {'Total road distance':<35} {total_road_m/1000:>5.3f} km")
    print(f"   {'Chainage method':<35} KML Reference")

    # Step 8: Pre-compress images
    precompress_all_images(all_detections)

    print(f"\n⏱️  Total pre-processing: {(time.time()-t0)/60:.1f} min")

    # Step 9: Generate reports
    print(f"\n{'='*80}")
    print(f"📄 Generating {len(grouped)} reports...")
    print(f"{'='*80}")

    timestamp     = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_reports = []

    for dtype, dets in sorted(grouped.items()):
        print(f"\n   📋 {dtype} — {len(dets)} detections")
        try:
            rpath = generate_individual_report(
                dtype, dets, total_road_m,
                OUTPUT_REPORT_DIR, timestamp, CHAINAGE_FOLDERS
            )
            saved_reports.append(rpath)
            size_mb = os.path.getsize(rpath) / (1024*1024)
            print(f"      ✅ {os.path.basename(rpath)} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"      ❌ Failed: {dtype}: {e}")
            import traceback; traceback.print_exc()

    # Cleanup
    print(f"\n🧹 Cleaning cache...")
    shutil.rmtree(LOCAL_CACHE_DIR, ignore_errors=True)

    total_time = time.time() - t0
    total_km   = total_road_m / 1000.0

    print(f"\n{'='*80}")
    print(f"✅ ALL DONE!")
    print(f"{'='*80}")
    print(f"   ⏱️  Total time          : {total_time/60:.1f} min")
    print(f"   📏 Total road distance  : {total_km:.3f} km (from KML)")
    print(f"   📊 Raw detections       : {total_raw}")
    print(f"   🔍 Duplicates removed   : {total_raw - total_unique}")
    print(f"   📊 Unique detections    : {total_unique}")
    print(f"   📄 Reports generated    : {len(saved_reports)}")
    print(f"   📂 Saved to Drive       : {OUTPUT_REPORT_DIR}")
    print(f"\n   Chainage Method: KML Reference (Ground Truth)")
    print(f"\n   Reports:")
    for r in saved_reports:
        size_mb = os.path.getsize(r) / (1024*1024)
        print(f"      • {os.path.basename(r)} ({size_mb:.1f} MB)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()