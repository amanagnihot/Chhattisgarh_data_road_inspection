# ============================================================
# core/pipeline.py — Main video processing pipeline
# ============================================================

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch

from config.settings import get_settings
from core.srt_parser import (
    calculate_cumulative_chainage,
    get_srt_data_for_frame,
    parse_srt,
)
from core.tracker import SegmentationTracker
from utils.file_manager import FileManager
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ── Class definitions (loaded once at import) ────────────────

IGNORED_CLASSES: set[str] = {
    "white_mark", "water_mark", "doubt", "bump",
    "guard_post", "overhead_sign_board", "sign_board", "kerb_damage",
}

CLASS_THRESHOLDS: dict[str, float] = {
    # Override per-class if needed — fallback is GLOBAL_THRESHOLD from settings
}

HEX_COLORS = [
    "#FF0000", "#008CFF", "#FF00FF", "#8000FF", "#00FF80",
    "#FF0080", "#B400B4", "#0000FF", "#02D32E", "#FF8000",
    "#00FFFF", "#9E4292", "#FF4040", "#4040FF", "#A0A0A0",
    "#006400", "#D2691E", "#008080", "#52240E", "#873CBE",
    "#00C8C8", "#0000C8", "#32CD32", "#FF69B4", "#696969",
    "#228B22", "#ADD8E6", "#F5F5F5",
]


def _hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def load_class_names(coco_json_path: str) -> list[str]:
    with open(coco_json_path, "r") as f:
        data = json.load(f)
    cats = sorted(data["categories"], key=lambda x: x["id"])
    names = [c["name"] for c in cats]
    logger.info("Classes loaded", extra={"count": len(names), "names": names})
    return names


# ── Helpers ───────────────────────────────────────────────────

class _ThreadedReader:
    """Background thread that feeds frames into a queue."""

    def __init__(self, video_path: Path, skip_n: int = 1, buffer_size: int = 16):
        self.cap     = cv2.VideoCapture(str(video_path))
        self.skip_n  = skip_n
        self.buffer  = queue.Queue(maxsize=buffer_size)

        self.width        = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height       = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps          = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.buffer.put(None)
                break
            if idx % self.skip_n == 0:
                self.buffer.put((idx, frame))
            idx += 1

    def read(self):
        return self.buffer.get()

    def release(self):
        self.cap.release()


class _AsyncSaver:
    """Thread pool that saves images asynchronously."""

    def __init__(self, num_workers: int = 4):
        self._pool    = ThreadPoolExecutor(max_workers=num_workers)
        self._futures = []

    def save(self, path: Path, image: np.ndarray):
        self._futures.append(self._pool.submit(cv2.imwrite, str(path), image))

    def wait_all(self):
        for f in self._futures:
            f.result()
        self._futures.clear()

    def shutdown(self):
        self.wait_all()
        self._pool.shutdown(wait=True)


# ── Core pipeline ─────────────────────────────────────────────

class VideoPipeline:
    """
    Encapsulates the full inference + tracking + saving pipeline
    for a single (video, SRT) pair.
    """

    def __init__(self, model, class_names: list[str]):
        self.model       = model
        self.class_names = class_names

        # Pad colors to match class count
        colors = list(HEX_COLORS)
        while len(colors) < len(class_names):
            colors.append("#FFFFFF")
        self.hex_colors = colors

        self.color_map: dict[str, tuple] = {
            class_names[i]: _hex_to_bgr(colors[i])
            for i in range(len(class_names))
        }

        # Supervision annotators
        palette = sv.ColorPalette.from_hex(self.hex_colors)
        text_scale     = sv.calculate_optimal_text_scale((settings.out_width, settings.out_height))
        line_thickness = sv.calculate_optimal_line_thickness((settings.out_width, settings.out_height))

        self.mask_annotator  = sv.MaskAnnotator(color=palette, opacity=settings.mask_opacity)
        self.bbox_annotator  = sv.BoxAnnotator(color=palette, thickness=line_thickness)
        self.label_annotator = sv.LabelAnnotator(
            color=palette,
            text_color=sv.Color.WHITE,
            text_scale=text_scale,
            text_thickness=max(1, line_thickness - 1),
        )

    # ── Detection helpers ─────────────────────────────────────

    def _filter(self, dets: sv.Detections) -> sv.Detections:
        if len(dets) == 0:
            return dets
        keep = []
        for i in range(len(dets)):
            cid  = int(dets.class_id[i])
            conf = float(dets.confidence[i])
            if cid < 0 or cid >= len(self.class_names):
                continue
            name = self.class_names[cid]
            if name in IGNORED_CLASSES:
                continue
            thresh = CLASS_THRESHOLDS.get(name, settings.global_threshold)
            if conf >= thresh:
                keep.append(i)
        if not keep:
            return dets[np.array([], dtype=int)]
        mask = np.zeros(len(dets), dtype=bool)
        mask[keep] = True
        return dets[mask]

    def _sv_to_list(self, dets: sv.Detections, h: int, w: int) -> list[dict]:
        result = []
        if len(dets) == 0:
            return result
        for i in range(len(dets)):
            cid  = int(dets.class_id[i])
            conf = float(dets.confidence[i])
            if cid < 0 or cid >= len(self.class_names):
                continue
            name = self.class_names[cid]
            if name in IGNORED_CLASSES:
                continue
            x1, y1, x2, y2 = dets.xyxy[i].tolist()
            mask = polygon = None
            if dets.mask is not None:
                raw = dets.mask[i]
                if raw.shape != (h, w):
                    raw = cv2.resize(
                        raw.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                mask = raw
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    polygon = contours[0].squeeze().tolist()
                    if polygon and isinstance(polygon[0], (int, float)):
                        polygon = [polygon]
                    if polygon and len(polygon) < 3:
                        polygon = None
            result.append(
                {"bbox": [x1, y1, x2, y2], "mask": mask, "polygon": polygon,
                 "class": name, "confidence": conf}
            )
        return result

    def _annotate(self, frame: np.ndarray, dets: sv.Detections) -> np.ndarray:
        labels = []
        for cid, conf in zip(dets.class_id, dets.confidence):
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else "Unknown"
            labels.append(f"{name} {conf:.2f}")
        out = frame.copy()
        out = self.mask_annotator.annotate(scene=out, detections=dets)
        out = self.bbox_annotator.annotate(scene=out, detections=dets)
        out = self.label_annotator.annotate(scene=out, detections=dets, labels=labels)
        self._draw_legend(out)
        return out

    def _draw_legend(self, frame: np.ndarray) -> None:
        visible = [(n, c) for n, c in self.color_map.items() if n not in IGNORED_CLASSES]
        w, h    = settings.out_width, settings.out_height
        scale   = max(0.6, min(w / 1920.0, 1.2))
        lx, ly  = int(0.02 * w), int(0.04 * h)
        line_h  = int(22 * scale)
        line_len = int(35 * scale)
        cv2.rectangle(frame, (lx - 12, ly - 12),
                      (lx + int(300 * scale), ly + line_h * len(visible) + 12),
                      (0, 0, 0), max(2, int(2 * scale)))
        for idx, (name, color) in enumerate(visible):
            y = ly + idx * line_h
            cv2.line(frame, (lx, y + 10), (lx + line_len, y + 10), color, 2)
            cv2.putText(frame, name, (lx + line_len + 12, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale,
                        (255, 255, 255), max(1, int(1.5 * scale)))

    # ── Main run ──────────────────────────────────────────────

    def run(
        self,
        video_path: Path,
        srt_path: Path,
        file_manager: FileManager,
        video_basename: str,
        starting_chainage_m: float = 0.0,
    ) -> tuple[list[dict], float]:
        """
        Run the full pipeline on one video.

        Returns:
            (all_detections, ending_chainage_m)
        """
        srt_data = parse_srt(srt_path)
        if srt_data is None:
            raise ValueError(f"Could not parse SRT: {srt_path}")

        ending_chainage = calculate_cumulative_chainage(srt_data, starting_chainage_m)

        reader = _ThreadedReader(
            video_path,
            skip_n=settings.process_every_n_frames,
        )
        logger.info(
            "Video opened",
            extra={
                "resolution": f"{reader.width}x{reader.height}",
                "fps":        reader.fps,
                "frames":     reader.total_frames,
            },
        )

        output_fps = max(1.0, reader.fps / settings.process_every_n_frames)
        video_out_path = file_manager.temp_annotated_video_path(video_basename)
        out = cv2.VideoWriter(
            str(video_out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (settings.out_width, settings.out_height),
        )

        tracker        = SegmentationTracker(
            settings.iou_threshold, settings.max_distance, settings.max_lost
        )
        saver          = _AsyncSaver(num_workers=4)
        track_history: dict  = {}
        all_detections: list = []
        det_id_counter       = 1
        processed_frames     = 0
        frame_batch: list    = []
        meta_batch:  list    = []
        start_time           = time.time()

        # ── Batch processor ──────────────────────────────────────
        def flush_batch():
            nonlocal processed_frames, det_id_counter, frame_batch, meta_batch

            if not frame_batch:
                return

            raw = self.model.predict(frame_batch, threshold=settings.global_threshold)
            batch_sv = list(raw) if isinstance(raw, (list, tuple)) else [raw]

            for i, (frame_resized, (frame_index, srt_entry)) in enumerate(
                zip(frame_batch, meta_batch)
            ):
                processed_frames += 1
                chainage_m = srt_entry["cumulative_chainage_m"]

                sv_dets = batch_sv[i].with_nms(threshold=settings.nms_threshold)
                sv_dets = self._filter(sv_dets)
                det_list = self._sv_to_list(sv_dets, settings.out_height, settings.out_width)

                tracked = tracker.update(det_list)
                annotated = self._annotate(frame_resized, sv_dets)

                # Update track history
                for tr in tracked:
                    tid      = tr["track_id"]
                    cls_name = tr["class"]
                    if cls_name in IGNORED_CLASSES:
                        continue
                    if tid not in track_history:
                        track_history[tid] = {
                            "type":            cls_name,
                            "first_frame":     frame_index + 1,
                            "last_frame":      frame_index + 1,
                            "first_chainage":  chainage_m,
                            "last_chainage":   chainage_m,
                            "first_timestamp": srt_entry["absolute_timestamp"],
                            "last_timestamp":  srt_entry["absolute_timestamp"],
                            "gps_lat":         srt_entry["latitude"],
                            "gps_lon":         srt_entry["longitude"],
                            "best_snapshot":   None,
                        }
                    th = track_history[tid]
                    th["last_frame"]     = frame_index + 1
                    th["last_chainage"]  = chainage_m
                    th["last_timestamp"] = srt_entry["absolute_timestamp"]

                    if th["best_snapshot"] is None:
                        x1, y1, x2, y2 = map(int, tr["bbox"])
                        crop = frame_resized[
                            max(0, y1): min(settings.out_height, y2),
                            max(0, x1): min(settings.out_width,  x2),
                        ].copy()
                        th["best_snapshot"] = {
                            "frame_idx":  frame_index + 1,
                            "crop":       crop,
                            "full_frame": annotated.copy(),
                            "polygon":    tr["polygon"],
                        }

                out.write(annotated)

                # Finalise lost tracks
                active_tids = {t["track_id"] for t in tracked}
                for tid in list(track_history.keys()):
                    if tid not in active_tids and tid not in tracker.tracks:
                        track = track_history.pop(tid)
                        if track["type"] in IGNORED_CLASSES or track["best_snapshot"] is None:
                            continue
                        _save_track(track, det_id_counter, video_basename,
                                    file_manager, saver, all_detections, srt_path.name)
                        det_id_counter += 1

                # Progress log
                if processed_frames % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_a   = processed_frames / elapsed if elapsed > 0 else 0
                    remain  = (reader.total_frames // settings.process_every_n_frames) - processed_frames
                    eta_min = (remain / fps_a / 60) if fps_a > 0 else 0
                    logger.info(
                        "Progress",
                        extra={
                            "frame": frame_index,
                            "total": reader.total_frames,
                            "fps":   round(fps_a, 1),
                            "eta_min": round(eta_min, 1),
                            "detections": len(all_detections),
                        },
                    )

            frame_batch.clear()
            meta_batch.clear()

        def _save_track(track, det_id, vbase, fm, sv_saver, dets_list, srt_name):
            dtype = track["type"]
            crop_path  = fm.temp_crop_path(dtype,  det_id, vbase)
            frame_path = fm.temp_frame_path(dtype, det_id, vbase)
            sv_saver.save(crop_path,  track["best_snapshot"]["crop"])
            sv_saver.save(frame_path, track["best_snapshot"]["full_frame"])

            dets_list.append(
                {
                    "id":               det_id,
                    "track_id":         None,
                    "defect_type":      dtype,
                    "video_name":       vbase,
                    "srt_name":         srt_name,
                    "frame_start":      track["first_frame"],
                    "frame_end":        track["last_frame"],
                    "timestamp_start":  track["first_timestamp"].isoformat(),
                    "timestamp_end":    track["last_timestamp"].isoformat(),
                    "chainage_start_m": track["first_chainage"],
                    "chainage_end_m":   track["last_chainage"],
                    "chainage_avg_m":   (track["first_chainage"] + track["last_chainage"]) / 2,
                    "gps": {
                        "latitude":  track["gps_lat"],
                        "longitude": track["gps_lon"],
                    },
                    "polygon": track["best_snapshot"]["polygon"],
                    "local_paths": {
                        "crop":  str(crop_path),
                        "frame": str(frame_path),
                    },
                }
            )

        # ── Main read loop ────────────────────────────────────────
        while True:
            item = reader.read()
            if item is None:
                flush_batch()
                break
            frame_index, frame = item
            srt_entry = get_srt_data_for_frame(frame_index + 1, srt_data)
            if srt_entry is None:
                continue

            frame_resized = cv2.resize(
                frame, (settings.out_width, settings.out_height),
                interpolation=cv2.INTER_LINEAR,
            )
            frame_batch.append(frame_resized)
            meta_batch.append((frame_index, srt_entry))

            if len(frame_batch) >= settings.batch_size:
                flush_batch()

        # ── Flush remaining active tracks ─────────────────────────
        for tid, track in track_history.items():
            if track["type"] in IGNORED_CLASSES or track["best_snapshot"] is None:
                continue
            _save_track(track, det_id_counter, video_basename,
                        file_manager, saver, all_detections, srt_path.name)
            det_id_counter += 1

        logger.info("Flushing image save queue...")
        saver.shutdown()
        reader.release()
        out.release()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        logger.info(
            "Pipeline complete",
            extra={
                "duration_min":    round(elapsed / 60, 2),
                "total_detections": len(all_detections),
                "processed_frames": processed_frames,
            },
        )
        return all_detections, ending_chainage