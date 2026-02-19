# ============================================================
# core/tracker.py — IoU + centroid-based segmentation tracker
# ============================================================

import math
import numpy as np
import cv2

from utils.logger import get_logger

logger = get_logger(__name__)


class SegmentationTracker:
    """
    Lightweight multi-object tracker that associates detections across frames
    using mask IoU and centroid distance.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_distance: int    = 50,
        max_lost: int        = 30,
    ):
        self.next_id       = 1
        self.tracks: dict  = {}
        self.iou_threshold = iou_threshold
        self.max_distance  = max_distance
        self.max_lost      = max_lost

    # ── Internal helpers ─────────────────────────────────────────

    @staticmethod
    def _mask_iou(m1: np.ndarray | None, m2: np.ndarray | None) -> float:
        if m1 is None or m2 is None:
            return 0.0
        inter = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()
        return float(inter / union) if union > 0 else 0.0

    @staticmethod
    def _centroid(bbox: list[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    # ── Public API ────────────────────────────────────────────────

    def update(self, detections: list[dict]) -> list[dict]:
        """
        Match detections to existing tracks.

        Args:
            detections: list of dicts with keys:
                        bbox, mask, polygon, class, confidence

        Returns:
            list of dicts with keys:
                track_id, bbox, mask, class, polygon
        """
        # Age all tracks
        for t in self.tracks.values():
            t["lost"] += 1

        assigned: set[int] = set()
        updated_tracks: list[dict] = []

        # ── Match existing tracks ────────────────────────────────
        for tid, t in list(self.tracks.items()):
            best_i, best_score = -1, 0.0
            for i, det in enumerate(detections):
                if i in assigned or t.get("class") != det.get("class"):
                    continue
                iou  = self._mask_iou(t.get("mask"), det.get("mask"))
                cx, cy = self._centroid(det["bbox"])
                dist = math.hypot(
                    t["centroid"][0] - cx,
                    t["centroid"][1] - cy,
                )
                if iou > self.iou_threshold or dist < self.max_distance:
                    score = iou - (dist / self.max_distance) * 0.5
                    if score > best_score:
                        best_score, best_i = score, i

            if best_i >= 0:
                det = detections[best_i]
                self.tracks[tid].update(
                    {
                        "bbox":     det["bbox"],
                        "centroid": self._centroid(det["bbox"]),
                        "mask":     det["mask"],
                        "lost":     0,
                        "class":    det["class"],
                    }
                )
                assigned.add(best_i)
                updated_tracks.append(
                    {
                        "track_id": tid,
                        "bbox":     det["bbox"],
                        "mask":     det["mask"],
                        "class":    det["class"],
                        "polygon":  det["polygon"],
                    }
                )
            elif t["lost"] > self.max_lost:
                del self.tracks[tid]

        # ── Register new tracks ──────────────────────────────────
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                "bbox":     det["bbox"],
                "centroid": self._centroid(det["bbox"]),
                "mask":     det["mask"],
                "class":    det["class"],
                "lost":     0,
            }
            updated_tracks.append(
                {
                    "track_id": tid,
                    "bbox":     det["bbox"],
                    "mask":     det["mask"],
                    "class":    det["class"],
                    "polygon":  det["polygon"],
                }
            )

        return updated_tracks