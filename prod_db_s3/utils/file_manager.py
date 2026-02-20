from __future__ import annotations

import shutil
from pathlib import Path

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


class FileManager:
    """
    Manages local temp/ and final/ directories for one job.

    Local structure:
        temp/{job_id}/
            input/          ← downloaded video + SRT
            working/        ← intermediate processing files

        final/{job_id}/
            temp/           ← crops + frames  → uploaded to S3 temp/
                defects/
                    Patch/
                        video_Patch_crop_000001.jpg
                frames/
                    video_Patch_frame_000001.jpg
            output/         ← annotated video + report → uploaded to S3 output/
                annotated_video/
                    video_annotated.mp4
                reports/
                    video_detections.json

    S3 mirrors this exactly:
        parent_folder/temp/   ← safe to delete anytime
        parent_folder/output/ ← keep forever
    """

    def __init__(self, job_id: str, class_names: list, ignored_classes: set):
        self.job_id          = job_id
        self.class_names     = class_names
        self.ignored_classes = ignored_classes
        s = get_settings()

        self.temp_root  = Path(s.temp_dir)  / job_id
        self.final_root = Path(s.final_dir) / job_id

        self._create_dirs()

    def _create_dirs(self):
        # Temp dirs
        (self.temp_root / "input").mkdir(parents=True, exist_ok=True)
        (self.temp_root / "working").mkdir(parents=True, exist_ok=True)
        logger.info("Directories created", extra={"job_id": self.job_id})

    # ── Temp paths (download area) ─────────────────────────────

    def temp_video_path(self, filename: str) -> Path:
        return self.temp_root / "input" / filename

    def temp_srt_path(self, filename: str) -> Path:
        return self.temp_root / "input" / filename

    def temp_working_dir(self) -> Path:
        return self.temp_root / "working"

    # ── Final: TEMP subfolder (crops + frames → S3 temp/) ──────

    def final_temp_dir(self, video_basename: str) -> Path:
        """crops + frames — uploaded to S3 temp/"""
        p = self.final_root / "temp"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def final_crop_dir(self, defect_type: str, video_basename: str) -> Path:
        p = self.final_root / "temp" / "defects" / defect_type
        p.mkdir(parents=True, exist_ok=True)
        return p

    def final_frames_dir(self, video_basename: str) -> Path:
        p = self.final_root / "temp" / "frames"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def final_crop_path(self, defect_type: str, det_id: int, video_basename: str) -> Path:
        return self.final_crop_dir(defect_type, video_basename) / \
               f"{video_basename}_{defect_type}_crop_{det_id:06d}.jpg"

    def final_frame_path(self, defect_type: str, det_id: int, video_basename: str) -> Path:
        return self.final_frames_dir(video_basename) / \
               f"{video_basename}_{defect_type}_frame_{det_id:06d}.jpg"

    # ── Final: OUTPUT subfolder (video + report → S3 output/) ──

    def final_output_dir(self, video_basename: str) -> Path:
        """annotated video + report — uploaded to S3 output/"""
        p = self.final_root / "output"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def final_annotated_video_dir(self, video_basename: str) -> Path:
        p = self.final_root / "output" / "annotated_video"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def final_annotated_video_path(self, video_basename: str) -> Path:
        return self.final_annotated_video_dir(video_basename) / \
               f"{video_basename}_annotated.mp4"

    def final_reports_dir(self, video_basename: str) -> Path:
        p = self.final_root / "output" / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def final_report_path(self, video_basename: str) -> Path:
        return self.final_reports_dir(video_basename) / \
               f"{video_basename}_detections.json"

    # ── Pipeline working paths (used by pipeline.py) ───────────

    def working_annotated_video_path(self, video_basename: str) -> Path:
        """Temp path where pipeline writes annotated video during processing."""
        p = self.temp_root / "working"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{video_basename}_annotated.mp4"

    def working_crop_path(self, defect_type: str, det_id: int, video_basename: str) -> Path:
        p = self.temp_root / "working" / "defects" / defect_type
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{video_basename}_{defect_type}_crop_{det_id:06d}.jpg"

    def working_frame_path(self, defect_type: str, det_id: int, video_basename: str) -> Path:
        p = self.temp_root / "working" / "frames"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{video_basename}_{defect_type}_frame_{det_id:06d}.jpg"

    # ── Move working → final ────────────────────────────────────

    def move_outputs_to_final(self, video_basename: str):
        """
        Move from temp/working/ → final/
            annotated video  → final/output/annotated_video/
            crops            → final/temp/defects/
            frames           → final/temp/frames/
        """
        working = self.temp_root / "working"

        # Annotated video → final/output/annotated_video/
        src_video = working / f"{video_basename}_annotated.mp4"
        if src_video.exists():
            dst = self.final_annotated_video_path(video_basename)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_video), str(dst))
            logger.info("Moved annotated video to final/output/")

        # Crops → final/temp/defects/
        src_defects = working / "defects"
        if src_defects.exists():
            dst_defects = self.final_root / "temp" / "defects"
            dst_defects.mkdir(parents=True, exist_ok=True)
            for cls_dir in src_defects.iterdir():
                if cls_dir.is_dir():
                    dst_cls = dst_defects / cls_dir.name
                    dst_cls.mkdir(parents=True, exist_ok=True)
                    for f in cls_dir.iterdir():
                        shutil.move(str(f), str(dst_cls / f.name))
            logger.info("Moved crops to final/temp/defects/")

        # Frames → final/temp/frames/
        src_frames = working / "frames"
        if src_frames.exists():
            dst_frames = self.final_root / "temp" / "frames"
            dst_frames.mkdir(parents=True, exist_ok=True)
            for f in src_frames.iterdir():
                shutil.move(str(f), str(dst_frames / f.name))
            logger.info("Moved frames to final/temp/frames/")

    # ── Cleanup ─────────────────────────────────────────────────

    def cleanup_temp(self):
        """Delete local temp/{job_id}/ folder after upload."""
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root)
            logger.info("Local temp cleaned up", extra={"job_id": self.job_id})

    def cleanup_final(self):
        """Delete local final/{job_id}/ folder after upload."""
        if self.final_root.exists():
            shutil.rmtree(self.final_root)
            logger.info("Local final cleaned up", extra={"job_id": self.job_id})