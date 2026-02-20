from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator, Optional

import sqlalchemy as sa
from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer,
    String, Text, Enum, JSON, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)

_engine       = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        s = get_settings()
        url = (
            f"mysql+pymysql://{s.db_user}:{s.db_password}"
            f"@{s.db_host}:{s.db_port}/{s.db_name}"
            f"?charset=utf8mb4"
        )
        _engine = create_engine(
            url,
            pool_size=s.db_pool_size,
            pool_recycle=s.db_pool_recycle,
            pool_pre_ping=True,
            echo=False,
        )
        logger.info("Database engine created", extra={"host": s.db_host, "db": s.db_name})
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autocommit=False, autoflush=False
        )
    return _SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class Base(DeclarativeBase):
    pass


class ProcessingJob(Base):
    __tablename__ = "processing_jobs"

    id                  = Column(String(36),  primary_key=True)
    status              = Column(
        Enum("pending", "downloading", "processing", "uploading",
             "completed", "failed", name="job_status"),
        nullable=False, default="pending", index=True,
    )

    # ── Input ──────────────────────────────────────────────────
    input_video_s3_url  = Column(String(2048), nullable=False)
    input_srt_s3_url    = Column(String(2048), nullable=False)
    input_s3_prefix     = Column(String(2048), nullable=True)   # S3 input/ folder

    # ── Processing ─────────────────────────────────────────────
    video_basename      = Column(String(255),  nullable=True)
    total_frames        = Column(Integer,      nullable=True)
    processed_frames    = Column(Integer,      nullable=True)
    video_fps           = Column(Float,        nullable=True)
    starting_chainage_m = Column(Float,        nullable=True, default=0.0)
    ending_chainage_m   = Column(Float,        nullable=True)
    total_detections    = Column(Integer,      nullable=True, default=0)

    # ── S3 Output Prefixes ─────────────────────────────────────
    temp_s3_prefix      = Column(String(2048), nullable=True)   # S3 temp/ folder  → crops + frames
    output_s3_prefix    = Column(String(2048), nullable=True)   # S3 output/ folder → video + report

    # ── Final S3 URLs ──────────────────────────────────────────
    annotated_video_s3  = Column(String(2048), nullable=True)   # output/annotated_video/
    report_json_s3      = Column(String(2048), nullable=True)   # output/reports/

    # ── Timestamps ─────────────────────────────────────────────
    created_at          = Column(DateTime, nullable=False, default=datetime.utcnow)
    started_at          = Column(DateTime, nullable=True)
    completed_at        = Column(DateTime, nullable=True)
    error_message       = Column(Text,     nullable=True)

    detections = relationship(
        "Detection", back_populates="job", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"

    id                  = Column(Integer,     primary_key=True, autoincrement=True)
    job_id              = Column(String(36),  ForeignKey("processing_jobs.id", ondelete="CASCADE"),
                                 nullable=False, index=True)
    det_sequence_id     = Column(Integer,     nullable=False)
    track_id            = Column(Integer,     nullable=True)
    defect_type         = Column(String(100), nullable=False, index=True)
    confidence          = Column(Float,       nullable=True)
    video_name          = Column(String(255), nullable=True)
    frame_start         = Column(Integer,     nullable=True)
    frame_end           = Column(Integer,     nullable=True)
    timestamp_start     = Column(String(50),  nullable=True)
    timestamp_end       = Column(String(50),  nullable=True)
    chainage_start_m    = Column(Float,       nullable=True)
    chainage_end_m      = Column(Float,       nullable=True)
    chainage_avg_m      = Column(Float,       nullable=True, index=True)
    gps_latitude        = Column(Float,       nullable=True)
    gps_longitude       = Column(Float,       nullable=True)
    crop_image_s3_url   = Column(String(2048), nullable=True)   # in S3 temp/
    frame_image_s3_url  = Column(String(2048), nullable=True)   # in S3 temp/
    polygon             = Column(JSON,        nullable=True)
    created_at          = Column(DateTime,    nullable=False, default=datetime.utcnow)

    job = relationship("ProcessingJob", back_populates="detections")


# ── Table Management ───────────────────────────────────────────

def create_all_tables() -> None:
    Base.metadata.create_all(bind=get_engine())
    logger.info("Database tables verified / created")


# ── CRUD ───────────────────────────────────────────────────────

def create_job(
    job_id: str,
    video_s3_url: str,
    srt_s3_url: str,
    starting_chainage_m: float = 0.0,
) -> None:
    with get_db_session() as session:
        job = ProcessingJob(
            id=job_id,
            status="pending",
            input_video_s3_url=video_s3_url,
            input_srt_s3_url=srt_s3_url,
            starting_chainage_m=starting_chainage_m,
        )
        session.add(job)
    logger.info("Job created in DB", extra={"job_id": job_id})


def update_job_status(job_id: str, status: str, **kwargs: Any) -> None:
    with get_db_session() as session:
        job = session.get(ProcessingJob, job_id)
        if job is None:
            logger.error("Job not found", extra={"job_id": job_id})
            return
        job.status = status
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        if status == "processing" and job.started_at is None:
            job.started_at = datetime.utcnow()
        if status in ("completed", "failed"):
            job.completed_at = datetime.utcnow()
    logger.info("Job status updated", extra={"job_id": job_id, "status": status})


def save_detections_bulk(job_id: str, detections: list[dict]) -> None:
    with get_db_session() as session:
        rows = []
        for d in detections:
            row = Detection(
                job_id=job_id,
                det_sequence_id=d["id"],
                track_id=d.get("track_id"),
                defect_type=d["defect_type"],
                video_name=d.get("video_name"),
                frame_start=d.get("frame_start"),
                frame_end=d.get("frame_end"),
                timestamp_start=d.get("timestamp_start"),
                timestamp_end=d.get("timestamp_end"),
                chainage_start_m=d.get("chainage_start_m"),
                chainage_end_m=d.get("chainage_end_m"),
                chainage_avg_m=d.get("chainage_avg_m"),
                gps_latitude=d.get("gps", {}).get("latitude"),
                gps_longitude=d.get("gps", {}).get("longitude"),
                crop_image_s3_url=d.get("s3_urls", {}).get("crop", ""),
                frame_image_s3_url=d.get("s3_urls", {}).get("frame", ""),
                polygon=d.get("polygon"),
            )
            rows.append(row)
        session.bulk_save_objects(rows)
    logger.info("Detections saved", extra={"job_id": job_id, "count": len(rows)})


def get_job(job_id: str) -> Optional[dict]:
    with get_db_session() as session:
        job = session.get(ProcessingJob, job_id)
        if job is None:
            return None
        return {
            "id":                   job.id,
            "status":               job.status,
            # Input
            "input_video_s3_url":   job.input_video_s3_url,
            "input_srt_s3_url":     job.input_srt_s3_url,
            "input_s3_prefix":      job.input_s3_prefix,
            # Processing info
            "video_basename":       job.video_basename,
            "total_frames":         job.total_frames,
            "processed_frames":     job.processed_frames,
            "starting_chainage_m":  job.starting_chainage_m,
            "ending_chainage_m":    job.ending_chainage_m,
            "total_detections":     job.total_detections,
            # S3 structure
            "temp_s3_prefix":       job.temp_s3_prefix,
            "output_s3_prefix":     job.output_s3_prefix,
            # Final URLs
            "annotated_video_s3":   job.annotated_video_s3,
            "report_json_s3":       job.report_json_s3,
            # Timestamps
            "created_at":           job.created_at.isoformat()   if job.created_at   else None,
            "started_at":           job.started_at.isoformat()   if job.started_at   else None,
            "completed_at":         job.completed_at.isoformat() if job.completed_at else None,
            "error_message":        job.error_message,
        }


def list_jobs(limit: int = 50, offset: int = 0) -> list[dict]:
    with get_db_session() as session:
        jobs = (
            session.query(ProcessingJob)
            .order_by(ProcessingJob.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )
        return [
            {
                "id":               j.id,
                "status":           j.status,
                "video_basename":   j.video_basename,
                "total_detections": j.total_detections,
                "temp_s3_prefix":   j.temp_s3_prefix,
                "output_s3_prefix": j.output_s3_prefix,
                "created_at":       j.created_at.isoformat()   if j.created_at   else None,
                "completed_at":     j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in jobs
        ]


def get_detections_for_job(job_id: str) -> list[dict]:
    with get_db_session() as session:
        rows = (
            session.query(Detection)
            .filter(Detection.job_id == job_id)
            .order_by(Detection.chainage_avg_m)
            .all()
        )
        return [
            {
                "id":                  r.id,
                "det_sequence_id":     r.det_sequence_id,
                "track_id":            r.track_id,
                "defect_type":         r.defect_type,
                "video_name":          r.video_name,
                "frame_start":         r.frame_start,
                "frame_end":           r.frame_end,
                "chainage_start_m":    r.chainage_start_m,
                "chainage_end_m":      r.chainage_end_m,
                "chainage_avg_m":      r.chainage_avg_m,
                "gps_latitude":        r.gps_latitude,
                "gps_longitude":       r.gps_longitude,
                "crop_image_s3_url":   r.crop_image_s3_url,    # in S3 temp/
                "frame_image_s3_url":  r.frame_image_s3_url,   # in S3 temp/
                "polygon":             r.polygon,
                "created_at":          r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]