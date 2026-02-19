# ============================================================
# config/settings.py — Centralised settings via pydantic-settings
# ============================================================

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── AWS S3 ───────────────────────────────────────────────
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "ap-south-1"
    s3_bucket_name: str
    s3_output_prefix: str = "road-assessments"

    # ── MySQL ─────────────────────────────────────────────────
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "road_assessment"
    db_user: str
    db_password: str
    db_pool_size: int = 10
    db_pool_recycle: int = 3600

    # ── Model ─────────────────────────────────────────────────
    model_checkpoint_path: str
    coco_json_path: str
    image_size: int = 432
    num_queries: int = 200

    # ── Processing ────────────────────────────────────────────
    process_every_n_frames: int = 1
    batch_size: int = 1
    out_width: int = 1280
    out_height: int = 720
    mask_opacity: float = 0.20
    global_threshold: float = 0.30
    nms_threshold: float = 0.50
    iou_threshold: float = 0.30
    max_distance: int = 50
    max_lost: int = 30

    # ── Paths ─────────────────────────────────────────────────
    temp_dir: str = "/tmp/road_assessment/temp"
    final_dir: str = "/tmp/road_assessment/final"
    log_dir: str = "/tmp/road_assessment/logs"
    log_level: str = "INFO"

    # ── API ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton — call get_settings() anywhere."""
    return Settings()