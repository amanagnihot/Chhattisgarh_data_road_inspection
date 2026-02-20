from __future__ import annotations

import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _build_client():
    s = get_settings()
    return boto3.client(
        "s3",
        region_name=s.aws_region,
        aws_access_key_id=s.aws_access_key_id,
        aws_secret_access_key=s.aws_secret_access_key,
    )


def parse_s3_url(s3_url: str) -> tuple[str, str]:
    parsed = urlparse(s3_url)
    if parsed.scheme == "s3":
        return parsed.netloc, parsed.path.lstrip("/")
    if parsed.scheme in ("http", "https"):
        host = parsed.netloc
        path = parsed.path.lstrip("/")
        vh = re.match(r"^(.+?)\.s3(?:\.[^.]+)?\.amazonaws\.com$", host)
        if vh:
            return vh.group(1), path
        ph = re.match(r"^s3(?:\.[^.]+)?\.amazonaws\.com$", host)
        if ph:
            parts = path.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
    raise ValueError(f"Cannot parse S3 URL: {s3_url!r}")


def extract_date_from_basename(video_basename: str) -> str:
    """Extract date from DJI filename. DJI_20260206174112 → 2026-02-06"""
    match = re.search(r"(\d{4})(\d{2})(\d{2})", video_basename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    from datetime import datetime
    return datetime.utcnow().strftime("%Y-%m-%d")


def build_parent_folder(video_basename: str) -> str:
    """
    Parent folder name = date_videoname
    e.g. 2026-02-06_DJI_20260206174112_0754_D
    """
    date_str = extract_date_from_basename(video_basename)
    return f"{date_str}_{video_basename}"


def build_s3_input_prefix(video_basename: str) -> str:
    """
    bucket/Chattisgarh_.../2026-02-06_DJI_.../input/
    """
    s = get_settings()
    parent = build_parent_folder(video_basename)
    return f"{s.s3_output_prefix}/{parent}/input"


def build_s3_temp_prefix(video_basename: str) -> str:
    """
    bucket/Chattisgarh_.../2026-02-06_DJI_.../temp/
    Contains: defects/ (crops) + frames/
    Safe to delete after review.
    """
    s = get_settings()
    parent = build_parent_folder(video_basename)
    return f"{s.s3_output_prefix}/{parent}/temp"


def build_s3_output_prefix(video_basename: str) -> str:
    """
    bucket/Chattisgarh_.../2026-02-06_DJI_.../output/
    Contains: annotated_video/ + reports/
    Keep forever.
    """
    s = get_settings()
    parent = build_parent_folder(video_basename)
    return f"{s.s3_output_prefix}/{parent}/output"


def download_from_s3(s3_url: str, local_path: Path) -> Path:
    bucket, key = parse_s3_url(s3_url)
    client = _build_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading from S3", extra={"key": key})
    try:
        client.download_file(bucket, key, str(local_path))
        size_mb = local_path.stat().st_size / 1_048_576
        logger.info("Download complete", extra={"key": key, "size_mb": round(size_mb, 2)})
        return local_path
    except ClientError as e:
        logger.error("S3 download failed", extra={"key": key, "error": str(e)})
        raise


def upload_file_to_s3(
    local_path: Path,
    s3_key: str,
    bucket: Optional[str] = None,
) -> str:
    s = get_settings()
    bucket = bucket or s.s3_bucket_name
    client = _build_client()
    content_type, _ = mimetypes.guess_type(str(local_path))
    extra = {"ContentType": content_type or "application/octet-stream"}
    try:
        client.upload_file(str(local_path), bucket, s3_key, ExtraArgs=extra)
        s3_url = f"s3://{bucket}/{s3_key}"
        logger.info("Uploaded", extra={"s3_url": s3_url})
        return s3_url
    except ClientError as e:
        logger.error("Upload failed", extra={"s3_key": s3_key, "error": str(e)})
        raise


def upload_directory_to_s3(
    local_dir: Path,
    s3_prefix: str,
    bucket: Optional[str] = None,
    max_workers: int = 8,
) -> dict[str, str]:
    s = get_settings()
    bucket = bucket or s.s3_bucket_name
    all_files = [f for f in local_dir.rglob("*") if f.is_file()]
    results: dict[str, str] = {}

    def _upload_one(fpath: Path) -> tuple[str, str]:
        rel    = fpath.relative_to(local_dir).as_posix()
        s3_key = f"{s3_prefix.rstrip('/')}/{rel}"
        s3_url = upload_file_to_s3(fpath, s3_key, bucket)
        return rel, s3_url

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_upload_one, f): f for f in all_files}
        for future in as_completed(futures):
            try:
                rel, url = future.result()
                results[rel] = url
            except Exception as exc:
                logger.error("Upload failed", extra={
                    "file": str(futures[future]), "error": str(exc)
                })

    logger.info("Directory upload complete", extra={"files": len(results), "prefix": s3_prefix})
    return results


def generate_presigned_url(s3_url: str, expiry_seconds: int = 86400) -> str:
    bucket, key = parse_s3_url(s3_url)
    client = _build_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expiry_seconds,
    )
