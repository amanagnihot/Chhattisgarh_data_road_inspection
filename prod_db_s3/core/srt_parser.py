# ============================================================
# core/srt_parser.py — DJI SRT file parser + chainage calc
# ============================================================

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def parse_srt(srt_path: Path) -> Optional[list[dict]]:
    """
    Parse DJI SRT files. Handles two known formats:

    Format A (older DJI):
        4
        00:00:00,049 --> 00:00:00,066
        <font size="28">FrameCnt: 4, DiffTime: 17ms
        2026-01-10 10:19:23.934
        [latitude: 20.665134] [longitude: 81.484726] [abs_alt: 309.551] ...

    Format B (newer DJI):
        8718
        00:04:50,838 --> 00:04:50,871
        <font size="36">SrtCnt : 8718, DiffTime : 33ms
        2026-01-09 15:31:21,948,194
        [latitude: 20.536977] [longitude: 80.962568] [altitude: 52.800000] ...
    """
    srt_data = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip HTML tags
    content = re.sub(r"<[^>]+>", "", content)

    blocks = re.split(r"\n\n+", content.strip())
    for block in blocks:
        if not block.strip():
            continue
        lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
        if len(lines) < 3:
            continue
        try:
            frame_num = int(lines[0])

            tc = re.search(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", lines[1])
            if not tc:
                continue
            h, m, s, ms = map(int, tc.groups())
            timestamp_ms = (h * 3600 + m * 60 + s) * 1000 + ms

            meta = " ".join(lines[2:])

            absolute_timestamp = None
            ts_a = re.search(
                r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})\.(\d+)", meta
            )
            if ts_a:
                ts_str = f"{ts_a.group(1)} {ts_a.group(2)}.{ts_a.group(3)[:6]}"
                absolute_timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
            else:
                ts_b = re.search(
                    r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}),(\d+)", meta
                )
                if ts_b:
                    ms_part = ts_b.group(3)[:6].ljust(6, "0")
                    ts_str  = f"{ts_b.group(1)} {ts_b.group(2)}.{ms_part}"
                    absolute_timestamp = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")

            if absolute_timestamp is None:
                continue

            lat = re.search(r"\[latitude\s*:\s*([-\d.]+)\]", meta)
            lon = re.search(r"\[longitude\s*:\s*([-\d.]+)\]", meta)
            if not (lat and lon):
                continue

            alt = re.search(r"\[(?:abs_alt|altitude)\s*:\s*([-\d.]+)", meta)

            srt_data.append(
                {
                    "frame_number":       frame_num,
                    "timestamp_ms":       timestamp_ms,
                    "absolute_timestamp": absolute_timestamp,
                    "latitude":           float(lat.group(1)),
                    "longitude":          float(lon.group(1)),
                    "altitude":           float(alt.group(1)) if alt else 50.0,
                }
            )
        except Exception:
            continue

    if not srt_data:
        logger.error("No valid GPS data parsed from SRT", extra={"path": str(srt_path)})
        return None

    logger.info("SRT parsed", extra={"entries": len(srt_data), "path": str(srt_path)})
    return srt_data


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R  = 6_371_000
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def calculate_cumulative_chainage(
    srt_data: list[dict], starting_chainage_m: float = 0.0
) -> float:
    """Attach cumulative_chainage_m to each entry; return ending chainage."""
    cum = starting_chainage_m
    for i, entry in enumerate(srt_data):
        if i == 0:
            entry["cumulative_chainage_m"] = starting_chainage_m
        else:
            prev = srt_data[i - 1]
            cum += haversine_distance(
                prev["latitude"], prev["longitude"],
                entry["latitude"], entry["longitude"],
            )
            entry["cumulative_chainage_m"] = cum

    logger.info(
        "Chainage calculated",
        extra={
            "start_m": starting_chainage_m,
            "end_m":   round(cum, 2),
            "km":      round((cum - starting_chainage_m) / 1000, 3),
        },
    )
    return cum


def get_srt_data_for_frame(frame_index: int, srt_data: list[dict]) -> Optional[dict]:
    """Return the SRT entry whose frame_number best matches frame_index."""
    for entry in srt_data:
        if entry["frame_number"] == frame_index:
            return entry
    if srt_data:
        return min(srt_data, key=lambda x: abs(x["frame_number"] - frame_index))
    return None