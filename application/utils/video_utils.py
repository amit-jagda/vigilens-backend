import subprocess
import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    has_timestamp: bool
    recording_started_at: Optional[datetime]   # None = not found
    timestamp_source: Optional[str]            # metadata field source
    duration_seconds: float
    fps: float
    width: Optional[int]
    height: Optional[int]
    codec: Optional[str]
    has_gps: bool
    gps_raw: Optional[str]
    recommendation: str                        # "AUTO_USE" or "ASK_USER"
    probe_error: Optional[str] = None


def probe_video(filepath: str, ffprobe_cmd: str = "ffprobe") -> VideoMetadata:
    """
    Runs ffprobe on the video file to extract recording timestamp and properties.
    """
    try:
        proc = subprocess.run(
            [
                ffprobe_cmd, "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                filepath
            ],
            capture_output=True,
            text=True,
            timeout=15
        )
        if proc.returncode != 0:
            return _empty_result(filepath, probe_error="ffprobe non-zero exit code")

        data = json.loads(proc.stdout)
        fmt = data.get("format", {})
        fmt_tags = fmt.get("tags", {})
        streams = data.get("streams", [])
        vid = next((s for s in streams if s.get("codec_type") == "video"), {})
        vid_tags = vid.get("tags", {})

    except Exception as e:
        return _empty_result(filepath, probe_error=str(e))

    candidates = [
        ("format.creation_time", fmt_tags.get("creation_time")),
        ("stream.creation_time", vid_tags.get("creation_time")),
        ("com.android.capture.fps", fmt_tags.get("com.android.capture.fps")),
    ]

    recording_started_at: Optional[datetime] = None
    timestamp_source: Optional[str] = None

    for field, raw in candidates:
        if not raw:
            continue
        dt = _parse_timestamp(raw)
        if dt is None:
            continue
        recording_started_at = dt
        timestamp_source = field
        break

    gps = fmt_tags.get("location") or fmt_tags.get("location-eng")
    fps = _parse_fps(vid.get("r_frame_rate", "0/1"))

    return VideoMetadata(
        has_timestamp=recording_started_at is not None,
        recording_started_at=recording_started_at,
        timestamp_source=timestamp_source,
        duration_seconds=round(float(fmt.get("duration") or 0), 2),
        fps=fps,
        width=vid.get("width"),
        height=vid.get("height"),
        codec=vid.get("codec_name"),
        has_gps=gps is not None,
        gps_raw=gps,
        recommendation="AUTO_USE" if recording_started_at else "ASK_USER"
    )


def _parse_timestamp(raw: str) -> Optional[datetime]:
    if not raw:
        return None
    try:
        raw = raw.strip().replace("Z", "+00:00")
        if "T" not in raw and " " in raw:
            raw = raw.replace(" ", "T", 1)
        raw = re.sub(r'([+-])(\d{2})(\d{2})$', r'\1\2:\3', raw)

        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        if dt.year < 2000 or dt.year > 2100:
            return None

        return dt
    except Exception:
        return None


def _parse_fps(raw: str) -> float:
    try:
        num, den = raw.split("/")
        return round(float(num) / float(den), 2)
    except Exception:
        return 0.0


def _empty_result(filepath: str, probe_error: Optional[str] = None) -> VideoMetadata:
    return VideoMetadata(
        has_timestamp=False, recording_started_at=None, timestamp_source=None,
        duration_seconds=0.0, fps=0.0, width=None, height=None, codec=None,
        has_gps=False, gps_raw=None, recommendation="ASK_USER", probe_error=probe_error
    )


def get_video_rotation(filepath: str) -> int:
    """Detect rotation angle (0, 90, 180, 270) from video stream metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                for side in stream.get("side_data_list", []):
                    if "rotation" in side:
                        rot = int(float(side["rotation"]))
                        return (rot % 360 + 360) % 360
                tags = stream.get("tags", {})
                if "rotate" in tags:
                    rot = int(float(tags["rotate"]))
                    return (rot % 360 + 360) % 360
    except Exception as e:
        logger.debug(f"Could not determine video rotation via ffprobe: {e}")
    return 0


def orient_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    """Apply cv2 rotation/flip based on rotation angle."""
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame
