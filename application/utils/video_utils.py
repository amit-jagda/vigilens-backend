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


def extract_masked_crop(
    frame: np.ndarray,
    mask: np.ndarray | None,
    x1: int, y1: int, x2: int, y2: int,
    background: str = "black"   # "black" | "white" | "blur"
) -> np.ndarray:
    """
    Extracts a person crop from the frame using the segmentation mask.
    Background pixels are zeroed (black), set to white, or blurred
    depending on the background parameter.

    Falls back to plain bounding box crop if mask is None or shape mismatch.

    Args:
        frame:      Full video frame (H, W, 3) BGR
        mask:       Full-frame boolean segmentation mask (H, W) for this person
                    Output of detections.mask[i] from YOLO-seg
        x1,y1,x2,y2: Bounding box coordinates (already clamped to frame bounds)
        background: How to fill non-person pixels

    Returns:
        Masked crop (y2-y1, x2-x1, 3) BGR — safe to pass directly to
        reid_service.extract_embedding() and face_rec_service.extract_faces()
    """
    body_crop = frame[y1:y2, x1:x2].copy()

    if mask is None:
        return body_crop  # graceful fallback

    person_mask = mask[y1:y2, x1:x2]  # crop mask to bounding box region

    if person_mask.shape[:2] != body_crop.shape[:2]:
        return body_crop  # shape mismatch — fallback

    if background == "white":
        body_crop[~person_mask] = 255
    elif background == "blur":
        blurred = cv2.GaussianBlur(body_crop, (21, 21), 0)
        body_crop[~person_mask] = blurred[~person_mask]
    else:
        body_crop[~person_mask] = 0  # black — default, best for ReID models

    return body_crop


def extract_head_crop_from_mask(
    frame: np.ndarray,
    mask: np.ndarray | None,
    x1: int, y1: int, x2: int, y2: int,
    head_fraction: float = 0.30
) -> np.ndarray:
    """
    Extracts the head/face region from the top `head_fraction` of the
    segmentation mask bounding box. Used for face recognition crop
    instead of the padded full-body crop.

    Falls back to padded bounding box if mask unavailable.
    """
    h = y2 - y1
    w = x2 - x1

    if mask is None:
        # Fallback: top 30% of bounding box with 10% padding
        pad_y = int(h * 0.10)
        pad_x = int(w * 0.10)
        fh, fw = frame.shape[:2]
        hy1 = max(0, y1 - pad_y)
        hy2 = min(fh, y1 + int(h * head_fraction) + pad_y)
        hx1 = max(0, x1 - pad_x)
        hx2 = min(fw, x2 + pad_x)
        return frame[hy1:hy2, hx1:hx2].copy()

    # Use mask: find topmost pixel rows with person pixels
    person_mask = mask[y1:y2, x1:x2]
    head_h = max(int(h * head_fraction), 60)  # minimum 60px head region

    head_crop = frame[y1:min(y2, y1 + head_h), x1:x2].copy()
    head_mask = person_mask[:head_crop.shape[0], :]

    if head_mask.shape[:2] == head_crop.shape[:2]:
        head_crop[~head_mask] = 0  # zero background in head region too

    return head_crop

