from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime
from typing import Optional, List, Dict

class PlateDetection(BaseModel):
    plate_text: str
    confidence: float
    ocr_confidence: float = 0.85
    first_seen: float
    last_seen: float
    total_appearances: int
    bbox_sample: Optional[List[int]] = None
    thumbnail_path: Optional[str] = None


class NumberplateReport(BaseModel):
    total_unique_plates: int
    plates: List[PlateDetection] = []
    most_frequent_plate: Optional[str] = None
    detection_timeline: List[Dict] = []


class DamageZone(BaseModel):
    damage_class: str
    severity_score: float
    affected_area_pct: float = 0.0
    confidence: float
    first_seen: float
    last_seen: float
    thumbnail_path: Optional[str] = None


class DamageReport(BaseModel):
    overall_condition: str
    overall_severity_score: float
    total_damage_zones: int
    damage_breakdown: Dict[str, int] = {}
    zones: List[DamageZone] = []
    inspection_verdict: str
    verdict_reason: str


class PersonPPEStatus(BaseModel):
    person_id: int
    is_compliant: bool
    detected_ppe: List[str] = []
    missing_ppe: List[str] = []
    compliance_score: float = 1.0
    first_seen: float
    last_seen: float
    thumbnail_path: Optional[str] = None


class PPEReport(BaseModel):
    total_persons_detected: int
    compliant_count: int
    non_compliant_count: int
    compliance_rate_pct: float
    ppe_item_stats: Dict[str, int] = {}
    persons: List[PersonPPEStatus] = []
    violation_timestamps: List[Dict] = []
    required_ppe: List[str] = []


class ObjectCountResultResponse(BaseModel):
    id: UUID
    media_id: UUID
    track_id: int
    class_name: str
    gender: Optional[str] = None
    first_frame: int
    last_frame: int
    total_frames: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ObjectCountMediaResponse(BaseModel):
    id: UUID
    gallery_media_id: UUID
    status: str
    classify_gender: bool
    classify_vehicle: bool
    detect_numberplate: bool = False
    detect_damage_parcel: bool = False
    detect_ppe: bool = False
    classes_to_track: Optional[List[str]] = None
    total_objects_count: Optional[int] = None
    peak_objects_count: Optional[int] = None
    average_objects_count: Optional[float] = None
    video_duration_seconds: Optional[float] = None
    report_summary: Optional[Dict] = None
    numberplate_results: Optional[Dict] = None
    damage_results: Optional[Dict] = None
    ppe_results: Optional[Dict] = None
    progress_percentage: int = 0
    created_at: datetime

    # Populated from gallery_media relationship
    filename: Optional[str] = None
    filepath: Optional[str] = None
    processed_filepath: Optional[str] = None
    media_type: Optional[str] = None

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        instance = super().model_validate(obj, *args, **kwargs)
        # Handle lazy/eager loading of gallery_media relationship
        gallery_media = getattr(obj, "gallery_media", None)
        if gallery_media:
            instance.filename = gallery_media.filename
            instance.filepath = gallery_media.filepath
            instance.processed_filepath = gallery_media.processed_filepath
            instance.media_type = gallery_media.media_type
        return instance

    class Config:
        from_attributes = True


class ObjectCountMediaDetailResponse(ObjectCountMediaResponse):
    results: List[ObjectCountResultResponse] = []


class ObjectCountAnalyzeRequest(BaseModel):
    gallery_media_id: UUID
    classes_to_track: Optional[List[str]] = None
    classify_gender: bool = False
    classify_vehicle: bool = False
    detect_numberplate: bool = False
    detect_damage_parcel: bool = False
    detect_ppe: bool = False
    required_ppe_items: Optional[List[str]] = None
    confidence_threshold: float = 0.35
    min_track_frames: int = 100
    track_buffer: int = 150
    gmc_method: str = "none"
    reid_classes: Optional[List[str]] = None
    imgsz: int = 480
    entry_exit_report: bool = False
    line_coords: Optional[List[List[int]]] = None
    device: Optional[str] = None
