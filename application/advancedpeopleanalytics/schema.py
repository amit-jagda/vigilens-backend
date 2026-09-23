import uuid
import datetime
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

# ==========================================
# CAMERA & SPATIAL TOPOLOGY SCHEMAS
# ==========================================

class CameraNodeCreate(BaseModel):
    name: str = Field(..., example="Reception")
    location_label: Optional[str] = Field(None, example="Main Entrance Floor 1")
    location_desc: Optional[str] = None
    label: Optional[str] = None


class CameraNodeUpdate(BaseModel):
    name: Optional[str] = None
    location_label: Optional[str] = None
    location_desc: Optional[str] = None
    label: Optional[str] = None


class CameraNodeResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    location_label: Optional[str] = None
    floor_plan_id: Optional[uuid.UUID] = None
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    fov_angle: Optional[float] = 0.0
    created_at: datetime.datetime
    update_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class CameraNodeLayoutUpdate(BaseModel):
    id: uuid.UUID
    x_coord: Optional[float] = None
    y_coord: Optional[float] = None
    fov_angle: Optional[float] = None


class SpatialLineUpsert(BaseModel):
    id: Optional[Union[uuid.UUID, str]] = None  # null or non-UUID = new line
    x1: float
    y1: float
    x2: float
    y2: float
    line_type: str = "wall"
    label: Optional[str] = None

    @field_validator("id", mode="before")
    @classmethod
    def sanitize_id(cls, v: Any) -> Optional[uuid.UUID]:
        if not v:
            return None
        if isinstance(v, uuid.UUID):
            return v
        try:
            return uuid.UUID(str(v))
        except (ValueError, AttributeError):
            return None


class SaveLayoutRequest(BaseModel):
    camera_nodes: List[CameraNodeLayoutUpdate] = Field(default_factory=list)
    lines: List[SpatialLineUpsert] = Field(default_factory=list)


class FloorPlanCreate(BaseModel):
    name: str = Field(..., example="Ground Floor Main Layout")
    image_filepath: Optional[str] = None
    canvas_width_px: int = 1920
    canvas_height_px: int = 1080
    scale_meters_per_px: Optional[float] = None


class SpatialLineResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    floor_plan_id: uuid.UUID
    x1: float
    y1: float
    x2: float
    y2: float
    line_type: str
    label: Optional[str] = None
    created_at: datetime.datetime
    update_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class FloorPlanResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    image_filepath: Optional[str] = None
    canvas_width_px: int
    canvas_height_px: int
    scale_meters_per_px: Optional[float] = None
    created_at: datetime.datetime
    update_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class FloorPlanLayoutResponse(BaseModel):
    floor_plan: FloorPlanResponse
    camera_nodes: List[CameraNodeResponse]
    lines: List[SpatialLineResponse]

    model_config = ConfigDict(from_attributes=True)


class CameraNodeLinkCreate(BaseModel):
    from_camera_id: uuid.UUID
    to_camera_id: uuid.UUID
    min_transit_seconds: float = Field(5.0, ge=0.0)
    avg_transit_seconds: float = Field(30.0, ge=0.0)
    max_transit_seconds: float = Field(300.0, ge=0.0)
    is_bidirectional: bool = Field(False, description="If true, also creates reverse link to_camera -> from_camera")


class CameraNodeLinkResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    from_camera_id: uuid.UUID
    to_camera_id: uuid.UUID
    from_camera_name: Optional[str] = None
    to_camera_name: Optional[str] = None
    min_transit_seconds: float
    avg_transit_seconds: float
    max_transit_seconds: float
    created_at: datetime.datetime
    update_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class CameraZoneCreate(BaseModel):
    zone_type: str = Field(..., example="exit") # "entry" | "exit" | "crossing"
    label: str = Field(..., example="Lobby Main Exit Gate")
    polygon: List[List[int]] = Field(..., example=[[100, 200], [300, 200], [300, 400], [100, 400]])


class CameraZoneResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    camera_id: uuid.UUID
    zone_type: str
    label: str
    polygon: List[List[int]]
    created_at: datetime.datetime
    update_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class CameraZoneLinkCreate(BaseModel):
    from_zone_id: uuid.UUID
    to_zone_id: uuid.UUID
    avg_transit_seconds: float = Field(30.0, ge=0.0)
    max_transit_seconds: float = Field(300.0, ge=0.0)


class CameraZoneLinkResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    from_zone_id: uuid.UUID
    to_zone_id: uuid.UUID
    avg_transit_seconds: float
    max_transit_seconds: float
    created_at: datetime.datetime
    update_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# TIMELINE & ASSOCIATION SCHEMAS
# ==========================================

class TimelineEventResponse(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    camera_name: str
    zone_name: Optional[str] = None
    event_type: str
    started_at: datetime.datetime
    ended_at: datetime.datetime
    identity_source: str
    identity_confidence: float
    tracker_id: int
    entry_crop_path: Optional[str] = None
    exit_crop_path: Optional[str] = None
    entry_crop_url: Optional[str] = None
    exit_crop_url: Optional[str] = None
    associated_objects: Optional[List[str]] = None

    @computed_field
    def duration_seconds(self) -> float:
        if self.ended_at and self.started_at:
            return round((self.ended_at - self.started_at).total_seconds(), 2)
        return 0.0

    model_config = ConfigDict(from_attributes=True)


class PersonTimelineResponse(BaseModel):
    person_id: uuid.UUID
    person_type: str
    date: Optional[datetime.date] = None
    events: List[TimelineEventResponse] = Field(default_factory=list)


class AssociationRequest(BaseModel):
    session_ids: List[uuid.UUID]


# ==========================================
# SESSION & ANALYTICS SCHEMAS
# ==========================================

class AdvancedVideoProcessItem(BaseModel):
    gallery_media_id: Optional[str] = None
    direct_video_path: Optional[str] = None
    line_start: Optional[List[int]] = None
    line_end: Optional[List[int]] = None
    similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    track_employees: Optional[bool] = None
    register_new_visitors: Optional[bool] = None
    track_repeat_visitors: Optional[bool] = None
    line_crossing_analysis: Optional[bool] = None
    track_occupancy: Optional[bool] = None
    track_objects: Optional[bool] = None
    classes_to_track: Optional[List[str]] = None
    generate_video: Optional[bool] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    camera_node_id: Optional[uuid.UUID] = None
    camera_id: Optional[uuid.UUID] = None
    camera_name: Optional[str] = None
    recording_started_at: Optional[datetime.datetime] = None


class ProcessAdvancedVideosRequest(BaseModel):
    videos: List[AdvancedVideoProcessItem]
    line_start: Optional[List[int]] = None
    line_end: Optional[List[int]] = None
    similarity_threshold: float = Field(0.85, ge=0.0, le=1.0)
    confidence_threshold: float = Field(0.3, ge=0.0, le=1.0)
    track_employees: bool = True
    register_new_visitors: bool = True
    track_repeat_visitors: bool = True
    line_crossing_analysis: bool = True
    track_occupancy: bool = True
    track_objects: bool = True
    classes_to_track: Optional[List[str]] = None
    generate_video: bool = False
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class AdvancedPeopleAnalyticsSessionResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    video_name: str
    video_path: str
    output_video_path: Optional[str] = None
    status: str
    session_type: Optional[str] = None
    camera_node_id: Optional[uuid.UUID] = None
    gallery_media_id: Optional[uuid.UUID] = None
    recording_started_at: Optional[datetime.datetime] = None
    line_start: Optional[List[int]] = None
    line_end: Optional[List[int]] = None
    similarity_threshold: float
    confidence_threshold: float
    track_employees: bool
    register_new_visitors: bool
    track_repeat_visitors: bool
    line_crossing_analysis: bool
    track_occupancy: bool
    track_objects: bool = True
    classes_to_track: Optional[List[str]] = None
    generate_video: bool = False
    start_time_sec: Optional[float] = None
    end_time_sec: Optional[float] = None
    unique_person_count: Optional[int] = None
    total_person_count: Optional[int] = None
    first_time_visitor_count: Optional[int] = None
    peak_occupancy: Optional[int] = None
    average_occupancy: Optional[float] = None
    entry_count: Optional[int] = None
    exit_count: Optional[int] = None
    employee_count: Optional[int] = None
    visitor_count: Optional[int] = None
    occupancy_timeline: Optional[List[dict]] = None
    detected_objects_summary: Optional[Dict[str, int]] = None
    completed_percentage: Optional[int] = 0
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PersonAppearanceSegment(BaseModel):
    first_seen_sec: float
    last_seen_sec: float
    duration_seconds: float
    formatted_time: str

    model_config = ConfigDict(from_attributes=True)


class SessionDetectedPerson(BaseModel):
    identity_id: Optional[uuid.UUID] = None
    employee_id: Optional[uuid.UUID] = None
    person_type: str
    name: str
    tracker_id: int
    crop_url: Optional[str] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    first_seen_sec: float = 0.0
    last_seen_sec: float = 0.0
    duration_seconds: float = 0.0
    formatted_time: Optional[str] = None
    zone_name: Optional[str] = None
    sequence_number: Optional[int] = None
    started_at: Optional[datetime.datetime] = None
    ended_at: Optional[datetime.datetime] = None
    confidence: float = 1.0
    identity_source: str = "tracking"
    camera_name: Optional[str] = None
    appearances_count: int = 1
    segments: List[PersonAppearanceSegment] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class FirstTimeVisitorDetail(BaseModel):
    identity_id: str
    visit_timestamp: datetime.datetime
    crop_url: Optional[str] = None


class VisitorAnalyticsReport(BaseModel):
    total_unique_visitors: int
    repeat_visit_rate: float
    new_visitors_today: int
    first_time_visitors: List[FirstTimeVisitorDetail] = Field(default_factory=list)


class VisitorAttendanceResponse(BaseModel):
    id: uuid.UUID
    identity_id: uuid.UUID
    session_id: Optional[uuid.UUID] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    first_seen: float
    last_seen: float
    occurrence_count: int
    visitor_entry_timestamp: datetime.datetime
    visitor_exit_timestamp: datetime.datetime
    photo_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class RegisterVisitorRequest(BaseModel):
    identity_id: uuid.UUID
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    registration_type: str = Field("visitor", description="'visitor' | 'new_employee' | 'link_existing_employee'")
    employee_code: Optional[str] = None
    existing_employee_id: Optional[uuid.UUID] = None
    department: Optional[str] = None
    retroactive_attendance: bool = Field(True, description="If true, converts visitor attendance logs to employee attendance")
    force: bool = Field(False, description="If true, bypasses the 0.15 low similarity safeguard")


class PersonSummaryItem(BaseModel):
    person_type: str  # "employee" | "visitor"
    person_id: uuid.UUID
    identity_id: Optional[uuid.UUID] = None
    name: str
    employee_code: Optional[str] = None
    crop_url: Optional[str] = None
    total_dwell_seconds: float
    camera_stops_count: int
    cameras_visited: List[str]
    first_seen_at: datetime.datetime
    last_seen_at: datetime.datetime
    latest_event_type: str = "presence"

    model_config = ConfigDict(from_attributes=True)


# ==========================================
# SUB-CLIP & PHOTO DWELL ANALYTICS SCHEMAS
# ==========================================

class CreateSubclipRequest(BaseModel):
    session_id: Optional[uuid.UUID] = None
    gallery_media_id: Optional[uuid.UUID] = None
    video_path: Optional[str] = None
    start_time: str = Field(..., description="Start time formatted as HH:MM:SS or seconds float/int (e.g. '00:12:00' or '720')")
    end_time: str = Field(..., description="End time formatted as HH:MM:SS or seconds float/int (e.g. '00:12:05' or '725')")
    save_to_gallery: bool = Field(True, description="If true, registers subclip into gallery_media")


class SubclipResponse(BaseModel):
    clip_path: str
    clip_url: str
    duration_seconds: float
    gallery_media_id: Optional[uuid.UUID] = None


class PlaceDwellItem(BaseModel):
    camera_name: str
    zone_name: Optional[str] = None
    duration_seconds: float
    formatted_duration: str
    visit_count: int


class PersonDwellByPhotoResponse(BaseModel):
    matched: bool
    identity_id: Optional[uuid.UUID] = None
    person_type: Optional[str] = None  # "employee" | "visitor"
    name: Optional[str] = None
    employee_code: Optional[str] = None
    confidence: float = 0.0
    total_dwell_seconds: float = 0.0
    formatted_total_dwell: str = "0s"
    first_seen_at: Optional[datetime.datetime] = None
    last_seen_at: Optional[datetime.datetime] = None
    placewise_dwell: List[PlaceDwellItem] = Field(default_factory=list)
    timeline_events: List[TimelineEventResponse] = Field(default_factory=list)


# ==========================================
# DAILY CHECK-IN, HOURLY DWELL & REVIEW QUEUE
# ==========================================

class DailyCheckinResponse(BaseModel):
    employee_id: uuid.UUID
    employee_name: str
    employee_code: Optional[str] = None
    checkin_date: datetime.date
    face_registered: bool
    appearance_anchored: bool
    face_photo_url: Optional[str] = None
    appearance_photo_url: Optional[str] = None
    message: str


class DailyCheckinRecordResponse(BaseModel):
    id: uuid.UUID
    employee_id: uuid.UUID
    employee_name: str
    employee_code: Optional[str] = None
    employee_photo: Optional[str] = None
    checkin_date: datetime.date
    face_photo_url: str
    appearance_photo_url: Optional[str] = None
    face_anchored: bool = True
    appearance_anchored: bool = False
    created_at: datetime.datetime



class HourlyAreaDwellItem(BaseModel):
    hour: str
    hour_int: int
    total_dwell_seconds: float
    portion_of_hour: float  # 0.0 to 1.0 (max 1 hour)
    formatted_duration: str
    areas: Dict[str, float] = Field(default_factory=dict)  # area_name -> seconds


class HourlyDwellResponse(BaseModel):
    target_date: datetime.date
    person_id: Optional[uuid.UUID] = None
    person_name: Optional[str] = None
    all_areas: List[str] = Field(default_factory=list)
    hourly_data: List[HourlyAreaDwellItem] = Field(default_factory=list)


class ReviewQueueCandidate(BaseModel):
    identity_id: uuid.UUID
    crop_url: Optional[str] = None
    first_seen_at: Optional[datetime.datetime] = None
    last_seen_at: Optional[datetime.datetime] = None
    total_dwell_seconds: float = 0.0
    camera_stops_count: int = 0
    cameras_visited: List[str] = Field(default_factory=list)
    suggested_employee_id: Optional[uuid.UUID] = None
    suggested_employee_name: Optional[str] = None
    suggested_similarity: Optional[float] = None


class ReconcileIdentityRequest(BaseModel):
    target_employee_id: Optional[uuid.UUID] = None
    or_visitor_name: Optional[str] = None
    auto_merge_similar: bool = True
    similarity_threshold: float = Field(0.65, ge=0.0, le=1.0)


# ==========================================
# SEARCH BY PHOTO / REFERENCE IMAGE
# ==========================================

class PhotoSearchAppearanceItem(BaseModel):
    session_id: Optional[uuid.UUID] = None
    camera_id: Optional[uuid.UUID] = None
    camera_name: Optional[str] = "Unknown Camera"
    zone_name: Optional[str] = None
    timestamp: datetime.datetime
    timestamp_offset_seconds: Optional[float] = None
    dwell_seconds: Optional[float] = None
    crop_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    event_type: Optional[str] = "appearance"


class PhotoSearchMatchItem(BaseModel):
    identity_type: str  # "employee" | "visitor"
    identity_id: uuid.UUID
    name: str
    code: Optional[str] = None
    similarity_score: float  # e.g. 0.88
    similarity_percentage: str  # e.g. "88%"
    matched_via: str  # "face" | "appearance" | "fusion"
    primary_photo_url: Optional[str] = None
    total_appearances: int = 0
    first_seen_at: Optional[datetime.datetime] = None
    last_seen_at: Optional[datetime.datetime] = None
    timeline_events: List[PhotoSearchAppearanceItem] = Field(default_factory=list)


class PhotoSearchResponse(BaseModel):
    query_processed: bool = True
    face_detected_in_query: bool = False
    appearance_extracted: bool = False
    total_matches_found: int = 0
    matches: List[PhotoSearchMatchItem] = Field(default_factory=list)

