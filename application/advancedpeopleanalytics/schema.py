import uuid
import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field, computed_field

# ==========================================
# CAMERA & SPATIAL TOPOLOGY SCHEMAS
# ==========================================

class CameraNodeCreate(BaseModel):
    name: str = Field(..., example="Reception")
    location_label: Optional[str] = Field(None, example="Main Entrance Floor 1")


class CameraNodeResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    location_label: Optional[str] = None
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

    @computed_field
    def duration_seconds(self) -> float:
        if self.ended_at and self.started_at:
            return round((self.ended_at - self.started_at).total_seconds(), 2)
        return 0.0

    model_config = ConfigDict(from_attributes=True)


class PersonTimelineResponse(BaseModel):
    person_id: uuid.UUID
    person_type: str
    date: datetime.date
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
    camera_node_id: Optional[uuid.UUID] = None
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


class AdvancedPeopleAnalyticsSessionResponse(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    video_name: str
    video_path: str
    output_video_path: Optional[str] = None
    status: str
    session_type: Optional[str] = None
    camera_node_id: Optional[uuid.UUID] = None
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
    completed_percentage: Optional[int] = 0
    created_at: datetime.datetime
    completed_at: Optional[datetime.datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SessionDetectedPerson(BaseModel):
    identity_id: uuid.UUID
    person_type: str
    name: str
    crop_url: Optional[str] = None
    first_seen: float
    last_seen: float
    occurrences_count: int

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
    first_name: str
    last_name: str
    registration_type: str = Field("visitor", example="visitor")
    employee_code: Optional[str] = None
    department: Optional[str] = None
