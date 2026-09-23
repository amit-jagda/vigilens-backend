import uuid
from typing import Optional, List
from datetime import datetime, date
from sqlalchemy import String, Integer, Float, ForeignKey, Boolean, DateTime, Date, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from pgvector.sqlalchemy import Vector

from database.base import BaseModel
from modules.gallery.model import GalleryMedia

class AdvancedUploadedVideo(BaseModel):
    """
    Tracks raw CCTV video files uploaded per tenant in Advanced People Analytics.
    """
    __tablename__ = "advanced_uploaded_videos"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    user_id: Mapped[uuid.UUID | None] = mapped_column(nullable=True)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    saved_path: Mapped[str] = mapped_column(String(512), nullable=False)


class AdvancedPeopleAnalyticsSession(BaseModel):
    """
    Represents an advanced video/image analytics processing session combining people and face ReID.
    """
    __tablename__ = "advanced_people_analytics_sessions"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    video_name: Mapped[str] = mapped_column(String(255), nullable=False)
    video_path: Mapped[str] = mapped_column(String(512), nullable=False)
    output_video_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    session_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    
    camera_node_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("camera_nodes.id", ondelete="SET NULL"), nullable=True)
    gallery_media_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("gallery_media.id", ondelete="SET NULL"), nullable=True
    )
    recording_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Coordinates of counting line
    line_start: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    line_end: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    
    similarity_threshold: Mapped[float] = mapped_column(Float, default=0.85)
    confidence_threshold: Mapped[float] = mapped_column(Float, default=0.3)
    
    # Execution flags
    track_employees: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    register_new_visitors: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    track_repeat_visitors: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    line_crossing_analysis: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    track_occupancy: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    track_objects: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    classes_to_track: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    generate_video: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    
    # Time Range / Sub-Clip Processing (seconds)
    start_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_time_sec: Mapped[float | None] = mapped_column(Float, nullable=True)
    
    # Results
    unique_person_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_person_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_time_visitor_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peak_occupancy: Mapped[int | None] = mapped_column(Integer, nullable=True)
    average_occupancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    entry_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    employee_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    visitor_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    
    occupancy_timeline: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)
    detected_objects_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    camera_node: Mapped[Optional["CameraNode"]] = relationship(back_populates="sessions")
    gallery_media: Mapped[Optional["GalleryMedia"]] = relationship()
    employee_attendance: Mapped[list["AdvancedEmployeeAttendanceLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    employee_session_detections: Mapped[list["AdvancedEmployeeSessionDetection"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    visitor_attendance: Mapped[list["AdvancedVisitorAttendanceLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    occurrences: Mapped[list["AdvancedPersonOccurrence"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    crossings: Mapped[list["AdvancedLineCrossingLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    zone_crossing_events: Mapped[list["ZoneCrossingEvent"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class AdvancedEmployeeAttendanceLog(BaseModel):
    """
    Logs presence of registered employees detected in an advanced session.
    """
    __tablename__ = "advanced_employee_attendance_logs"

    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=True)
    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    first_seen: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_seen: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)

    employee_entry_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"), nullable=False)
    employee_exit_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"), nullable=False)

    session: Mapped[Optional["AdvancedPeopleAnalyticsSession"]] = relationship(back_populates="employee_attendance")
    employee: Mapped["Employee"] = relationship()


class AdvancedPersonIdentity(BaseModel):
    """
    Represents a unique visitor or employee tracked across runs (class_id=0 for ReID body/fusion, 1 for Face-only).
    """
    __tablename__ = "advanced_person_identities"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    class_id: Mapped[int] = mapped_column(Integer, default=0) # 0 = person/ReID, 1 = face
    first_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    visitor_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    is_employee: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    employee_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("employees.id", ondelete="SET NULL"), nullable=True)

    employee: Mapped[Optional["Employee"]] = relationship()
    embeddings: Mapped[list["AdvancedPersonEmbedding"]] = relationship(
        back_populates="identity", cascade="all, delete-orphan"
    )
    occurrences: Mapped[list["AdvancedPersonOccurrence"]] = relationship(
        back_populates="identity", cascade="all, delete-orphan"
    )
    crossings: Mapped[list["AdvancedLineCrossingLog"]] = relationship(
        back_populates="identity", cascade="all, delete-orphan"
    )
    attendance_logs: Mapped[list["AdvancedVisitorAttendanceLog"]] = relationship(
        back_populates="identity", cascade="all, delete-orphan"
    )


class AdvancedPersonEmbedding(BaseModel):
    """
    Stores 512-dimensional visual embeddings of visitors.
    """
    __tablename__ = "advanced_person_embeddings"

    identity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="CASCADE"), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(512), nullable=False)
    bbox: Mapped[list[int] | None] = mapped_column(ARRAY(Integer), nullable=True)
    timestamp: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding_type: Mapped[str] = mapped_column(
        String(20), default="appearance", server_default="appearance"
    )  # "face" | "appearance"

    # NEW — segmentation metadata
    is_segmented: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="false"
    )  # True = extracted from masked crop, False = plain bbox crop
    mask_coverage: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )  # fraction of bbox pixels that are person pixels (0.0–1.0)
    recorded_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    face_anchored: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")

    identity: Mapped["AdvancedPersonIdentity"] = relationship(back_populates="embeddings")


class AdvancedPersonOccurrence(BaseModel):
    """
    Logs separate track appearances of a person inside an advanced session.
    """
    __tablename__ = "advanced_person_occurrences"

    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=False)
    identity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="CASCADE"), nullable=False)
    tracker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    first_seen: Mapped[float] = mapped_column(Float, nullable=False)
    last_seen: Mapped[float] = mapped_column(Float, nullable=False)
    crop_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    associated_objects: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    session: Mapped["AdvancedPeopleAnalyticsSession"] = relationship(back_populates="occurrences")
    identity: Mapped["AdvancedPersonIdentity"] = relationship(back_populates="occurrences")


class AdvancedLineCrossingLog(BaseModel):
    """
    Logs line crossing events in advanced sessions.
    """
    __tablename__ = "advanced_line_crossing_logs"

    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=False)
    identity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="CASCADE"), nullable=False)
    tracker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[float] = mapped_column(Float, nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)

    session: Mapped["AdvancedPeopleAnalyticsSession"] = relationship(back_populates="crossings")
    identity: Mapped["AdvancedPersonIdentity"] = relationship(back_populates="crossings")


class AdvancedVisitorAttendanceLog(BaseModel):
    """
    Logs presence of unique visitors detected in advanced sessions day-wise.
    """
    __tablename__ = "advanced_visitor_attendance_logs"

    session_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=True)
    identity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="CASCADE"), nullable=False)
    first_seen: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_seen: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)

    visitor_entry_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"), nullable=False)
    visitor_exit_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=text("now()"), nullable=False)

    session: Mapped[Optional["AdvancedPeopleAnalyticsSession"]] = relationship(back_populates="visitor_attendance")
    identity: Mapped["AdvancedPersonIdentity"] = relationship(back_populates="attendance_logs")

    @property
    def photo_path(self) -> Optional[str]:
        if self.identity and self.identity.occurrences:
            for occ in self.identity.occurrences:
                if occ.crop_path:
                    return occ.crop_path
        return None

    @property
    def first_name(self) -> Optional[str]:
        return self.identity.first_name if self.identity else None

    @property
    def last_name(self) -> Optional[str]:
        return self.identity.last_name if self.identity else None


class AdvancedEmployeeSessionDetection(BaseModel):
    """
    Logs presence of registered employees detected in an advanced session.
    """
    __tablename__ = "advanced_employee_session_detections"

    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=False)
    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    first_seen: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    last_seen: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    occurrence_count: Mapped[int] = mapped_column(Integer, default=1)

    session: Mapped["AdvancedPeopleAnalyticsSession"] = relationship(back_populates="employee_session_detections")
    employee: Mapped["Employee"] = relationship()


# ==========================================
# CAMERA TOPOLOGY & SPATIAL GRAPH MODELS
# ==========================================

class FloorPlan(BaseModel):
    """
    Represents an architectural floor plan or layout canvas for a tenant.
    """
    __tablename__ = "floor_plans"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    image_filepath: Mapped[str | None] = mapped_column(String(512), nullable=True)
    canvas_width_px: Mapped[int] = mapped_column(Integer, default=1920, server_default="1920", nullable=False)
    canvas_height_px: Mapped[int] = mapped_column(Integer, default=1080, server_default="1080", nullable=False)
    scale_meters_per_px: Mapped[float | None] = mapped_column(Float, nullable=True)

    camera_nodes: Mapped[list["CameraNode"]] = relationship("CameraNode", back_populates="floor_plan")
    spatial_lines: Mapped[list["SpatialLine"]] = relationship("SpatialLine", back_populates="floor_plan", cascade="all, delete-orphan")


class SpatialLine(BaseModel):
    """
    Represents wall, corridor, or boundary lines drawn on a floor plan canvas.
    """
    __tablename__ = "spatial_lines"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    floor_plan_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("floor_plans.id", ondelete="CASCADE"), nullable=False)
    x1: Mapped[float] = mapped_column(Float, nullable=False)
    y1: Mapped[float] = mapped_column(Float, nullable=False)
    x2: Mapped[float] = mapped_column(Float, nullable=False)
    y2: Mapped[float] = mapped_column(Float, nullable=False)
    line_type: Mapped[str] = mapped_column(String(50), default="wall", server_default="wall", nullable=False)
    label: Mapped[str | None] = mapped_column(String(255), nullable=True)

    floor_plan: Mapped["FloorPlan"] = relationship("FloorPlan", back_populates="spatial_lines")


class CameraNode(BaseModel):
    """
    Represents a physical or virtual camera node location.
    """
    __tablename__ = "camera_nodes"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False) # e.g. "Reception"
    location_label: Mapped[str | None] = mapped_column(String(255), nullable=True) # e.g. "Main Entrance Floor 1"
    floor_plan_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("floor_plans.id", ondelete="SET NULL"), nullable=True)
    x_coord: Mapped[float | None] = mapped_column(Float, nullable=True)
    y_coord: Mapped[float | None] = mapped_column(Float, nullable=True)
    fov_angle: Mapped[float | None] = mapped_column(Float, default=0.0, server_default="0.0", nullable=True)

    floor_plan: Mapped["FloorPlan | None"] = relationship("FloorPlan", back_populates="camera_nodes")
    sessions: Mapped[list["AdvancedPeopleAnalyticsSession"]] = relationship(back_populates="camera_node")
    zones: Mapped[list["CameraZone"]] = relationship(back_populates="camera", cascade="all, delete-orphan")
    outgoing_node_links: Mapped[list["CameraNodeLink"]] = relationship(
        "CameraNodeLink",
        foreign_keys="[CameraNodeLink.from_camera_id]",
        back_populates="from_camera",
        cascade="all, delete-orphan"
    )
    incoming_node_links: Mapped[list["CameraNodeLink"]] = relationship(
        "CameraNodeLink",
        foreign_keys="[CameraNodeLink.to_camera_id]",
        back_populates="to_camera",
        cascade="all, delete-orphan"
    )


class CameraNodeLink(BaseModel):
    """
    Represents directed spatial connections and travel time parameters between camera nodes (Camera-to-Camera Topology).
    """
    __tablename__ = "camera_node_links"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    from_camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("camera_nodes.id", ondelete="CASCADE"), nullable=False)
    to_camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("camera_nodes.id", ondelete="CASCADE"), nullable=False)
    min_transit_seconds: Mapped[float] = mapped_column(Float, default=5.0, nullable=False)
    avg_transit_seconds: Mapped[float] = mapped_column(Float, default=30.0, nullable=False)
    max_transit_seconds: Mapped[float] = mapped_column(Float, default=300.0, nullable=False)

    from_camera: Mapped["CameraNode"] = relationship(foreign_keys=[from_camera_id], back_populates="outgoing_node_links")
    to_camera: Mapped["CameraNode"] = relationship(foreign_keys=[to_camera_id], back_populates="incoming_node_links")


class CameraZone(BaseModel):
    """
    Represents a designated spatial zone inside a camera's field of view.
    """
    __tablename__ = "camera_zones"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("camera_nodes.id", ondelete="CASCADE"), nullable=False)
    zone_type: Mapped[str] = mapped_column(String(20), nullable=False) # 'entry' | 'exit' | 'crossing'
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    polygon: Mapped[dict] = mapped_column(JSONB, nullable=False) # list of points [[x1, y1], [x2, y2], ...]

    camera: Mapped["CameraNode"] = relationship(back_populates="zones")
    outgoing_links: Mapped[list["CameraZoneLink"]] = relationship(
        foreign_keys="[CameraZoneLink.from_zone_id]", cascade="all, delete-orphan"
    )
    incoming_links: Mapped[list["CameraZoneLink"]] = relationship(
        foreign_keys="[CameraZoneLink.to_zone_id]", cascade="all, delete-orphan"
    )


class CameraZoneLink(BaseModel):
    """
    Represents spatial connections and travel time parameters between camera zones.
    """
    __tablename__ = "camera_zone_links"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    from_zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("camera_zones.id", ondelete="CASCADE"), nullable=False)
    to_zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("camera_zones.id", ondelete="CASCADE"), nullable=False)
    avg_transit_seconds: Mapped[float] = mapped_column(Float, default=30.0, nullable=False)
    max_transit_seconds: Mapped[float] = mapped_column(Float, default=300.0, nullable=False)

    from_zone: Mapped["CameraZone"] = relationship(foreign_keys=[from_zone_id], back_populates="outgoing_links")
    to_zone: Mapped["CameraZone"] = relationship(foreign_keys=[to_zone_id], back_populates="incoming_links")


class ZoneCrossingEvent(BaseModel):
    """
    Logs track entry/exit/crossing events through camera zones.
    """
    __tablename__ = "zone_crossing_events"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=False)
    zone_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("camera_zones.id", ondelete="CASCADE"), nullable=False)
    identity_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="SET NULL"), nullable=True)
    tracker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False) # 'entry' | 'exit' | 'crossing'
    real_world_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reid_embedding: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)
    face_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    mask_coverage: Mapped[float | None] = mapped_column(Float, nullable=True)

    session: Mapped["AdvancedPeopleAnalyticsSession"] = relationship(back_populates="zone_crossing_events")
    zone: Mapped["CameraZone"] = relationship()
    identity: Mapped[Optional["AdvancedPersonIdentity"]] = relationship()


class CrossCameraIdentityLink(BaseModel):
    """
    Stores cross-camera association scores between person identities across sessions.
    """
    __tablename__ = "cross_camera_identity_links"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    from_session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=False)
    to_session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=False)
    from_identity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="CASCADE"), nullable=False)
    to_identity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="CASCADE"), nullable=False)
    reid_score: Mapped[float] = mapped_column(Float, nullable=False)
    time_score: Mapped[float] = mapped_column(Float, nullable=False)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(50), default="spatial_reid_fusion")
    is_confirmed: Mapped[bool] = mapped_column(Boolean, default=False)


class PersonTimelineEvent(BaseModel):
    """
    Logs complete person journey timeline across cameras and zones.
    """
    __tablename__ = "person_timeline_events"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    person_type: Mapped[str] = mapped_column(String(20), nullable=False) # 'employee' | 'visitor'
    employee_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("employees.id", ondelete="SET NULL"), nullable=True)
    identity_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("advanced_person_identities.id", ondelete="SET NULL"), nullable=True)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("advanced_people_analytics_sessions.id", ondelete="CASCADE"), nullable=False)
    camera_name: Mapped[str] = mapped_column(String(255), nullable=False)
    zone_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False) # 'entry' | 'exit' | 'presence'
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    identity_source: Mapped[str] = mapped_column(String(50), nullable=False) # 'face' | 'reid' | 'face+reid' | 'tracking'
    identity_confidence: Mapped[float] = mapped_column(Float, default=1.0)
    tracker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entry_crop_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    exit_crop_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    associated_objects: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)

    @property
    def duration_seconds(self) -> float:
        if hasattr(self, "ended_at") and hasattr(self, "started_at") and self.ended_at and self.started_at:
            return max(0.0, float((self.ended_at - self.started_at).total_seconds()))
        return 0.0


class AdvancedEmployeeDailyCheckin(BaseModel):
    """
    Stores employee daily check-in anchors (face photo and outfit/appearance photo).
    """
    __tablename__ = "advanced_employee_daily_checkins"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False)
    employee_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("employees.id", ondelete="CASCADE"), nullable=False)
    checkin_date: Mapped[date] = mapped_column(Date, nullable=False)
    face_photo_path: Mapped[str] = mapped_column(String(512), nullable=False)
    appearance_photo_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    face_anchored: Mapped[bool] = mapped_column(Boolean, default=True, server_default="true")
    appearance_anchored: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false")
    notes: Mapped[str | None] = mapped_column(String(512), nullable=True)

    employee: Mapped["Employee"] = relationship()


