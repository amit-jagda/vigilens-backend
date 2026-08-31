import uuid
import datetime
from typing import List, Optional, Tuple
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from application.advancedpeopleanalytics.model import (
    AdvancedPeopleAnalyticsSession,
    AdvancedPersonIdentity,
    AdvancedPersonEmbedding,
    AdvancedPersonOccurrence,
    AdvancedLineCrossingLog,
    AdvancedEmployeeAttendanceLog,
    AdvancedVisitorAttendanceLog,
    AdvancedEmployeeSessionDetection,
    AdvancedUploadedVideo,
    CameraNode,
    CameraZone,
    CameraZoneLink,
    ZoneCrossingEvent,
    CrossCameraIdentityLink,
    PersonTimelineEvent
)
from modules.employees.model import Employee

class AdvancedPeopleAnalyticsRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ==========================================
    # CAMERA & SPATIAL TOPOLOGY CRUD
    # ==========================================

    async def create_camera_node(
        self,
        tenant_id: uuid.UUID,
        name: str,
        location_label: Optional[str] = None
    ) -> CameraNode:
        node = CameraNode(
            tenant_id=tenant_id,
            name=name,
            location_label=location_label
        )
        self.db.add(node)
        await self.db.flush()
        return node

    async def get_camera_nodes(self, tenant_id: uuid.UUID) -> List[CameraNode]:
        stmt = select(CameraNode).where(
            CameraNode.tenant_id == tenant_id,
            CameraNode.is_delete == False
        ).order_by(CameraNode.created_at.desc())
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def get_camera_node_by_id(self, node_id: uuid.UUID, tenant_id: uuid.UUID) -> Optional[CameraNode]:
        stmt = select(CameraNode).where(
            CameraNode.id == node_id,
            CameraNode.tenant_id == tenant_id,
            CameraNode.is_delete == False
        )
        res = await self.db.execute(stmt)
        return res.scalars().first()

    async def create_camera_zone(
        self,
        tenant_id: uuid.UUID,
        camera_id: uuid.UUID,
        zone_type: str,
        label: str,
        polygon: list
    ) -> CameraZone:
        zone = CameraZone(
            tenant_id=tenant_id,
            camera_id=camera_id,
            zone_type=zone_type,
            label=label,
            polygon=polygon
        )
        self.db.add(zone)
        await self.db.flush()
        return zone

    async def get_camera_zones(self, camera_id: uuid.UUID) -> List[CameraZone]:
        stmt = select(CameraZone).where(
            CameraZone.camera_id == camera_id,
            CameraZone.is_delete == False
        ).order_by(CameraZone.created_at.asc())
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def create_zone_link(
        self,
        tenant_id: uuid.UUID,
        from_zone_id: uuid.UUID,
        to_zone_id: uuid.UUID,
        avg_transit_seconds: float = 30.0,
        max_transit_seconds: float = 300.0
    ) -> CameraZoneLink:
        link = CameraZoneLink(
            tenant_id=tenant_id,
            from_zone_id=from_zone_id,
            to_zone_id=to_zone_id,
            avg_transit_seconds=avg_transit_seconds,
            max_transit_seconds=max_transit_seconds
        )
        self.db.add(link)
        await self.db.flush()
        return link

    async def get_zone_links_from(self, from_zone_id: uuid.UUID) -> List[CameraZoneLink]:
        stmt = select(CameraZoneLink).where(
            CameraZoneLink.from_zone_id == from_zone_id,
            CameraZoneLink.is_delete == False
        )
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def create_zone_crossing_event(
        self,
        tenant_id: uuid.UUID,
        session_id: uuid.UUID,
        zone_id: uuid.UUID,
        tracker_id: int,
        event_type: str,
        real_world_time: datetime.datetime,
        identity_id: Optional[uuid.UUID] = None,
        reid_embedding: Optional[list[float]] = None,
        face_confirmed: bool = False,
        confidence: float = 1.0
    ) -> ZoneCrossingEvent:
        evt = ZoneCrossingEvent(
            tenant_id=tenant_id,
            session_id=session_id,
            zone_id=zone_id,
            identity_id=identity_id,
            tracker_id=tracker_id,
            event_type=event_type,
            real_world_time=real_world_time,
            reid_embedding=reid_embedding,
            face_confirmed=face_confirmed,
            confidence=confidence
        )
        self.db.add(evt)
        await self.db.flush()
        return evt

    async def get_zone_crossing_events(
        self,
        zone_id: uuid.UUID,
        time_window_start: datetime.datetime,
        time_window_end: datetime.datetime
    ) -> List[ZoneCrossingEvent]:
        stmt = select(ZoneCrossingEvent).where(
            ZoneCrossingEvent.zone_id == zone_id,
            ZoneCrossingEvent.real_world_time >= time_window_start,
            ZoneCrossingEvent.real_world_time <= time_window_end,
            ZoneCrossingEvent.is_delete == False
        )
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def create_cross_camera_link(
        self,
        tenant_id: uuid.UUID,
        from_session_id: uuid.UUID,
        to_session_id: uuid.UUID,
        from_identity_id: uuid.UUID,
        to_identity_id: uuid.UUID,
        reid_score: float,
        time_score: float,
        final_score: float,
        method: str = "spatial_reid_fusion",
        is_confirmed: bool = False
    ) -> CrossCameraIdentityLink:
        link = CrossCameraIdentityLink(
            tenant_id=tenant_id,
            from_session_id=from_session_id,
            to_session_id=to_session_id,
            from_identity_id=from_identity_id,
            to_identity_id=to_identity_id,
            reid_score=reid_score,
            time_score=time_score,
            final_score=final_score,
            method=method,
            is_confirmed=is_confirmed
        )
        self.db.add(link)
        await self.db.flush()
        return link

    async def create_timeline_event(
        self,
        tenant_id: uuid.UUID,
        person_type: str,
        session_id: uuid.UUID,
        camera_name: str,
        event_type: str,
        started_at: datetime.datetime,
        ended_at: datetime.datetime,
        identity_source: str,
        identity_confidence: float,
        tracker_id: int,
        employee_id: Optional[uuid.UUID] = None,
        identity_id: Optional[uuid.UUID] = None,
        zone_name: Optional[str] = None
    ) -> PersonTimelineEvent:
        event = PersonTimelineEvent(
            tenant_id=tenant_id,
            person_type=person_type,
            employee_id=employee_id,
            identity_id=identity_id,
            session_id=session_id,
            camera_name=camera_name,
            zone_name=zone_name,
            event_type=event_type,
            started_at=started_at,
            ended_at=ended_at,
            identity_source=identity_source,
            identity_confidence=identity_confidence,
            tracker_id=tracker_id
        )
        self.db.add(event)
        await self.db.flush()
        return event

    async def get_person_timeline(
        self,
        tenant_id: uuid.UUID,
        person_type: str,
        person_id: uuid.UUID,
        target_date: datetime.date
    ) -> List[PersonTimelineEvent]:
        start_dt = datetime.datetime.combine(target_date, datetime.time.min, tzinfo=datetime.timezone.utc)
        end_dt = datetime.datetime.combine(target_date, datetime.time.max, tzinfo=datetime.timezone.utc)

        if person_type == "employee":
            stmt = select(PersonTimelineEvent).where(
                PersonTimelineEvent.tenant_id == tenant_id,
                PersonTimelineEvent.person_type == "employee",
                PersonTimelineEvent.employee_id == person_id,
                PersonTimelineEvent.started_at >= start_dt,
                PersonTimelineEvent.started_at <= end_dt,
                PersonTimelineEvent.is_delete == False
            ).order_by(PersonTimelineEvent.started_at.asc())
        else:
            stmt = select(PersonTimelineEvent).where(
                PersonTimelineEvent.tenant_id == tenant_id,
                PersonTimelineEvent.person_type == "visitor",
                PersonTimelineEvent.identity_id == person_id,
                PersonTimelineEvent.started_at >= start_dt,
                PersonTimelineEvent.started_at <= end_dt,
                PersonTimelineEvent.is_delete == False
            ).order_by(PersonTimelineEvent.started_at.asc())

        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    # ==========================================
    # SESSION MANAGEMENT
    # ==========================================

    async def create_analytics_session(
        self,
        tenant_id: uuid.UUID,
        video_name: str,
        video_path: str,
        line_start: Optional[list[int]] = None,
        line_end: Optional[list[int]] = None,
        similarity_threshold: float = 0.85,
        confidence_threshold: float = 0.3,
        track_employees: bool = True,
        register_new_visitors: bool = True,
        track_repeat_visitors: bool = True,
        line_crossing_analysis: bool = True,
        track_occupancy: bool = True,
        camera_node_id: Optional[uuid.UUID] = None,
        recording_started_at: Optional[datetime.datetime] = None
    ) -> AdvancedPeopleAnalyticsSession:
        session = AdvancedPeopleAnalyticsSession(
            tenant_id=tenant_id,
            video_name=video_name,
            video_path=video_path,
            line_start=line_start,
            line_end=line_end,
            similarity_threshold=similarity_threshold,
            confidence_threshold=confidence_threshold,
            track_employees=track_employees,
            register_new_visitors=register_new_visitors,
            track_repeat_visitors=track_repeat_visitors,
            line_crossing_analysis=line_crossing_analysis,
            track_occupancy=track_occupancy,
            camera_node_id=camera_node_id,
            recording_started_at=recording_started_at,
            status="pending"
        )
        self.db.add(session)
        await self.db.flush()
        return session

    async def get_session_by_id(self, session_id: uuid.UUID, tenant_id: uuid.UUID) -> Optional[AdvancedPeopleAnalyticsSession]:
        stmt = select(AdvancedPeopleAnalyticsSession).where(
            AdvancedPeopleAnalyticsSession.id == session_id,
            AdvancedPeopleAnalyticsSession.tenant_id == tenant_id,
            AdvancedPeopleAnalyticsSession.is_delete == False
        )
        res = await self.db.execute(stmt)
        return res.scalars().first()

    async def get_all_sessions(self, tenant_id: uuid.UUID) -> List[AdvancedPeopleAnalyticsSession]:
        stmt = select(AdvancedPeopleAnalyticsSession).where(
            AdvancedPeopleAnalyticsSession.tenant_id == tenant_id,
            AdvancedPeopleAnalyticsSession.is_delete == False
        ).order_by(AdvancedPeopleAnalyticsSession.created_at.desc())
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def update_session_results(
        self,
        session_id: uuid.UUID,
        unique_person_count: Optional[int] = None,
        total_person_count: Optional[int] = None,
        first_time_visitor_count: Optional[int] = None,
        peak_occupancy: Optional[int] = None,
        average_occupancy: Optional[float] = None,
        entry_count: Optional[int] = None,
        exit_count: Optional[int] = None,
        employee_count: Optional[int] = None,
        visitor_count: Optional[int] = None,
        occupancy_timeline: Optional[List[dict]] = None,
        output_video_path: Optional[str] = None,
        status: str = "completed"
    ):
        stmt = select(AdvancedPeopleAnalyticsSession).where(
            AdvancedPeopleAnalyticsSession.id == session_id,
            AdvancedPeopleAnalyticsSession.is_delete == False
        )
        res = await self.db.execute(stmt)
        session = res.scalars().first()
        if not session:
            return

        session.status = status
        if unique_person_count is not None:
            session.unique_person_count = unique_person_count
        if total_person_count is not None:
            session.total_person_count = total_person_count
        if first_time_visitor_count is not None:
            session.first_time_visitor_count = first_time_visitor_count
        if peak_occupancy is not None:
            session.peak_occupancy = peak_occupancy
        if average_occupancy is not None:
            session.average_occupancy = average_occupancy
        if entry_count is not None:
            session.entry_count = entry_count
        if exit_count is not None:
            session.exit_count = exit_count
        if employee_count is not None:
            session.employee_count = employee_count
        if visitor_count is not None:
            session.visitor_count = visitor_count
        if occupancy_timeline is not None:
            session.occupancy_timeline = occupancy_timeline
        if output_video_path is not None:
            session.output_video_path = output_video_path
        session.completed_at = datetime.datetime.now(datetime.timezone.utc)

    async def delete_session(self, session_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        session = await self.get_session_by_id(session_id, tenant_id)
        if not session:
            return False
        session.is_delete = True
        await self.db.flush()
        return True

    # ==========================================
    # VISITOR & RE-ID EMBEDDINGS
    # ==========================================

    async def find_similar_visitor(
        self,
        tenant_id: uuid.UUID,
        embedding: List[float],
        threshold: float,
        class_id: int = 0
    ) -> Optional[Tuple[AdvancedPersonIdentity, float]]:
        stmt = (
            select(AdvancedPersonEmbedding, AdvancedPersonIdentity)
            .join(AdvancedPersonIdentity, AdvancedPersonEmbedding.identity_id == AdvancedPersonIdentity.id)
            .where(
                AdvancedPersonIdentity.tenant_id == tenant_id,
                AdvancedPersonIdentity.class_id == class_id,
                AdvancedPersonIdentity.is_delete == False,
                AdvancedPersonEmbedding.is_delete == False
            )
            .order_by(AdvancedPersonEmbedding.embedding.cosine_distance(embedding))
            .limit(1)
        )
        res = await self.db.execute(stmt)
        row = res.first()
        if not row:
            return None

        emb_obj, identity = row
        import numpy as np
        vec1 = np.array(embedding, dtype=np.float32)
        vec2 = np.array(emb_obj.embedding, dtype=np.float32)
        
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return None
        sim = float(np.dot(vec1, vec2) / (n1 * n2))

        if sim >= threshold:
            return identity, sim
        return None

    async def create_person_identity(self, tenant_id: uuid.UUID, class_id: int = 0) -> AdvancedPersonIdentity:
        identity = AdvancedPersonIdentity(tenant_id=tenant_id, class_id=class_id)
        self.db.add(identity)
        await self.db.flush()
        return identity

    async def create_person_embedding(
        self,
        identity_id: uuid.UUID,
        embedding: List[float],
        bbox: Optional[List[int]] = None,
        timestamp: Optional[float] = None
    ) -> AdvancedPersonEmbedding:
        emb = AdvancedPersonEmbedding(
            identity_id=identity_id,
            embedding=embedding,
            bbox=bbox,
            timestamp=timestamp
        )
        self.db.add(emb)
        await self.db.flush()
        return emb

    async def create_person_occurrence(
        self,
        session_id: uuid.UUID,
        identity_id: uuid.UUID,
        tracker_id: int,
        first_seen: float,
        last_seen: float,
        crop_path: Optional[str] = None
    ) -> AdvancedPersonOccurrence:
        occ = AdvancedPersonOccurrence(
            session_id=session_id,
            identity_id=identity_id,
            tracker_id=tracker_id,
            first_seen=first_seen,
            last_seen=last_seen,
            crop_path=crop_path
        )
        self.db.add(occ)
        await self.db.flush()
        return occ

    async def log_visitor_attendance(
        self,
        session_id: uuid.UUID,
        identity_id: uuid.UUID,
        first_seen_sec: float,
        last_seen_sec: float,
        occurrence_increment: int = 1
    ) -> AdvancedVisitorAttendanceLog:
        session = await self.db.get(AdvancedPeopleAnalyticsSession, session_id)
        session_created = session.recording_started_at or session.created_at or datetime.datetime.now(datetime.timezone.utc)
        
        entry_ts = session_created + datetime.timedelta(seconds=first_seen_sec)
        exit_ts = session_created + datetime.timedelta(seconds=last_seen_sec)
        target_date = entry_ts.date()
        
        stmt = select(AdvancedVisitorAttendanceLog).where(
            AdvancedVisitorAttendanceLog.identity_id == identity_id,
            func.date(AdvancedVisitorAttendanceLog.visitor_entry_timestamp) == target_date,
            AdvancedVisitorAttendanceLog.is_delete == False
        )
        res = await self.db.execute(stmt)
        existing = res.scalars().first()
        
        if existing:
            existing.occurrence_count += occurrence_increment
            if exit_ts > existing.visitor_exit_timestamp:
                existing.visitor_exit_timestamp = exit_ts
                existing.last_seen = last_seen_sec
            if entry_ts < existing.visitor_entry_timestamp:
                existing.visitor_entry_timestamp = entry_ts
                existing.first_seen = first_seen_sec
            await self.db.flush()
            return existing
        else:
            log = AdvancedVisitorAttendanceLog(
                session_id=session_id,
                identity_id=identity_id,
                first_seen=first_seen_sec,
                last_seen=last_seen_sec,
                occurrence_count=occurrence_increment,
                visitor_entry_timestamp=entry_ts,
                visitor_exit_timestamp=exit_ts
            )
            self.db.add(log)
            await self.db.flush()
            return log

    async def create_employee_attendance(
        self,
        session_id: uuid.UUID,
        employee_id: uuid.UUID,
        first_seen: float,
        last_seen: float,
        occurrence_count: int = 1
    ) -> AdvancedEmployeeAttendanceLog:
        session = await self.db.get(AdvancedPeopleAnalyticsSession, session_id)
        session_created = session.recording_started_at or session.created_at or datetime.datetime.now(datetime.timezone.utc)
        
        entry_ts = session_created + datetime.timedelta(seconds=first_seen)
        exit_ts = session_created + datetime.timedelta(seconds=last_seen)

        log = AdvancedEmployeeAttendanceLog(
            session_id=session_id,
            employee_id=employee_id,
            first_seen=first_seen,
            last_seen=last_seen,
            occurrence_count=occurrence_count,
            employee_entry_timestamp=entry_ts,
            employee_exit_timestamp=exit_ts
        )
        self.db.add(log)
        await self.db.flush()
        return log

    async def create_line_crossing_log(
        self,
        session_id: uuid.UUID,
        identity_id: uuid.UUID,
        tracker_id: int,
        timestamp: float,
        direction: str
    ) -> AdvancedLineCrossingLog:
        log = AdvancedLineCrossingLog(
            session_id=session_id,
            identity_id=identity_id,
            tracker_id=tracker_id,
            timestamp=timestamp,
            direction=direction
        )
        self.db.add(log)
        await self.db.flush()
        return log
