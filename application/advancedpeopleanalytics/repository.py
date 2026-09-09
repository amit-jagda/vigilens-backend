import uuid
import datetime
from typing import List, Optional, Tuple
from sqlalchemy import select, update, delete, func, and_, or_, text
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
    CameraNodeLink,
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

    async def update_camera_node(
        self,
        node_id: uuid.UUID,
        tenant_id: uuid.UUID,
        name: Optional[str] = None,
        location_label: Optional[str] = None
    ) -> Optional[CameraNode]:
        node = await self.get_camera_node_by_id(node_id, tenant_id)
        if not node:
            return None
        if name is not None:
            node.name = name
        if location_label is not None:
            node.location_label = location_label
        await self.db.flush()
        return node

    async def delete_camera_node(self, node_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        stmt = select(CameraNode).where(
            CameraNode.id == node_id,
            CameraNode.tenant_id == tenant_id,
            CameraNode.is_delete == False
        )
        res = await self.db.execute(stmt)
        node = res.scalars().first()
        if not node:
            return False
        node.is_delete = True
        
        # Also mark connected camera links as deleted
        link_stmt = update(CameraNodeLink).where(
            or_(
                CameraNodeLink.from_camera_id == node_id,
                CameraNodeLink.to_camera_id == node_id
            ),
            CameraNodeLink.tenant_id == tenant_id
        ).values(is_delete=True)
        await self.db.execute(link_stmt)
        await self.db.flush()
        return True

    # ==========================================
    # CAMERA-TO-CAMERA TOPOLOGY LINKS
    # ==========================================

    async def create_camera_node_link(
        self,
        tenant_id: uuid.UUID,
        from_camera_id: uuid.UUID,
        to_camera_id: uuid.UUID,
        min_transit_seconds: float = 5.0,
        avg_transit_seconds: float = 30.0,
        max_transit_seconds: float = 300.0
    ) -> CameraNodeLink:
        link = CameraNodeLink(
            tenant_id=tenant_id,
            from_camera_id=from_camera_id,
            to_camera_id=to_camera_id,
            min_transit_seconds=min_transit_seconds,
            avg_transit_seconds=avg_transit_seconds,
            max_transit_seconds=max_transit_seconds
        )
        self.db.add(link)
        await self.db.flush()
        return link

    async def get_camera_node_links(self, tenant_id: uuid.UUID) -> List[CameraNodeLink]:
        stmt = select(CameraNodeLink).options(
            selectinload(CameraNodeLink.from_camera),
            selectinload(CameraNodeLink.to_camera)
        ).where(
            CameraNodeLink.tenant_id == tenant_id,
            CameraNodeLink.is_delete == False
        ).order_by(CameraNodeLink.created_at.desc())
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def get_camera_node_links_from(self, from_camera_id: uuid.UUID) -> List[CameraNodeLink]:
        stmt = select(CameraNodeLink).where(
            CameraNodeLink.from_camera_id == from_camera_id,
            CameraNodeLink.is_delete == False
        )
        res = await self.db.execute(stmt)
        return list(res.scalars().all())

    async def delete_camera_node_link(self, link_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        stmt = select(CameraNodeLink).where(
            CameraNodeLink.id == link_id,
            CameraNodeLink.tenant_id == tenant_id,
            CameraNodeLink.is_delete == False
        )
        res = await self.db.execute(stmt)
        link = res.scalars().first()
        if not link:
            return False
        link.is_delete = True

        # Also soft-delete reverse / reciprocal link between the same pair of cameras if it exists
        stmt_reverse = select(CameraNodeLink).where(
            CameraNodeLink.from_camera_id == link.to_camera_id,
            CameraNodeLink.to_camera_id == link.from_camera_id,
            CameraNodeLink.tenant_id == tenant_id,
            CameraNodeLink.is_delete == False
        )
        res_reverse = await self.db.execute(stmt_reverse)
        for rev in res_reverse.scalars().all():
            rev.is_delete = True

        await self.db.flush()
        return True

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
        confidence: float = 1.0,
        mask_coverage: Optional[float] = None
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
            confidence=confidence,
            mask_coverage=mask_coverage
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
        target_date: Optional[datetime.date] = None
    ) -> List[PersonTimelineEvent]:
        if person_type == "employee":
            stmt = select(PersonTimelineEvent).where(
                PersonTimelineEvent.tenant_id == tenant_id,
                PersonTimelineEvent.person_type == "employee",
                PersonTimelineEvent.employee_id == person_id,
                PersonTimelineEvent.is_delete == False
            )
            if target_date:
                start_dt = datetime.datetime.combine(target_date, datetime.time.min, tzinfo=datetime.timezone.utc)
                end_dt = datetime.datetime.combine(target_date, datetime.time.max, tzinfo=datetime.timezone.utc)
                stmt = stmt.where(
                    PersonTimelineEvent.started_at >= start_dt,
                    PersonTimelineEvent.started_at <= end_dt
                )
            stmt = stmt.order_by(PersonTimelineEvent.started_at.asc())
            res = await self.db.execute(stmt)
            return list(res.scalars().all())
        else:
            # Find all cross-camera linked identity IDs for this visitor
            linked_ids = {person_id}
            link_stmt = select(CrossCameraIdentityLink).where(
                CrossCameraIdentityLink.tenant_id == tenant_id,
                or_(
                    CrossCameraIdentityLink.from_identity_id == person_id,
                    CrossCameraIdentityLink.to_identity_id == person_id
                ),
                CrossCameraIdentityLink.is_delete == False
            )
            link_res = await self.db.execute(link_stmt)
            for lk in link_res.scalars().all():
                linked_ids.add(lk.from_identity_id)
                linked_ids.add(lk.to_identity_id)

            stmt = select(PersonTimelineEvent).where(
                PersonTimelineEvent.tenant_id == tenant_id,
                PersonTimelineEvent.person_type == "visitor",
                PersonTimelineEvent.identity_id.in_(list(linked_ids)),
                PersonTimelineEvent.is_delete == False
            )
            if target_date:
                start_dt = datetime.datetime.combine(target_date, datetime.time.min, tzinfo=datetime.timezone.utc)
                end_dt = datetime.datetime.combine(target_date, datetime.time.max, tzinfo=datetime.timezone.utc)
                stmt = stmt.where(
                    PersonTimelineEvent.started_at >= start_dt,
                    PersonTimelineEvent.started_at <= end_dt
                )
            stmt = stmt.order_by(PersonTimelineEvent.started_at.asc())
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
        generate_video: bool = False,
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
            generate_video=generate_video,
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
        await self.db.execute(
            delete(PersonTimelineEvent).where(
                PersonTimelineEvent.session_id == session_id,
                PersonTimelineEvent.tenant_id == tenant_id
            )
        )
        await self.db.flush()
        return True

    async def clear_session_results(self, session_id: uuid.UUID):
        """
        Clears all past detection records, logs, and events for a session prior to rerunning.
        """
        await self.db.execute(delete(AdvancedEmployeeAttendanceLog).where(AdvancedEmployeeAttendanceLog.session_id == session_id))
        await self.db.execute(delete(AdvancedEmployeeSessionDetection).where(AdvancedEmployeeSessionDetection.session_id == session_id))
        await self.db.execute(delete(AdvancedVisitorAttendanceLog).where(AdvancedVisitorAttendanceLog.session_id == session_id))
        await self.db.execute(delete(AdvancedPersonOccurrence).where(AdvancedPersonOccurrence.session_id == session_id))
        await self.db.execute(delete(AdvancedLineCrossingLog).where(AdvancedLineCrossingLog.session_id == session_id))
        await self.db.execute(delete(ZoneCrossingEvent).where(ZoneCrossingEvent.session_id == session_id))
        await self.db.execute(delete(PersonTimelineEvent).where(PersonTimelineEvent.session_id == session_id))
        await self.db.flush()

    async def get_or_create_employee_identity(
        self,
        tenant_id: uuid.UUID,
        employee_id: uuid.UUID,
        employee_name: Optional[str] = None
    ) -> AdvancedPersonIdentity:
        stmt = select(AdvancedPersonIdentity).where(
            AdvancedPersonIdentity.tenant_id == tenant_id,
            AdvancedPersonIdentity.is_employee == True,
            AdvancedPersonIdentity.employee_id == employee_id,
            AdvancedPersonIdentity.is_delete == False
        )
        res = await self.db.execute(stmt)
        ident = res.scalars().first()
        if not ident:
            ident = AdvancedPersonIdentity(
                tenant_id=tenant_id,
                class_id=0,
                is_employee=True,
                employee_id=employee_id,
                visitor_name=employee_name
            )
            self.db.add(ident)
            await self.db.flush()
        return ident

    # ==========================================
    # VISITOR & RE-ID EMBEDDINGS
    # ==========================================

    async def find_similar_visitor(
        self,
        tenant_id: uuid.UUID,
        target_embedding: list[float],
        threshold: float,
        class_id: int = 0,
        embedding_type: str = "appearance",
        prefer_segmented: bool = True
    ) -> Optional[Tuple[AdvancedPersonIdentity, float]]:

        distance_limit = 1.0 - threshold

        stmt = (
            select(
                AdvancedPersonIdentity,
                AdvancedPersonEmbedding.embedding.cosine_distance(target_embedding).label("distance"),
                AdvancedPersonEmbedding.is_segmented,
                AdvancedPersonEmbedding.mask_coverage
            )
            .join(AdvancedPersonEmbedding, AdvancedPersonEmbedding.identity_id == AdvancedPersonIdentity.id)
            .where(
                AdvancedPersonIdentity.tenant_id == tenant_id,
                AdvancedPersonIdentity.class_id == class_id,
                AdvancedPersonIdentity.is_delete == False,
                AdvancedPersonEmbedding.is_delete == False,
                AdvancedPersonEmbedding.is_active == True,
                AdvancedPersonEmbedding.embedding_type == embedding_type,
                AdvancedPersonEmbedding.embedding.cosine_distance(target_embedding) <= distance_limit
            )
            .order_by(
                AdvancedPersonEmbedding.is_segmented.desc() if prefer_segmented else text("1"),
                AdvancedPersonEmbedding.mask_coverage.desc().nulls_last(),
                text("distance ASC")
            )
            .limit(1)
        )

        result = await self.db.execute(stmt)
        row = result.first()
        if row:
            identity, distance, is_seg, coverage = row
            similarity = 1.0 - distance
            return identity, similarity
        return None

    async def get_active_identities_with_embeddings(
        self,
        tenant_id: uuid.UUID,
        class_id: int = 0,
        target_date: Optional[datetime.date] = None
    ) -> List[Tuple[AdvancedPersonIdentity, List[float]]]:
        conditions = [
            AdvancedPersonIdentity.tenant_id == tenant_id,
            AdvancedPersonIdentity.class_id == class_id,
            AdvancedPersonIdentity.is_delete == False,
            AdvancedPersonEmbedding.is_delete == False,
            AdvancedPersonEmbedding.embedding_type == "appearance"
        ]

        if target_date is not None:
            conditions.append(
                or_(
                    AdvancedPersonEmbedding.recorded_date == target_date,
                    and_(
                        AdvancedPersonEmbedding.is_active == True,
                        AdvancedPersonEmbedding.recorded_date.is_(None)
                    )
                )
            )
        else:
            conditions.append(AdvancedPersonEmbedding.is_active == True)

        stmt = (
            select(AdvancedPersonIdentity, AdvancedPersonEmbedding.embedding)
            .join(AdvancedPersonEmbedding, AdvancedPersonEmbedding.identity_id == AdvancedPersonIdentity.id)
            .where(*conditions)
            .order_by(AdvancedPersonEmbedding.created_at.desc())
        )
        result = await self.db.execute(stmt)
        return [(row[0], list(row[1])) for row in result.all() if row[1] is not None]

    async def get_identity_by_id(self, identity_id: uuid.UUID, tenant_id: uuid.UUID) -> Optional[AdvancedPersonIdentity]:
        stmt = (
            select(AdvancedPersonIdentity)
            .options(selectinload(AdvancedPersonIdentity.embeddings))
            .where(
                AdvancedPersonIdentity.id == identity_id,
                AdvancedPersonIdentity.tenant_id == tenant_id,
                AdvancedPersonIdentity.is_delete == False
            )
        )
        res = await self.db.execute(stmt)
        return res.scalars().first()

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
        timestamp: Optional[float] = None,
        embedding_type: str = "appearance",
        is_segmented: bool = False,
        mask_coverage: Optional[float] = None,
        recorded_date: Optional[datetime.date] = None,
        is_active: bool = True,
        face_anchored: bool = False
    ) -> AdvancedPersonEmbedding:
        emb = AdvancedPersonEmbedding(
            identity_id=identity_id,
            embedding=embedding,
            bbox=bbox,
            timestamp=timestamp,
            embedding_type=embedding_type,
            is_segmented=is_segmented,
            mask_coverage=mask_coverage,
            recorded_date=recorded_date,
            is_active=is_active,
            face_anchored=face_anchored
        )
        self.db.add(emb)
        await self.db.flush()
        return emb

    async def deactivate_old_appearance_embeddings(
        self,
        identity_id: uuid.UUID,
        current_date: datetime.date
    ) -> int:
        """
        Deactivates older appearance embeddings (recorded_date < current_date or recorded_date is None)
        for a given identity when fresh face-confirmed appearance embeddings are recorded today.
        """
        stmt = (
            update(AdvancedPersonEmbedding)
            .where(
                AdvancedPersonEmbedding.identity_id == identity_id,
                AdvancedPersonEmbedding.embedding_type == "appearance",
                AdvancedPersonEmbedding.is_active == True,
                or_(
                    AdvancedPersonEmbedding.recorded_date < current_date,
                    AdvancedPersonEmbedding.recorded_date.is_(None)
                )
            )
            .values(is_active=False)
        )
        res = await self.db.execute(stmt)
        return res.rowcount

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

    create_employee_attendance_log = create_employee_attendance

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

    async def get_session_detected_people(
        self,
        session_id: uuid.UUID,
        tenant_id: uuid.UUID
    ) -> List[dict]:
        import os
        # Query PersonTimelineEvent records for this session
        stmt = select(PersonTimelineEvent).where(
            PersonTimelineEvent.session_id == session_id,
            PersonTimelineEvent.tenant_id == tenant_id,
            PersonTimelineEvent.is_delete == False
        ).order_by(PersonTimelineEvent.started_at.asc())
        res = await self.db.execute(stmt)
        events = list(res.scalars().all())

        people = []
        seen_trackers = set()

        for evt in events:
            if evt.tracker_id in seen_trackers:
                continue
            seen_trackers.add(evt.tracker_id)

            # Resolve person display name
            name = "Visitor"
            if evt.person_type == "employee" and evt.employee_id:
                emp = await self.db.get(Employee, evt.employee_id)
                if emp:
                    name = f"{emp.first_name} {emp.last_name}"
            elif evt.identity_id:
                ident = await self.db.get(AdvancedPersonIdentity, evt.identity_id)
                if ident and ident.visitor_name:
                    name = ident.visitor_name
                else:
                    name = f"Visitor #{str(evt.identity_id)[:8]}"
            else:
                name = f"Visitor #{evt.tracker_id}"

            crop_filename = f"advanced_{session_id}_{evt.tracker_id}.jpg"
            crop_path_disk = os.path.join("storage", "visitor_crops", crop_filename)
            crop_url = f"/storage/visitor_crops/{crop_filename}" if os.path.exists(crop_path_disk) else None

            first_seen_sec = 0.0
            last_seen_sec = max(1.0, (evt.ended_at - evt.started_at).total_seconds())

            people.append({
                "identity_id": evt.identity_id,
                "employee_id": evt.employee_id,
                "person_type": evt.person_type,
                "name": name,
                "tracker_id": evt.tracker_id,
                "crop_url": crop_url,
                "first_seen": first_seen_sec,
                "last_seen": last_seen_sec,
                "started_at": evt.started_at,
                "ended_at": evt.ended_at,
                "confidence": evt.identity_confidence,
                "identity_source": evt.identity_source,
                "camera_name": evt.camera_name,
            })

        return people

    async def get_tenant_people_summary(
        self,
        tenant_id: uuid.UUID,
        target_date: Optional[datetime.date] = None,
        person_type: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[dict]:
        import os
        from collections import defaultdict

        stmt = select(PersonTimelineEvent).where(
            PersonTimelineEvent.tenant_id == tenant_id,
            PersonTimelineEvent.is_delete == False
        )
        if target_date:
            start_dt = datetime.datetime.combine(target_date, datetime.time.min, tzinfo=datetime.timezone.utc)
            end_dt = datetime.datetime.combine(target_date, datetime.time.max, tzinfo=datetime.timezone.utc)
            stmt = stmt.where(
                PersonTimelineEvent.started_at >= start_dt,
                PersonTimelineEvent.started_at <= end_dt
            )
        if person_type:
            stmt = stmt.where(PersonTimelineEvent.person_type == person_type)

        stmt = stmt.order_by(PersonTimelineEvent.started_at.asc())
        res = await self.db.execute(stmt)
        events = list(res.scalars().all())

        # Group events by (person_type, person_id)
        grouped = defaultdict(list)
        for evt in events:
            p_id = evt.employee_id if evt.person_type == "employee" else evt.identity_id
            if not p_id:
                p_id = uuid.UUID(int=evt.tracker_id)
            grouped[(evt.person_type, str(p_id))].append(evt)

        results = []
        for (p_type, p_id_str), p_events in grouped.items():
            p_uuid = uuid.UUID(p_id_str)
            total_dwell = sum(
                max(0.0, float((e.ended_at - e.started_at).total_seconds()))
                if getattr(e, "ended_at", None) and getattr(e, "started_at", None) else 0.0
                for e in p_events
            )
            cameras_visited = []
            seen_cams = set()
            for e in p_events:
                if e.camera_name and e.camera_name not in seen_cams:
                    seen_cams.add(e.camera_name)
                    cameras_visited.append(e.camera_name)

            first_seen = p_events[0].started_at
            last_seen = max(e.ended_at for e in p_events)

            name = "Visitor"
            emp_code = None
            crop_url = None

            if p_type == "employee":
                emp = await self.db.get(Employee, p_uuid)
                if emp:
                    name = f"{emp.first_name} {emp.last_name}"
                    emp_code = emp.employee_code
                emp_crop = os.path.join("storage", "visitor_crops", f"emp_{p_uuid}.jpg")
                if os.path.exists(emp_crop):
                    crop_url = f"/storage/visitor_crops/emp_{p_uuid}.jpg"
            else:
                ident = await self.db.get(AdvancedPersonIdentity, p_uuid)
                if ident and ident.visitor_name:
                    name = ident.visitor_name
                else:
                    name = f"Visitor #{p_id_str[:8]}"
                vis_crop = os.path.join("storage", "visitor_crops", f"visitor_{p_uuid}.jpg")
                if os.path.exists(vis_crop):
                    crop_url = f"/storage/visitor_crops/visitor_{p_uuid}.jpg"

            # Fallback to session crop if specific crop not found
            if not crop_url and p_events:
                first_evt = p_events[0]
                crop_fname = f"advanced_{first_evt.session_id}_{first_evt.tracker_id}.jpg"
                if os.path.exists(os.path.join("storage", "visitor_crops", crop_fname)):
                    crop_url = f"/storage/visitor_crops/{crop_fname}"

            # Filter by search string if supplied
            if search:
                s_lower = search.lower().strip()
                if s_lower not in name.lower() and (not emp_code or s_lower not in emp_code.lower()) and s_lower not in p_id_str:
                    continue

            results.append({
                "person_type": p_type,
                "person_id": p_uuid,
                "name": name,
                "employee_code": emp_code,
                "crop_url": crop_url,
                "total_dwell_seconds": float(total_dwell),
                "camera_stops_count": len(cameras_visited),
                "cameras_visited": cameras_visited,
                "first_seen_at": first_seen,
                "last_seen_at": last_seen,
                "latest_event_type": p_events[-1].event_type if p_events else "presence",
            })

        # Sort by latest activity
        results.sort(key=lambda x: x["last_seen_at"], reverse=True)
        return results

