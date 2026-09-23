import os
import uuid
import datetime
import subprocess
import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, delete, update

from application.advancedpeopleanalytics.repository import AdvancedPeopleAnalyticsRepository
from application.advancedpeopleanalytics.model import (
    AdvancedPeopleAnalyticsSession,
    AdvancedPersonIdentity,
    AdvancedPersonEmbedding,
    AdvancedPersonOccurrence,
    AdvancedEmployeeAttendanceLog,
    AdvancedVisitorAttendanceLog,
    AdvancedLineCrossingLog,
    CameraNode,
    CameraNodeLink,
    CameraZone,
    CameraZoneLink,
    PersonTimelineEvent,
    AdvancedEmployeeDailyCheckin,
    FloorPlan,
    SpatialLine
)
from application.advancedpeopleanalytics.schema import (
    AdvancedVideoProcessItem,
    SessionDetectedPerson,
    VisitorAnalyticsReport,
    FirstTimeVisitorDetail,
    VisitorAttendanceResponse,
    RegisterVisitorRequest,
    CameraNodeCreate,
    CameraNodeUpdate,
    CameraNodeResponse,
    CameraNodeLinkCreate,
    CameraNodeLinkResponse,
    CameraZoneCreate,
    CameraZoneResponse,
    CameraZoneLinkCreate,
    CameraZoneLinkResponse,
    PersonTimelineResponse,
    TimelineEventResponse,
    CreateSubclipRequest,
    SubclipResponse,
    PlaceDwellItem,
    PersonDwellByPhotoResponse,
    DailyCheckinResponse,
    DailyCheckinRecordResponse,
    HourlyDwellResponse,
    HourlyAreaDwellItem,
    ReviewQueueCandidate,
    ReconcileIdentityRequest,
    PhotoSearchAppearanceItem,
    PhotoSearchMatchItem,
    PhotoSearchResponse,
    FloorPlanCreate,
    FloorPlanResponse,
    FloorPlanLayoutResponse,
    SaveLayoutRequest,
    SpatialLineResponse
)


def format_seconds(seconds: float) -> str:
    secs = int(round(seconds))
    hrs = secs // 3600
    mins = (secs % 3600) // 60
    rem_secs = secs % 60
    parts = []
    if hrs > 0:
        parts.append(f"{hrs}h")
    if mins > 0 or hrs > 0:
        parts.append(f"{mins}m")
    parts.append(f"{rem_secs}s")
    return " ".join(parts)

def format_crop_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    norm = path.replace("\\", "/")
    if "storage/" in norm:
        return "/" + norm[norm.find("storage/"):]
    return norm

ADVANCED_OUTPUTS_DIR = os.path.join("storage", "advanced_people_analytics_outputs")
os.makedirs(ADVANCED_OUTPUTS_DIR, exist_ok=True)

class AdvancedPeopleAnalyticsService:
    def __init__(self, db: AsyncSession):
        self.repo = AdvancedPeopleAnalyticsRepository(db)
        self.db = db

    # ==========================================
    # CAMERA TOPOLOGY & SPATIAL GRAPH
    # ==========================================

    async def create_camera_node(self, tenant_id: uuid.UUID, data: CameraNodeCreate) -> CameraNode:
        effective_label = data.location_label or data.location_desc or data.label
        node = await self.repo.create_camera_node(
            tenant_id=tenant_id,
            name=data.name,
            location_label=effective_label
        )
        await self.db.commit()
        await self.db.refresh(node)
        return node

    async def update_camera_node(self, tenant_id: uuid.UUID, node_id: uuid.UUID, data: CameraNodeUpdate) -> CameraNode:
        effective_label = data.location_label or data.location_desc or data.label
        node = await self.repo.update_camera_node(
            node_id=node_id,
            tenant_id=tenant_id,
            name=data.name,
            location_label=effective_label
        )
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera node not found."
            )
        await self.db.commit()
        await self.db.refresh(node)
        return node

    async def get_camera_nodes(self, tenant_id: uuid.UUID) -> List[CameraNode]:
        return await self.repo.get_camera_nodes(tenant_id)

    async def delete_camera_node(self, node_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        res = await self.repo.delete_camera_node(node_id, tenant_id)
        if not res:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera node not found."
            )
        await self.db.commit()
        return True

    async def create_camera_node_link(self, tenant_id: uuid.UUID, data: CameraNodeLinkCreate) -> CameraNodeLinkResponse:
        from_node = await self.repo.get_camera_node_by_id(data.from_camera_id, tenant_id)
        to_node = await self.repo.get_camera_node_by_id(data.to_camera_id, tenant_id)
        if not from_node or not to_node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both CameraNodes not found or access denied."
            )
        if data.from_camera_id == data.to_camera_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot link a camera to itself."
            )

        link = await self.repo.create_camera_node_link(
            tenant_id=tenant_id,
            from_camera_id=data.from_camera_id,
            to_camera_id=data.to_camera_id,
            min_transit_seconds=data.min_transit_seconds,
            avg_transit_seconds=data.avg_transit_seconds,
            max_transit_seconds=data.max_transit_seconds
        )

        if data.is_bidirectional:
            await self.repo.create_camera_node_link(
                tenant_id=tenant_id,
                from_camera_id=data.to_camera_id,
                to_camera_id=data.from_camera_id,
                min_transit_seconds=data.min_transit_seconds,
                avg_transit_seconds=data.avg_transit_seconds,
                max_transit_seconds=data.max_transit_seconds
            )

        await self.db.commit()
        await self.db.refresh(link)

        return CameraNodeLinkResponse(
            id=link.id,
            tenant_id=link.tenant_id,
            from_camera_id=link.from_camera_id,
            to_camera_id=link.to_camera_id,
            from_camera_name=from_node.name,
            to_camera_name=to_node.name,
            min_transit_seconds=link.min_transit_seconds,
            avg_transit_seconds=link.avg_transit_seconds,
            max_transit_seconds=link.max_transit_seconds,
            created_at=link.created_at,
            update_at=link.update_at
        )

    async def get_camera_node_links(self, tenant_id: uuid.UUID) -> List[CameraNodeLinkResponse]:
        links = await self.repo.get_camera_node_links(tenant_id)
        result = []
        for l in links:
            result.append(CameraNodeLinkResponse(
                id=l.id,
                tenant_id=l.tenant_id,
                from_camera_id=l.from_camera_id,
                to_camera_id=l.to_camera_id,
                from_camera_name=l.from_camera.name if l.from_camera else None,
                to_camera_name=l.to_camera.name if l.to_camera else None,
                min_transit_seconds=l.min_transit_seconds,
                avg_transit_seconds=l.avg_transit_seconds,
                max_transit_seconds=l.max_transit_seconds,
                created_at=l.created_at,
                update_at=l.update_at
            ))
        return result

    async def delete_camera_node_link(self, link_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        res = await self.repo.delete_camera_node_link(link_id, tenant_id)
        if not res:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Camera node link not found."
            )
        await self.db.commit()
        return True

    # ==========================================
    # FLOOR PLANS & SPATIAL LINES
    # ==========================================

    async def create_floor_plan(self, tenant_id: uuid.UUID, data: FloorPlanCreate) -> FloorPlanResponse:
        floor_plan = await self.repo.create_floor_plan(
            tenant_id=tenant_id,
            name=data.name,
            canvas_width_px=data.canvas_width_px,
            canvas_height_px=data.canvas_height_px,
            scale_meters_per_px=data.scale_meters_per_px,
            image_filepath=data.image_filepath
        )
        await self.db.commit()
        await self.db.refresh(floor_plan)
        return FloorPlanResponse.model_validate(floor_plan)

    async def get_floor_plans(self, tenant_id: uuid.UUID) -> List[FloorPlanResponse]:
        floor_plans = await self.repo.get_floor_plans(tenant_id)
        return [FloorPlanResponse.model_validate(fp) for fp in floor_plans]

    async def get_floor_plan_layout(self, tenant_id: uuid.UUID, floor_plan_id: uuid.UUID) -> FloorPlanLayoutResponse:
        floor_plan = await self.repo.get_floor_plan_by_id(floor_plan_id, tenant_id)
        if not floor_plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Floor plan not found."
            )
        camera_nodes = await self.repo.get_camera_nodes_for_floor_plan(floor_plan_id, tenant_id)
        lines = await self.repo.get_spatial_lines_for_floor_plan(floor_plan_id, tenant_id)
        return FloorPlanLayoutResponse(
            floor_plan=FloorPlanResponse.model_validate(floor_plan),
            camera_nodes=[CameraNodeResponse.model_validate(c) for c in camera_nodes],
            lines=[SpatialLineResponse.model_validate(l) for l in lines]
        )

    async def save_floor_plan_layout(
        self,
        tenant_id: uuid.UUID,
        floor_plan_id: uuid.UUID,
        data: SaveLayoutRequest
    ) -> FloorPlanLayoutResponse:
        try:
            floor_plan, updated_nodes, active_lines = await self.repo.save_floor_plan_layout(
                tenant_id=tenant_id,
                floor_plan_id=floor_plan_id,
                camera_nodes_data=data.camera_nodes,
                lines_data=data.lines
            )
            await self.db.commit()
            await self.db.refresh(floor_plan)
            return FloorPlanLayoutResponse(
                floor_plan=FloorPlanResponse.model_validate(floor_plan),
                camera_nodes=[CameraNodeResponse.model_validate(c) for c in updated_nodes],
                lines=[SpatialLineResponse.model_validate(l) for l in active_lines]
            )
        except ValueError as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            await self.db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save floor plan layout: {str(e)}"
            )

    async def create_camera_zone(self, tenant_id: uuid.UUID, camera_id: uuid.UUID, data: CameraZoneCreate) -> CameraZone:
        node = await self.repo.get_camera_node_by_id(camera_id, tenant_id)
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"CameraNode '{camera_id}' not found or access denied."
            )
        zone = await self.repo.create_camera_zone(
            tenant_id=tenant_id,
            camera_id=camera_id,
            zone_type=data.zone_type,
            label=data.label,
            polygon=data.polygon
        )
        await self.db.commit()
        await self.db.refresh(zone)
        return zone

    async def get_camera_zones(self, camera_id: uuid.UUID) -> List[CameraZone]:
        return await self.repo.get_camera_zones(camera_id)

    async def create_zone_link(self, tenant_id: uuid.UUID, data: CameraZoneLinkCreate) -> CameraZoneLink:
        link = await self.repo.create_zone_link(
            tenant_id=tenant_id,
            from_zone_id=data.from_zone_id,
            to_zone_id=data.to_zone_id,
            avg_transit_seconds=data.avg_transit_seconds,
            max_transit_seconds=data.max_transit_seconds
        )
        await self.db.commit()
        await self.db.refresh(link)
        return link

    # ==========================================
    # TIMELINE & JOURNEY QUERYING
    # ==========================================

    async def get_person_timeline(
        self,
        tenant_id: uuid.UUID,
        person_type: str,
        person_id: uuid.UUID,
        target_date: datetime.date
    ) -> PersonTimelineResponse:
        events = await self.repo.get_person_timeline(tenant_id, person_type, person_id, target_date)
        timeline_events = []
        for e in events:
            t_resp = TimelineEventResponse.model_validate(e)
            t_resp.entry_crop_url = format_crop_url(e.entry_crop_path)
            t_resp.exit_crop_url = format_crop_url(e.exit_crop_path)
            timeline_events.append(t_resp)
        return PersonTimelineResponse(
            person_id=person_id,
            person_type=person_type,
            date=target_date,
            events=timeline_events
        )

    async def trigger_manual_association(self, session_ids: List[uuid.UUID]):
        from application.advancedpeopleanalytics.tasks import run_cross_camera_association_task
        for sid in session_ids:
            run_cross_camera_association_task.delay(str(sid))
        return len(session_ids)

    # ==========================================
    # SESSION MANAGEMENT
    # ==========================================

    async def create_and_start_sessions(
        self,
        tenant_id: uuid.UUID,
        videos: List[AdvancedVideoProcessItem],
        global_line_start: Optional[List[int]] = None,
        global_line_end: Optional[List[int]] = None,
        global_similarity_threshold: float = 0.85,
        global_confidence_threshold: float = 0.3,
        global_track_employees: bool = True,
        global_register_new_visitors: bool = True,
        global_track_repeat_visitors: bool = True,
        global_line_crossing_analysis: bool = True,
        global_track_occupancy: bool = True,
        global_track_objects: bool = True,
        global_classes_to_track: Optional[List[str]] = None,
        global_generate_video: bool = False,
        global_start_time: Optional[float] = None,
        global_end_time: Optional[float] = None,
        user_id: Optional[uuid.UUID] = None
    ) -> List[AdvancedPeopleAnalyticsSession]:
        from modules.gallery.repository import GalleryRepository
        gallery_repo = GalleryRepository(self.db)

        resolved_items = []
        for item in videos:
            filepath = None
            video_name = "Analytics_Video.mp4"
            item_gallery_id = None

            if item.gallery_media_id:
                item_gallery_id = uuid.UUID(str(item.gallery_media_id))
                gallery_media = await gallery_repo.get_media_by_id(item_gallery_id, tenant_id)
                if not gallery_media:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Gallery media '{item.gallery_media_id}' not found or access denied."
                    )
                filepath = gallery_media.filepath
                video_name = gallery_media.filename
                gallery_media.status = "processing"
            elif item.direct_video_path:
                filepath = item.direct_video_path
                video_name = os.path.basename(filepath)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Each video item must specify either 'gallery_media_id' or 'direct_video_path'."
                )

            if not os.path.exists(filepath):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video file does not exist on server storage: '{filepath}'."
                )

            resolved_items.append((item, filepath, video_name, item_gallery_id))

        sessions = []

        for item, filepath, video_name, item_gallery_id in resolved_items:
            line_start = item.line_start if item.line_start is not None else global_line_start
            line_end = item.line_end if item.line_end is not None else global_line_end
            similarity_threshold = item.similarity_threshold if item.similarity_threshold is not None else global_similarity_threshold
            confidence_threshold = item.confidence_threshold if item.confidence_threshold is not None else global_confidence_threshold
            track_employees = item.track_employees if item.track_employees is not None else global_track_employees
            register_new_visitors = item.register_new_visitors if item.register_new_visitors is not None else global_register_new_visitors
            track_repeat_visitors = item.track_repeat_visitors if item.track_repeat_visitors is not None else global_track_repeat_visitors
            line_crossing_analysis = item.line_crossing_analysis if item.line_crossing_analysis is not None else global_line_crossing_analysis
            track_occupancy = item.track_occupancy if item.track_occupancy is not None else global_track_occupancy
            track_objects = item.track_objects if item.track_objects is not None else global_track_objects
            classes_to_track = item.classes_to_track if item.classes_to_track is not None else global_classes_to_track
            generate_video = item.generate_video if item.generate_video is not None else global_generate_video
            start_time = item.start_time if item.start_time is not None else global_start_time
            end_time = item.end_time if item.end_time is not None else global_end_time

            effective_camera_node_id = item.camera_node_id or getattr(item, "camera_id", None)
            if not effective_camera_node_id:
                clean_vname = os.path.splitext(video_name)[0].lower().replace("-t", "").replace("_", " ").strip()
                try:
                    all_cams = await self.repo.get_camera_nodes(tenant_id)
                    best_cam = None
                    clean_normalized = clean_vname.replace(" ", "").replace("-", "")
                    for cam in all_cams:
                        cam_normalized = cam.name.lower().replace(" ", "").replace("-", "")
                        if cam_normalized in clean_normalized or clean_normalized in cam_normalized:
                            if cam.location_label:
                                best_cam = cam
                                break
                            elif best_cam is None:
                                best_cam = cam
                    if best_cam:
                        effective_camera_node_id = best_cam.id
                except Exception:
                    pass

            session = await self.repo.create_analytics_session(
                tenant_id=tenant_id,
                video_name=video_name,
                video_path=filepath,
                line_start=line_start,
                line_end=line_end,
                similarity_threshold=similarity_threshold,
                confidence_threshold=confidence_threshold,
                track_employees=track_employees,
                register_new_visitors=register_new_visitors,
                track_repeat_visitors=track_repeat_visitors,
                line_crossing_analysis=line_crossing_analysis,
                track_occupancy=track_occupancy,
                track_objects=track_objects,
                classes_to_track=classes_to_track,
                generate_video=generate_video,
                camera_node_id=effective_camera_node_id,
                recording_started_at=item.recording_started_at,
                gallery_media_id=item_gallery_id,
                start_time_sec=start_time,
                end_time_sec=end_time
            )
            await self.db.commit()
            await self.db.refresh(session)

            # Schedule Celery background task
            from application.advancedpeopleanalytics.tasks import process_advanced_people_analytics_task
            process_advanced_people_analytics_task.delay(
                str(session.id),
                filepath,
                line_start,
                line_end,
                similarity_threshold,
                confidence_threshold,
                str(user_id) if user_id else None,
                track_employees,
                register_new_visitors,
                track_repeat_visitors,
                line_crossing_analysis,
                track_occupancy,
                track_objects,
                generate_video
            )

            sessions.append(session)

        return sessions

    async def get_all_sessions(self, tenant_id: uuid.UUID) -> List[AdvancedPeopleAnalyticsSession]:
        sessions = await self.repo.get_all_sessions(tenant_id)
        from database.redis import get_redis_client
        redis_client = get_redis_client()
        
        for s in sessions:
            if s.status == "completed":
                s.completed_percentage = 100
            elif s.status == "failed":
                s.completed_percentage = 0
            elif redis_client:
                try:
                    val = await redis_client.get(f"advancedpeopleanalytics:progress:{s.id}")
                    if val is not None:
                        s.completed_percentage = int(val)
                except Exception:
                    s.completed_percentage = 0
        return sessions

    async def get_session(self, session_id: uuid.UUID, tenant_id: uuid.UUID) -> AdvancedPeopleAnalyticsSession:
        session = await self.repo.get_session_by_id(session_id, tenant_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analytics session not found or unauthorized access."
            )
        
        progress = 0
        if session.status == "completed":
            progress = 100
        elif session.status == "failed":
            progress = 0
        else:
            from database.redis import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                try:
                    val = await redis_client.get(f"advancedpeopleanalytics:progress:{session.id}")
                    if val is not None:
                        progress = int(val)
                except Exception:
                    progress = 0
        session.completed_percentage = progress
        return session

    async def get_session_detected_people(self, session_id: uuid.UUID, tenant_id: uuid.UUID) -> List[SessionDetectedPerson]:
        await self.get_session(session_id, tenant_id)
        people_data = await self.repo.get_session_detected_people(session_id, tenant_id)
        return [SessionDetectedPerson.model_validate(p) for p in people_data]

    async def delete_session(self, session_id: uuid.UUID, tenant_id: uuid.UUID) -> bool:
        session = await self.get_session(session_id, tenant_id)

        # Signal immediate background task cancellation via Redis
        try:
            from database.redis import get_redis_client, init_redis
            redis_client = get_redis_client()
            if redis_client is None:
                await init_redis()
                redis_client = get_redis_client()
            if redis_client:
                await redis_client.setex(f"advancedpeopleanalytics:cancelled:{session_id}", 3600, "1")
                await redis_client.delete(f"advancedpeopleanalytics:progress:{session_id}")
        except Exception:
            pass

        if session.output_video_path and os.path.exists(session.output_video_path):
            try:
                os.remove(session.output_video_path)
            except Exception:
                pass
        await self.repo.delete_session(session_id, tenant_id)
        await self.db.commit()
        return True

    async def rerun_session(
        self,
        session_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None
    ) -> AdvancedPeopleAnalyticsSession:
        session = await self.repo.get_session_by_id(session_id, tenant_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analytics session not found or unauthorized access."
            )

        if not session.video_path or not os.path.exists(session.video_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Original video file for this session is missing or no longer available on disk."
            )

        # Clear existing detection logs, crossings, and events
        await self.repo.clear_session_results(session.id)

        # Reset session state
        session.status = "processing"
        session.unique_person_count = None
        session.total_person_count = None
        session.first_time_visitor_count = None
        session.peak_occupancy = None
        session.average_occupancy = None
        session.entry_count = None
        session.exit_count = None
        session.employee_count = None
        session.visitor_count = None
        session.occupancy_timeline = None
        session.completed_at = None

        if session.output_video_path and os.path.exists(session.output_video_path):
            try:
                os.remove(session.output_video_path)
            except Exception:
                pass
        session.output_video_path = None

        await self.db.commit()
        await self.db.refresh(session)

        # Reset redis progress & cancellation flags
        try:
            from database.redis import get_redis_client, init_redis
            redis_client = get_redis_client()
            if redis_client is None:
                await init_redis()
                redis_client = get_redis_client()
            if redis_client:
                await redis_client.delete(f"advancedpeopleanalytics:cancelled:{session.id}")
                await redis_client.setex(f"advancedpeopleanalytics:progress:{session.id}", 3600, "0")
        except Exception:
            pass

        # Re-dispatch Celery worker task with stored session parameters
        from application.advancedpeopleanalytics.tasks import process_advanced_people_analytics_task
        process_advanced_people_analytics_task.delay(
            str(session.id),
            session.video_path,
            session.line_start,
            session.line_end,
            session.similarity_threshold,
            session.confidence_threshold,
            str(user_id) if user_id else (str(session.created_by_id) if session.created_by_id else None),
            session.track_employees,
            session.register_new_visitors,
            session.track_repeat_visitors,
            session.line_crossing_analysis,
            session.track_occupancy,
            session.generate_video
        )

        session.completed_percentage = 0
        return session

    async def register_or_update_visitor(
        self,
        tenant_id: uuid.UUID,
        identity_id: uuid.UUID,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        registration_type: str = "visitor",
        employee_code: Optional[str] = None,
        existing_employee_id: Optional[uuid.UUID] = None,
        retroactive_attendance: bool = True,
        force: bool = False
    ) -> Dict[str, Any]:
        identity = await self.repo.get_identity_by_id(identity_id, tenant_id)
        if not identity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Person identity not found."
            )

        from modules.employees.repository import EmployeeRepository
        from modules.employees.cache import invalidate_employee_embeddings_cache
        emp_repo = EmployeeRepository(self.db)

        # Fetch occurrences explicitly
        occ_stmt = select(AdvancedPersonOccurrence).where(
            AdvancedPersonOccurrence.identity_id == identity.id,
            AdvancedPersonOccurrence.is_delete == False
        )
        occ_res = await self.db.execute(occ_stmt)
        identity_occurrences = list(occ_res.scalars().all())

        # Fetch embeddings explicitly
        emb_stmt = select(AdvancedPersonEmbedding).where(
            AdvancedPersonEmbedding.identity_id == identity.id,
            AdvancedPersonEmbedding.is_delete == False
        )
        emb_res = await self.db.execute(emb_stmt)
        identity_embeddings = list(emb_res.scalars().all())

        if registration_type in ("new_employee", "employee"):
            if not first_name or not last_name or not employee_code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="first_name, last_name, and employee_code are required when registering as a new employee."
                )
            full_name = f"{first_name.strip()} {last_name.strip()}"
            
            # Find best crop from occurrences, timeline events, or session visitor crops
            photo_path = None
            for occ in identity_occurrences:
                if occ.crop_path and os.path.exists(occ.crop_path):
                    photo_path = occ.crop_path
                    break

            if not photo_path:
                t_stmt = select(PersonTimelineEvent).where(
                    PersonTimelineEvent.identity_id == identity.id,
                    PersonTimelineEvent.is_delete == False
                ).order_by(PersonTimelineEvent.started_at.desc())
                t_res = await self.db.execute(t_stmt)
                for tev in t_res.scalars().all():
                    if tev.entry_crop_path and os.path.exists(tev.entry_crop_path):
                        photo_path = tev.entry_crop_path
                        break
                    cand_crop = os.path.join("storage", "visitor_crops", f"advanced_{tev.session_id}_{tev.tracker_id}.jpg")
                    if os.path.exists(cand_crop):
                        photo_path = cand_crop
                        break

            if not photo_path:
                vis_crop = os.path.join("storage", "visitor_crops", f"visitor_{identity.id}.jpg")
                if os.path.exists(vis_crop):
                    photo_path = vis_crop

            from modules.employees.model import Employee, EmployeeEmbedding
            from modules.employees.service import EMPLOYEE_PHOTOS_DIR
            import shutil

            os.makedirs(EMPLOYEE_PHOTOS_DIR, exist_ok=True)
            emp_id = uuid.uuid4()
            emp_photo_rel = f"storage/employee_photos/{emp_id}.jpg"
            emp_photo_disk = os.path.join(EMPLOYEE_PHOTOS_DIR, f"{emp_id}.jpg")

            if photo_path and os.path.exists(photo_path):
                try:
                    shutil.copy(photo_path, emp_photo_disk)
                    emp_photo_path = emp_photo_rel
                    shutil.copy(photo_path, os.path.join("storage", "visitor_crops", f"emp_{emp_id}.jpg"))
                except Exception:
                    emp_photo_path = photo_path.replace("\\", "/")
            else:
                emp_photo_path = emp_photo_rel

            emp = Employee(
                id=emp_id,
                tenant_id=tenant_id,
                first_name=first_name.strip(),
                last_name=last_name.strip(),
                employee_code=employee_code.strip(),
                photo_path=emp_photo_path
            )
            self.db.add(emp)
            await self.db.flush()

            # Copy face embedding(s)
            face_embs = [e for e in identity_embeddings if getattr(e, "embedding_type", "") == "face"]
            target_embs = face_embs if face_embs else identity_embeddings
            for emb in target_embs[:6]:
                await emp_repo.add_employee_face_exemplar(emp.id, emb.embedding)

            identity.is_employee = True
            identity.employee_id = emp.id
            identity.first_name = first_name.strip()
            identity.last_name = last_name.strip()
            identity.visitor_name = full_name

            # Retroactive attendance conversion
            if retroactive_attendance:
                v_logs_stmt = select(AdvancedVisitorAttendanceLog).where(
                    AdvancedVisitorAttendanceLog.identity_id == identity.id,
                    AdvancedVisitorAttendanceLog.is_delete == False
                )
                v_logs_res = await self.db.execute(v_logs_stmt)
                v_logs = list(v_logs_res.scalars().all())

                for vlog in v_logs:
                    await self.repo.create_employee_attendance(
                        session_id=vlog.session_id,
                        employee_id=emp.id,
                        first_seen=vlog.first_seen,
                        last_seen=vlog.last_seen,
                        occurrence_count=vlog.occurrence_count
                    )
                    try:
                        await emp_repo.log_employee_attendance(
                            tenant_id=tenant_id,
                            employee_id=emp.id,
                            session_id=vlog.session_id,
                            first_seen_sec=vlog.first_seen,
                            last_seen_sec=vlog.last_seen,
                            occurrence_increment=vlog.occurrence_count,
                            entry_time=vlog.visitor_entry_timestamp,
                            exit_time=vlog.visitor_exit_timestamp
                        )
                    except Exception:
                        pass
                    vlog.is_delete = True

                # Rewire timeline events
                t_stmt = select(PersonTimelineEvent).where(
                    PersonTimelineEvent.identity_id == identity.id,
                    PersonTimelineEvent.is_delete == False
                )
                t_res = await self.db.execute(t_stmt)
                for tev in t_res.scalars().all():
                    tev.person_type = "employee"
                    tev.employee_id = emp.id

            await self.db.commit()
            await invalidate_employee_embeddings_cache(tenant_id)
            return {
                "message": f"Visitor successfully registered as Employee {full_name} ({emp.employee_code})",
                "employee_id": str(emp.id),
                "identity_id": str(identity.id),
                "person_type": "employee",
                "name": full_name,
                "employee_code": emp.employee_code
            }

        elif registration_type == "link_existing_employee":
            if not existing_employee_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="existing_employee_id is required when linking to an existing employee."
                )
            target_emp = await emp_repo.get_employee_by_id(existing_employee_id, tenant_id)
            if not target_emp:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Target employee not found."
                )

            # 0.15 Safeguard: verify facial similarity
            visitor_face_embs = [e for e in identity_embeddings if getattr(e, "embedding_type", "") == "face"]
            max_sim = 0.0
            emp_embs = list(target_emp.embeddings) if target_emp.embeddings else []
            if visitor_face_embs and emp_embs:
                for ve in visitor_face_embs:
                    v_vec = np.array(ve.embedding, dtype=np.float32)
                    v_norm = np.linalg.norm(v_vec)
                    if v_norm <= 0:
                        continue
                    v_u = v_vec / v_norm
                    for ee in emp_embs:
                        e_vec = np.array(ee.embedding, dtype=np.float32)
                        e_norm = np.linalg.norm(e_vec)
                        if e_norm <= 0:
                            continue
                        sim = float(np.dot(v_u, e_vec / e_norm))
                        if sim > max_sim:
                            max_sim = sim

                if max_sim < 0.15 and not force:
                    raise HTTPException(
                        status_code=status.HTTP_409_CONFLICT,
                        detail={
                            "warning": f"Low facial correlation ({max_sim*100:.1f}% < 15%). This visitor appears to be a different person than {target_emp.first_name} {target_emp.last_name}. Pass force=true to confirm merge.",
                            "similarity": round(max_sim, 3)
                        }
                    )

            # Add visitor face as a new exemplar if valid
            if visitor_face_embs:
                await emp_repo.add_employee_face_exemplar(target_emp.id, visitor_face_embs[0].embedding)

            # Check if this employee already has an identity
            prim_stmt = select(AdvancedPersonIdentity).where(
                AdvancedPersonIdentity.tenant_id == tenant_id,
                AdvancedPersonIdentity.employee_id == target_emp.id,
                AdvancedPersonIdentity.id != identity.id,
                AdvancedPersonIdentity.is_delete == False
            )
            prim_res = await self.db.execute(prim_stmt)
            prim_ident = prim_res.scalars().first()

            target_identity_id = prim_ident.id if prim_ident else identity.id
            if prim_ident:
                # Merge occurrences and timeline events into primary identity
                occ_stmt = select(AdvancedPersonOccurrence).where(AdvancedPersonOccurrence.identity_id == identity.id)
                occ_res = await self.db.execute(occ_stmt)
                for occ in occ_res.scalars().all():
                    occ.identity_id = prim_ident.id

                identity.is_delete = True
            else:
                identity.is_employee = True
                identity.employee_id = target_emp.id
                identity.visitor_name = f"{target_emp.first_name} {target_emp.last_name}"

            # Retroactive attendance conversion
            if retroactive_attendance:
                v_logs_stmt = select(AdvancedVisitorAttendanceLog).where(
                    AdvancedVisitorAttendanceLog.identity_id == identity.id,
                    AdvancedVisitorAttendanceLog.is_delete == False
                )
                v_logs_res = await self.db.execute(v_logs_stmt)
                for vlog in v_logs_res.scalars().all():
                    await self.repo.create_employee_attendance(
                        session_id=vlog.session_id,
                        employee_id=target_emp.id,
                        first_seen=vlog.first_seen,
                        last_seen=vlog.last_seen,
                        occurrence_count=vlog.occurrence_count
                    )
                    try:
                        await emp_repo.log_employee_attendance(
                            tenant_id=tenant_id,
                            employee_id=target_emp.id,
                            session_id=vlog.session_id,
                            first_seen_sec=vlog.first_seen,
                            last_seen_sec=vlog.last_seen,
                            occurrence_increment=vlog.occurrence_count,
                            entry_time=vlog.visitor_entry_timestamp,
                            exit_time=vlog.visitor_exit_timestamp
                        )
                    except Exception:
                        pass
                    vlog.is_delete = True

                # Rewire timeline events
                t_stmt = select(PersonTimelineEvent).where(
                    PersonTimelineEvent.identity_id == identity.id,
                    PersonTimelineEvent.is_delete == False
                )
                t_res = await self.db.execute(t_stmt)
                for tev in t_res.scalars().all():
                    tev.person_type = "employee"
                    tev.employee_id = target_emp.id
                    tev.identity_id = target_identity_id

            await self.db.commit()
            await invalidate_employee_embeddings_cache(tenant_id)
            return {
                "message": f"Visitor successfully linked to Employee {target_emp.first_name} {target_emp.last_name} ({target_emp.employee_code}).",
                "employee_id": str(target_emp.id),
                "identity_id": str(target_identity_id)
            }

        else:
            full_name = f"{(first_name or '').strip()} {(last_name or '').strip()}".strip()
            identity.visitor_name = full_name or "Visitor"
            await self.db.commit()
            return {"message": f"Visitor name successfully updated to '{identity.visitor_name}'", "identity_id": str(identity.id)}

    async def add_person_from_face_photo(
        self,
        tenant_id: uuid.UUID,
        image_bytes: bytes,
        first_name: str,
        last_name: str,
        registration_type: str = "visitor",
        employee_code: Optional[str] = None
    ) -> Dict[str, Any]:
        from services.ai.face_recognition import face_rec_service
        import cv2
        import numpy as np

        # 1. Decode image and extract face embeddings
        faces = face_rec_service.extract_faces(image_bytes)
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No clear face detected in the uploaded photo. Please upload a clear frontal face image."
            )
        best_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        face_emb = best_face["embedding"]

        full_name = f"{first_name.strip()} {last_name.strip()}"
        os.makedirs("storage/visitor_crops", exist_ok=True)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if registration_type == "employee":
            if not employee_code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Employee code is required for employee registration."
                )
            from modules.employees.model import Employee, EmployeeEmbedding
            emp = Employee(
                tenant_id=tenant_id,
                first_name=first_name.strip(),
                last_name=last_name.strip(),
                employee_code=employee_code.strip()
            )
            self.db.add(emp)
            await self.db.flush()

            emp_emb = EmployeeEmbedding(
                employee_id=emp.id,
                embedding=face_emb
            )
            self.db.add(emp_emb)
            await self.db.commit()

            if img_np is not None:
                cv2.imwrite(f"storage/visitor_crops/emp_{emp.id}.jpg", img_np)

            from modules.employees.cache import invalidate_employee_embeddings_cache
            await invalidate_employee_embeddings_cache(tenant_id)
            return {
                "message": f"Successfully registered Employee {full_name} ({employee_code}).",
                "person_type": "employee",
                "id": str(emp.id),
                "name": full_name
            }
        else:
            identity = await self.repo.create_person_identity(tenant_id=tenant_id, class_id=0)
            identity.visitor_name = full_name
            await self.repo.create_person_embedding(
                identity_id=identity.id,
                embedding=face_emb,
                bbox=best_face["bbox"]
            )
            await self.db.commit()

            if img_np is not None:
                cv2.imwrite(f"storage/visitor_crops/visitor_{identity.id}.jpg", img_np)

            return {
                "message": f"Successfully registered Visitor {full_name}.",
                "person_type": "visitor",
                "id": str(identity.id),
                "name": full_name
            }

    async def get_tenant_people_summary(
        self,
        tenant_id: uuid.UUID,
        target_date: Optional[datetime.date] = None,
        person_type: Optional[str] = None,
        search: Optional[str] = None
    ) -> List[dict]:
        return await self.repo.get_tenant_people_summary(
            tenant_id=tenant_id,
            target_date=target_date,
            person_type=person_type,
            search=search
        )

    async def reset_all_analytics_data(self, tenant_id: uuid.UUID) -> dict:
        """
        Clears all analytics sessions, person embeddings, identities, timeline events, and cached progress.
        """
        import glob
        from sqlalchemy import delete
        from application.advancedpeopleanalytics.model import (
            AdvancedPeopleAnalyticsSession,
            AdvancedPersonIdentity,
            AdvancedPersonEmbedding,
            AdvancedPersonOccurrence,
            AdvancedVisitorAttendanceLog,
            AdvancedLineCrossingLog,
            ZoneCrossingEvent,
            CrossCameraIdentityLink,
            PersonTimelineEvent
        )
        
        # 1. Delete DB records
        await self.db.execute(delete(PersonTimelineEvent).where(PersonTimelineEvent.tenant_id == tenant_id))
        await self.db.execute(delete(CrossCameraIdentityLink).where(CrossCameraIdentityLink.tenant_id == tenant_id))
        await self.db.execute(delete(ZoneCrossingEvent).where(ZoneCrossingEvent.tenant_id == tenant_id))
        await self.db.execute(delete(AdvancedVisitorAttendanceLog))
        await self.db.execute(delete(AdvancedLineCrossingLog))
        await self.db.execute(delete(AdvancedPersonOccurrence))
        await self.db.execute(delete(AdvancedPersonEmbedding))
        await self.db.execute(delete(AdvancedPersonIdentity).where(AdvancedPersonIdentity.tenant_id == tenant_id))
        await self.db.execute(delete(AdvancedPeopleAnalyticsSession).where(AdvancedPeopleAnalyticsSession.tenant_id == tenant_id))
        await self.db.commit()

        # 2. Delete crop files and outputs from disk
        crop_patterns = [
            "storage/visitor_crops/advanced_*",
            "storage/visitor_crops/visitor_*",
            "storage/visitor_crops/emp_*",
            "storage/advanced_people_analytics_outputs/*",
            "storage/advanced_outputs/*"
        ]
        for pattern in crop_patterns:
            for f in glob.glob(pattern):
                try:
                    if os.path.isfile(f):
                        os.remove(f)
                except Exception:
                    pass

        # 3. Clear Redis progress keys
        from database.redis import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            try:
                keys = await redis_client.keys("advancedpeopleanalytics:*")
                if keys:
                    await redis_client.delete(*keys)
            except Exception:
                pass

        return {"message": "All person embeddings, video sessions, and analytics data successfully reset."}

    async def get_gallery_videos(self, tenant_id: uuid.UUID) -> List[dict]:
        from modules.gallery.repository import GalleryRepository
        gallery_repo = GalleryRepository(self.db)
        media_list = await gallery_repo.get_all_media(tenant_id, media_type="video")
        return [
            {
                "id": str(m.id),
                "filename": m.filename,
                "filepath": m.filepath,
                "processed_filepath": m.processed_filepath,
                "status": m.status,
                "created_at": m.created_at
            }
            for m in media_list
        ]

    async def create_subclip(
        self,
        tenant_id: uuid.UUID,
        data: CreateSubclipRequest
    ) -> SubclipResponse:
        from modules.gallery.repository import GalleryRepository
        gallery_repo = GalleryRepository(self.db)

        input_path = None
        if data.gallery_media_id:
            gallery_media = await gallery_repo.get_media_by_id(data.gallery_media_id, tenant_id)
            if not gallery_media:
                raise HTTPException(status_code=404, detail="Gallery media not found.")
            input_path = gallery_media.filepath
        elif data.session_id:
            session = await self.repo.get_session_by_id(data.session_id, tenant_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found.")
            input_path = session.video_path
        elif data.video_path:
            input_path = data.video_path
        else:
            raise HTTPException(status_code=400, detail="Must provide gallery_media_id, session_id, or video_path.")

        if not os.path.exists(input_path):
            raise HTTPException(status_code=400, detail=f"Source video file not found: {input_path}")

        subclips_dir = os.path.join("storage", "subclips")
        os.makedirs(subclips_dir, exist_ok=True)
        clip_filename = f"clip_{uuid.uuid4().hex[:8]}_{os.path.basename(input_path)}"
        output_path = os.path.join(subclips_dir, clip_filename)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(data.start_time),
            "-to", str(data.end_time),
            "-i", input_path,
            "-c", "copy",
            output_path
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0 or not os.path.exists(output_path):
            cmd_reencode = [
                "ffmpeg", "-y",
                "-ss", str(data.start_time),
                "-to", str(data.end_time),
                "-i", input_path,
                "-c:v", "libx264", "-c:a", "aac",
                output_path
            ]
            proc2 = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc2.returncode != 0 or not os.path.exists(output_path):
                raise HTTPException(status_code=500, detail=f"FFmpeg subclip generation failed: {proc.stderr.decode()}")

        created_gallery_id = None
        if data.save_to_gallery:
            new_gm = await gallery_repo.create_media(
                tenant_id=tenant_id,
                filename=clip_filename,
                filepath=output_path,
                media_type="video"
            )
            new_gm.status = "completed"
            await self.db.commit()
            created_gallery_id = new_gm.id

        cap = cv2.VideoCapture(output_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        dur = frames / fps if fps > 0 else 0.0

        return SubclipResponse(
            clip_path=output_path,
            clip_url=f"/storage/subclips/{clip_filename}",
            duration_seconds=round(dur, 2),
            gallery_media_id=created_gallery_id
        )

    async def get_person_dwell_by_photo(
        self,
        tenant_id: uuid.UUID,
        image_bytes: bytes,
        threshold: float = 0.35
    ) -> PersonDwellByPhotoResponse:
        from application.utils.face_rec import face_rec_service
        faces = face_rec_service.extract_faces(image_bytes)
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No clear face detected in the query photo. Please upload a clear frontal face image."
            )
        best_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        face_emb = best_face["embedding"]

        # Search across registered employees first
        from modules.employees.repository import EmployeeRepository
        emp_repo = EmployeeRepository(self.db)
        matched_emp = await emp_repo.find_similar_employee(tenant_id, face_emb, threshold=threshold)

        matched_identity = None
        similarity_score = 0.0
        employee_id = None
        person_type = "visitor"
        person_name = "Unknown Person"
        employee_code = None

        if matched_emp:
            employee, sim = matched_emp
            employee_id = employee.id
            person_type = "employee"
            person_name = f"{employee.first_name} {employee.last_name}"
            employee_code = employee.employee_code
            similarity_score = sim

            ident_stmt = select(AdvancedPersonIdentity).where(
                AdvancedPersonIdentity.tenant_id == tenant_id,
                AdvancedPersonIdentity.employee_id == employee.id,
                AdvancedPersonIdentity.is_delete == False
            )
            ident_res = await self.db.execute(ident_stmt)
            matched_identity = ident_res.scalars().first()
        else:
            match_res = await self.repo.find_matching_identity(
                tenant_id=tenant_id,
                target_embedding=face_emb,
                threshold=threshold,
                embedding_type="face"
            )
            if match_res:
                identity, sim = match_res
                matched_identity = identity
                similarity_score = sim
                if identity.is_employee and identity.employee_id:
                    person_type = "employee"
                    employee_id = identity.employee_id
                    emp = await emp_repo.get_employee_by_id(identity.employee_id, tenant_id)
                    if emp:
                        person_name = f"{emp.first_name} {emp.last_name}"
                        employee_code = emp.employee_code
                    else:
                        person_name = identity.visitor_name or "Employee"
                else:
                    person_type = "visitor"
                    person_name = identity.visitor_name or f"Visitor {str(identity.id)[:8]}"

        if not matched_identity and not employee_id:
            return PersonDwellByPhotoResponse(
                matched=False,
                confidence=0.0
            )

        conds = [PersonTimelineEvent.tenant_id == tenant_id, PersonTimelineEvent.is_delete == False]
        ident_conditions = []
        if employee_id:
            ident_conditions.append(PersonTimelineEvent.employee_id == employee_id)
        if matched_identity:
            ident_conditions.append(PersonTimelineEvent.identity_id == matched_identity.id)
        conds.append(or_(*ident_conditions))

        t_stmt = select(PersonTimelineEvent).where(*conds).order_by(PersonTimelineEvent.started_at.asc())
        t_res = await self.db.execute(t_stmt)
        timeline_events = list(t_res.scalars().all())

        total_dwell_sec = sum(e.duration_seconds for e in timeline_events)
        first_seen = timeline_events[0].started_at if timeline_events else None
        last_seen = timeline_events[-1].ended_at if timeline_events else None

        placewise_map = {}
        for e in timeline_events:
            key = (e.camera_name, e.zone_name)
            if key not in placewise_map:
                placewise_map[key] = {"duration": 0.0, "count": 0}
            placewise_map[key]["duration"] += e.duration_seconds
            placewise_map[key]["count"] += 1

        placewise_list = [
            PlaceDwellItem(
                camera_name=cam,
                zone_name=zn,
                duration_seconds=round(data["duration"], 2),
                formatted_duration=format_seconds(data["duration"]),
                visit_count=data["count"]
            )
            for (cam, zn), data in placewise_map.items()
        ]

        timeline_event_responses = [
            TimelineEventResponse(
                id=e.id,
                session_id=e.session_id,
                camera_name=e.camera_name,
                zone_name=e.zone_name,
                event_type=e.event_type,
                started_at=e.started_at,
                ended_at=e.ended_at,
                identity_source=e.identity_source,
                identity_confidence=e.identity_confidence,
                tracker_id=e.tracker_id,
                entry_crop_path=e.entry_crop_path,
                exit_crop_path=e.exit_crop_path,
                entry_crop_url=format_crop_url(e.entry_crop_path),
                exit_crop_url=format_crop_url(e.exit_crop_path)
            )
            for e in timeline_events
        ]

        return PersonDwellByPhotoResponse(
            matched=True,
            identity_id=matched_identity.id if matched_identity else None,
            person_type=person_type,
            name=person_name,
            employee_code=employee_code,
            confidence=round(similarity_score, 3),
            total_dwell_seconds=round(total_dwell_sec, 2),
            formatted_total_dwell=format_seconds(total_dwell_sec),
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            placewise_dwell=placewise_list,
            timeline_events=timeline_event_responses
        )

    # ==========================================
    # DAILY CHECK-IN BRIDGE (MORNING ANCHOR)
    # ==========================================

    async def daily_employee_checkin(
        self,
        tenant_id: uuid.UUID,
        employee_id: uuid.UUID,
        face_bytes: bytes,
        appearance_bytes: Optional[bytes] = None,
        checkin_date: Optional[datetime.date] = None
    ) -> DailyCheckinResponse:
        """
        Anchors an employee's face and daily outfit before CCTV analysis.
        Flexible inputs: Face is required; Outfit is optional.
        """
        if checkin_date is None:
            checkin_date = datetime.datetime.now(datetime.timezone.utc).date()

        from modules.employees.repository import EmployeeRepository
        emp_repo = EmployeeRepository(self.db)
        employee = await emp_repo.get_employee_by_id(employee_id, tenant_id)
        if not employee:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Employee not found or unauthorized."
            )

        from application.utils.face_rec import face_rec_service
        from application.utils.reid import reid_service
        import cv2
        import numpy as np

        # 1. Face Extraction & Exemplar Registration
        faces = face_rec_service.extract_faces(face_bytes)
        if not faces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No clear face detected in the face photo. Please upload a clear photo of the employee's face."
            )

        best_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
        face_emb = best_face["embedding"]

        # Add face exemplar to employee directory
        await emp_repo.add_employee_face_exemplar(employee.id, face_emb)
        from modules.employees.cache import invalidate_employee_embeddings_cache
        await invalidate_employee_embeddings_cache(tenant_id)

        # Get or create AdvancedPersonIdentity for this employee
        emp_identity = await self.repo.get_or_create_employee_identity(
            tenant_id=tenant_id,
            employee_id=employee.id,
            employee_name=f"{employee.first_name} {employee.last_name}"
        )

        # Save face embedding to AdvancedPersonEmbedding
        await self.repo.create_person_embedding(
            identity_id=emp_identity.id,
            embedding=list(face_emb),
            bbox=best_face.get("bbox"),
            timestamp=0.0,
            embedding_type="face",
            is_segmented=False,
            mask_coverage=None,
            recorded_date=checkin_date,
            is_active=True,
            face_anchored=True
        )

        # 2. Appearance / Outfit ReID Extraction (Flexible)
        reid_emb = None
        source_app_bytes = appearance_bytes if appearance_bytes else face_bytes
        
        try:
            nparr = np.frombuffer(source_app_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                h, w = img.shape[:2]
                from configs.base import settings
                from shared.utils.model_loader import get_model_path
                from ultralytics import YOLO
                seg_model_filename = os.path.basename(settings.YOLO_SEG_MODEL)
                seg_weights_path = get_model_path("attendance", seg_model_filename)
                yolo_model = YOLO(seg_weights_path)
                
                results = yolo_model(img, imgsz=640, classes=[0], verbose=False)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    # Pick largest person detection
                    largest_box = max(boxes, key=lambda b: (float(b.xyxy[0][2] - b.xyxy[0][0]) * float(b.xyxy[0][3] - b.xyxy[0][1])))
                    bx1, by1, bx2, by2 = map(int, largest_box.xyxy[0])
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(w, bx2), min(h, by2)
                    bbox_h = by2 - by1
                    # If appearance_bytes was provided, or if face_bytes contains upper/full body (>35% of img height)
                    if appearance_bytes or (bbox_h / max(h, 1)) >= 0.35:
                        body_crop = img[by1:by2, bx1:bx2]
                        if body_crop is not None and body_crop.size > 0:
                            reid_vec = reid_service.extract_embedding(body_crop)
                            if reid_vec is not None:
                                reid_emb = reid_vec.tolist()
                                # Deactivate older appearance embeddings for this employee
                                await self.repo.deactivate_old_appearance_embeddings(emp_identity.id, checkin_date)
                                # Save new appearance embedding anchored to checkin_date
                                await self.repo.create_person_embedding(
                                    identity_id=emp_identity.id,
                                    embedding=reid_emb,
                                    bbox=[bx1, by1, bx2, by2],
                                    timestamp=0.0,
                                    embedding_type="appearance",
                                    is_segmented=True,
                                    mask_coverage=0.9,
                                    recorded_date=checkin_date,
                                    is_active=True,
                                    face_anchored=True
                                )
        except Exception as e:
            logger.warning(f"Appearance extraction encountered issue during daily check-in: {e}")

        # Save uploaded check-in photos to storage/checkin_photos
        checkin_dir = os.path.join("storage", "checkin_photos")
        os.makedirs(checkin_dir, exist_ok=True)
        unique_suffix = f"{uuid.uuid4().hex[:8]}"
        face_filename = f"face_{employee.id}_{checkin_date}_{unique_suffix}.jpg"
        face_disk_path = os.path.join(checkin_dir, face_filename)
        with open(face_disk_path, "wb") as f:
            f.write(face_bytes)

        app_filename = None
        if appearance_bytes:
            app_filename = f"app_{employee.id}_{checkin_date}_{unique_suffix}.jpg"
            app_disk_path = os.path.join(checkin_dir, app_filename)
            with open(app_disk_path, "wb") as f:
                f.write(appearance_bytes)

        face_url = f"/storage/checkin_photos/{face_filename}"
        app_url = f"/storage/checkin_photos/{app_filename}" if app_filename else None

        # Record daily check-in log entry
        checkin_log = AdvancedEmployeeDailyCheckin(
            tenant_id=tenant_id,
            employee_id=employee.id,
            checkin_date=checkin_date,
            face_photo_path=face_url,
            appearance_photo_path=app_url,
            face_anchored=True,
            appearance_anchored=(reid_emb is not None)
        )
        self.db.add(checkin_log)

        await self.db.commit()

        return DailyCheckinResponse(
            employee_id=employee.id,
            employee_name=f"{employee.first_name} {employee.last_name}",
            employee_code=employee.employee_code,
            checkin_date=checkin_date,
            face_registered=True,
            appearance_anchored=(reid_emb is not None),
            face_photo_url=face_url,
            appearance_photo_url=app_url,
            message=f"Successfully anchored daily check-in for {employee.first_name} {employee.last_name} for {checkin_date}."
        )

    async def get_daily_checkins(
        self,
        tenant_id: uuid.UUID,
        checkin_date: Optional[datetime.date] = None,
        employee_id: Optional[uuid.UUID] = None,
        start_date: Optional[datetime.date] = None,
        end_date: Optional[datetime.date] = None
    ) -> List[DailyCheckinRecordResponse]:
        """
        Retrieves all daily check-in anchors with optional date and employee filters.
        """
        from modules.employees.model import Employee
        stmt = (
            select(AdvancedEmployeeDailyCheckin, Employee)
            .join(Employee, AdvancedEmployeeDailyCheckin.employee_id == Employee.id)
            .where(
                AdvancedEmployeeDailyCheckin.tenant_id == tenant_id,
                AdvancedEmployeeDailyCheckin.is_delete == False
            )
            .order_by(
                AdvancedEmployeeDailyCheckin.checkin_date.desc(),
                AdvancedEmployeeDailyCheckin.created_at.desc()
            )
        )

        if checkin_date is not None:
            stmt = stmt.where(AdvancedEmployeeDailyCheckin.checkin_date == checkin_date)
        if employee_id is not None:
            stmt = stmt.where(AdvancedEmployeeDailyCheckin.employee_id == employee_id)
        if start_date is not None:
            stmt = stmt.where(AdvancedEmployeeDailyCheckin.checkin_date >= start_date)
        if end_date is not None:
            stmt = stmt.where(AdvancedEmployeeDailyCheckin.checkin_date <= end_date)

        res = await self.db.execute(stmt)
        rows = res.all()

        results = []
        for checkin, emp in rows:
            results.append(
                DailyCheckinRecordResponse(
                    id=checkin.id,
                    employee_id=emp.id,
                    employee_name=f"{emp.first_name} {emp.last_name}",
                    employee_code=emp.employee_code,
                    employee_photo=emp.photo_path,
                    checkin_date=checkin.checkin_date,
                    face_photo_url=checkin.face_photo_path,
                    appearance_photo_url=checkin.appearance_photo_path,
                    face_anchored=checkin.face_anchored,
                    appearance_anchored=checkin.appearance_anchored,
                    created_at=checkin.created_at
                )
            )
        return results


    # ==========================================
    # AREA-WISE HOURLY DWELL TIME ANALYTICS
    # ==========================================

    async def get_hourly_area_dwell(
        self,
        tenant_id: uuid.UUID,
        target_date: Optional[datetime.date] = None,
        person_id: Optional[uuid.UUID] = None,
        session_id: Optional[uuid.UUID] = None
    ) -> HourlyDwellResponse:
        """
        Aggregates time spent across specific camera areas hour-by-hour (00:00 to 23:00).
        Portion of hour is capped at 1.0 (60 minutes / 3600 seconds max per hour).
        """
        if target_date is None:
            target_date = datetime.datetime.now(datetime.timezone.utc).date()

        day_start = datetime.datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=datetime.timezone.utc)
        day_end = day_start + datetime.timedelta(days=1)

        conds = [
            PersonTimelineEvent.tenant_id == tenant_id,
            PersonTimelineEvent.is_delete == False,
            PersonTimelineEvent.started_at < day_end,
            PersonTimelineEvent.ended_at > day_start
        ]

        person_name = None
        if person_id:
            conds.append(
                or_(
                    PersonTimelineEvent.identity_id == person_id,
                    PersonTimelineEvent.employee_id == person_id
                )
            )
            # Fetch person name for response metadata
            ident = await self.repo.get_identity_by_id(person_id, tenant_id)
            if ident:
                person_name = ident.visitor_name or f"{ident.first_name or ''} {ident.last_name or ''}".strip() or "Visitor"
            else:
                from modules.employees.repository import EmployeeRepository
                emp_repo = EmployeeRepository(self.db)
                emp = await emp_repo.get_employee_by_id(person_id, tenant_id)
                if emp:
                    person_name = f"{emp.first_name} {emp.last_name}"

        if session_id:
            conds.append(PersonTimelineEvent.session_id == session_id)

        stmt = select(PersonTimelineEvent).where(*conds).order_by(PersonTimelineEvent.started_at.asc())
        res = await self.db.execute(stmt)
        events = list(res.scalars().all())

        # Collect unique areas
        all_areas_set = set()
        for e in events:
            area = e.zone_name or e.camera_name or "General Area"
            all_areas_set.add(area)
        all_areas = sorted(list(all_areas_set))

        # Hourly buckets (0 to 23)
        hourly_data: List[HourlyAreaDwellItem] = []
        for h in range(24):
            h_start = day_start + datetime.timedelta(hours=h)
            h_end = h_start + datetime.timedelta(hours=1)
            hour_str = f"{h:02d}:00"

            area_dwell: Dict[str, float] = {a: 0.0 for a in all_areas}
            total_h_dwell = 0.0

            for e in events:
                e_start = e.started_at if e.started_at.tzinfo else e.started_at.replace(tzinfo=datetime.timezone.utc)
                e_end = e.ended_at if e.ended_at.tzinfo else e.ended_at.replace(tzinfo=datetime.timezone.utc)

                overlap_s = max(e_start, h_start)
                overlap_e = min(e_end, h_end)

                if overlap_e > overlap_s:
                    dur = (overlap_e - overlap_s).total_seconds()
                    area = e.zone_name or e.camera_name or "General Area"
                    area_dwell[area] = area_dwell.get(area, 0.0) + dur
                    total_h_dwell += dur

            # Cap max hour dwell at 3600 seconds (1 hr)
            portion = min(1.0, round(total_h_dwell / 3600.0, 3))
            
            hourly_data.append(
                HourlyAreaDwellItem(
                    hour=hour_str,
                    hour_int=h,
                    total_dwell_seconds=round(total_h_dwell, 1),
                    portion_of_hour=portion,
                    formatted_duration=format_seconds(total_h_dwell),
                    areas={k: round(v, 1) for k, v in area_dwell.items() if v > 0}
                )
            )

        return HourlyDwellResponse(
            target_date=target_date,
            person_id=person_id,
            person_name=person_name,
            all_areas=all_areas,
            hourly_data=hourly_data
        )

    # ==========================================
    # REVIEW QUEUE & CASCADING RECONCILIATION
    # ==========================================

    async def get_review_queue(
        self,
        tenant_id: uuid.UUID,
        target_date: Optional[datetime.date] = None
    ) -> List[ReviewQueueCandidate]:
        """
        Retrieves unconfirmed/candidate visitor identities detected on target_date
        with camera stops and suggested employee matches.
        """
        if target_date is None:
            target_date = datetime.datetime.now(datetime.timezone.utc).date()

        day_start = datetime.datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=datetime.timezone.utc)
        day_end = day_start + datetime.timedelta(days=1)

        # Find unconfirmed visitor identities with activity on this date
        stmt = (
            select(AdvancedPersonIdentity)
            .where(
                AdvancedPersonIdentity.tenant_id == tenant_id,
                AdvancedPersonIdentity.is_delete == False,
                AdvancedPersonIdentity.is_employee == False
            )
            .order_by(AdvancedPersonIdentity.created_at.desc())
        )
        res = await self.db.execute(stmt)
        candidate_idents = list(res.scalars().all())

        from modules.employees.repository import EmployeeRepository
        from modules.employees.cache import get_cached_employee_embeddings
        emp_cache = await get_cached_employee_embeddings(self.db, tenant_id)

        results: List[ReviewQueueCandidate] = []
        for ident in candidate_idents:
            # Query timeline events for this identity on this date
            tev_stmt = select(PersonTimelineEvent).where(
                PersonTimelineEvent.identity_id == ident.id,
                PersonTimelineEvent.is_delete == False,
                PersonTimelineEvent.started_at < day_end,
                PersonTimelineEvent.ended_at > day_start
            ).order_by(PersonTimelineEvent.started_at.asc())
            tev_res = await self.db.execute(tev_stmt)
            tevs = list(tev_res.scalars().all())

            if not tevs:
                continue

            first_seen = tevs[0].started_at
            last_seen = tevs[-1].ended_at
            total_dwell = sum(t.duration_seconds for t in tevs)
            cams = list(dict.fromkeys(t.camera_name for t in tevs if t.camera_name))
            crop_path = tevs[0].entry_crop_path or tevs[0].exit_crop_path

            # Check for closest employee match in cache
            sugg_emp_id = None
            sugg_emp_name = None
            sugg_sim = None

            if emp_cache:
                # Fetch face embedding for this identity if any
                f_emb_stmt = select(AdvancedPersonEmbedding.embedding).where(
                    AdvancedPersonEmbedding.identity_id == ident.id,
                    AdvancedPersonEmbedding.embedding_type == "face",
                    AdvancedPersonEmbedding.is_delete == False
                ).limit(1)
                f_res = await self.db.execute(f_emb_stmt)
                f_emb_raw = f_res.scalars().first()
                if f_emb_raw:
                    import numpy as np
                    fv = np.array(f_emb_raw, dtype=np.float32)
                    fn = np.linalg.norm(fv)
                    if fn > 0:
                        fu = fv / fn
                        best_s = 0.0
                        best_e = None
                        for emp, ref_emb in emp_cache:
                            rn = np.linalg.norm(ref_emb)
                            if rn > 0:
                                s = float(np.dot(fu, ref_emb / rn))
                                if s > best_s:
                                    best_s = s
                                    best_e = emp
                        if best_e and best_s > 0.30:
                            sugg_emp_id = best_e.id
                            sugg_emp_name = f"{best_e.first_name} {best_e.last_name}"
                            sugg_sim = round(best_s, 3)

            results.append(
                ReviewQueueCandidate(
                    identity_id=ident.id,
                    crop_url=format_crop_url(crop_path),
                    first_seen_at=first_seen,
                    last_seen_at=last_seen,
                    total_dwell_seconds=round(total_dwell, 1),
                    camera_stops_count=len(cams),
                    cameras_visited=cams,
                    suggested_employee_id=sugg_emp_id,
                    suggested_employee_name=sugg_emp_name,
                    suggested_similarity=sugg_sim
                )
            )

        return results

    async def reconcile_identity(
        self,
        tenant_id: uuid.UUID,
        identity_id: uuid.UUID,
        data: ReconcileIdentityRequest
    ) -> dict:
        """
        Reconciles an unconfirmed visitor identity into an employee or named visitor,
        with automatic cascading merge of related appearances across cameras.
        """
        ident = await self.repo.get_identity_by_id(identity_id, tenant_id)
        if not ident:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Identity not found or unauthorized."
            )

        import numpy as np
        from modules.employees.repository import EmployeeRepository
        emp_repo = EmployeeRepository(self.db)

        # Get reference embeddings for identity
        ref_embs_stmt = select(AdvancedPersonEmbedding).where(
            AdvancedPersonEmbedding.identity_id == identity_id,
            AdvancedPersonEmbedding.is_delete == False
        )
        ref_embs_res = await self.db.execute(ref_embs_stmt)
        ref_embs = list(ref_embs_res.scalars().all())

        ref_reid = next((e.embedding for e in ref_embs if e.embedding_type == "appearance"), None)
        ref_face = next((e.embedding for e in ref_embs if e.embedding_type == "face"), None)

        merged_identity_ids = [identity_id]
        target_name = ""

        if data.target_employee_id:
            emp = await emp_repo.get_employee_by_id(data.target_employee_id, tenant_id)
            if not emp:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Target employee not found."
                )
            target_name = f"{emp.first_name} {emp.last_name}"
            ident.is_employee = True
            ident.employee_id = emp.id
            ident.visitor_name = target_name

            # Add face exemplar to employee if face was captured
            if ref_face:
                await emp_repo.add_employee_face_exemplar(emp.id, ref_face)
                from modules.employees.cache import invalidate_employee_embeddings_cache
                await invalidate_employee_embeddings_cache(tenant_id)

            # Rewire existing timeline events for this identity
            await self.db.execute(
                update(PersonTimelineEvent)
                .where(PersonTimelineEvent.identity_id == identity_id)
                .values(
                    person_type="employee",
                    employee_id=emp.id,
                    identity_source="manual_review"
                )
            )

        elif data.or_visitor_name:
            target_name = data.or_visitor_name.strip()
            ident.visitor_name = target_name
            ident.is_employee = False
            ident.employee_id = None

        # Automatic Cascading Merge for related visitor identities
        if data.auto_merge_similar and (ref_reid or ref_face):
            # Query candidate visitor identities for this tenant
            other_stmt = select(AdvancedPersonIdentity).where(
                AdvancedPersonIdentity.tenant_id == tenant_id,
                AdvancedPersonIdentity.id != identity_id,
                AdvancedPersonIdentity.is_delete == False,
                AdvancedPersonIdentity.is_employee == False
            )
            other_res = await self.db.execute(other_stmt)
            other_idents = list(other_res.scalars().all())

            for o_id in other_idents:
                o_embs_stmt = select(AdvancedPersonEmbedding).where(
                    AdvancedPersonEmbedding.identity_id == o_id.id,
                    AdvancedPersonEmbedding.is_delete == False
                )
                o_embs = list((await self.db.execute(o_embs_stmt)).scalars().all())

                is_match = False
                # Check ReID similarity
                if ref_reid:
                    o_reid = next((e.embedding for e in o_embs if e.embedding_type == "appearance"), None)
                    if o_reid:
                        v1, v2 = np.array(ref_reid, dtype=np.float32), np.array(o_reid, dtype=np.float32)
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 0 and n2 > 0 and float(np.dot(v1, v2) / (n1 * n2)) >= data.similarity_threshold:
                            is_match = True

                # Check Face similarity
                if not is_match and ref_face:
                    o_face = next((e.embedding for e in o_embs if e.embedding_type == "face"), None)
                    if o_face:
                        f1, f2 = np.array(ref_face, dtype=np.float32), np.array(o_face, dtype=np.float32)
                        fn1, fn2 = np.linalg.norm(f1), np.linalg.norm(f2)
                        if fn1 > 0 and fn2 > 0 and float(np.dot(f1, f2) / (fn1 * fn2)) >= 0.50:
                            is_match = True

                if is_match:
                    merged_identity_ids.append(o_id.id)
                    # Rewire timeline events
                    if data.target_employee_id:
                        await self.db.execute(
                            update(PersonTimelineEvent)
                            .where(PersonTimelineEvent.identity_id == o_id.id)
                            .values(
                                identity_id=identity_id,
                                person_type="employee",
                                employee_id=data.target_employee_id,
                                identity_source="review_cascade"
                            )
                        )
                    else:
                        await self.db.execute(
                            update(PersonTimelineEvent)
                            .where(PersonTimelineEvent.identity_id == o_id.id)
                            .values(
                                identity_id=identity_id,
                                identity_source="review_cascade"
                            )
                        )

                    # Rewire occurrences
                    await self.db.execute(
                        update(AdvancedPersonOccurrence)
                        .where(AdvancedPersonOccurrence.identity_id == o_id.id)
                        .values(identity_id=identity_id)
                    )

                    # Mark duplicate identity deleted
                    o_id.is_delete = True

        await self.db.commit()

        return {
            "primary_identity_id": str(identity_id),
            "assigned_name": target_name,
            "is_employee": ident.is_employee,
            "merged_cards_count": len(merged_identity_ids),
            "merged_identity_ids": [str(i) for i in merged_identity_ids],
            "message": f"Successfully reconciled {len(merged_identity_ids)} card(s) into {target_name}."
        }

    async def delete_visitor_identity(self, tenant_id: uuid.UUID, identity_id: uuid.UUID) -> Dict[str, Any]:
        stmt = select(AdvancedPersonIdentity).where(
            AdvancedPersonIdentity.id == identity_id,
            AdvancedPersonIdentity.tenant_id == tenant_id,
            AdvancedPersonIdentity.is_delete == False
        )
        res = await self.db.execute(stmt)
        identity = res.scalars().first()
        if not identity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Visitor profile not found."
            )

        # Soft delete identity
        identity.is_delete = True

        # Soft delete associated occurrences
        await self.db.execute(
            update(AdvancedPersonOccurrence)
            .where(AdvancedPersonOccurrence.identity_id == identity_id)
            .values(is_delete=True)
        )

        # Soft delete associated embeddings
        await self.db.execute(
            update(AdvancedPersonEmbedding)
            .where(AdvancedPersonEmbedding.identity_id == identity_id)
            .values(is_delete=True)
        )

        # Soft delete associated timeline events
        await self.db.execute(
            update(PersonTimelineEvent)
            .where(PersonTimelineEvent.identity_id == identity_id)
            .values(is_delete=True)
        )

        # Soft delete associated visitor attendance logs
        await self.db.execute(
            update(AdvancedVisitorAttendanceLog)
            .where(AdvancedVisitorAttendanceLog.identity_id == identity_id)
            .values(is_delete=True)
        )

        try:
            await self.db.execute(
                update(AdvancedLineCrossingLog)
                .where(AdvancedLineCrossingLog.identity_id == identity_id)
                .values(is_delete=True)
            )
        except Exception:
            pass

        await self.db.commit()

        return {
            "message": "Visitor profile deleted successfully.",
            "identity_id": str(identity_id)
        }

    async def search_by_photo(
        self,
        tenant_id: uuid.UUID,
        image_bytes: bytes,
        similarity_threshold: float = 0.55,
        limit: int = 20
    ) -> PhotoSearchResponse:
        import cv2
        import numpy as np
        from application.utils.face_rec import face_rec_service
        from application.utils.reid import reid_service

        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        face_emb = None
        face_detected = False
        try:
            faces = face_rec_service.extract_faces(image_bytes)
            if faces:
                best_face = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                face_emb = best_face.get("embedding")
                face_detected = True
        except Exception:
            faces = []

        reid_emb = None
        appearance_extracted = False
        if img_bgr is not None and img_bgr.size > 0:
            try:
                reid_emb = reid_service.extract_embedding(img_bgr)
                if reid_emb:
                    appearance_extracted = True
            except Exception:
                reid_emb = None

        if not face_emb and not reid_emb:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract face or appearance features from the uploaded photo. Please upload a clear JPG/PNG image."
            )

        # Collect candidate matches
        matched_candidates_map = {}

        # 1. Search via Face Embedding
        if face_emb:
            face_matches = await self.repo.search_identities_by_embedding(
                tenant_id=tenant_id,
                target_embedding=face_emb,
                embedding_type="face",
                similarity_threshold=similarity_threshold,
                limit=limit
            )
            for m in face_matches:
                k = (m["identity_type"], m["identity_id"])
                matched_candidates_map[k] = {
                    **m,
                    "matched_via": "face"
                }

        # 2. Search via Appearance / ReID Embedding
        if reid_emb:
            app_matches = await self.repo.search_identities_by_embedding(
                tenant_id=tenant_id,
                target_embedding=reid_emb,
                embedding_type="appearance",
                similarity_threshold=similarity_threshold,
                limit=limit
            )
            for m in app_matches:
                k = (m["identity_type"], m["identity_id"])
                if k in matched_candidates_map:
                    if m["similarity_score"] > matched_candidates_map[k]["similarity_score"]:
                        matched_candidates_map[k]["similarity_score"] = m["similarity_score"]
                        matched_candidates_map[k]["matched_via"] = "appearance"
                    else:
                        matched_candidates_map[k]["matched_via"] = "face+reid"
                else:
                    matched_candidates_map[k] = {
                        **m,
                        "matched_via": "appearance"
                    }

        all_matches = sorted(matched_candidates_map.values(), key=lambda x: x["similarity_score"], reverse=True)[:limit]

        # For each match, assemble timeline events & appearances
        result_items: List[PhotoSearchMatchItem] = []
        for cand in all_matches:
            conds = [
                PersonTimelineEvent.tenant_id == tenant_id,
                PersonTimelineEvent.is_delete == False
            ]
            id_filters = []
            if cand.get("employee_id"):
                id_filters.append(PersonTimelineEvent.employee_id == cand["employee_id"])
            if cand.get("advanced_identity_id"):
                id_filters.append(PersonTimelineEvent.identity_id == cand["advanced_identity_id"])
            if cand.get("identity_id"):
                id_filters.append(PersonTimelineEvent.identity_id == cand["identity_id"])

            if id_filters:
                conds.append(or_(*id_filters))

            tev_stmt = (
                select(PersonTimelineEvent)
                .where(*conds)
                .order_by(PersonTimelineEvent.started_at.asc())
            )
            tev_res = await self.db.execute(tev_stmt)
            timeline_events = list(tev_res.scalars().all())

            occ_conds = [AdvancedPersonOccurrence.is_delete == False]
            occ_ids = []
            if cand.get("advanced_identity_id"):
                occ_ids.append(AdvancedPersonOccurrence.identity_id == cand["advanced_identity_id"])
            if cand.get("identity_id"):
                occ_ids.append(AdvancedPersonOccurrence.identity_id == cand["identity_id"])
            if occ_ids:
                occ_conds.append(or_(*occ_ids))
                occ_stmt = select(AdvancedPersonOccurrence).where(*occ_conds).order_by(AdvancedPersonOccurrence.first_seen.asc())
                occ_res = await self.db.execute(occ_stmt)
                occurrences = list(occ_res.scalars().all())
            else:
                occurrences = []

            primary_photo = format_crop_url(cand.get("photo_path"))
            if not primary_photo and occurrences:
                for occ in occurrences:
                    if occ.crop_path and os.path.exists(occ.crop_path):
                        primary_photo = format_crop_url(occ.crop_path)
                        break

            if not primary_photo and timeline_events:
                for tev in timeline_events:
                    if tev.entry_crop_path and os.path.exists(tev.entry_crop_path):
                        primary_photo = format_crop_url(tev.entry_crop_path)
                        break

            appearance_items: List[PhotoSearchAppearanceItem] = []
            for tev in timeline_events:
                offset_sec = 0.0
                matched_occ = next((o for o in occurrences if o.session_id == tev.session_id), None)
                if matched_occ and matched_occ.first_seen is not None:
                    offset_sec = float(matched_occ.first_seen)

                crop_u = format_crop_url(tev.entry_crop_path) or format_crop_url(tev.exit_crop_path)
                if not crop_u and matched_occ:
                    crop_u = format_crop_url(matched_occ.crop_path)

                appearance_items.append(
                    PhotoSearchAppearanceItem(
                        session_id=tev.session_id,
                        camera_name=tev.camera_name or "Camera",
                        zone_name=tev.zone_name,
                        timestamp=tev.started_at,
                        timestamp_offset_seconds=round(offset_sec, 2),
                        dwell_seconds=round(tev.duration_seconds, 2),
                        crop_url=crop_u,
                        thumbnail_url=crop_u,
                        event_type=tev.event_type
                    )
                )

            if not appearance_items and occurrences:
                for occ in occurrences:
                    crop_u = format_crop_url(occ.crop_path)
                    appearance_items.append(
                        PhotoSearchAppearanceItem(
                            session_id=occ.session_id,
                            camera_name="Main Camera",
                            zone_name=None,
                            timestamp=occ.created_at or datetime.datetime.now(datetime.timezone.utc),
                            timestamp_offset_seconds=round(float(occ.first_seen or 0.0), 2),
                            dwell_seconds=round(float((occ.last_seen or 0.0) - (occ.first_seen or 0.0)), 2),
                            crop_url=crop_u,
                            thumbnail_url=crop_u,
                            event_type="appearance"
                        )
                    )

            first_seen = appearance_items[0].timestamp if appearance_items else None
            last_seen = appearance_items[-1].timestamp if appearance_items else None
            sim_score = cand["similarity_score"]

            result_items.append(
                PhotoSearchMatchItem(
                    identity_type=cand["identity_type"],
                    identity_id=cand["identity_id"],
                    name=cand["name"],
                    code=cand["code"],
                    similarity_score=round(sim_score, 3),
                    similarity_percentage=f"{int(round(sim_score * 100))}%",
                    matched_via=cand["matched_via"],
                    primary_photo_url=primary_photo,
                    total_appearances=len(appearance_items),
                    first_seen_at=first_seen,
                    last_seen_at=last_seen,
                    timeline_events=appearance_items
                )
            )

        return PhotoSearchResponse(
            query_processed=True,
            face_detected_in_query=face_detected,
            appearance_extracted=appearance_extracted,
            total_matches_found=len(result_items),
            matches=result_items
        )



