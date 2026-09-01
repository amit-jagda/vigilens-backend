import os
import uuid
import datetime
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from application.advancedpeopleanalytics.repository import AdvancedPeopleAnalyticsRepository
from application.advancedpeopleanalytics.model import (
    AdvancedPeopleAnalyticsSession,
    AdvancedVisitorAttendanceLog,
    CameraNode,
    CameraNodeLink,
    CameraZone,
    CameraZoneLink,
    PersonTimelineEvent
)
from application.advancedpeopleanalytics.schema import (
    AdvancedVideoProcessItem,
    SessionDetectedPerson,
    VisitorAnalyticsReport,
    FirstTimeVisitorDetail,
    VisitorAttendanceResponse,
    RegisterVisitorRequest,
    CameraNodeCreate,
    CameraNodeResponse,
    CameraNodeLinkCreate,
    CameraNodeLinkResponse,
    CameraZoneCreate,
    CameraZoneResponse,
    CameraZoneLinkCreate,
    CameraZoneLinkResponse,
    PersonTimelineResponse,
    TimelineEventResponse
)

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
        node = await self.repo.create_camera_node(
            tenant_id=tenant_id,
            name=data.name,
            location_label=data.location_label
        )
        await self.db.commit()
        await self.db.refresh(node)
        return node

    async def get_camera_nodes(self, tenant_id: uuid.UUID) -> List[CameraNode]:
        return await self.repo.get_camera_nodes(tenant_id)

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
        timeline_events = [TimelineEventResponse.model_validate(e) for e in events]
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
        user_id: Optional[uuid.UUID] = None
    ) -> List[AdvancedPeopleAnalyticsSession]:
        from modules.gallery.repository import GalleryRepository
        gallery_repo = GalleryRepository(self.db)

        resolved_items = []
        for item in videos:
            filepath = None
            video_name = "Analytics_Video.mp4"

            if item.gallery_media_id:
                gallery_media_id = uuid.UUID(item.gallery_media_id)
                gallery_media = await gallery_repo.get_media_by_id(gallery_media_id, tenant_id)
                if not gallery_media:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Gallery media '{item.gallery_media_id}' not found or access denied."
                    )
                filepath = gallery_media.filepath
                video_name = gallery_media.filename
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

            resolved_items.append((item, filepath, video_name))

        sessions = []

        for item, filepath, video_name in resolved_items:
            line_start = item.line_start if item.line_start is not None else global_line_start
            line_end = item.line_end if item.line_end is not None else global_line_end
            similarity_threshold = item.similarity_threshold if item.similarity_threshold is not None else global_similarity_threshold
            confidence_threshold = item.confidence_threshold if item.confidence_threshold is not None else global_confidence_threshold
            track_employees = item.track_employees if item.track_employees is not None else global_track_employees
            register_new_visitors = item.register_new_visitors if item.register_new_visitors is not None else global_register_new_visitors
            track_repeat_visitors = item.track_repeat_visitors if item.track_repeat_visitors is not None else global_track_repeat_visitors
            line_crossing_analysis = item.line_crossing_analysis if item.line_crossing_analysis is not None else global_line_crossing_analysis
            track_occupancy = item.track_occupancy if item.track_occupancy is not None else global_track_occupancy

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
                camera_node_id=item.camera_node_id,
                recording_started_at=item.recording_started_at
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
                track_occupancy
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
        if session.output_video_path and os.path.exists(session.output_video_path):
            try:
                os.remove(session.output_video_path)
            except Exception:
                pass
        await self.repo.delete_session(session_id, tenant_id)
        await self.db.commit()
        return True

    async def register_or_update_visitor(
        self,
        tenant_id: uuid.UUID,
        identity_id: uuid.UUID,
        first_name: str,
        last_name: str,
        registration_type: str = "visitor",
        employee_code: Optional[str] = None
    ) -> Dict[str, Any]:
        identity = await self.repo.get_identity_by_id(identity_id, tenant_id)
        if not identity:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Person identity not found."
            )

        full_name = f"{first_name.strip()} {last_name.strip()}"
        if registration_type == "employee":
            if not employee_code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Employee code is required when registering as employee."
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

            if identity.embeddings:
                for emb in identity.embeddings:
                    emp_emb = EmployeeEmbedding(
                        employee_id=emp.id,
                        embedding=emb.embedding
                    )
                    self.db.add(emp_emb)

            identity.is_employee = True
            identity.employee_id = emp.id
            identity.visitor_name = full_name
            await self.db.commit()

            from modules.employees.cache import invalidate_employee_embeddings_cache
            await invalidate_employee_embeddings_cache(tenant_id)
            return {"message": f"Visitor successfully registered as Employee {full_name} ({emp.employee_code})", "employee_id": str(emp.id)}
        else:
            identity.visitor_name = full_name
            await self.db.commit()
            return {"message": f"Visitor name successfully updated to '{full_name}'", "identity_id": str(identity.id)}

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



