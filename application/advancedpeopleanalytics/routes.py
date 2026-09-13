import uuid
import datetime
import os
from typing import List, Optional, Union
from fastapi import APIRouter, Depends, status, HTTPException, Query, File, UploadFile, Form, Request
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from shared.schemas.response import StandardResponse
from shared.dependencies.auth import require_admin, require_viewer
from modules.users.model import User
from application.advancedpeopleanalytics.service import AdvancedPeopleAnalyticsService
from application.advancedpeopleanalytics.schema import (
    AdvancedVideoProcessItem,
    ProcessAdvancedVideosRequest,
    AdvancedPeopleAnalyticsSessionResponse,
    SessionDetectedPerson,
    VisitorAnalyticsReport,
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
    PersonSummaryItem,
    AssociationRequest,
    CreateSubclipRequest,
    SubclipResponse,
    PersonDwellByPhotoResponse,
    DailyCheckinResponse,
    HourlyDwellResponse,
    ReviewQueueCandidate,
    ReconcileIdentityRequest,
    PhotoSearchResponse
)

router = APIRouter(prefix="/advancedpeopleanalytics", tags=["Advanced People Analytics Suite"])


def verify_tenant(user: User) -> uuid.UUID:
    """Helper to verify and return the tenant_id from the authenticated user."""
    if not user.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The active user account is not associated with any tenant namespace."
        )
    return user.tenant_id


# ==========================================
# CAMERA & SPATIAL TOPOLOGY ENDPOINTS
# ==========================================

@router.post(
    "/cameras",
    response_model=StandardResponse[CameraNodeResponse],
    status_code=status.HTTP_201_CREATED
)
async def create_camera_node(
    data: CameraNodeCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    node = await service.create_camera_node(tenant_id, data)
    return StandardResponse(
        message="CameraNode created successfully.",
        status=status.HTTP_201_CREATED,
        data=CameraNodeResponse.model_validate(node)
    )


@router.get(
    "/cameras",
    response_model=StandardResponse[List[CameraNodeResponse]],
    status_code=status.HTTP_200_OK
)
async def list_camera_nodes(
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    nodes = await service.get_camera_nodes(tenant_id)
    return StandardResponse(
        message=f"Retrieved {len(nodes)} CameraNode(s).",
        status=status.HTTP_200_OK,
        data=[CameraNodeResponse.model_validate(n) for n in nodes]
    )


@router.put(
    "/cameras/{camera_id}",
    response_model=StandardResponse[CameraNodeResponse],
    status_code=status.HTTP_200_OK
)
async def update_camera_node(
    camera_id: uuid.UUID,
    data: CameraNodeUpdate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update camera node name or location label.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    node = await service.update_camera_node(tenant_id, camera_id, data)
    return StandardResponse(
        message="Camera node updated successfully.",
        status=status.HTTP_200_OK,
        data=CameraNodeResponse.model_validate(node)
    )


@router.delete(
    "/cameras/{camera_id}",
    response_model=StandardResponse[None],
    status_code=status.HTTP_200_OK
)
async def delete_camera_node(
    camera_id: uuid.UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    await service.delete_camera_node(camera_id, tenant_id)
    return StandardResponse(
        message="Camera node deleted successfully.",
        status=status.HTTP_200_OK,
        data=None
    )


@router.post(
    "/cameras/links",
    response_model=StandardResponse[CameraNodeLinkResponse],
    status_code=status.HTTP_201_CREATED
)
async def create_camera_link(
    data: CameraNodeLinkCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Creates a directed spatial connection between two Camera Nodes with transit time window constraints.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    link = await service.create_camera_node_link(tenant_id, data)
    return StandardResponse(
        message="Camera-to-Camera link created successfully.",
        status=status.HTTP_201_CREATED,
        data=link
    )


@router.get(
    "/cameras/links",
    response_model=StandardResponse[List[CameraNodeLinkResponse]],
    status_code=status.HTTP_200_OK
)
async def list_camera_links(
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    Lists all Camera-to-Camera connections and spatial graph edges for the tenant.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    links = await service.get_camera_node_links(tenant_id)
    return StandardResponse(
        message=f"Retrieved {len(links)} CameraLink(s).",
        status=status.HTTP_200_OK,
        data=links
    )


@router.delete(
    "/cameras/links/{link_id}",
    response_model=StandardResponse[None],
    status_code=status.HTTP_200_OK
)
async def delete_camera_link(
    link_id: uuid.UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Deletes a Camera-to-Camera connection.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    await service.delete_camera_node_link(link_id, tenant_id)
    return StandardResponse(
        message="Camera link deleted successfully.",
        status=status.HTTP_200_OK,
        data=None
    )


@router.post(
    "/cameras/{camera_id}/zones",
    response_model=StandardResponse[CameraZoneResponse],
    status_code=status.HTTP_201_CREATED
)
async def add_camera_zone(
    camera_id: uuid.UUID,
    data: CameraZoneCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    zone = await service.create_camera_zone(tenant_id, camera_id, data)
    return StandardResponse(
        message="CameraZone added successfully.",
        status=status.HTTP_201_CREATED,
        data=CameraZoneResponse.model_validate(zone)
    )


@router.get(
    "/cameras/{camera_id}/zones",
    response_model=StandardResponse[List[CameraZoneResponse]],
    status_code=status.HTTP_200_OK
)
async def list_camera_zones(
    camera_id: uuid.UUID,
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    service = AdvancedPeopleAnalyticsService(db)
    zones = await service.get_camera_zones(camera_id)
    return StandardResponse(
        message=f"Retrieved {len(zones)} zone(s) for camera.",
        status=status.HTTP_200_OK,
        data=[CameraZoneResponse.model_validate(z) for z in zones]
    )


@router.post(
    "/cameras/zone-links",
    response_model=StandardResponse[CameraZoneLinkResponse],
    status_code=status.HTTP_201_CREATED
)
async def link_camera_zones(
    data: CameraZoneLinkCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    link = await service.create_zone_link(tenant_id, data)
    return StandardResponse(
        message="CameraZoneLink created successfully.",
        status=status.HTTP_201_CREATED,
        data=CameraZoneLinkResponse.model_validate(link)
    )


# ==========================================
# CROSS-CAMERA & TIMELINE JOURNEY
# ==========================================

@router.post(
    "/associate",
    response_model=StandardResponse[int],
    status_code=status.HTTP_202_ACCEPTED
)
async def trigger_cross_camera_association(
    data: AssociationRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Manually triggers cross-camera identity association for a list of session IDs.
    """
    service = AdvancedPeopleAnalyticsService(db)
    count = await service.trigger_manual_association(data.session_ids)
    return StandardResponse(
        message=f"Initiated cross-camera association for {count} session(s).",
        status=status.HTTP_202_ACCEPTED,
        data=count
    )


@router.get(
    "/timeline",
    response_model=StandardResponse[PersonTimelineResponse],
    status_code=status.HTTP_200_OK
)
async def get_person_journey_timeline(
    person_type: str = Query(..., description="'employee' or 'visitor'"),
    person_id: uuid.UUID = Query(..., description="UUID of employee or visitor identity"),
    date: Optional[datetime.date] = Query(None, description="Optional target date (YYYY-MM-DD)"),
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieves a person's complete spatio-temporal journey across camera nodes and zones for a given date.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    timeline = await service.get_person_timeline(tenant_id, person_type, person_id, date)
    return StandardResponse(
        message="Person journey timeline retrieved successfully.",
        status=status.HTTP_200_OK,
        data=timeline
    )


@router.get(
    "/people",
    response_model=StandardResponse[List[PersonSummaryItem]],
    status_code=status.HTTP_200_OK
)
async def list_people_directory(
    date: Optional[datetime.date] = Query(None, description="Optional target date filter (YYYY-MM-DD)"),
    person_type: Optional[str] = Query(None, description="'employee' or 'visitor'"),
    search: Optional[str] = Query(None, description="Search query by name, ID, or employee code"),
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    List all detected individuals across cameras with dwell times, cameras visited, and latest status.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    people = await service.get_tenant_people_summary(
        tenant_id=tenant_id,
        target_date=date,
        person_type=person_type,
        search=search
    )
    return StandardResponse(
        message=f"Retrieved {len(people)} person profile(s).",
        status=status.HTTP_200_OK,
        data=[PersonSummaryItem.model_validate(p) for p in people]
    )


@router.post(
    "/reset",
    response_model=StandardResponse[dict],
    status_code=status.HTTP_200_OK
)
async def reset_all_analytics_data(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Clears all stored sessions, person identities, appearance/face embeddings, and timeline history.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    result = await service.reset_all_analytics_data(tenant_id)
    return StandardResponse(
        message=result["message"],
        status=status.HTTP_200_OK,
        data=result
    )


# ==========================================
# VISITOR & PERSON REGISTRATION
# ==========================================

@router.post(
    "/visitors/register",
    response_model=StandardResponse[dict],
    status_code=status.HTTP_200_OK
)
async def register_or_update_visitor(
    data: RegisterVisitorRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Update visitor name or convert a detected visitor to an employee.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    result = await service.register_or_update_visitor(
        tenant_id=tenant_id,
        identity_id=data.identity_id,
        first_name=data.first_name,
        last_name=data.last_name,
        registration_type=data.registration_type,
        employee_code=data.employee_code,
        existing_employee_id=data.existing_employee_id,
        retroactive_attendance=data.retroactive_attendance,
        force=data.force
    )
    return StandardResponse(
        message=result["message"],
        status=status.HTTP_200_OK,
        data=result
    )


@router.post(
    "/visitors/add-from-face",
    response_model=StandardResponse[dict],
    status_code=status.HTTP_201_CREATED
)
async def add_person_from_face_photo(
    file: UploadFile = File(...),
    first_name: str = Form(...),
    last_name: str = Form(...),
    registration_type: str = Form("visitor"),
    employee_code: Optional[str] = Form(None),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Directly enroll a new person (Visitor or Employee) by uploading a clear face photo.
    """
    tenant_id = verify_tenant(current_user)
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded face photo is empty."
        )
    service = AdvancedPeopleAnalyticsService(db)
    result = await service.add_person_from_face_photo(
        tenant_id=tenant_id,
        image_bytes=contents,
        first_name=first_name,
        last_name=last_name,
        registration_type=registration_type,
        employee_code=employee_code
    )
    return StandardResponse(
        message=result["message"],
        status=status.HTTP_201_CREATED,
        data=result
    )


@router.delete(
    "/visitors/{identity_id}",
    response_model=StandardResponse[dict],
    status_code=status.HTTP_200_OK
)
async def delete_visitor_identity(
    identity_id: uuid.UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Soft-delete a visitor identity profile and its associated occurrences, embeddings, and timeline events.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    result = await service.delete_visitor_identity(
        tenant_id=tenant_id,
        identity_id=identity_id
    )
    return StandardResponse(
        message=result["message"],
        status=status.HTTP_200_OK,
        data=result
    )


# ==========================================
# ANALYTICS SESSION MANAGEMENT
# ==========================================

@router.post(
    "/process",
    response_model=StandardResponse[List[AdvancedPeopleAnalyticsSessionResponse]],
    status_code=status.HTTP_202_ACCEPTED
)
async def process_batch_sessions(
    request: Union[ProcessAdvancedVideosRequest, List[AdvancedVideoProcessItem]],
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)

    if isinstance(request, list):
        req_obj = ProcessAdvancedVideosRequest(videos=request)
    else:
        req_obj = request

    sessions = await service.create_and_start_sessions(
        tenant_id=tenant_id,
        videos=req_obj.videos,
        global_line_start=req_obj.line_start,
        global_line_end=req_obj.line_end,
        global_similarity_threshold=req_obj.similarity_threshold,
        global_confidence_threshold=req_obj.confidence_threshold,
        global_track_employees=req_obj.track_employees,
        global_register_new_visitors=req_obj.register_new_visitors,
        global_track_repeat_visitors=req_obj.track_repeat_visitors,
        global_line_crossing_analysis=req_obj.line_crossing_analysis,
        global_track_occupancy=req_obj.track_occupancy,
        global_generate_video=req_obj.generate_video,
        global_start_time=req_obj.start_time,
        global_end_time=req_obj.end_time,
        user_id=current_user.id
    )
    return StandardResponse(
        message=f"Successfully registered and started processing for {len(sessions)} advanced session(s).",
        status=status.HTTP_202_ACCEPTED,
        data=[AdvancedPeopleAnalyticsSessionResponse.model_validate(s) for s in sessions]
    )


@router.get(
    "/sessions",
    response_model=StandardResponse[List[AdvancedPeopleAnalyticsSessionResponse]],
    status_code=status.HTTP_200_OK
)
async def list_analytics_sessions(
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    sessions = await service.get_all_sessions(tenant_id)
    return StandardResponse(
        message=f"Retrieved {len(sessions)} session(s).",
        status=status.HTTP_200_OK,
        data=[AdvancedPeopleAnalyticsSessionResponse.model_validate(s) for s in sessions]
    )


@router.get(
    "/sessions/{session_id}",
    response_model=StandardResponse[AdvancedPeopleAnalyticsSessionResponse],
    status_code=status.HTTP_200_OK
)
async def get_session_details(
    session_id: uuid.UUID,
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    session = await service.get_session(session_id, tenant_id)
    return StandardResponse(
        message="Session details retrieved successfully.",
        status=status.HTTP_200_OK,
        data=AdvancedPeopleAnalyticsSessionResponse.model_validate(session)
    )


@router.get(
    "/sessions/{session_id}/people",
    response_model=StandardResponse[List[SessionDetectedPerson]],
    status_code=status.HTTP_200_OK
)
async def get_session_detected_people(
    session_id: uuid.UUID,
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    people = await service.get_session_detected_people(session_id, tenant_id)
    return StandardResponse(
        message=f"Successfully retrieved {len(people)} detected person(s).",
        status=status.HTTP_200_OK,
        data=people
    )


def stream_video_file_with_range(request: Request, file_path: str, media_type: str = "video/mp4", filename: str = "video.mp4"):
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("range")

    if not range_header:
        def full_iter(chunk_size: int = 1024 * 1024):
            with open(file_path, "rb") as f:
                while chunk := f.read(chunk_size):
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
            "Content-Type": media_type,
            "Content-Disposition": f"inline; filename={filename}",
        }
        return StreamingResponse(full_iter(), status_code=200, headers=headers)

    range_val = range_header.strip()
    if range_val.startswith("bytes="):
        range_val = range_val[6:]
    
    parts = range_val.split("-")
    start_str = parts[0].strip() if len(parts) > 0 else ""
    end_str = parts[1].strip() if len(parts) > 1 else ""

    if start_str and end_str:
        start = int(start_str)
        end = int(end_str)
    elif start_str:
        start = int(start_str)
        end = file_size - 1
    elif end_str:
        start = max(0, file_size - int(end_str))
        end = file_size - 1
    else:
        start = 0
        end = file_size - 1

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    content_length = (end - start) + 1

    def range_iter(start_pos: int, length: int, chunk_size: int = 1024 * 512):
        with open(file_path, "rb") as f:
            f.seek(start_pos)
            remaining = length
            while remaining > 0:
                read_amount = min(remaining, chunk_size)
                data = f.read(read_amount)
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(content_length),
        "Content-Type": media_type,
        "Content-Disposition": f"inline; filename={filename}",
    }
    return StreamingResponse(range_iter(start, content_length), status_code=206, headers=headers)


@router.get(
    "/sessions/{session_id}/video",
    status_code=status.HTTP_200_OK
)
async def stream_annotated_video(
    session_id: uuid.UUID,
    request: Request,
    token: Optional[str] = Query(None),
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    session = await service.get_session(session_id, tenant_id)
    
    if session.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Video cannot be retrieved. Session status is currently '{session.status}'."
        )

    target_video_path = session.output_video_path if (session.output_video_path and os.path.exists(session.output_video_path)) else session.video_path
    if not target_video_path or not os.path.exists(target_video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Video file is missing or not found on disk."
        )

    file_ext = os.path.splitext(target_video_path)[1].lower()
    media_type = "image/jpeg" if file_ext in (".jpg", ".jpeg") else "video/mp4"

    if media_type == "image/jpeg":
        return FileResponse(
            path=target_video_path,
            media_type=media_type,
            filename=f"{session_id}{file_ext}"
        )

    return stream_video_file_with_range(
        request=request,
        file_path=target_video_path,
        media_type=media_type,
        filename=f"{session_id}{file_ext}"
    )


@router.post(
    "/sessions/{session_id}/rerun",
    response_model=StandardResponse[AdvancedPeopleAnalyticsSessionResponse],
    status_code=status.HTTP_202_ACCEPTED
)
async def rerun_analytics_session(
    session_id: uuid.UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Rerun video analytics for a previously created session using its original video file and parameters.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    session = await service.rerun_session(session_id, tenant_id, current_user.id)
    return StandardResponse(
        message=f"Session {session_id} rerun initiated successfully.",
        status=status.HTTP_202_ACCEPTED,
        data=AdvancedPeopleAnalyticsSessionResponse.model_validate(session)
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=StandardResponse[None],
    status_code=status.HTTP_200_OK
)
async def delete_analytics_session(
    session_id: uuid.UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    await service.delete_session(session_id, tenant_id)
    return StandardResponse(
        message="Analytics session and output video files deleted successfully.",
        status=status.HTTP_200_OK,
        data=None
    )


# ==========================================
# GALLERY REUSE & SUBCLIP & PHOTO ANALYTICS
# ==========================================

@router.get(
    "/gallery",
    response_model=StandardResponse[List[dict]],
    status_code=status.HTTP_200_OK
)
async def list_gallery_videos(
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    List available raw CCTV videos stored in the centralized gallery repository for the tenant.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    videos = await service.get_gallery_videos(tenant_id)
    return StandardResponse(
        message=f"Retrieved {len(videos)} gallery video(s).",
        status=status.HTTP_200_OK,
        data=videos
    )


@router.post(
    "/sessions/subclip",
    response_model=StandardResponse[SubclipResponse],
    status_code=status.HTTP_201_CREATED
)
async def create_video_subclip(
    data: CreateSubclipRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Extract a sub-clip from an existing video/session/gallery media using 0-second FFmpeg stream copy.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    subclip = await service.create_subclip(tenant_id, data)
    return StandardResponse(
        message="Video sub-clip generated successfully.",
        status=status.HTTP_201_CREATED,
        data=subclip
    )


@router.post(
    "/analytics/person-dwell-by-photo",
    response_model=StandardResponse[PersonDwellByPhotoResponse],
    status_code=status.HTTP_200_OK
)
async def get_person_dwell_by_photo(
    file: UploadFile = File(...),
    threshold: float = Query(0.35, ge=0.0, le=1.0, description="Cosine distance threshold (<=0.35 means >=65% similarity)"),
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a face photo to query how much total and place-wise dwell time that person has spent.
    """
    tenant_id = verify_tenant(current_user)
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded face photo is empty."
        )
    service = AdvancedPeopleAnalyticsService(db)
    result = await service.get_person_dwell_by_photo(tenant_id, contents, threshold=threshold)
    return StandardResponse(
        message="Person dwell analytics calculated successfully." if result.matched else "No matching face found in records.",
        status=status.HTTP_200_OK,
        data=result
    )


# ==========================================
# SEARCH BY PHOTO / REFERENCE IMAGE (VECTOR DB)
# ==========================================

@router.post(
    "/search/photo",
    response_model=StandardResponse[PhotoSearchResponse],
    status_code=status.HTTP_200_OK
)
async def search_by_photo(
    file: UploadFile = File(..., description="Query photo (face portrait or full body image)"),
    threshold: float = Query(0.55, ge=0.0, le=1.0, description="Similarity threshold for vector match"),
    limit: int = Query(20, ge=1, le=100, description="Max matches to return"),
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    Reverse searches vector DB using uploaded photo.
    Extracts 512-dim Face embedding & 512-dim Body ReID embedding,
    returns all matching people ranked by similarity with full timeline moments & seek timestamps.
    """
    tenant_id = verify_tenant(current_user)
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded query photo is empty."
        )
    service = AdvancedPeopleAnalyticsService(db)
    result = await service.search_by_photo(
        tenant_id=tenant_id,
        image_bytes=contents,
        similarity_threshold=threshold,
        limit=limit
    )
    return StandardResponse(
        message=f"Found {result.total_matches_found} matching identity/timeline appearance(s).",
        status=status.HTTP_200_OK,
        data=result
    )


# ==========================================
# DAILY CHECK-IN BRIDGE (MORNING ANCHOR)
# ==========================================

@router.post(
    "/employees/{employee_id}/daily-checkin",
    response_model=StandardResponse[DailyCheckinResponse],
    status_code=status.HTTP_200_OK
)
async def daily_employee_checkin(
    employee_id: uuid.UUID,
    face_image: UploadFile = File(..., description="Selfie, headshot, or portrait of employee (Required)"),
    appearance_image: Optional[UploadFile] = File(None, description="Full-body outfit/clothing photo of employee today (Optional)"),
    checkin_date: Optional[datetime.date] = Query(None, description="Date to anchor check-in to (defaults to today)"),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Morning check-in anchor: registers clear face and daily outfit before CCTV footages are analyzed.
    Flexible: Face image is required; Outfit is optional (or extracted from face_image if full-body).
    """
    tenant_id = verify_tenant(current_user)
    face_bytes = await face_image.read()
    if not face_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded face photo is empty."
        )
    appearance_bytes = await appearance_image.read() if appearance_image else None

    service = AdvancedPeopleAnalyticsService(db)
    res = await service.daily_employee_checkin(
        tenant_id=tenant_id,
        employee_id=employee_id,
        face_bytes=face_bytes,
        appearance_bytes=appearance_bytes,
        checkin_date=checkin_date
    )
    return StandardResponse(
        message=res.message,
        status=status.HTTP_200_OK,
        data=res
    )


# ==========================================
# AREA-WISE HOURLY DWELL TIME ANALYTICS
# ==========================================

@router.get(
    "/analytics/hourly-dwell",
    response_model=StandardResponse[HourlyDwellResponse],
    status_code=status.HTTP_200_OK
)
async def get_hourly_area_dwell(
    target_date: Optional[datetime.date] = Query(None, description="Date to aggregate (defaults to today)"),
    person_id: Optional[uuid.UUID] = Query(None, description="Filter for specific employee/visitor identity UUID"),
    session_id: Optional[uuid.UUID] = Query(None, description="Filter for specific video session UUID"),
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    Area-wise dwell time bar chart data hour-by-hour (00:00 to 23:00).
    Max bar height represents 1 hour (3600 seconds / 60 minutes).
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    res = await service.get_hourly_area_dwell(
        tenant_id=tenant_id,
        target_date=target_date,
        person_id=person_id,
        session_id=session_id
    )
    return StandardResponse(
        message="Hourly area dwell analytics retrieved successfully.",
        status=status.HTTP_200_OK,
        data=res
    )


# ==========================================
# REVIEW QUEUE & CASCADING RECONCILIATION
# ==========================================

@router.get(
    "/review-queue",
    response_model=StandardResponse[List[ReviewQueueCandidate]],
    status_code=status.HTTP_200_OK
)
async def get_review_queue(
    target_date: Optional[datetime.date] = Query(None, description="Date to review (defaults to today)"),
    current_user: User = Depends(require_viewer),
    db: AsyncSession = Depends(get_db)
):
    """
    Lists unconfirmed visitor identities with recurring camera appearances and suggested employee matches.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    items = await service.get_review_queue(tenant_id, target_date=target_date)
    return StandardResponse(
        message=f"Retrieved {len(items)} candidate(s) awaiting review.",
        status=status.HTTP_200_OK,
        data=items
    )


@router.post(
    "/identities/{identity_id}/reconcile",
    response_model=StandardResponse[dict],
    status_code=status.HTTP_200_OK
)
async def reconcile_identity(
    identity_id: uuid.UUID,
    data: ReconcileIdentityRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Assigns an unconfirmed visitor identity to an employee or named visitor,
    and automatically cascades the match to merge related cards across cameras.
    """
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    res = await service.reconcile_identity(tenant_id, identity_id, data)
    return StandardResponse(
        message=res["message"],
        status=status.HTTP_200_OK,
        data=res
    )
