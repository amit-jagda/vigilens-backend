import uuid
import datetime
import os
from typing import List, Optional
from fastapi import APIRouter, Depends, status, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db
from shared.schemas.response import StandardResponse
from shared.dependencies.auth import require_admin, require_viewer
from modules.users.model import User
from application.advancedpeopleanalytics.service import AdvancedPeopleAnalyticsService
from application.advancedpeopleanalytics.schema import (
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
    AssociationRequest
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
        employee_code=data.employee_code
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


# ==========================================
# ANALYTICS SESSION MANAGEMENT
# ==========================================

@router.post(
    "/process",
    response_model=StandardResponse[List[AdvancedPeopleAnalyticsSessionResponse]],
    status_code=status.HTTP_202_ACCEPTED
)
async def process_batch_sessions(
    request: ProcessAdvancedVideosRequest,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    tenant_id = verify_tenant(current_user)
    service = AdvancedPeopleAnalyticsService(db)
    sessions = await service.create_and_start_sessions(
        tenant_id=tenant_id,
        videos=request.videos,
        global_line_start=request.line_start,
        global_line_end=request.line_end,
        global_similarity_threshold=request.similarity_threshold,
        global_confidence_threshold=request.confidence_threshold,
        global_track_employees=request.track_employees,
        global_register_new_visitors=request.register_new_visitors,
        global_track_repeat_visitors=request.track_repeat_visitors,
        global_line_crossing_analysis=request.line_crossing_analysis,
        global_track_occupancy=request.track_occupancy,
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


@router.get(
    "/sessions/{session_id}/video",
    status_code=status.HTTP_200_OK
)
async def stream_annotated_video(
    session_id: uuid.UUID,
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
    if not session.output_video_path or not os.path.exists(session.output_video_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Annotated video file is missing or not generated on disk."
        )

    file_ext = os.path.splitext(session.output_video_path)[1].lower()
    media_type = "image/jpeg" if file_ext == ".jpg" else "video/mp4"

    return FileResponse(
        path=session.output_video_path,
        media_type=media_type,
        filename=f"{session_id}{file_ext}"
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
