import os
import uuid
import cv2
import logging
from datetime import datetime, timezone, timedelta
import numpy as np
from sqlalchemy import select, delete
from collections import defaultdict
from ultralytics import YOLO
import supervision as sv
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)
cv2.setNumThreads(0)

import torch
if torch.cuda.is_available():
    torch.set_num_threads(1)

from workers.celery import celery_app
from database.session import SessionLocal
from workers.utils import run_async

from application.advancedpeopleanalytics.model import (
    AdvancedPeopleAnalyticsSession,
    AdvancedPersonIdentity,
    AdvancedPersonEmbedding,
    CameraZone,
    CameraZoneLink,
    ZoneCrossingEvent,
    CrossCameraIdentityLink,
    PersonTimelineEvent
)
from application.advancedpeopleanalytics.repository import AdvancedPeopleAnalyticsRepository
from modules.employees.model import Employee
from modules.employees.cache import get_cached_employee_embeddings

# Import 100% self-contained module & utility functions
from application.utils.math_utils import (
    map_similarity_threshold,
    find_best_match_in_cache,
    is_face_occluded
)
from application.utils.line_counter import LineCrossingCounter
from application.utils.video_utils import get_video_rotation, orient_frame, probe_video
from application.utils.face_rec import face_rec_service
from application.utils.reid import reid_service

ADVANCED_OUTPUTS_DIR = os.path.join("storage", "advanced_people_analytics_outputs")
os.makedirs(ADVANCED_OUTPUTS_DIR, exist_ok=True)

VISITOR_CROPS_DIR = os.path.join("storage", "visitor_crops")
os.makedirs(VISITOR_CROPS_DIR, exist_ok=True)


@celery_app.task(name="application.advancedpeopleanalytics.tasks.process_advanced_people_analytics_task")
def process_advanced_people_analytics_task(
    session_id_str: str,
    filepath: str,
    line_start: list[int] | None = None,
    line_end: list[int] | None = None,
    similarity_threshold: float = 0.85,
    confidence_threshold: float = 0.3,
    user_id_str: str | None = None,
    track_employees: bool = True,
    register_new_visitors: bool = True,
    track_repeat_visitors: bool = True,
    line_crossing_analysis: bool = True,
    track_occupancy: bool = True
):
    """
    Celery background task for Advanced People Analytics Suite with ReID & Spatial Topology.
    """
    session_id = uuid.UUID(session_id_str)

    async def run():
        async with SessionLocal() as db:
            repo = AdvancedPeopleAnalyticsRepository(db)
            
            stmt = select(AdvancedPeopleAnalyticsSession).where(
                AdvancedPeopleAnalyticsSession.id == session_id,
                AdvancedPeopleAnalyticsSession.is_delete == False
            )
            res = await db.execute(stmt)
            session = res.scalars().first()
            if not session:
                return

            session.status = "processing"
            await db.commit()

            try:
                logger.info(f"Advanced People Analytics processing started for session ID {session_id}. File: {filepath}")
                from services.storage import storage_client
                local_filepath = storage_client.get_file_path(filepath)

                # Auto-probe video metadata & recording timestamp
                probe_meta = probe_video(local_filepath)
                if probe_meta.has_timestamp and probe_meta.recording_started_at:
                    session.recording_started_at = probe_meta.recording_started_at
                    await db.commit()

                file_ext = os.path.splitext(local_filepath)[1].lower()
                is_image = file_ext in {".jpg", ".jpeg", ".png", ".heic", ".heif"}

                from database.redis import get_redis_client, init_redis
                if get_redis_client() is None:
                    await init_redis()

                employee_cache = await get_cached_employee_embeddings(db, session.tenant_id)

                from configs.base import settings
                from shared.utils.model_loader import get_model_path
                model_filename = os.path.basename(settings.YOLO_MODEL)
                weights_path = get_model_path("attendance", model_filename)
                model = YOLO(weights_path)

                if is_image:
                    await _process_image_job(
                        db=db, repo=repo, session=session, filepath=local_filepath, model=model,
                        employee_cache=employee_cache, similarity_threshold=similarity_threshold,
                        confidence_threshold=confidence_threshold, user_id_str=user_id_str,
                        track_employees=track_employees, register_new_visitors=register_new_visitors,
                        track_repeat_visitors=track_repeat_visitors, track_occupancy=track_occupancy
                    )
                else:
                    await _process_video_job(
                        db=db, repo=repo, session=session, filepath=local_filepath, model=model,
                        employee_cache=employee_cache, line_start=line_start, line_end=line_end,
                        similarity_threshold=similarity_threshold, confidence_threshold=confidence_threshold,
                        user_id_str=user_id_str, track_employees=track_employees,
                        register_new_visitors=register_new_visitors, track_repeat_visitors=track_repeat_visitors,
                        line_crossing_analysis=line_crossing_analysis, track_occupancy=track_occupancy
                    )

                logger.info(f"Advanced People Analytics completed successfully for session ID {session_id}.")

                # Automatically trigger cross-camera association task on session completion
                run_cross_camera_association_task.delay(str(session_id))

            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Advanced People Analytics processing failed for session ID {session_id}: {str(e)}")
                session.status = "failed"
                await db.commit()

    run_async(run())


async def _process_image_job(
    db, repo, session, filepath, model, employee_cache, similarity_threshold, confidence_threshold, user_id_str=None,
    track_employees: bool = True, register_new_visitors: bool = True, track_repeat_visitors: bool = True, track_occupancy: bool = True
):
    try:
        with Image.open(filepath) as pil_img:
            pil_img_transposed = ImageOps.exif_transpose(pil_img)
            img = cv2.cvtColor(np.array(pil_img_transposed), cv2.COLOR_RGB2BGR)
    except Exception:
        img = cv2.imread(filepath)

    if img is None:
        raise ValueError("Could not read image file.")

    h_orig, w_orig = img.shape[:2]
    results = model(img, conf=float(confidence_threshold), classes=[0], verbose=False)
    boxes = results[0].boxes

    total_person_count = len(boxes)
    output_filename = f"{uuid.uuid4()}_annotated.jpg"
    user_outputs_dir = os.path.join(ADVANCED_OUTPUTS_DIR, user_id_str) if user_id_str else ADVANCED_OUTPUTS_DIR
    os.makedirs(user_outputs_dir, exist_ok=True)
    output_path = os.path.join(user_outputs_dir, output_filename)
    cv2.imwrite(output_path, img)

    from services.storage import storage_client
    storage_client.upload_file(output_path, output_path)

    await repo.update_session_results(
        session_id=session.id,
        total_person_count=total_person_count if track_occupancy else None,
        output_video_path=output_path
    )
    await db.commit()


async def _process_video_job(
    db, repo, session, filepath, model, employee_cache, line_start, line_end, similarity_threshold, confidence_threshold, user_id_str=None,
    track_employees: bool = True, register_new_visitors: bool = True, track_repeat_visitors: bool = True, line_crossing_analysis: bool = True, track_occupancy: bool = True
):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_rotation = get_video_rotation(filepath)
    out_w, out_h = (orig_height, orig_width) if video_rotation in (90, 270) else (orig_width, orig_height)

    output_filename = f"{uuid.uuid4()}_annotated.mp4"
    user_outputs_dir = os.path.join(ADVANCED_OUTPUTS_DIR, user_id_str) if user_id_str else ADVANCED_OUTPUTS_DIR
    os.makedirs(user_outputs_dir, exist_ok=True)
    output_path = os.path.join(user_outputs_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    tracker = sv.ByteTrack(frame_rate=int(fps), lost_track_buffer=int(fps * 10))
    line_counter = None
    if line_crossing_analysis and line_start and line_end and len(line_start) == 2 and len(line_end) == 2 and line_start != line_end:
        line_counter = LineCrossingCounter(line_start, line_end)

    # Load Spatial Camera Zones if linked
    camera_zones = []
    if session.camera_node_id:
        camera_zones = await repo.get_camera_zones(session.camera_node_id)

    # Helper function for real-world datetime conversion
    def to_real_time(video_sec: float) -> datetime:
        base_time = session.recording_started_at or datetime.now(timezone.utc)
        return base_time + timedelta(seconds=video_sec)

    active_tracks = {}
    occupancy_history = []
    peak_occupancy_so_far = 0
    frame_step = max(1, int(fps / 5))
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if video_rotation != 0:
            frame = orient_frame(frame, video_rotation)

        if frame_idx % 10 == 0:
            progress = int((frame_idx / total_frames) * 100) if total_frames > 0 else 0
            try:
                from database.redis import get_redis_client
                r_client = get_redis_client()
                if r_client:
                    await r_client.setex(f"advancedpeopleanalytics:progress:{session.id}", 3600, str(progress))
            except Exception:
                pass

        timestamp_sec = frame_idx / fps
        real_time_now = to_real_time(timestamp_sec)
        frame_idx += 1

        results = model(frame, conf=float(confidence_threshold), classes=[0], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        detections = tracker.update_with_detections(detections)

        current_occupancy = len(detections) if detections.tracker_id is not None else 0
        occupancy_history.append({"time_sec": round(timestamp_sec, 2), "occupancy": current_occupancy})
        if current_occupancy > peak_occupancy_so_far:
            peak_occupancy_so_far = current_occupancy

        if detections.tracker_id is not None:
            for xyxy, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
                if tracker_id is None:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(out_w, x2), min(out_h, y2)

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                if tracker_id not in active_tracks:
                    body_crop = frame[y1:y2, x1:x2]
                    active_tracks[tracker_id] = {
                        "first_seen": timestamp_sec,
                        "last_seen": timestamp_sec,
                        "person_type": "visitor",
                        "employee_id": None,
                        "identity_id": None,
                        "identity_source": "tracking",
                        "identity_confidence": 0.5,
                        "reid_embedding": None,
                        "current_zones": set(),
                        "face_confirmed": False
                    }
                track_info = active_tracks[tracker_id]
                track_info["last_seen"] = timestamp_sec

                # Extract ReID & Face embedding every frame_step
                if frame_idx % frame_step == 0 and (not track_info["face_confirmed"] or track_info["person_type"] == "visitor"):
                    body_crop = frame[y1:y2, x1:x2]
                    if body_crop is not None and body_crop.size > 0:
                        # Extract 512D ReID Embedding
                        reid_vec = reid_service.extract_embedding(body_crop)
                        if reid_vec is not None:
                            track_info["reid_embedding"] = reid_vec.tolist()

                        # Extract Face embedding with 10% padded crop for higher face detection accuracy
                        pad_y = int((y2 - y1) * 0.10)
                        pad_x = int((x2 - x1) * 0.10)
                        crop_y1, crop_y2 = max(0, y1 - pad_y), min(out_h, y2 + pad_y)
                        crop_x1, crop_x2 = max(0, x1 - pad_x), min(out_w, x2 + pad_x)
                        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                        faces = []
                        if face_crop is not None and face_crop.size > 0:
                            _, encoded_crop = cv2.imencode(".jpg", face_crop)
                            try:
                                faces = face_rec_service.extract_faces(encoded_crop.tobytes())
                            except Exception:
                                faces = []

                        valid_faces = []
                        for face in faces:
                            fx1, fy1, fx2, fy2 = map(int, face["bbox"])
                            if (fx2 - fx1) >= 35 and (fy2 - fy1) >= 35 and face.get("det_score", 0.0) >= 0.65:
                                valid_faces.append(face)

                        if valid_faces:
                            matched_face = max(valid_faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                            face_embedding = np.array(matched_face["embedding"], dtype=np.float32)
                            mapped_threshold = map_similarity_threshold(similarity_threshold)

                            # 1. Match against registered employees cache
                            match_emp = None
                            if track_employees and employee_cache:
                                match_emp = find_best_match_in_cache(face_embedding, employee_cache, mapped_threshold)

                            if match_emp:
                                employee, sim = match_emp
                                track_info["person_type"] = "employee"
                                track_info["employee_id"] = employee.id
                                track_info["employee_name"] = f"{employee.first_name} {employee.last_name}"
                                track_info["employee_code"] = employee.employee_code
                                track_info["identity_source"] = "face"
                                track_info["identity_confidence"] = float(sim)
                                track_info["face_confirmed"] = True
                            else:
                                # 2. Match or create visitor
                                match_vis = None
                                if track_repeat_visitors and track_info["reid_embedding"]:
                                    match_vis = await repo.find_similar_visitor(
                                        session.tenant_id, track_info["reid_embedding"], mapped_threshold, class_id=0
                                    )
                                if match_vis:
                                    visitor, sim = match_vis
                                    track_info["identity_id"] = visitor.id
                                    track_info["identity_source"] = "reid"
                                    track_info["identity_confidence"] = float(sim)
                                elif register_new_visitors and track_info["reid_embedding"]:
                                    visitor = await repo.create_person_identity(session.tenant_id, class_id=0)
                                    track_info["identity_id"] = visitor.id
                                    track_info["identity_source"] = "reid"
                                    track_info["identity_confidence"] = 0.80

                                # Update appearance vector ONLY IF identity_confidence >= 0.85 and face confirmed
                                if track_info["identity_id"] and track_info["identity_confidence"] >= 0.85 and track_info["face_confirmed"]:
                                    await repo.create_person_embedding(
                                        identity_id=track_info["identity_id"],
                                        embedding=face_embedding.tolist(),
                                        bbox=matched_face["bbox"],
                                        timestamp=timestamp_sec
                                    )

                # Check Virtual Line Crossing Vector
                if line_counter:
                    direction = line_counter.update(tracker_id, (center_x, center_y))
                    if direction:
                        await repo.create_line_crossing_log(
                            session_id=session.id,
                            identity_id=track_info["identity_id"] or uuid.UUID(int=0),
                            tracker_id=tracker_id,
                            timestamp=timestamp_sec,
                            direction=direction
                        )

                # Check spatial Camera Zones polygon containment
                for zone in camera_zones:
                    poly_points = np.array(zone.polygon, dtype=np.int32)
                    is_inside = cv2.pointPolygonTest(poly_points, (center_x, center_y), False) >= 0
                    
                    if is_inside and zone.id not in track_info["current_zones"]:
                        track_info["current_zones"].add(zone.id)
                        await repo.create_zone_crossing_event(
                            tenant_id=session.tenant_id,
                            session_id=session.id,
                            zone_id=zone.id,
                            identity_id=track_info["identity_id"],
                            tracker_id=tracker_id,
                            event_type=zone.zone_type,
                            real_world_time=real_time_now,
                            reid_embedding=track_info.get("reid_embedding"),
                            face_confirmed=track_info["face_confirmed"],
                            confidence=track_info["identity_confidence"]
                        )
                    elif not is_inside and zone.id in track_info["current_zones"]:
                        track_info["current_zones"].remove(zone.id)

                # --- DRAW OVERLAYS FOR THIS TRACKED PERSON ---
                if track_info["person_type"] == "employee":
                    color = (0, 255, 0) # Green for Employee
                    emp_name = track_info.get("employee_name", "Employee")
                    emp_code = track_info.get("employee_code", "")
                    sim_pct = int(track_info["identity_confidence"] * 100)
                    code_str = f" [{emp_code}]" if emp_code else ""
                    label = f"#{tracker_id} | {emp_name}{code_str} ({sim_pct}%)"
                elif track_info["identity_source"] == "reid":
                    color = (255, 215, 0) # Cyan for ReID matched Visitor
                    label = f"#{tracker_id} | Vis ReID ({int(track_info['identity_confidence']*100)}%)"
                else:
                    color = (0, 165, 255) # Orange for Visitor
                    label = f"#{tracker_id} | Visitor"

                # Draw Bounding Box & Centroid
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (int(center_x), int(center_y)), 4, color, -1)

                # Label text background box
                (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - h_lbl - 8)), (x1 + w_lbl + 6, max(h_lbl, y1)), color, -1)
                cv2.putText(frame, label, (x1 + 3, max(h_lbl - 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # --- DRAW VIRTUAL GATE LINE VECTOR OVERLAY ---
        if line_counter:
            p1 = (int(line_counter.start_point[0]), int(line_counter.start_point[1]))
            p2 = (int(line_counter.end_point[0]), int(line_counter.end_point[1]))
            cv2.line(frame, p1, p2, (255, 255, 0), 3) # Yellow-Cyan Gate Line
            cv2.circle(frame, p1, 6, (0, 255, 0), -1) # Start (Green dot)
            cv2.circle(frame, p2, 6, (0, 0, 255), -1) # End (Red dot)

        # --- DRAW HUD SUMMARY BOX ---
        if track_occupancy or line_crossing_analysis:
            hud_w, hud_h = 340, 100
            if out_w > hud_w + 20 and out_h > hud_h + 20:
                sub = frame[10:10+hud_h, 10:10+hud_w]
                bg = np.zeros(sub.shape, dtype=np.uint8) + 15
                frame[10:10+hud_h, 10:10+hud_w] = cv2.addWeighted(sub, 0.3, bg, 0.7, 0)
                
                in_cnt = line_counter.in_count if line_counter else 0
                out_cnt = line_counter.out_count if line_counter else 0
                
                cv2.putText(frame, "ADVANCED PEOPLE & SPATIAL ANALYTICS", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"Occupancy: {current_occupancy}  |  Peak: {peak_occupancy_so_far}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.putText(frame, f"Gate Line  --> IN: {in_cnt}   <-- OUT: {out_cnt}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        output_writer.write(frame)

    cap.release()
    output_writer.release()

    # Create PersonTimelineEvent records at track completion
    camera_name = session.video_name
    if session.camera_node_id:
        c_node = await repo.get_camera_node_by_id(session.camera_node_id, session.tenant_id)
        if c_node:
            camera_name = c_node.name
    for tracker_id, info in active_tracks.items():
        started_at = to_real_time(info["first_seen"])
        ended_at = to_real_time(info["last_seen"])
        await repo.create_timeline_event(
            tenant_id=session.tenant_id,
            person_type=info["person_type"],
            session_id=session.id,
            camera_name=camera_name,
            event_type="presence",
            started_at=started_at,
            ended_at=ended_at,
            identity_source=info["identity_source"],
            identity_confidence=info["identity_confidence"],
            tracker_id=tracker_id,
            employee_id=info["employee_id"],
            identity_id=info["identity_id"]
        )

    # Transcode output video to browser-compatible H.264 format using shared video utility
    try:
        from shared.utils.video_format import videoFormatChanger
        videoFormatChanger(output_path, formats="h264", overwrite_input=True)
    except Exception as e:
        print(f"Video transcoding failed: {e}")

    from services.storage import storage_client
    storage_client.upload_file(output_path, output_path)

    avg_occ = float(np.mean([item["occupancy"] for item in occupancy_history])) if occupancy_history else 0.0
    total_person_count = len(active_tracks)
    emp_count = sum(1 for info in active_tracks.values() if info["person_type"] == "employee")
    vis_count = sum(1 for info in active_tracks.values() if info["person_type"] == "visitor")
    new_vis_count = sum(1 for info in active_tracks.values() if info["person_type"] == "visitor" and info["identity_source"] != "reid")
    unique_ids = set(info["employee_id"] or info["identity_id"] for info in active_tracks.values() if info["employee_id"] or info["identity_id"])
    unique_person_count = len(unique_ids) if unique_ids else total_person_count

    in_count = line_counter.in_count if line_counter else 0
    out_count = line_counter.out_count if line_counter else 0

    await repo.update_session_results(
        session_id=session.id,
        unique_person_count=unique_person_count,
        total_person_count=total_person_count,
        first_time_visitor_count=new_vis_count,
        peak_occupancy=peak_occupancy_so_far if track_occupancy else None,
        average_occupancy=avg_occ if track_occupancy else None,
        entry_count=in_count if line_crossing_analysis else None,
        exit_count=out_count if line_crossing_analysis else None,
        employee_count=emp_count,
        visitor_count=vis_count,
        occupancy_timeline=occupancy_history if track_occupancy else None,
        output_video_path=output_path
    )
    await db.commit()


# ==========================================
# CROSS-CAMERA IDENTITY ASSOCIATION TASK
# ==========================================

@celery_app.task(name="application.advancedpeopleanalytics.tasks.run_cross_camera_association_task")
def run_cross_camera_association_task(session_id_str: str):
    """
    Celery task that matches exit zone events from this session against linked entry zone events
    across other cameras within the spatio-temporal transit window.
    """
    session_id = uuid.UUID(session_id_str)

    async def run():
        async with SessionLocal() as db:
            repo = AdvancedPeopleAnalyticsRepository(db)
            
            session = await repo.get_session_by_id(session_id, uuid.UUID(int=0)) # Skip tenant filter for system task query
            if not session or not session.camera_node_id:
                # Fallback search without tenant lock
                stmt = select(AdvancedPeopleAnalyticsSession).where(
                    AdvancedPeopleAnalyticsSession.id == session_id,
                    AdvancedPeopleAnalyticsSession.is_delete == False
                )
                res = await db.execute(stmt)
                session = res.scalars().first()

            if not session or not session.camera_node_id:
                return

            # Fetch camera zones for this session's camera
            zones = await repo.get_camera_zones(session.camera_node_id)
            exit_zones = [z for z in zones if z.zone_type in {"exit", "crossing"}]

            for exit_z in exit_zones:
                # Get outgoing links to other camera zones
                links = await repo.get_zone_links_from(exit_z.id)
                for link in links:
                    # Query exit events from exit_z
                    start_t = session.recording_started_at or (datetime.now(timezone.utc) - timedelta(hours=24))
                    end_t = session.completed_at or datetime.now(timezone.utc)
                    exit_events = await repo.get_zone_crossing_events(exit_z.id, start_t, end_t)

                    for exit_evt in exit_events:
                        if not exit_evt.reid_embedding:
                            continue

                        # Target time window: exit_time + 5s to exit_time + max_transit_seconds
                        window_start = exit_evt.real_world_time + timedelta(seconds=5)
                        window_end = exit_evt.real_world_time + timedelta(seconds=link.max_transit_seconds)

                        entry_events = await repo.get_zone_crossing_events(link.to_zone_id, window_start, window_end)

                        for entry_evt in entry_events:
                            if not entry_evt.reid_embedding or not exit_evt.identity_id or not entry_evt.identity_id:
                                continue

                            # 1. Cosine ReID Score
                            vec1 = np.array(exit_evt.reid_embedding, dtype=np.float32)
                            vec2 = np.array(entry_evt.reid_embedding, dtype=np.float32)
                            n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                            reid_score = float(np.dot(vec1, vec2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0

                            # 2. Time Transit Score
                            actual_transit = (entry_evt.real_world_time - exit_evt.real_world_time).total_seconds()
                            time_score = max(0.0, 1.0 - (abs(actual_transit - link.avg_transit_seconds) / link.max_transit_seconds))

                            # 3. Face Bonus
                            face_bonus = 0.15 if (exit_evt.face_confirmed and entry_evt.face_confirmed and exit_evt.identity_id == entry_evt.identity_id) else 0.0

                            final_score = (0.60 * reid_score) + (0.25 * time_score) + (0.15 * face_bonus)

                            if final_score >= 0.65:
                                await repo.create_cross_camera_link(
                                    tenant_id=session.tenant_id,
                                    from_session_id=exit_evt.session_id,
                                    to_session_id=entry_evt.session_id,
                                    from_identity_id=exit_evt.identity_id,
                                    to_identity_id=entry_evt.identity_id,
                                    reid_score=reid_score,
                                    time_score=time_score,
                                    final_score=final_score,
                                    is_confirmed=(final_score >= 0.85)
                                )

            await db.commit()

    run_async(run())
