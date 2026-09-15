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
    CameraNode,
    CameraNodeLink,
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
from application.utils.video_utils import (
    get_video_rotation,
    orient_frame,
    probe_video,
    extract_masked_crop,
    extract_head_crop_from_mask
)
from application.utils.face_rec import face_rec_service
from application.utils.reid import reid_service
from application.advancedpeopleanalytics.coco_registry import (
    COCO_CLASS_TO_ID,
    COCO_ID_TO_CLASS,
    DEFAULT_APA_TRACKED_CLASSES,
    resolve_class_ids,
    calculate_containment_ratio,
    calculate_bbox_iou,
)
from configs.base import settings

ADVANCED_OUTPUTS_DIR = os.path.join("storage", "advanced_people_analytics_outputs")
os.makedirs(ADVANCED_OUTPUTS_DIR, exist_ok=True)

VISITOR_CROPS_DIR = os.path.join("storage", "visitor_crops")
os.makedirs(VISITOR_CROPS_DIR, exist_ok=True)


def get_model_scale_label(model_name: str) -> str:
    name = str(model_name).lower()
    if any(k in name for k in ["nano", "8n", "11n", "26n", "yolon", "-n."]):
        return "NANO"
    elif any(k in name for k in ["small", "8s", "11s", "26s", "yolos", "-s."]):
        return "SMALL"
    elif any(k in name for k in ["medium", "8m", "11m", "26m", "yolom", "-m."]):
        return "MEDIUM"
    elif any(k in name for k in ["large", "8l", "11l", "26l", "yolol", "-l."]):
        return "LARGE"
    elif any(k in name for k in ["xlarge", "8x", "11x", "26x", "yolox", "-x."]):
        return "XLARGE"
    return "CUSTOM"


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
    track_occupancy: bool = True,
    track_objects: bool = True,
    generate_video: bool = False
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
                logger.info(f"Advanced People Analytics processing started for session ID {session_id}. File: {filepath} (generate_video={generate_video})")
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
                logger.info(f"👥 [Session {session.id}] Loaded {len(employee_cache)} registered employee face embedding(s) for tenant {session.tenant_id}")

                from configs.base import settings
                from shared.utils.model_loader import get_model_path
                seg_model_filename = os.path.basename(settings.YOLO_SEG_MODEL)
                seg_weights_path = get_model_path("attendance", seg_model_filename)
                model = YOLO(seg_weights_path)

                if is_image:
                    await _process_image_job(
                        db=db, repo=repo, session=session, filepath=local_filepath, model=model,
                        employee_cache=employee_cache, similarity_threshold=similarity_threshold,
                        confidence_threshold=confidence_threshold, user_id_str=user_id_str,
                        track_employees=track_employees, register_new_visitors=register_new_visitors,
                        track_repeat_visitors=track_repeat_visitors, track_occupancy=track_occupancy,
                        generate_video=generate_video
                    )
                else:
                    await _process_video_job(
                        db=db, repo=repo, session=session, filepath=local_filepath, model=model,
                        employee_cache=employee_cache, line_start=line_start, line_end=line_end,
                        similarity_threshold=similarity_threshold, confidence_threshold=confidence_threshold,
                        user_id_str=user_id_str, track_employees=track_employees,
                        register_new_visitors=register_new_visitors, track_repeat_visitors=track_repeat_visitors,
                        line_crossing_analysis=line_crossing_analysis, track_occupancy=track_occupancy,
                        track_objects=track_objects, generate_video=generate_video
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
    track_employees: bool = True, register_new_visitors: bool = True, track_repeat_visitors: bool = True, track_occupancy: bool = True,
    generate_video: bool = False
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
    output_path = None
    if generate_video:
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
    track_employees: bool = True, register_new_visitors: bool = True, track_repeat_visitors: bool = True, line_crossing_analysis: bool = True, track_occupancy: bool = True,
    track_objects: bool = True, generate_video: bool = False
):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- TIME-RANGE / SUB-CLIP SLICING ---
    start_time_sec = getattr(session, "start_time_sec", None)
    end_time_sec = getattr(session, "end_time_sec", None)

    start_frame = 0
    if start_time_sec is not None and start_time_sec > 0:
        start_frame = max(0, int(start_time_sec * fps))

    end_frame = total_frames
    if end_time_sec is not None and end_time_sec > 0:
        end_frame = min(total_frames, int(end_time_sec * fps))

    if end_frame <= start_frame:
        end_frame = total_frames
        start_frame = 0

    total_subclip_frames = max(1, end_frame - start_frame)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        logger.info(f"⏩ [Session {session.id}] SUB-CLIP SLICING ACTIVE: Frames {start_frame} -> {end_frame} ({start_frame/fps:.2f}s -> {end_frame/fps:.2f}s, total {total_subclip_frames} frames)")

    video_rotation = get_video_rotation(filepath)
    out_w, out_h = (orig_height, orig_width) if video_rotation in (90, 270) else (orig_width, orig_height)

    # Determine active model and scale label (LARGE / MEDIUM / SMALL / NANO)
    seg_model_filename = os.path.basename(getattr(settings, "YOLO_SEG_MODEL", "yolov8n-seg.pt"))
    model_scale_tag = get_model_scale_label(seg_model_filename)
    logger.info(f"🤖 [Session {session.id}] ACTIVE MODEL: {seg_model_filename} [{model_scale_tag}]")

    # --- CONFIGURABLE ANALYSIS MODE: Full-Frame vs Smart 5 FPS Sampling ---
    env_full_frame = os.getenv("PROCESS_EVERY_FRAME", str(getattr(settings, "PROCESS_EVERY_FRAME", False))).lower() in ("true", "1", "yes")
    if env_full_frame:
        TARGET_FPS = fps
        frame_interval = 1
        processing_fps = fps
        logger.info(f"🚀 [Session {session.id}] [{model_scale_tag}] FULL-FRAME MODE ACTIVE: Analyzing 100% of frames ({fps:.1f} FPS, frame_interval=1)")
    else:
        TARGET_FPS = float(os.getenv("PROCESSING_FPS", os.getenv("ADVANCED_PROCESSING_FPS", getattr(settings, "PROCESSING_FPS", 5.0))))
        frame_interval = max(1, int(round(fps / TARGET_FPS)))
        processing_fps = fps / frame_interval
        logger.info(f"⚡ [Session {session.id}] [{model_scale_tag}] SMART SAMPLING ACTIVE: Target {TARGET_FPS} FPS (processing 1 out of every {frame_interval} frames)")

    # Output video max 720p resolution for ultra-fast software encoding
    output_writer = None
    output_path = None
    scale_factor = 1.0
    vid_w, vid_h = out_w, out_h

    if generate_video:
        MAX_OUT_W, MAX_OUT_H = 1280, 720
        scale_factor = min(1.0, MAX_OUT_W / out_w, MAX_OUT_H / out_h)
        vid_w = int(out_w * scale_factor)
        vid_h = int(out_h * scale_factor)
        vid_w = vid_w if vid_w % 2 == 0 else vid_w - 1
        vid_h = vid_h if vid_h % 2 == 0 else vid_h - 1

        output_filename = f"{uuid.uuid4()}_annotated.mp4"
        user_outputs_dir = os.path.join(ADVANCED_OUTPUTS_DIR, user_id_str) if user_id_str else ADVANCED_OUTPUTS_DIR
        os.makedirs(user_outputs_dir, exist_ok=True)
        output_path = os.path.join(user_outputs_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, processing_fps, (vid_w, vid_h))
    else:
        logger.info(f"⚡ [Session {session.id}] FAST METADATA-ONLY PIPELINE ACTIVE: Skipping video generation & rendering overlays.")

    tracker = sv.ByteTrack(frame_rate=int(processing_fps), lost_track_buffer=int(processing_fps * 8))
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

    video_date = (
        session.recording_started_at.date()
        if session.recording_started_at is not None
        else datetime.now(timezone.utc).date()
    )

    employee_dict = {str(emp.id): emp for emp, emb in (employee_cache or []) if hasattr(emp, "id")}
    active_tracks = {}
    occupancy_history = []
    peak_occupancy_so_far = 0

    # Configure multi-class COCO tracking (Person + 12 Carried/Office Object Classes)
    track_objects_flag = getattr(session, "track_objects", True) if track_objects is None else track_objects
    raw_classes = getattr(session, "classes_to_track", None) or getattr(settings, "APA_TRACKED_CLASSES", DEFAULT_APA_TRACKED_CLASSES)
    target_class_ids = resolve_class_ids(raw_classes if track_objects_flag else ["person"])
    session_detected_objects = defaultdict(int)
    logger.info(f"🎯 [Session {session.id}] TRACKED COCO CLASSES ({len(target_class_ids)}): {[COCO_ID_TO_CLASS.get(cid, str(cid)) for cid in target_class_ids]}")

    # Pre-cache active identities and their ReID embeddings for this video's specific date in memory (0ms lookup)
    db_identities = await repo.get_active_identities_with_embeddings(session.tenant_id, class_id=0, target_date=video_date)
    session_reid_cache = []
    for ident, emb_raw in db_identities:
        if emb_raw and len(emb_raw) > 0:
            arr = np.array(emb_raw, dtype=np.float32)
            n = np.linalg.norm(arr)
            if n > 0:
                session_reid_cache.append((ident, arr / n))

    raw_frame_idx = start_frame
    processed_frame_idx = 0

    # High-contrast Terminal ANSI Color Codes
    CLR_RST = "\033[0m"
    CLR_YG = "\033[1;38;5;154m"      # Vibrant Yellow-Green (Bold)
    CLR_YEL = "\033[1;33m"           # Bold Yellow
    CLR_GRN = "\033[1;32m"           # Bold Emerald Green
    CLR_CYAN = "\033[1;36m"          # Bold Cyan
    CLR_DIM = "\033[38;5;245m"       # Muted / Dim Gray

    while cap.isOpened():
        if raw_frame_idx >= end_frame:
            logger.info(f"⏹️ [Session {session.id}] Reached end of sub-clip range at frame {raw_frame_idx}/{end_frame}. Completing video job.")
            break

        # Clean & fast packet grabbing
        grabbed = cap.grab()
        if not grabbed:
            break
        raw_frame_idx += 1

        # Only decode the actual pixel image for target FPS intervals
        if raw_frame_idx % frame_interval != 0:
            continue

        ret, frame = cap.retrieve()
        if not ret or frame is None:
            break

        processed_frame_idx += 1

        if video_rotation != 0:
            frame = orient_frame(frame, video_rotation)

        timestamp_sec = raw_frame_idx / fps
        real_time_now = to_real_time(timestamp_sec)

        # Check for user cancellation every 10 processed frames
        if processed_frame_idx % 10 == 0:
            try:
                from database.redis import get_redis_client
                r_client = get_redis_client()
                if r_client:
                    is_cancelled = await r_client.get(f"advancedpeopleanalytics:cancelled:{session.id}")
                    if is_cancelled:
                        logger.info(f"🛑 Session {session.id} was deleted by user. Aborting background analysis immediately.")
                        cap.release()
                        output_writer.release()
                        if os.path.exists(output_path):
                            try:
                                os.remove(output_path)
                            except Exception:
                                pass
                        return
            except Exception:
                pass

        # Smooth progress reporting (1% to 99%)
        progress = max(1, min(99, int(((raw_frame_idx - start_frame) / total_subclip_frames) * 100)))
        try:
            from database.redis import get_redis_client
            r_client = get_redis_client()
            if r_client:
                await r_client.setex(f"advancedpeopleanalytics:progress:{session.id}", 3600, str(progress))
        except Exception:
            pass

        # Run accelerated YOLO inference on every sampled frame with target COCO classes (imgsz=640)
        results = model(frame, imgsz=640, conf=float(confidence_threshold), classes=target_class_ids, verbose=False)
        raw_detections = sv.Detections.from_ultralytics(results[0])

        # Separate person detections (class_id == 0) and object detections (class_id != 0)
        if len(raw_detections) > 0:
            person_mask_indices = (raw_detections.class_id == 0)
            person_detections = raw_detections[person_mask_indices]
            object_detections = raw_detections[~person_mask_indices] if track_objects_flag else None
        else:
            person_detections = raw_detections
            object_detections = None

        detections = tracker.update_with_detections(person_detections)

        # Process detected objects and associate with active person tracks
        if object_detections is not None and len(object_detections) > 0:
            for obj_xyxy, obj_cid in zip(object_detections.xyxy, object_detections.class_id):
                obj_name = COCO_ID_TO_CLASS.get(int(obj_cid), f"object_{obj_cid}")
                session_detected_objects[obj_name] += 1
                
                # Spatial association with persons in current frame
                if detections.tracker_id is not None:
                    for p_xyxy, p_tid in zip(detections.xyxy, detections.tracker_id):
                        if p_tid is not None and p_tid in active_tracks:
                            containment = calculate_containment_ratio(obj_xyxy, p_xyxy)
                            iou = calculate_bbox_iou(obj_xyxy, p_xyxy)
                            if containment > 0.35 or iou > 0.15:
                                active_tracks[p_tid]["associated_objects"].add(obj_name)

        current_occupancy = len(detections) if detections.tracker_id is not None else 0
        occupancy_history.append({"time_sec": round(timestamp_sec, 2), "occupancy": current_occupancy})
        if current_occupancy > peak_occupancy_so_far:
            peak_occupancy_so_far = current_occupancy

        # Clear, informative per-frame logging
        if current_occupancy > 0:
            logger.info(
                f"{CLR_YG}🔍 [{model_scale_tag}] [Frame {raw_frame_idx}/{total_frames} | {timestamp_sec:.1f}s | {progress}%] "
                f"PROCESSED -> People: {current_occupancy} | Active Tracks: {len(active_tracks)}{CLR_RST}"
            )
        elif processed_frame_idx % 10 == 0:
            logger.info(
                f"{CLR_DIM}⏳ [{model_scale_tag}] [Frame {raw_frame_idx}/{total_frames} | {timestamp_sec:.1f}s | {progress}%] "
                f"Scanning frame (0 people present){CLR_RST}"
            )

        if detections.tracker_id is not None:
            for xyxy, class_id, tracker_id in zip(detections.xyxy, detections.class_id, detections.tracker_id):
                if tracker_id is None:
                    continue

                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(out_w, x2), min(out_h, y2)
                bbox_height = y2 - y1

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                person_mask = None
                mask_coverage = None
                if detections.mask is not None:
                    det_indices = np.where(detections.tracker_id == tracker_id)[0]
                    if len(det_indices) > 0:
                        raw_mask = detections.mask[det_indices[0]]
                        if raw_mask.shape == (out_h, out_w):
                            person_mask = raw_mask
                        else:
                            person_mask = cv2.resize(
                                raw_mask.astype(np.uint8), (out_w, out_h),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)

                        bbox_mask = person_mask[y1:y2, x1:x2]
                        bbox_area = max(bbox_height * (x2 - x1), 1)
                        mask_coverage = float(np.sum(bbox_mask) / bbox_area)

                if tracker_id not in active_tracks:
                    body_crop = extract_masked_crop(frame, person_mask, x1, y1, x2, y2, background="black")
                    entry_fpath = None
                    if body_crop is not None and body_crop.size > 0:
                        try:
                            os.makedirs(VISITOR_CROPS_DIR, exist_ok=True)
                            entry_fname = f"entry_{session.id}_{tracker_id}.jpg"
                            entry_fpath = os.path.join(VISITOR_CROPS_DIR, entry_fname)
                            if not os.path.exists(entry_fpath):
                                cv2.imwrite(entry_fpath, body_crop)
                            legacy_crop_fname = f"advanced_{session.id}_{tracker_id}.jpg"
                            legacy_crop_fpath = os.path.join(VISITOR_CROPS_DIR, legacy_crop_fname)
                            if not os.path.exists(legacy_crop_fpath):
                                cv2.imwrite(legacy_crop_fpath, body_crop)
                        except Exception:
                            pass

                    active_tracks[tracker_id] = {
                        "first_seen": timestamp_sec,
                        "last_seen": timestamp_sec,
                        "entry_crop_path": entry_fpath,
                        "exit_crop_path": None,
                        "last_crop": body_crop if (body_crop is not None and body_crop.size > 0) else None,
                        "person_type": "visitor",
                        "employee_id": None,
                        "identity_id": None,
                        "identity_source": "tracking",
                        "identity_confidence": 0.5,
                        "reid_embedding": None,
                        "is_segmented": False,
                        "mask_coverage": None,
                        "current_zones": set(),
                        "associated_objects": set(),
                        "face_confirmed": False,
                        "best_face_quality": 0.0,
                        "best_bbox_height": bbox_height,
                        "face_shots_count": 0,
                    }
                track_info = active_tracks[tracker_id]
                track_info["last_seen"] = timestamp_sec
                if body_crop is not None and body_crop.size > 0:
                    track_info["last_crop"] = body_crop

                # CONTINUOUS MULTI-SHOT QUALITY-WEIGHTED FACE REFINEMENT
                # Keeps sampling as the person approaches the camera or when a clearer face frame is available
                is_close_enough_for_face = bbox_height >= 40
                max_face_shots = int(os.getenv("FEATURE_LOCK_SHOTS", getattr(settings, "FEATURE_LOCK_SHOTS", 8)))
                face_shots = track_info.get("face_shots_count", 0)
                is_significantly_closer = bbox_height > (track_info.get("best_bbox_height", 0) * 1.20)
                is_confidence_marginal = track_info.get("identity_confidence", 0.5) < 0.82

                should_check_face = (
                    face_shots < max_face_shots and
                    is_close_enough_for_face and
                    (not track_info["face_confirmed"] or is_significantly_closer or is_confidence_marginal) and
                    (processed_frame_idx % 2 == 0 or face_shots == 0)
                )
                should_extract_reid = (
                    track_info["reid_embedding"] is None and
                    (mask_coverage is None or mask_coverage >= 0.35)
                )

                if should_extract_reid or should_check_face:
                    body_crop = extract_masked_crop(frame, person_mask, x1, y1, x2, y2, background="black")
                    
                    if should_extract_reid and body_crop is not None and body_crop.size > 0:
                        reid_vec = reid_service.extract_embedding(body_crop)
                        if reid_vec is not None:
                            track_info["reid_embedding"] = reid_vec.tolist()
                            track_info["is_segmented"] = person_mask is not None
                            track_info["mask_coverage"] = mask_coverage

                    faces = []
                    face_crop = None
                    if should_check_face:
                        face_crop = extract_head_crop_from_mask(frame, person_mask, x1, y1, x2, y2, head_fraction=0.45)
                        if face_crop is not None and face_crop.size > 0:
                            try:
                                faces = face_rec_service.extract_faces_from_image(face_crop)
                            except Exception:
                                faces = []

                    min_face_size = int(os.getenv("ADVANCED_FACE_MIN_SIZE", getattr(settings, "ADVANCED_FACE_MIN_SIZE", 14)))
                    min_face_det_score = float(os.getenv("ADVANCED_FACE_MIN_DET_SCORE", getattr(settings, "ADVANCED_FACE_MIN_DET_SCORE", 0.35)))

                    valid_faces = []
                    for face in faces:
                        fx1, fy1, fx2, fy2 = map(int, face["bbox"])
                        f_w = fx2 - fx1
                        f_h = fy2 - fy1
                        if f_w >= min_face_size and f_h >= min_face_size and face.get("det_score", 0.0) >= min_face_det_score:
                            face["quality_score"] = float(f_w * f_h * face.get("det_score", 0.5))
                            valid_faces.append(face)

                    matched_face = None
                    face_embedding = None
                    current_face_quality = 0.0

                    if valid_faces:
                        matched_face = max(valid_faces, key=lambda f: f.get("quality_score", 0.0))
                        current_face_quality = matched_face.get("quality_score", 0.0)
                        face_embedding = np.array(matched_face["embedding"], dtype=np.float32)
                        track_info["face_shots_count"] = face_shots + 1
                        if bbox_height > track_info.get("best_bbox_height", 0):
                            track_info["best_bbox_height"] = bbox_height

                        if current_face_quality > track_info.get("best_face_quality", 0.0):
                            track_info["best_face_quality"] = current_face_quality
                            try:
                                if face_crop is not None:
                                    crop_fname = f"advanced_{session.id}_{tracker_id}.jpg"
                                    cv2.imwrite(os.path.join(VISITOR_CROPS_DIR, crop_fname), face_crop)
                            except Exception:
                                pass

                    env_face_thresh = os.getenv("ADVANCED_FACE_SIMILARITY_THRESHOLD")
                    FACE_SIMILARITY_THRESHOLD = float(env_face_thresh) if env_face_thresh is not None else map_similarity_threshold(similarity_threshold)
                    REID_SIMILARITY_THRESHOLD = float(os.getenv("ADVANCED_REID_SIMILARITY_THRESHOLD", getattr(settings, "ADVANCED_REID_SIMILARITY_THRESHOLD", 0.65)))

                    # PHASE 1: Face Recognition against Employee Cache (with quality-based identity refinement)
                    match_emp = None
                    best_cand_emp = None
                    best_cand_sim = -1.0
                    if track_employees and employee_cache and face_embedding is not None:
                        match_emp = find_best_match_in_cache(face_embedding, employee_cache, FACE_SIMILARITY_THRESHOLD)
                        if not match_emp:
                            t_norm = np.linalg.norm(face_embedding)
                            t_normed = face_embedding / t_norm if t_norm > 0 else face_embedding
                            for emp, ref_emb in employee_cache:
                                r_norm = np.linalg.norm(ref_emb)
                                r_normed = ref_emb / r_norm if r_norm > 0 else ref_emb
                                sim_val = float(np.dot(t_normed, r_normed))
                                if sim_val > best_cand_sim:
                                    best_cand_sim = sim_val
                                    best_cand_emp = emp

                    if match_emp:
                        employee, sim = match_emp
                        claimed_by_other = any(
                            t_id != tracker_id and t.get("employee_id") == employee.id and t.get("face_confirmed")
                            for t_id, t in active_tracks.items()
                        )
                        
                        # Check if this match is stronger or upgrades previous classification
                        prev_sim = track_info.get("identity_confidence", 0.0)
                        should_upgrade_match = (
                            not claimed_by_other and
                            (not track_info["face_confirmed"] or sim > (prev_sim + 0.03) or track_info.get("employee_id") != employee.id)
                        )

                        if should_upgrade_match:
                            track_info["person_type"] = "employee"
                            track_info["employee_id"] = employee.id
                            track_info["employee_name"] = f"{employee.first_name} {employee.last_name}"
                            track_info["employee_code"] = employee.employee_code
                            track_info["identity_source"] = "face"
                            track_info["identity_confidence"] = float(sim)
                            track_info["face_confirmed"] = True

                            logger.info(
                                f"{CLR_GRN}  👤 [Track #{tracker_id}] EMPLOYEE IDENTIFIED / REFINED: {track_info['employee_name']} "
                                f"(Code: {track_info['employee_code']}) | Face Confidence: {sim*100:.1f}% (Quality: {current_face_quality:.1f}){CLR_RST}"
                            )

                            # Register/retrieve employee identity in Advanced Analytics
                            emp_identity = await repo.get_or_create_employee_identity(
                                tenant_id=session.tenant_id,
                                employee_id=employee.id,
                                employee_name=track_info["employee_name"]
                            )
                            track_info["identity_id"] = emp_identity.id

                            # Deactivate older appearance embeddings for this employee (older than video_date)
                            await repo.deactivate_old_appearance_embeddings(emp_identity.id, video_date)

                            # Save Video Date's Body/Clothes Appearance Embedding ONLY on genuine face confirmation
                            if track_info["reid_embedding"]:
                                await repo.create_person_embedding(
                                    identity_id=emp_identity.id,
                                    embedding=track_info["reid_embedding"],
                                    bbox=[x1, y1, x2, y2],
                                    timestamp=timestamp_sec,
                                    embedding_type="appearance",
                                    is_segmented=track_info.get("is_segmented", person_mask is not None),
                                    mask_coverage=track_info.get("mask_coverage"),
                                    recorded_date=video_date,
                                    is_active=True,
                                    face_anchored=True
                                )

                            # Save PERMANENT Face Embedding (Never deactivated across dates)
                            if face_embedding is not None:
                                await repo.create_person_embedding(
                                    identity_id=emp_identity.id,
                                    embedding=face_embedding.tolist() if hasattr(face_embedding, "tolist") else list(face_embedding),
                                    bbox=[x1, y1, x2, y2],
                                    timestamp=timestamp_sec,
                                    embedding_type="face",
                                    is_segmented=False,
                                    mask_coverage=None,
                                    recorded_date=video_date,
                                    is_active=True,
                                    face_anchored=True
                                )
                    else:
                        if not track_info.get("face_confirmed") and best_cand_emp is not None and best_cand_sim > 0.15:
                            logger.info(
                                f"  🔍 [Track #{tracker_id}] Face scanned -> Closest Match: {best_cand_emp.first_name} {best_cand_emp.last_name} "
                                f"({best_cand_sim*100:.1f}%) | Required Threshold: {FACE_SIMILARITY_THRESHOLD*100:.1f}%"
                            )

                    # PHASE 2: If no face matched or face is not visible, check Full-Body ReID Appearance in memory (0ms)
                    if track_info["reid_embedding"] and not track_info["face_confirmed"]:
                        match_vis = None
                        if track_repeat_visitors and session_reid_cache:
                            t_vec = np.array(track_info["reid_embedding"], dtype=np.float32)
                            t_norm = np.linalg.norm(t_vec)
                            if t_norm > 0:
                                t_unit = t_vec / t_norm
                                best_vis, best_sim = None, 0.0
                                for v_ident, v_emb in session_reid_cache:
                                    sim = float(np.dot(t_unit, v_emb))
                                    if sim >= REID_SIMILARITY_THRESHOLD and sim > best_sim:
                                        best_sim = sim
                                        best_vis = v_ident
                                if best_vis is not None:
                                    match_vis = (best_vis, best_sim)

                        if match_vis:
                            visitor, sim = match_vis
                            claimed_by_other = any(
                                t_id != tracker_id and t.get("identity_id") == visitor.id
                                for t_id, t in active_tracks.items()
                            )
                            if not claimed_by_other:
                                track_info["identity_id"] = visitor.id
                                track_info["identity_source"] = "reid"
                                track_info["identity_confidence"] = float(sim)
                                if visitor.is_employee and visitor.employee_id:
                                    emp_obj = employee_dict.get(str(visitor.employee_id))
                                    if emp_obj:
                                        track_info["person_type"] = "employee"
                                        track_info["employee_id"] = visitor.employee_id
                                        track_info["employee_name"] = f"{emp_obj.first_name} {emp_obj.last_name}"
                                        track_info["employee_code"] = emp_obj.employee_code
                                        logger.info(
                                            f"{CLR_YG}  👤 [Track #{tracker_id}] EMPLOYEE RE-MATCHED VIA REID: {track_info['employee_name']} "
                                            f"(Code: {track_info['employee_code']}) | ReID Similarity: {sim*100:.1f}%{CLR_RST}"
                                        )
                                    else:
                                        # Deleted / unlisted employee: downgrade to visitor, do NOT use old employee name
                                        track_info["person_type"] = "visitor"
                                        track_info["employee_id"] = None
                                        track_info["employee_name"] = None
                                        logger.info(
                                            f"{CLR_CYAN}  🏷️ [Track #{tracker_id}] VISITOR MATCHED (Inactive employee record): Identity {str(visitor.id)[:8]} "
                                            f"| ReID Similarity: {sim*100:.1f}%{CLR_RST}"
                                        )
                                elif visitor.visitor_name:
                                    track_info["visitor_name"] = visitor.visitor_name
                                    logger.info(
                                        f"{CLR_CYAN}  🏷️ [Track #{tracker_id}] RETURNING VISITOR MATCHED: {visitor.visitor_name} "
                                        f"| ReID Similarity: {sim*100:.1f}%{CLR_RST}"
                                    )
                                else:
                                    logger.info(
                                        f"{CLR_CYAN}  🏷️ [Track #{tracker_id}] VISITOR MATCHED: Identity {str(visitor.id)[:8]} "
                                        f"| ReID Similarity: {sim*100:.1f}%{CLR_RST}"
                                    )

                        elif register_new_visitors and track_info["identity_id"] is None:
                            visitor = await repo.create_person_identity(session.tenant_id, class_id=0)
                            track_info["identity_id"] = visitor.id
                            track_info["identity_source"] = "reid"
                            track_info["identity_confidence"] = 0.40

                            # Add to in-memory cache for instant matching on future frames
                            t_v = np.array(track_info["reid_embedding"], dtype=np.float32)
                            t_n = np.linalg.norm(t_v)
                            if t_n > 0:
                                session_reid_cache.append((visitor, t_v / t_n))

                            logger.info(f"{CLR_YEL}  ✨ [Track #{tracker_id}] NEW VISITOR REGISTERED: Identity {str(visitor.id)[:8]}{CLR_RST}")

                            await repo.create_person_embedding(
                                identity_id=visitor.id,
                                embedding=track_info["reid_embedding"],
                                bbox=[x1, y1, x2, y2],
                                timestamp=timestamp_sec,
                                embedding_type="appearance",
                                is_segmented=track_info.get("is_segmented", person_mask is not None),
                                mask_coverage=track_info.get("mask_coverage"),
                                recorded_date=video_date,
                                is_active=True,
                                face_anchored=False
                            )

                            if face_embedding is not None:
                                await repo.create_person_embedding(
                                    identity_id=visitor.id,
                                    embedding=face_embedding.tolist() if hasattr(face_embedding, "tolist") else list(face_embedding),
                                    bbox=[x1, y1, x2, y2],
                                    timestamp=timestamp_sec,
                                    embedding_type="face",
                                    is_segmented=False,
                                    mask_coverage=None,
                                    recorded_date=video_date,
                                    is_active=True,
                                    face_anchored=False
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
                            confidence=track_info["identity_confidence"],
                            mask_coverage=track_info.get("mask_coverage")
                        )
                    elif not is_inside and zone.id in track_info["current_zones"]:
                        track_info["current_zones"].remove(zone.id)

                # --- DRAW OVERLAYS FOR THIS TRACKED PERSON (ONLY IF VIDEO GENERATION REQUESTED) ---
                if generate_video:
                    if track_info["person_type"] == "employee":
                        color = (0, 255, 0) # Green for Employee
                        emp_name = track_info.get("employee_name", "Employee")
                        emp_code = track_info.get("employee_code", "")
                        sim_pct = int(track_info["identity_confidence"] * 100)
                        code_str = f" [{emp_code}]" if emp_code else ""
                        source_str = " (ReID)" if track_info.get("identity_source") == "reid" else ""
                        label = f"#{tracker_id} | {emp_name}{code_str}{source_str} ({sim_pct}%)"
                    elif track_info.get("visitor_name"):
                        color = (255, 215, 0) # Cyan for Named Visitor
                        v_name = track_info["visitor_name"]
                        sim_pct = int(track_info["identity_confidence"] * 100)
                        label = f"#{tracker_id} | {v_name} ({sim_pct}%)"
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

        if generate_video:
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

            if output_writer is not None:
                out_frame = cv2.resize(frame, (vid_w, vid_h)) if scale_factor < 1.0 else frame
                output_writer.write(out_frame)

    cap.release()
    if output_writer is not None:
        output_writer.release()

    # Create PersonTimelineEvent records and Employee Attendance logs at track completion
    from modules.employees.repository import EmployeeRepository
    emp_repo = EmployeeRepository(db)

    camera_name = session.video_name
    zone_name = None

    if not session.camera_node_id:
        clean_vname = os.path.splitext(session.video_name)[0].lower().replace("-t", "").replace("_", " ").strip()
        try:
            all_cams = await repo.get_camera_nodes(session.tenant_id)
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
                session.camera_node_id = best_cam.id
                await db.commit()
        except Exception:
            pass

    if session.camera_node_id:
        c_node = await repo.get_camera_node_by_id(session.camera_node_id, session.tenant_id)
        if c_node:
            camera_name = c_node.name
            zone_name = c_node.location_label or c_node.name
    else:
        base_name = os.path.splitext(session.video_name)[0]
        camera_name = base_name.replace("-t", "").replace("_", " ").title()
        zone_name = f"{camera_name} Area"

    for tracker_id, info in active_tracks.items():
        if info.get("last_crop") is not None and info["last_crop"].size > 0:
            try:
                exit_fname = f"exit_{session.id}_{tracker_id}.jpg"
                exit_fpath = os.path.join(VISITOR_CROPS_DIR, exit_fname)
                cv2.imwrite(exit_fpath, info["last_crop"])
                info["exit_crop_path"] = exit_fpath
            except Exception:
                pass
            info["last_crop"] = None

        started_at = to_real_time(info["first_seen"])
        ended_at = to_real_time(info["last_seen"])
        
        track_zone = ", ".join(info.get("current_zones", [])) if info.get("current_zones") else zone_name

        await repo.create_timeline_event(
            tenant_id=session.tenant_id,
            person_type=info["person_type"],
            session_id=session.id,
            camera_name=camera_name,
            zone_name=track_zone,
            event_type="presence",
            started_at=started_at,
            ended_at=ended_at,
            identity_source=info["identity_source"],
            identity_confidence=info["identity_confidence"],
            tracker_id=tracker_id,
            employee_id=info["employee_id"],
            identity_id=info["identity_id"],
            entry_crop_path=info.get("entry_crop_path"),
            exit_crop_path=info.get("exit_crop_path"),
            associated_objects=list(info["associated_objects"]) if info.get("associated_objects") else None
        )

        # Log employee attendance when matched in video analysis
        if info["person_type"] == "employee" and info["employee_id"]:
            try:
                await repo.create_employee_attendance(
                    session_id=session.id,
                    employee_id=info["employee_id"],
                    first_seen=info["first_seen"],
                    last_seen=info["last_seen"],
                    occurrence_count=1
                )
            except Exception as e:
                logger.warning(f"Failed to create advanced employee attendance record: {e}")

            try:
                await emp_repo.log_employee_attendance(
                    tenant_id=session.tenant_id,
                    employee_id=info["employee_id"],
                    session_id=None,
                    first_seen_sec=info["first_seen"],
                    last_seen_sec=info["last_seen"],
                    occurrence_increment=1,
                    entry_time=started_at,
                    exit_time=ended_at
                )
            except Exception as e:
                logger.warning(f"Failed to log employee attendance: {e}")

    # Transcode and upload output video if video generation was requested
    if generate_video and output_path and os.path.exists(output_path):
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
        detected_objects_summary=dict(session_detected_objects) if track_objects_flag else None,
        output_video_path=output_path if generate_video else None
    )
    await db.commit()


# ==========================================
# CROSS-CAMERA IDENTITY ASSOCIATION TASK
# ==========================================

@celery_app.task(name="application.advancedpeopleanalytics.tasks.run_cross_camera_association_task")
def run_cross_camera_association_task(session_id_str: str):
    """
    Celery task that matches person tracks from this session across linked destination camera nodes
    (Camera-to-Camera Topology) within spatio-temporal transit windows.
    Also supports employee identity transfer across camera feeds.
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

            if not session.camera_node_id:
                clean_vname = os.path.splitext(session.video_name)[0].lower().replace("-t", "").replace("_", " ").strip()
                try:
                    all_cams = await repo.get_camera_nodes(session.tenant_id)
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
                        session.camera_node_id = best_cam.id
                        await db.commit()
                except Exception:
                    pass

            if not session or not session.camera_node_id:
                return

            tenant_id = session.tenant_id

            # ----------------------------------------------------
            # 1. CAMERA-TO-CAMERA DIRECTED TOPOLOGY LINKS
            # ----------------------------------------------------
            camera_links = await repo.get_camera_node_links_from(session.camera_node_id)

            # Query all timeline events from this session
            from_events_stmt = select(PersonTimelineEvent).where(
                PersonTimelineEvent.session_id == session.id,
                PersonTimelineEvent.is_delete == False
            )
            from_res = await db.execute(from_events_stmt)
            from_events = list(from_res.scalars().all())

            # For each outgoing link to target camera
            for cam_link in camera_links:
                target_cam_id = cam_link.to_camera_id
                
                # Find sessions on target camera
                target_sess_stmt = select(AdvancedPeopleAnalyticsSession).where(
                    AdvancedPeopleAnalyticsSession.tenant_id == tenant_id,
                    AdvancedPeopleAnalyticsSession.camera_node_id == target_cam_id,
                    AdvancedPeopleAnalyticsSession.is_delete == False
                )
                target_sess_res = await db.execute(target_sess_stmt)
                target_sessions = list(target_sess_res.scalars().all())

                for target_sess in target_sessions:
                    # Query timeline events from target session
                    to_events_stmt = select(PersonTimelineEvent).where(
                        PersonTimelineEvent.session_id == target_sess.id,
                        PersonTimelineEvent.is_delete == False
                    )
                    to_res = await db.execute(to_events_stmt)
                    to_events = list(to_res.scalars().all())

                    for from_evt in from_events:
                        exit_time = from_evt.ended_at
                        window_start = exit_time + timedelta(seconds=cam_link.min_transit_seconds)
                        window_end = exit_time + timedelta(seconds=cam_link.max_transit_seconds)

                        # Check candidate events in target camera within transit window
                        for to_evt in to_events:
                            if not (window_start <= to_evt.started_at <= window_end):
                                continue

                            # Fetch embeddings for from_evt and to_evt
                            from_emb_stmt = select(AdvancedPersonEmbedding).where(
                                AdvancedPersonEmbedding.identity_id == from_evt.identity_id,
                                AdvancedPersonEmbedding.is_delete == False
                            ).limit(1) if from_evt.identity_id else None
                            to_emb_stmt = select(AdvancedPersonEmbedding).where(
                                AdvancedPersonEmbedding.identity_id == to_evt.identity_id,
                                AdvancedPersonEmbedding.is_delete == False
                            ).limit(1) if to_evt.identity_id else None

                            from_emb = (await db.execute(from_emb_stmt)).scalars().first() if from_emb_stmt is not None else None
                            to_emb = (await db.execute(to_emb_stmt)).scalars().first() if to_emb_stmt is not None else None

                            reid_score = 0.5
                            if from_emb and to_emb and from_emb.embedding and to_emb.embedding:
                                vec1 = np.array(from_emb.embedding, dtype=np.float32)
                                vec2 = np.array(to_emb.embedding, dtype=np.float32)
                                n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                                if n1 > 0 and n2 > 0:
                                    reid_score = float(np.dot(vec1, vec2) / (n1 * n2))

                            from_quality = from_emb.mask_coverage if from_emb and from_emb.mask_coverage else 0.5
                            to_quality = to_emb.mask_coverage if to_emb and to_emb.mask_coverage else 0.5
                            quality_weight = (from_quality + to_quality) / 2.0  # 0.0–1.0

                            # Apply quality weight to reid_score — poor crops get penalised
                            weighted_reid_score = reid_score * (0.5 + 0.5 * quality_weight)

                            actual_transit = (to_evt.started_at - exit_time).total_seconds()
                            time_score = max(0.0, 1.0 - (abs(actual_transit - cam_link.avg_transit_seconds) / cam_link.max_transit_seconds))
                            face_bonus = 0.15 if (from_evt.employee_id and to_evt.employee_id and from_evt.employee_id == to_evt.employee_id) else 0.0

                            final_score = (0.60 * weighted_reid_score) + (0.25 * time_score) + (0.15 * face_bonus)
                            cross_camera_min_score = float(os.getenv("ADVANCED_CROSS_CAMERA_MIN_SCORE", getattr(settings, "ADVANCED_CROSS_CAMERA_MIN_SCORE", 0.65)))

                            if final_score >= cross_camera_min_score and from_evt.identity_id and to_evt.identity_id:
                                await repo.create_cross_camera_link(
                                    tenant_id=tenant_id,
                                    from_session_id=from_evt.session_id,
                                    to_session_id=to_evt.session_id,
                                    from_identity_id=from_evt.identity_id,
                                    to_identity_id=to_evt.identity_id,
                                    reid_score=reid_score,
                                    time_score=time_score,
                                    final_score=final_score,
                                    method="camera_topology_reid_fusion",
                                    is_confirmed=(final_score >= 0.70)
                                )

                                # Propagate employee or visitor identity across cameras
                                if from_evt.employee_id and not to_evt.employee_id:
                                    to_evt.employee_id = from_evt.employee_id
                                    to_evt.person_type = "employee"
                                    to_evt.identity_source = "face+reid"
                                    to_evt.identity_confidence = final_score
                                    to_evt.identity_id = from_evt.identity_id
                                elif not to_evt.employee_id and not from_evt.employee_id and from_evt.identity_id:
                                    to_evt.identity_id = from_evt.identity_id
                                    to_evt.person_type = "visitor"
                                    to_evt.identity_source = "cross_camera_reid"
                                    to_evt.identity_confidence = final_score
                                await db.commit()

            # ----------------------------------------------------
            # 2. LEGACY CAMERA ZONE LINKS (FALLBACK)
            # ----------------------------------------------------
            zones = await repo.get_camera_zones(session.camera_node_id)
            exit_zones = [z for z in zones if z.zone_type in {"exit", "crossing"}]

            for exit_z in exit_zones:
                links = await repo.get_zone_links_from(exit_z.id)
                for link in links:
                    start_t = session.recording_started_at or (datetime.now(timezone.utc) - timedelta(hours=24))
                    end_t = session.completed_at or datetime.now(timezone.utc)
                    exit_events = await repo.get_zone_crossing_events(exit_z.id, start_t, end_t)

                    for exit_evt in exit_events:
                        if not exit_evt.reid_embedding:
                            continue

                        window_start = exit_evt.real_world_time + timedelta(seconds=5)
                        window_end = exit_evt.real_world_time + timedelta(seconds=link.max_transit_seconds)
                        entry_events = await repo.get_zone_crossing_events(link.to_zone_id, window_start, window_end)

                        for entry_evt in entry_events:
                            if not entry_evt.reid_embedding or not exit_evt.identity_id or not entry_evt.identity_id:
                                continue

                            vec1 = np.array(exit_evt.reid_embedding, dtype=np.float32)
                            vec2 = np.array(entry_evt.reid_embedding, dtype=np.float32)
                            n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                            reid_score = float(np.dot(vec1, vec2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0

                            from_quality = exit_evt.mask_coverage if exit_evt and exit_evt.mask_coverage else 0.5
                            to_quality = entry_evt.mask_coverage if entry_evt and entry_evt.mask_coverage else 0.5
                            quality_weight = (from_quality + to_quality) / 2.0

                            weighted_reid_score = reid_score * (0.5 + 0.5 * quality_weight)

                            actual_transit = (entry_evt.real_world_time - exit_evt.real_world_time).total_seconds()
                            time_score = max(0.0, 1.0 - (abs(actual_transit - link.avg_transit_seconds) / link.max_transit_seconds))
                            face_bonus = 0.15 if (exit_evt.face_confirmed and entry_evt.face_confirmed and exit_evt.identity_id == entry_evt.identity_id) else 0.0

                            final_score = (0.60 * weighted_reid_score) + (0.25 * time_score) + (0.15 * face_bonus)

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
