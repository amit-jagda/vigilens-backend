import os
import uuid
import cv2
import numpy as np
from sqlalchemy import select

from workers.celery import celery_app
from database.session import SessionLocal
from modules.peoplefind.model import FaceSearchSession, MediaSource, FaceSearchResult
from modules.peoplefind.repository import PeopleFindRepository
from services.ai.face_recognition import face_rec_service
from services.ai.math_utils import group_timestamps
from services.ai.visualization import draw_bbox_on_image, format_time
from workers.utils import run_async

VIDEO_MATCHES_DIR = os.path.join("storage", "video_matches")
os.makedirs(VIDEO_MATCHES_DIR, exist_ok=True)


@celery_app.task(name="modules.peoplefind.tasks.index_video_task")
def index_video_task(media_source_id_str: str, filepath: str, interval: float = 1.0):
    """
    Celery task to index a video's faces asynchronously.
    """
    media_id = uuid.UUID(media_source_id_str)
    
    async def run():
        async with SessionLocal() as db:
            from modules.peoplefind.service import PeopleFindService
            service = PeopleFindService(db)
            await service._index_video_sync(media_id, filepath, interval)
            
    run_async(run())

@celery_app.task(name="modules.peoplefind.tasks.index_photo_task")
def index_photo_task(media_source_id_str: str, filepath: str):
    """
    Celery task to index a photo's faces asynchronously.
    """
    media_id = uuid.UUID(media_source_id_str)
    
    async def run():
        async with SessionLocal() as db:
            from modules.peoplefind.service import PeopleFindService
            service = PeopleFindService(db)
            await service._index_photo_sync(media_id, filepath)
            
    run_async(run())


@celery_app.task(name="modules.peoplefind.tasks.process_video_search_task")
def process_video_search_task(
    video_id_str: str, 
    session_id_str: str, 
    threshold: float = 0.45, 
    interval: float = 1.0, 
    model_name: str = "buffalo_s"
):
    """
    Celery task to search a target video file on-demand for a target face, 
    annotating and saving keyframes and generating a summary presence report.
    """
    video_id = uuid.UUID(video_id_str)
    session_id = uuid.UUID(session_id_str)

    async def run():
        try:
            async with SessionLocal() as db:
                print(f"[Vigilens Task] Starting video search task. Session ID: {session_id}, Video ID: {video_id}")
                # 1. Load entities from DB without tenant checks for celery background runner
                stmt_session = select(FaceSearchSession).where(
                    FaceSearchSession.id == session_id,
                    FaceSearchSession.is_delete == False
                )
                res_session = await db.execute(stmt_session)
                session = res_session.scalars().first()
                if not session:
                    print(f"[Vigilens Task ERROR] Search Session {session_id} not found in DB.")
                    return

                stmt_media = select(MediaSource).where(
                    MediaSource.id == video_id,
                    MediaSource.is_delete == False
                )
                res_media = await db.execute(stmt_media)
                media = res_media.scalars().first()
                if not media:
                    print(f"[Vigilens Task ERROR] Media Source {video_id} not found in DB.")
                    session.status = "failed"
                    await db.commit()
                    return

                # Setup matching directories
                session_out_dir = os.path.join(VIDEO_MATCHES_DIR, str(session.id))
                os.makedirs(session_out_dir, exist_ok=True)

                from services.storage import storage_client
                
                # 2. Download reference selfie and load embeddings
                print(f"[Vigilens Task] Downloading reference selfie: {session.selfie_path}")
                local_selfie_path = storage_client.get_file_path(session.selfie_path) if session.selfie_path else None
                
                print(f"[Vigilens Task] Loading search embeddings using model {model_name}...")
                group_embeddings = face_rec_service.load_search_embeddings(
                    local_selfie_path, session.selfie_embedding, model_name=model_name
                )
                print(f"[Vigilens Task] Loaded {len(group_embeddings)} reference embedding(s).")

                # 3. Download target video from storage provider
                print(f"[Vigilens Task] Downloading target video: {media.filepath}")
                local_media_path = storage_client.get_file_path(media.filepath)
                print(f"[Vigilens Task] Target video downloaded/accessible locally at: {local_media_path}")

                matched_seconds = []
                repo = PeopleFindRepository(db)

                print(f"[Vigilens Task] Starting face matching loop with threshold {threshold}...")
                
                try:
                    for match in face_rec_service.search_face_in_video(
                        local_media_path, group_embeddings, threshold, interval, model_name
                    ):
                        sec = match["timestamp"]
                        best_sim = match["similarity"]
                        orig_bbox = match["bbox"]
                        frame = match["frame"]

                        time_str = format_time(sec)
                        print(f"[Vigilens Task] Match found at {time_str} (similarity: {best_sim:.3f})")
                        matched_seconds.append(sec)

                        # Save FaceSearchResult in database
                        await repo.create_search_result(
                            session_id=session.id,
                            media_source_id=media.id,
                            similarity=best_sim,
                            bbox=orig_bbox,
                            timestamp=sec
                        )

                        # Annotate frame and save keyframe image
                        label = f"Sim: {best_sim:.3f} | {time_str}"
                        vis_frame = draw_bbox_on_image(frame, orig_bbox, label)
                        time_filename = time_str.replace(":", "_")
                        out_img_path = os.path.join(session_out_dir, f"frame_{time_filename}.jpg")
                        cv2.imwrite(out_img_path, vis_frame)
                        
                        # Upload keyframe to storage provider (S3/Local)
                        storage_client.upload_file(out_img_path, out_img_path)
                except Exception as loop_err:
                    print(f"[Vigilens Task ERROR] Exception in face search loop: {loop_err}")
                    raise loop_err

                print(f"[Vigilens Task] Finished analysis. Found {len(matched_seconds)} matches total. Grouping matches into intervals...")
                # Group matches into intervals
                max_gap = interval * 2.0
                intervals = group_timestamps(matched_seconds, max_gap)

                # Generate and write txt report
                report_path = os.path.join(session_out_dir, "report.txt")
                print(f"[Vigilens Task] Generating presence report at {report_path}...")
                with open(report_path, "w") as rf:
                    rf.write("=====================================================\n")
                    rf.write("  ON-DEMAND VIDEO SEARCH REPORT\n")
                    rf.write("=====================================================\n")
                    rf.write(f"Video File ID    : {media.id}\n")
                    rf.write(f"Original Name    : {media.filename}\n")
                    rf.write(f"Search Session   : {session.id}\n")
                    rf.write(f"Threshold        : {threshold}\n")
                    rf.write(f"Sample Interval  : {interval}s\n")
                    rf.write(f"Total Matches    : {len(matched_seconds)} frame(s)\n")
                    rf.write("-----------------------------------------------------\n\n")

                    if intervals:
                        rf.write("Detected Timestamps (Intervals):\n")
                        for start, end in intervals:
                            if start == end:
                                rf.write(f"  • {format_time(start)}\n")
                            else:
                                rf.write(f"  • {format_time(start)} to {format_time(end)}\n")
                    else:
                        rf.write("No matching face detected in the video.\n")

                # Upload report file to storage provider
                storage_client.upload_file(report_path, report_path)

                # Update session status
                session.status = "completed"
                await db.commit()
                print("[Vigilens Task] Task completed successfully!")
        except Exception as e:
            import traceback
            print(f"[Vigilens Task FATAL ERROR] Task failed with exception: {e}")
            traceback.print_exc()
            try:
                async with SessionLocal() as db_err:
                    stmt_session_err = select(FaceSearchSession).where(
                        FaceSearchSession.id == session_id
                    )
                    res_session_err = await db_err.execute(stmt_session_err)
                    session_err = res_session_err.scalars().first()
                    if session_err:
                        session_err.status = "failed"
                        await db_err.commit()
                        print("[Vigilens Task] Database status successfully updated to 'failed'.")
            except Exception as db_ex:
                print(f"[Vigilens Task FATAL ERROR] Failed to set status to 'failed': {db_ex}")

    run_async(run())
