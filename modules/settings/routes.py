import os
import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, status
from pydantic import BaseModel

from configs.base import settings
from shared.schemas.response import StandardResponse

router = APIRouter(prefix="/settings", tags=["System & Environment Settings"])


class SettingItem(BaseModel):
    key: str
    label: str
    value: Any
    type: str  # "string" | "number" | "boolean"
    category: str
    unit: Optional[str] = None
    recommended_range: Optional[str] = None
    description: str
    env_source: str = ".env / container environment"


class SettingsCategory(BaseModel):
    id: str
    name: str
    description: str
    parameters: List[SettingItem]


class SystemSettingsData(BaseModel):
    raw_parameters: Dict[str, Any]
    categories: List[SettingsCategory]
    environment: str
    server_time: str
    instructions: str


def _get_live_env_val(key: str, default_val: Any, val_type: type):
    """Retrieve the active live runtime value from os.environ or configs.base.settings."""
    raw = os.getenv(key)
    if raw is not None:
        if val_type == bool:
            return raw.strip().lower() in ("true", "1", "yes", "on", "t")
        elif val_type == int:
            try:
                return int(raw.strip())
            except ValueError:
                return default_val
        elif val_type == float:
            try:
                return float(raw.strip())
            except ValueError:
                return default_val
        return raw.strip()
    
    return getattr(settings, key, default_val)


@router.get("", response_model=StandardResponse[SystemSettingsData])
@router.get("/system", response_model=StandardResponse[SystemSettingsData])
async def get_system_settings():
    """
    Returns all backend and Computer Vision runtime parameters loaded from the .env configuration.
    Values are dynamic and update immediately when the container/backend restarts with new environment variables.
    """
    # 1. Database & Storage
    postgres_db = _get_live_env_val("POSTGRES_DB", settings.POSTGRES_DB, str)
    storage_provider = _get_live_env_val("STORAGE_PROVIDER", settings.STORAGE_PROVIDER, str)
    local_storage_path = _get_live_env_val("LOCAL_STORAGE_PATH", settings.LOCAL_STORAGE_PATH, str)

    # 2. Computer Vision & Advanced People Analytics Thresholds
    face_similarity_threshold = _get_live_env_val(
        "ADVANCED_FACE_SIMILARITY_THRESHOLD", settings.ADVANCED_FACE_SIMILARITY_THRESHOLD, float
    )
    reid_similarity_threshold = _get_live_env_val(
        "ADVANCED_REID_SIMILARITY_THRESHOLD", settings.ADVANCED_REID_SIMILARITY_THRESHOLD, float
    )
    face_min_size = _get_live_env_val("ADVANCED_FACE_MIN_SIZE", settings.ADVANCED_FACE_MIN_SIZE, int)
    face_min_det_score = _get_live_env_val(
        "ADVANCED_FACE_MIN_DET_SCORE", settings.ADVANCED_FACE_MIN_DET_SCORE, float
    )
    cross_camera_min_score = _get_live_env_val(
        "ADVANCED_CROSS_CAMERA_MIN_SCORE", settings.ADVANCED_CROSS_CAMERA_MIN_SCORE, float
    )
    feature_lock_shots = _get_live_env_val("FEATURE_LOCK_SHOTS", settings.FEATURE_LOCK_SHOTS, int)
    reid_memory_bank_ttl = _get_live_env_val(
        "REID_MEMORY_BANK_TTL_SEC", settings.REID_MEMORY_BANK_TTL_SEC, float
    )
    reid_memory_bank_thresh = _get_live_env_val(
        "REID_MEMORY_BANK_SIMILARITY_THRESHOLD", settings.REID_MEMORY_BANK_SIMILARITY_THRESHOLD, float
    )

    # 3. Video Pipeline & Sampling Optimization
    process_every_frame = _get_live_env_val("PROCESS_EVERY_FRAME", settings.PROCESS_EVERY_FRAME, bool)
    processing_fps = _get_live_env_val("PROCESSING_FPS", settings.PROCESSING_FPS, float)
    rtsp_retry_interval = _get_live_env_val("RTSP_RETRY_INTERVAL", settings.RTSP_RETRY_INTERVAL, int)

    # 4. SQL Agent & AI Configuration
    sql_agent_timeout = _get_live_env_val(
        "SQL_AGENT_TIMEOUT_SECONDS", settings.SQL_AGENT_TIMEOUT_SECONDS, float
    )
    sql_agent_max_rows = _get_live_env_val("SQL_AGENT_MAX_ROWS", settings.SQL_AGENT_MAX_ROWS, int)
    sql_agent_max_retries = _get_live_env_val("SQL_AGENT_MAX_RETRIES", settings.SQL_AGENT_MAX_RETRIES, int)
    llm_provider = _get_live_env_val("LLM_PROVIDER", settings.LLM_PROVIDER, str)
    llm_model = _get_live_env_val("LLM_MODEL", settings.LLM_MODEL, str)
    ollama_base_url = _get_live_env_val("OLLAMA_BASE_URL", settings.OLLAMA_BASE_URL, str)
    ollama_model = _get_live_env_val("OLLAMA_MODEL", settings.OLLAMA_MODEL, str)

    raw_parameters = {
        "POSTGRES_DB": postgres_db,
        "ADVANCED_FACE_SIMILARITY_THRESHOLD": face_similarity_threshold,
        "ADVANCED_REID_SIMILARITY_THRESHOLD": reid_similarity_threshold,
        "ADVANCED_FACE_MIN_SIZE": face_min_size,
        "ADVANCED_FACE_MIN_DET_SCORE": face_min_det_score,
        "ADVANCED_CROSS_CAMERA_MIN_SCORE": cross_camera_min_score,
        "PROCESS_EVERY_FRAME": process_every_frame,
        "PROCESSING_FPS": processing_fps,
        "FEATURE_LOCK_SHOTS": feature_lock_shots,
        "REID_MEMORY_BANK_TTL_SEC": reid_memory_bank_ttl,
        "REID_MEMORY_BANK_SIMILARITY_THRESHOLD": reid_memory_bank_thresh,
        "SQL_AGENT_TIMEOUT_SECONDS": sql_agent_timeout,
        "SQL_AGENT_MAX_ROWS": sql_agent_max_rows,
        "SQL_AGENT_MAX_RETRIES": sql_agent_max_retries,
        "STORAGE_PROVIDER": storage_provider,
        "LOCAL_STORAGE_PATH": local_storage_path,
        "RTSP_RETRY_INTERVAL": rtsp_retry_interval,
        "LLM_PROVIDER": llm_provider,
        "LLM_MODEL": llm_model,
        "OLLAMA_BASE_URL": ollama_base_url,
        "OLLAMA_MODEL": ollama_model,
        "PROJECT_NAME": settings.PROJECT_NAME,
    }

    categories = [
        SettingsCategory(
            id="cv_reid",
            name="Computer Vision & ReID Thresholds",
            description="Core facial matching, full-body OSNet ReID similarity, and spatio-temporal tracking thresholds.",
            parameters=[
                SettingItem(
                    key="ADVANCED_FACE_SIMILARITY_THRESHOLD",
                    label="Face Recognition Similarity Threshold",
                    value=face_similarity_threshold,
                    type="number",
                    category="cv_reid",
                    unit="cosine",
                    recommended_range="0.25 - 0.45",
                    description="ArcFace cosine similarity cutoff. Higher values reduce false positives; lower values improve distant/angled recognition.",
                ),
                SettingItem(
                    key="ADVANCED_REID_SIMILARITY_THRESHOLD",
                    label="ReID Appearance Similarity Threshold",
                    value=reid_similarity_threshold,
                    type="number",
                    category="cv_reid",
                    unit="cosine",
                    recommended_range="0.65 - 0.88",
                    description="OSNet appearance embedding similarity cutoff for associating person tracks and recognizing returning individuals.",
                ),
                SettingItem(
                    key="ADVANCED_FACE_MIN_SIZE",
                    label="Minimum Face Bounding Box Size",
                    value=face_min_size,
                    type="number",
                    category="cv_reid",
                    unit="px",
                    recommended_range="12 - 32 px",
                    description="Minimum face bounding box dimension in pixels. Filters out ultra-low resolution artifacts.",
                ),
                SettingItem(
                    key="ADVANCED_FACE_MIN_DET_SCORE",
                    label="Minimum Face Detection Confidence",
                    value=face_min_det_score,
                    type="number",
                    category="cv_reid",
                    unit="score",
                    recommended_range="0.20 - 0.50",
                    description="Confidence threshold for InsightFace detection. Lower thresholds capture angled or partly occluded faces.",
                ),
                SettingItem(
                    key="ADVANCED_CROSS_CAMERA_MIN_SCORE",
                    label="Cross-Camera Association Min Score",
                    value=cross_camera_min_score,
                    type="number",
                    category="cv_reid",
                    unit="score",
                    recommended_range="0.55 - 0.75",
                    description="Combined spatio-temporal transit and visual similarity threshold required to link trajectories across distinct CCTV cameras.",
                ),
                SettingItem(
                    key="FEATURE_LOCK_SHOTS",
                    label="Feature Lock Face Sampling Shots",
                    value=feature_lock_shots,
                    type="number",
                    category="cv_reid",
                    unit="shots",
                    recommended_range="4 - 10 shots",
                    description="Maximum number of high-quality facial and body embedding samples captured per unique tracked person.",
                ),
                SettingItem(
                    key="REID_MEMORY_BANK_TTL_SEC",
                    label="ReID Memory Bank Window (TTL)",
                    value=reid_memory_bank_ttl,
                    type="number",
                    category="cv_reid",
                    unit="sec",
                    recommended_range="30.0 - 90.0 sec",
                    description="Time window during which lost or occluded tracks remain in active memory for track stitching.",
                ),
                SettingItem(
                    key="REID_MEMORY_BANK_SIMILARITY_THRESHOLD",
                    label="ReID Memory Bank Stitching Threshold",
                    value=reid_memory_bank_thresh,
                    type="number",
                    category="cv_reid",
                    unit="cosine",
                    recommended_range="0.70 - 0.85",
                    description="Cosine similarity threshold required to stitch a re-emerging track back into a recent memory bank entity.",
                ),
            ],
        ),
        SettingsCategory(
            id="pipeline",
            name="Video Pipeline & Frame Sampling",
            description="Frame rate decimation and video ingestion settings balancing throughput and GPU compute.",
            parameters=[
                SettingItem(
                    key="PROCESS_EVERY_FRAME",
                    label="Process Every Frame Mode",
                    value=process_every_frame,
                    type="boolean",
                    category="pipeline",
                    unit="toggle",
                    recommended_range="TRUE / FALSE",
                    description="When TRUE, analyzes 100% of video frames (native FPS). When FALSE, applies smart sampling at PROCESSING_FPS for 3-5x faster inference.",
                ),
                SettingItem(
                    key="PROCESSING_FPS",
                    label="Smart Sampling Frame Rate",
                    value=processing_fps,
                    type="number",
                    category="pipeline",
                    unit="FPS",
                    recommended_range="5.0 - 15.0 FPS",
                    description="Analyzed frames per second when PROCESS_EVERY_FRAME is FALSE (7.5 FPS is optimal for standard 15-30 FPS CCTV).",
                ),
                SettingItem(
                    key="RTSP_RETRY_INTERVAL",
                    label="RTSP Stream Retry Interval",
                    value=rtsp_retry_interval,
                    type="number",
                    category="pipeline",
                    unit="sec",
                    recommended_range="3 - 15 sec",
                    description="Interval between automatic reconnection attempts if an IP RTSP camera stream drops.",
                ),
            ],
        ),
        SettingsCategory(
            id="database_storage",
            name="Database & Media Storage",
            description="Active database name and media persistence destinations.",
            parameters=[
                SettingItem(
                    key="POSTGRES_DB",
                    label="PostgreSQL Database Name",
                    value=postgres_db,
                    type="string",
                    category="database_storage",
                    unit="db",
                    recommended_range="cv_db",
                    description="Name of the active PostgreSQL database instance storing analytics sessions, timeline events, and embeddings.",
                ),
                SettingItem(
                    key="STORAGE_PROVIDER",
                    label="Media Storage Provider",
                    value=storage_provider,
                    type="string",
                    category="database_storage",
                    unit="provider",
                    recommended_range="local | s3",
                    description="Storage engine used for CCTV footage, person crops, face thumbnails, and synthesized timeline videos.",
                ),
                SettingItem(
                    key="LOCAL_STORAGE_PATH",
                    label="Local Media Storage Directory",
                    value=local_storage_path,
                    type="string",
                    category="database_storage",
                    unit="path",
                    recommended_range="storage or absolute path",
                    description="File system directory where local CCTV footage, face crops, and generated subclips are stored.",
                ),
            ],
        ),
        SettingsCategory(
            id="sql_agent",
            name="SQL Agent & AI Assistant",
            description="LLM provider configuration, query timeouts, and safety bounds for the AI assistant.",
            parameters=[
                SettingItem(
                    key="LLM_PROVIDER",
                    label="Active LLM Provider",
                    value=llm_provider,
                    type="string",
                    category="sql_agent",
                    unit="provider",
                    recommended_range="gemini | groq | openai | ollama",
                    description="LLM provider servicing natural language SQL generation and forensic journey summaries.",
                ),
                SettingItem(
                    key="LLM_MODEL",
                    label="Active LLM Model",
                    value=llm_model,
                    type="string",
                    category="sql_agent",
                    unit="model",
                    recommended_range="gemini-flash-latest | llama-3.3-70b-versatile",
                    description="Model identifier currently deployed for assistant queries.",
                ),
                SettingItem(
                    key="SQL_AGENT_TIMEOUT_SECONDS",
                    label="SQL Agent Query Timeout",
                    value=sql_agent_timeout,
                    type="number",
                    category="sql_agent",
                    unit="sec",
                    recommended_range="5.0 - 30.0 sec",
                    description="Maximum allowed execution duration for dynamically generated SQL analytical queries.",
                ),
                SettingItem(
                    key="SQL_AGENT_MAX_ROWS",
                    label="SQL Agent Max Rows Limit",
                    value=sql_agent_max_rows,
                    type="number",
                    category="sql_agent",
                    unit="rows",
                    recommended_range="50 - 500 rows",
                    description="Hard limit on number of rows fetched by LLM queries to protect server memory.",
                ),
                SettingItem(
                    key="SQL_AGENT_MAX_RETRIES",
                    label="SQL Agent Error Self-Correction Retries",
                    value=sql_agent_max_retries,
                    type="number",
                    category="sql_agent",
                    unit="retries",
                    recommended_range="2 - 5 retries",
                    description="Number of automated self-correction iterations attempted if an LLM SQL syntax error occurs.",
                ),
                SettingItem(
                    key="OLLAMA_BASE_URL",
                    label="Ollama Local Base URL",
                    value=ollama_base_url,
                    type="string",
                    category="sql_agent",
                    unit="url",
                    recommended_range="http://localhost:11434",
                    description="Endpoint for offline on-premise LLM execution when running air-gapped.",
                ),
                SettingItem(
                    key="OLLAMA_MODEL",
                    label="Ollama Local Model",
                    value=ollama_model,
                    type="string",
                    category="sql_agent",
                    unit="model",
                    recommended_range="qwen2.5-coder:latest",
                    description="Local Ollama model name used for offline intelligence.",
                ),
            ],
        ),
    ]

    return StandardResponse(
        status=status.HTTP_200_OK,
        message="System environment settings retrieved successfully",
        data=SystemSettingsData(
            raw_parameters=raw_parameters,
            categories=categories,
            environment=os.getenv("ENV", "development"),
            server_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            instructions="To update parameters, modify your .env file and restart the backend container/process.",
        ),
    )
