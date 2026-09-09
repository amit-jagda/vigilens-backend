from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
import json

class Settings(BaseSettings):
    PROJECT_NAME: str = "Computer Vision API"
    API_V1_STR: str = "/api/v1"
    
    # Database Configuration
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/cv_db"
    
    # Security Configuration
    SECRET_KEY: str = "super-secret-key-for-development-only-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # 1 hour
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:5173",
        "http://localhost:8000",
    ]
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            if isinstance(v, str):
                try:
                    return json.loads(v)
                except Exception:
                    return [v]
            return v
        return v
    
    # Storage Configuration
    STORAGE_PROVIDER: str = "local"
    LOCAL_STORAGE_PATH: str = "storage"
    
    # AWS S3 Settings
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_S3_BUCKET_NAME: str = ""
    AWS_S3_REGION: str = "us-east-1"
    AWS_S3_KEY_PREFIX: str = ""

    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Redis Cache Configuration
    REDIS_URL: str = "redis://localhost:6379/1"


    # RTSP Configuration
    RTSP_RETRY_INTERVAL: int = 5  # seconds

    # YOLO Configuration
    YOLO_MODEL: str = "models/yolo26n.pt"
    YOLO_SEG_MODEL: str = "yolov8l-seg.pt"
    YOLO_IMGSZ: int = 384

    # Advanced People Analytics Configuration
    PROCESS_EVERY_FRAME: bool = False  # Set to True to analyze 100% of video frames (native FPS) without skipping
    PROCESSING_FPS: float = 7.5  # Analyzed frames per second when PROCESS_EVERY_FRAME is False (7.5 FPS optimal for 15 FPS native CCTV)
    ADVANCED_FACE_SIMILARITY_THRESHOLD: float = 0.28
    ADVANCED_REID_SIMILARITY_THRESHOLD: float = 0.75
    ADVANCED_FACE_MIN_SIZE: int = 14
    ADVANCED_FACE_MIN_DET_SCORE: float = 0.35
    ADVANCED_CROSS_CAMERA_MIN_SCORE: float = 0.65
    FEATURE_LOCK_SHOTS: int = 6  # Allow up to 6 face sampling attempts per person
    REID_MEMORY_BANK_TTL_SEC: float = 45.0  # Time window to stitch lost tracks on occlusions
    REID_MEMORY_BANK_SIMILARITY_THRESHOLD: float = 0.75
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
