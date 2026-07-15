import os
from configs.base import settings
from .s3 import S3Storage

class StorageClient:
    def __init__(self):
        self.provider = settings.STORAGE_PROVIDER.lower()
        if self.provider == "s3":
            self.s3 = S3Storage()

    def upload_file(self, local_path: str, remote_path: str) -> None:
        if self.provider == "local":
            return
        remote_path = remote_path.replace("\\", "/")
        self.s3.upload_file(local_path, remote_path)

    def get_file_path(self, remote_path: str) -> str:
        if not remote_path:
            return ""
        remote_path = remote_path.replace("\\", "/")
        local_path = os.path.normpath(remote_path)
        
        if self.provider == "local":
            return local_path

        # Cache hit: If local file exists and is not empty, use it
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path

        # Cache miss: Download from S3
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(remote_path, local_path)
        return local_path

    def delete_file(self, remote_path: str) -> None:
        if not remote_path:
            return
        remote_path = remote_path.replace("\\", "/")
        local_path = os.path.normpath(remote_path)
        
        # Always remove from local cache/filesystem if it exists
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception as e:
                print(f"Error removing local file {local_path}: {e}")

        if self.provider == "local":
            return

        # Remove from S3
        try:
            self.s3.delete_file(remote_path)
        except Exception as e:
            print(f"Error deleting file from S3 {remote_path}: {e}")

    def get_url(self, remote_path: str) -> str:
        if not remote_path:
            return ""
        remote_path = remote_path.replace("\\", "/")
        if self.provider == "local":
            return f"/{remote_path}"
        try:
            return self.s3.get_url(remote_path)
        except Exception as e:
            print(f"Error generating presigned S3 URL for {remote_path}: {e}")
            return f"/{remote_path}"

storage_client = StorageClient()
