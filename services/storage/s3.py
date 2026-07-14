import os
from configs.base import settings

class S3Storage:
    def __init__(self):
        import boto3
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID or None,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY or None,
            region_name=settings.AWS_S3_REGION or "us-east-1",
        )
        self.bucket_name = settings.AWS_S3_BUCKET_NAME

    def _get_full_path(self, remote_path: str) -> str:
        """
        Resolves the remote path prepending the AWS_S3_KEY_PREFIX configuration if set.
        """
        remote_path = remote_path.replace("\\", "/")
        prefix = getattr(settings, "AWS_S3_KEY_PREFIX", "")
        if prefix:
            prefix = prefix.strip("/")
            return f"{prefix}/{remote_path.lstrip('/')}"
        return remote_path

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """
        Uploads a local file to S3.
        """
        from botocore.exceptions import ClientError
        resolved_path = self._get_full_path(remote_path)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file {local_path} does not exist for upload.")
        
        try:
            import mimetypes
            content_type, _ = mimetypes.guess_type(local_path)
            extra_args = {}
            if content_type:
                extra_args["ContentType"] = content_type
            
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                resolved_path,
                ExtraArgs=extra_args
            )
        except ClientError as e:
            print(f"AWS S3 upload error: {e}")
            raise e

    def download_file(self, remote_path: str, local_path: str) -> None:
        """
        Downloads a file from S3 to local path.
        """
        from botocore.exceptions import ClientError
        resolved_path = self._get_full_path(remote_path)
        try:
            self.s3_client.download_file(self.bucket_name, resolved_path, local_path)
        except ClientError as e:
            print(f"AWS S3 download error for key {resolved_path}: {e}")
            raise e

    def delete_file(self, remote_path: str) -> None:
        """
        Deletes a file from S3.
        """
        from botocore.exceptions import ClientError
        resolved_path = self._get_full_path(remote_path)
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=resolved_path)
        except ClientError as e:
            print(f"AWS S3 delete error for key {resolved_path}: {e}")
            raise e

    def get_url(self, remote_path: str) -> str:
        """
        Generates a pre-signed URL for temporary access to a private S3 object.
        """
        from botocore.exceptions import ClientError
        resolved_path = self._get_full_path(remote_path)
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": resolved_path},
                ExpiresIn=3600
            )
            return url
        except ClientError as e:
            print(f"AWS S3 presigned URL generation error: {e}")
            raise e
