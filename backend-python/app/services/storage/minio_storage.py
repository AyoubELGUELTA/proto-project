import io
import boto3
import uuid
import json
import logging
from PIL import Image
from botocore.client import Config
from app.services.storage.base import BaseStorage
from app.core.settings import settings

logger = logging.getLogger(__name__)

class MinioStorage(BaseStorage):
    """
    S3-compatible storage service (MinIO) for persisting document assets.
    
    This service handles bucket lifecycle, public access policies, 
    and optimized image uploads for the GraphRAG pipeline.
    """

    def __init__(self):
        # Centralized configuration from settings
        self.bucket = settings.s3_bucket_name
        self.endpoint = settings.s3_endpoint
        self.public_url = settings.s3_public_url

        self.s3 = boto3.client(
            's3',
            endpoint_url=f"http://{self.endpoint}" if not self.endpoint.startswith('http') else self.endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        logger.info(f"📦 Storage Service initialized - Bucket: {self.bucket} - Endpoint: {self.endpoint}")
        self._ensure_bucket()
        self.set_bucket_public()

    def _ensure_bucket(self):
        """Verifies bucket existence or creates it if missing."""
        if not self.bucket:
            logger.critical("❌ S3_BUCKET_NAME is missing in configuration!")
            raise ValueError("S3_BUCKET_NAME is required for MinioStorage.")
            
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception:
            logger.info(f"✨ Bucket not found. Creating bucket: {self.bucket}")
            self.s3.create_bucket(Bucket=self.bucket)

    def set_bucket_public(self):
        """Sets a Read-Only public policy for the current bucket."""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"AWS": ["*"]},
                    "Action": ["s3:GetObject"],
                    "Resource": [f"arn:aws:s3:::{self.bucket}/*"]
                }
            ]
        }
        try:
            self.s3.put_bucket_policy(Bucket=self.bucket, Policy=json.dumps(policy))
            logger.debug(f"🔓 Public access policy applied to {self.bucket}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to set public policy on bucket {self.bucket}: {e}")

    def upload_image(self, pil_image: Image.Image) -> str:
        """
        Processes and uploads a PIL image to the S3 bucket.
        
        Optimizations:
        - Resize to max 1024px while preserving aspect ratio.
        - Compress as JPEG with 80% quality.
        - Generates a unique UUID filename.
        """
        try:
            filename = f"{uuid.uuid4()}.jpg"
            
            # Optimization: Thumbnail preserves aspect ratio
            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=80)
            buffer.seek(0)
            file_size = buffer.getbuffer().nbytes

            logger.debug(f"📤 Uploading image {filename} ({file_size // 1024} KB)...")
            
            self.s3.upload_fileobj(
                buffer, 
                self.bucket, 
                filename,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # Use public URL from settings for final mapping
            public_path = f"{self.public_url}/{self.bucket}/{filename}"
            logger.debug(f"✅ Upload successful: {public_path}")
            return public_path
            
        except Exception as e:
            logger.error(f"❌ Storage Upload failed: {e}", exc_info=True)
            return ""