
import os
import io
import boto3
from PIL import Image
from botocore.client import Config
import uuid
import json
from app.services.storage.base import BaseStorage

class MinioStorage(BaseStorage):
    def __init__(self):
        self.s3 = boto3.client(
            's3',
            endpoint_url=f"http://{os.getenv('S3_ENDPOINT')}",
            aws_access_key_id=os.getenv('S3_ACCESS_KEY'),
            aws_secret_access_key=os.getenv('S3_SECRET_KEY'),
            config=Config(signature_version='s3v4'),
            region_name='us-east-1' # Requis par boto3 mais ignoré par MinIO
        )
        self.bucket = os.getenv('S3_BUCKET_NAME')
        self._ensure_bucket()
        self.set_bucket_public()

    def _ensure_bucket(self):
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except:
            self.s3.create_bucket(Bucket=self.bucket)

    def set_bucket_public(self):
        """Configure le bucket pour être accessible sans signature (Lecture seule)."""
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
        self.s3.put_bucket_policy(Bucket=self.bucket, Policy=json.dumps(policy))

    def upload_image(self, pil_image: Image.Image) -> str:
        try:
            filename = f"{uuid.uuid4()}.jpg"

            pil_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=80)
            buffer.seek(0)

            self.s3.upload_fileobj(
                buffer, self.bucket, filename,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            return f"{os.getenv('S3_PUBLIC_URL')}/{self.bucket}/{filename}"
        except Exception as e:
            print(f"❌ Erreur Storage: {e}")
            return ""