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
    def __init__(self):
        # Centralisation sur settings
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
        
        logger.info(f"📦 Storage Initialisé - Bucket: {self.bucket} - Endpoint: {self.endpoint}")
        self._ensure_bucket()
        self.set_bucket_public()

    def _ensure_bucket(self):
        if not self.bucket:
            raise ValueError("❌ S3_BUCKET_NAME est manquant dans la configuration !")
        try:
            self.s3.head_bucket(Bucket=self.bucket)
        except Exception:
            logger.info(f"✨ Création du bucket : {self.bucket}")
            self.s3.create_bucket(Bucket=self.bucket)

    def set_bucket_public(self):
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
            # Utilisation de l'URL publique des settings
            return f"{self.public_url}/{self.bucket}/{filename}"
        except Exception as e:
            logger.error(f"❌ Erreur Upload Storage: {e}")
            return ""