import os
import io
import boto3
from PIL import Image
from botocore.client import Config
import uuid
import json

class S3Storage:
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

    def upload_image(self, pil_image):
        """Redimensionne, compresse et upload l'image."""
        try:
            filename = f"{uuid.uuid4()}.jpg"

            # 1. Redimensionnement (Max 1024px pour GPT-5 Vision)
            max_size = (1024, 1024)
            pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)

            # 2. Compression JPEG
            buffer = io.BytesIO()
            pil_image.save(buffer, format="JPEG", quality=80)
            buffer.seek(0)

            # 3. Upload
            self.s3.upload_fileobj(
                buffer, 
                self.bucket, 
                filename,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            public_base_url = os.getenv("S3_PUBLIC_URL").rstrip('/')            
            # 4. URL Publique Ngrok pour OpenAI

            return f"{public_base_url}/{self.bucket}/{filename}"
        except Exception as e:
            print(f"❌ Erreur S3 Upload: {e}")
            return None

storage = S3Storage()