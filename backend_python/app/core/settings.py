import os
from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- DÉTECTION DYNAMIQUE DU ROOT DIR (Racine du Monorepo) ---
def get_root_dir() -> Path:
    # On part de app/core/settings.py
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        # On remonte jusqu'à trouver le fichier .env qui est à la racine du Monorepo
        if (parent / ".env").exists():
            return parent
    # Fallback : si on ne trouve pas de .env, on prend le parent du projet python
    return Path(__file__).resolve().parent.parent.parent.parent

ROOT_DIR = get_root_dir()
ENV_FILE = ROOT_DIR / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE, 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # --- LLM ---
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # --- REDIS ---
    redis_url: str = Field("redis://localhost:6379", alias="REDIS_URL")

    # --- NEO4J ---
    neo4j_uri: str = Field("bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field("neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    # --- POSTGRES ---
    db_user: str = Field("postgres", alias="DB_USER")
    db_password: str = Field(..., alias="DB_PASSWORD")
    db_host: str = Field("localhost", alias="DB_HOST")
    db_port: int = Field(5432, alias="DB_PORT")
    db_name: str = Field("sira_db", alias="DB_NAME")

    # --- S3 / MINIO ---
    s3_endpoint: str = Field(..., alias="S3_ENDPOINT")
    s3_access_key: str = Field(..., alias="S3_ACCESS_KEY")
    s3_secret_key: str = Field(..., alias="S3_SECRET_KEY")
    s3_bucket_name: str = Field("sira-assets", alias="S3_BUCKET_NAME")
    s3_public_url: str = Field("http://localhost:9000", alias="S3_PUBLIC_URL")

    # --- STORAGE LOCAL ---
    # D'après ton tree, c'est dans backend-python/data/storage
    local_storage_path: Path = Field(default=ROOT_DIR / "backend-python" / "data" / "storage")

    @computed_field
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

    def __init__(self, **values):
        super().__init__(**values)
        # Création automatique du dossier s'il n'existe pas
        self.local_storage_path.mkdir(parents=True, exist_ok=True)

# Instanciation
try:
    settings = Settings()
    print(f"✅ Configuration Monorepo validée")
    print(f"📍 Root (Monorepo): {ROOT_DIR}")
    print(f"💾 Storage: {settings.local_storage_path}")
except Exception as e:
    print(f"❌ Erreur de config : {e}")
    raise e