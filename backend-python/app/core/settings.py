from pathlib import Path
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Localisation du fichier .env
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

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

    @computed_field
    @property
    def database_url(self) -> str:
        """Reconstruit l'URL Postgres propre pour l'application."""
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"

try:
    settings = Settings()
except Exception as e:
    print(f"❌ Erreur de configuration. Assurez-vous que le fichier {ENV_FILE} contient toutes les clés requises.")
    raise e