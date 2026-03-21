"""
Configuration centralisée - Load .env une seule fois
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (une seule fois au import)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # app/core/ → proto-project/
ENV_PATH = PROJECT_ROOT / '.env'

load_dotenv(ENV_PATH, override=True)

# Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# PostgreSQL settings
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate required vars
if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD not set in .env")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in .env")