from pydantic import BaseModel

class ChunkingConfig(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 100
    strategy: str = "docling_native"