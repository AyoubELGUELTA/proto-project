from pydantic import BaseModel

class ChunkingConfig(BaseModel):
    chunk_size: int = 800
    chunk_overlap: int = 100
    strategy: str = "docling_native"


chunking_config = ChunkingConfig()

CHUNK_SIZE = chunking_config.chunk_size
CHUNK_OVERLAP = chunking_config.chunk_overlap
