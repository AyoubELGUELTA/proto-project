import re
from typing import List, Any, Dict
from pydantic import Field
from app.core.data_model.base import BaseModel


class TextUnit(BaseModel):
    """
    Represents a structured fragment of text enriched with visual and spatial data.
    
    This is the primary unit of work for the GraphRAG extraction process. 
    It encapsulates everything needed for the LLM to understand context: 
    narrative text, structural headings, tables, and associated images.
    """
    id: str = Field(..., description="Unique SHA-256 hash of the content")
    text: str = Field(..., description="The primary narrative text")
    headings: List[str] = Field(default_factory=list, description="Structural hierarchy (TOC path)")
    page_numbers: List[int] = Field(default_factory=list, description="Source pages in the original PDF")
    tables: List[str] = Field(default_factory=list, description="Markdown representation of tables")
    images_b64: List[str] = Field(default_factory=list, description="Base64 encoded visual assets (pre-storage)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extensible metadata (bbox, refinement flags)")

    class Config:
        """Pydantic configuration."""
        populate_by_name = True 
        arbitrary_types_allowed = True # Allows to store non trivial type objetcs in the df