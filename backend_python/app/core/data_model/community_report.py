from typing import List, Optional, Dict, Any
from pydantic import Field
from .base import NamedModel

class CommunityReportModel(NamedModel):
    """
    The LLM-generated summary report for a specific community.
    """
    community_id: str = Field(..., description="The ID of the community this report describes")
    
    summary: str = Field("", description="A brief summary of the community's importance")
    full_content: str = Field("", description="The complete text of the generated report")
    
    rank: float = Field(1.0, description="Quality rank of the report (higher is better)")
    full_content_embedding: Optional[List[float]] = Field(None, description="Semantic vector of the full report")
    
    attributes: Dict[str, Any] = Field(default_factory=dict)
    size: Optional[int] = Field(None, description="Complexity or size indicator")
    period: Optional[str] = Field(None, description="Temporal context of the report")