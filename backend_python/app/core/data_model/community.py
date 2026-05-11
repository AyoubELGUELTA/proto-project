from typing import List, Optional, Dict, Any
from pydantic import Field
from app.core.data_model.base import NamedModel

class CommunityModel(NamedModel):
    """
    A community (cluster) within the system, representing a group of related entities.
    """
    
    level: str = Field(..., description="Community level (0, 1, 2, etc.)")
    parent: Optional[str] = Field(None, description="ID of the parent community")
    children: List[str] = Field(default_factory=list, description="IDs of sub-communities")
    
    entity_ids: Optional[List[str]] = Field(default_factory=list)
    relationship_ids: Optional[List[str]] = Field(default_factory=list)
    text_unit_ids: Optional[List[str]] = Field(default_factory=list)
    
    # Pour les 'claims' ou autres métadonnées extraites
    covariate_ids: Dict[str, List[str]] = Field(default_factory=dict)
    
    attributes: Dict[str, Any] = Field(default_factory=dict)
    size: Optional[int] = Field(None, description="Amount of text units in this community")
    period: Optional[str] = Field(None, description="Time period associated with this community")