from typing import List, Optional, Any, Dict
from pydantic import  Field, field_validator, ConfigDict, model_validator
from app.core.data_model.base import slugify_entity, IdentifiedModel

class RelationshipModel(IdentifiedModel):
    source_id: str = Field(..., description="Unique ID of the source entity")

    target_id: str = Field(..., description="Unique ID of the target entity")

    source_slug: Optional[str] = None
    
    target_slug: Optional[str] = None
    
    description: str
  
    weight: float = 1.0
    
    # --- MC-Compliant Fields ---
    rank: int = Field(1, description="The importance of the relation")
    
    source_ids: List[str] = Field(default_factory=list)
   
    attributes: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source", "target", mode="before")
    @classmethod
    def normalize_nodes(cls, v: Any) -> str:
        if isinstance(v, str):
            return slugify_entity(v)
        return v