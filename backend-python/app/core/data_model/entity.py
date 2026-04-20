import re
from typing import List, Optional, Any, Dict
from pydantic import Field, field_validator, ConfigDict, model_validator
from app.models.domain import SiraEntityType
from app.core.data_model.base import DescriptiveModel, slugify_entity

import hashlib

class EntityModel(DescriptiveModel):
    """
    Represents an entity in the Knowledge Graph.
    Acts as a bridge between raw extractions and the authoritative Encyclopedia.
    
    Attributes:
        ...
        canonical_id: If present, points to the corresponding entry in the Encyclopedia.
        review_status: Tracks the curation state (PENDING for manual review, VALIDATED for verified nodes).
    """

    model_config = ConfigDict(populate_by_name=True)
        
    type: str = Field("UNKNOWN", description="The category of the entity") #Ex: 'SAHABA', 'BATTLE'
    
    category: Optional[str] = None # Ex: 'HUMAN', 'EVENT'
    
    description: str = Field("", description="Consolidated summary of the entity")
    
    frequency: int = Field(1, description="Number of occurrences in the text")
    
    source_ids: List[str] = Field(default_factory=list)

    rank: int = Field(1, description="Importance de l'entité (centralité)")

    community_ids: List[str] = Field(default_factory=list)

    attributes: Dict[str, Any] = Field(default_factory=dict) # e.g. for periods (start/end),...

    # TODO lookup embedded attrbutes

    # For our synchro system
    canonical_id: Optional[str] = Field(None, description="The link to the Encyclopedia entry.") # if this attribute isn't empty, then the entity extracted should be linked/merged with the ency. entity
    
    review_status: str = Field("NOT_KNOWN", description="""  
                               Updated Statuses: 
                                - NOT_KNOWN: Newly extracted, no match found yet.
                                - PENDING: Match(es) found in Encyclopedia, awaiting final validation.
                                - LLM_VALIDATED: The LLM successfully anchored this entity.
                                - CORE_VALIDATED: Deterministic 1-to-1 match from CoreResolver.
                                - OFFICIAL: Record comes from the Professor/Source of Truth.""")
    
    @model_validator(mode="before")
    @classmethod
    def ensure_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "id" not in data:
            title = data.get("title", "unknown")
            etype = data.get("type", "unknown")
            # Hash déterministe
            data["id"] = hashlib.sha256(f"{title}_{etype}".encode()).hexdigest()[:16]
        return data


    @model_validator(mode="after")
    def assign_category(self) -> "EntityModel":
        """
        Automatically assigns a broad category based on the specific type.
        This happens ONCE at creation, so the rest of the app doesn't have 
        to calculate it repeatedly.
        """
        if not self.category:
            # On centralise la logique SiraEntityType ici
            self.category = SiraEntityType.get_category(self.type)
        return self