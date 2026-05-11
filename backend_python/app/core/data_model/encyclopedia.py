from typing import List, Optional, Dict, Any
from app.core.data_model.base import DescriptiveModel
from pydantic import  Field, field_validator, ConfigDict, model_validator
from app.models.domain import SiraEntityType


class EncyclopediaEntry(DescriptiveModel): #Or EncyclopediaEntity(DescriptiveModel)
    """
    The authoritative record for an entity.
    Stored in SQL, used to resolve and enrich the Knowledge Graph.
    """
    model_config = ConfigDict(populate_by_name=True)

    type: str = Field(..., description="Specific category (SAHABI, BATTLE, CITY, etc.)")

    category: Optional[str] = Field(None, description="Broad category (HUMAN, EVENT, etc.)")

    core_summary: str = Field(..., description="The verified historical summary")

    properties: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Type-specific attributes (e.g., nasab for humans, revelation_order for surahs)"
    )
    
    review_status: str = Field("OFFICIAL", description="""  
                               Updated Statuses: 
                                - NOT_KNOWN: Newly extracted, no match found yet.
                                - PENDING: Match(es) found in Encyclopedia, awaiting final validation.
                                - LLM_VALIDATED: The LLM successfully anchored this entity.
                                - CORE_VALIDATED: Deterministic 1-to-1 match from CoreResolver.
                                - OFFICIAL: Record comes from the Professor/Source of Truth.""")
    
    # Metadata for the UI
    is_verified: bool = Field(True)
    last_updated_by: Optional[str] = None
   
    @model_validator(mode="after")
    def assign_category(self) -> "EncyclopediaEntry":
        """Assign the broad category (HUMAN, EVENT, etc.) based on the specific type."""
        if not self.category:
            self.category = SiraEntityType.get_category(self.type)
        return self


