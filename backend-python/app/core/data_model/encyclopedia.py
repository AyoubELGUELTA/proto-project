from typing import List, Optional, Dict, Any
from pydantic import Field
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
    
    # Metadata for the UI
    is_verified: bool = Field(True)
    last_updated_by: Optional[str] = None
   
    @model_validator(mode="after")
    def assign_category(self) -> "EncyclopediaEntry":
        """Calcule la catégorie broad (HUMAN, EVENT, etc.) à partir du type spécifique."""
        if not self.category:
            self.category = SiraEntityType.get_category(self.type)
        return self


