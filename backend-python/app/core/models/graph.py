import re
from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from app.models.domain import SiraEntityType

def slugify_entity(value: str) -> str:
    """
    Standardizes entity titles:
    'Maymuna bint al-Harith' -> 'MAYMUNA_BINT_AL_HARITH'
    """
    if not value:
        return ""
    # 1. Convert to Uppercase
    value = value.upper()
    # 2. Remove special characters (apostrophes, etc.)
    value = re.sub(r"[^A-Z0-9\s_-]", "", value)
    # 3. Replace spaces and hyphens with underscores
    value = re.sub(r"[\s-]+", "_", value)
    # 4. Strip leading/trailing underscores
    return value.strip("_")

class EntityModel(BaseModel):
    """
    Represents a resolved entity in the graph.
    """
    model_config = ConfigDict(populate_by_name=True)

    title: str = Field(..., description="The human-readable title of the entity")
    original_id: Optional[str] = Field(None, alias="id")
    type: str = Field("UNKNOWN", description="The category of the entity") #Ex: 'SAHABA', 'BATTLE'
    category: Optional[str] = None # Ex: 'HUMAN', 'EVENT'
    description: str = Field("", description="Consolidated summary of the entity")
    frequency: int = Field(1, description="Number of occurrences in the text")
    source_ids: List[str] = Field(default_factory=list)
    canonical_id: Optional[str] = None

    @field_validator("title", mode="before")
    @classmethod
    def normalize_title(cls, v: Any) -> str:
        if isinstance(v, str):
            return slugify_entity(v)
        return v
    
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

class RelationshipModel(BaseModel):
    """
    Represents a link between two entities.
    """
    source: str
    target: str
    description: str
    weight: float = 1.0
    source_ids: List[str] = Field(default_factory=list)

    @field_validator("source", "target", mode="before")
    @classmethod
    def normalize_nodes(cls, v: Any) -> str:
        if isinstance(v, str):
            return slugify_entity(v)
        return v