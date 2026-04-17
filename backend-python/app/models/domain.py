import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class EntityCategory(str, Enum):
    """
    High-level classification for Sira entities.
    Used for broad filtering and graph visualization layers.
    """
    GOD = "God"
    HUMAN = "Human"           # Physical persons
    GEOGRAPHIC = "Geographic" # Cities, mountains, GPS points
    CONCEPTUAL = "Conceptual" # Tribes, groups, theology, ideas
    EVENT = "Event"           # Battles, treaties, specific moments
    OBJECT = "Object"         # Documents, animals, physical relics

class SiraEntityType(str, Enum):
    """
    Specific entity types with descriptions and category mapping.
    
    This Enum uses a custom constructor to attach semantic metadata 
    to each member, which can be used to ground LLM prompts.
    """
    # DIVINE
    GOD = "God", "Allah, The One and Only One", EntityCategory.GOD

    # HUMANS
    PROPHET_MUHAMMAD = "Prophet", "The Prophet Muhammad ﷺ specifically", EntityCategory.HUMAN
    MOTHER_OF_BELIEVERS = "MotherBeliever", "The wives of the Prophet ﷺ", EntityCategory.HUMAN
    AHL_BAYT = "AhlBayt", "The immediate family of the Prophet ﷺ", EntityCategory.HUMAN
    SAHABI = "Sahabi", "Male companions", EntityCategory.HUMAN
    SAHABIYA = "Sahabiya", "Female companions", EntityCategory.HUMAN
    CHILD = "Child", "Children mentioned", EntityCategory.HUMAN
    OPPONENT = "Opponent", "Notable adversaries", EntityCategory.HUMAN
    PERSON = "Person", "Generic person", EntityCategory.HUMAN

    # GEOGRAPHIC
    CITY = "City", "Cities and major settlements", EntityCategory.GEOGRAPHIC
    PLACE = "Place", "Specific geographic locations", EntityCategory.GEOGRAPHIC
    LOCATION = "Location", "Generic or uncertain locations", EntityCategory.GEOGRAPHIC

    # EVENTS
    BATTLE = "Battle", "Armed conflicts", EntityCategory.EVENT
    EVENT = "Event", "Historical events (Hijra, treaties)", EntityCategory.EVENT

    # CONCEPTUAL / GROUPS
    TRIBE = "Tribe", "Clans and tribes", EntityCategory.CONCEPTUAL
    CONCEPT = "Concept", "Socio-political groups or terms", EntityCategory.CONCEPTUAL
    CONCEPT_RELIGIOUS = "Religious Concept", "Theological concepts", EntityCategory.CONCEPTUAL
    GROUP = "Group", "Generic groups of people", EntityCategory.CONCEPTUAL

    # OBJECTS
    DOCUMENT = "Document", "Treaties, letters, or sacred texts", EntityCategory.OBJECT
    ANIMAL = "Animal", "Specifically named animals", EntityCategory.OBJECT

    def __new__(cls, value, description, category):
        """Custom constructor to store metadata within the Enum member."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        obj.category = category
        return obj

    @classmethod
    def get_category(cls, type_name: str) -> Optional[EntityCategory]:
        """
        Retrieves the parent category from a string value (usually from LLM output).
        """
        try:
            if not type_name:
                return None
                
            name_upper = type_name.strip().upper()
            
            # 1. Try to match by Enum Name (e.g., "SAHABI")
            if name_upper in cls.__members__:
                return cls[name_upper].category

            # 2. Try to match by Enum Value (e.g., "Sahabi")
            for member in cls:
                if member.value.upper() == name_upper:
                    return member.category
                    
            logger.debug(f"⚠️ Unknown entity type: '{type_name}'. No category mapped.")
            return None
        except ValueError:
            logger.debug(f"⚠️ Unknown entity type: '{type_name}'. No category mapped.")
            return None

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