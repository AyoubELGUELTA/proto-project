from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field

class SiraEntityType(str, Enum):
    """
    Syra entities for GraphRAG extraction. 
    Strictly English for internal LLM processing.
    """
    GOD = "God", "Allah, The One and Only One ('azzawajel)."
    PROPHET = "Prophet", "The Prophet Muhammad ﷺ specifically."
    MOTHER_BELIEVER = "MotherBeliever", "The wives of the Prophet ﷺ (Mothers of the Believers)."
    AHL_BAYT = "AhlBayt", "The immediate family of the Prophet ﷺ."
    SAHABI = "Sahabi", "Male companions of the Prophet ﷺ."
    SAHABIYA = "Sahabiya", "Female companions of the Prophet ﷺ."
    OPPONENT = "Opponent", "Notable adversaries during the prophetic period."
    CITY = "City", "Cities and major settlements (e.g., Makkah, Madinah)."
    PLACE = "Place", "Specific geographic locations (mountains, wells, valleys)."
    BATTLE = "Battle", "Armed conflicts and expeditions (Ghazwa, Sariya)."
    EVENT = "Event", "Historical events (e.g., Hijra, Treaties)."
    TRIBE = "Tribe", "Clans, tribes, and confederations."
    CONCEPT = "Concept", "Technical terms or socio-political groups (e.g., Ansar, Muhajirun)."
    DOCUMENT = "Document", "Treaties, letters, or sacred texts."
    ANIMAL = "Animal", "Specifically named animals."
    CHILD = "Child", "Children mentioned in the text."
    GROUP = "Group", "Generic groups of people."
    LOCATION = "Location", "Generic or uncertain locations."
    CONCEPT_RELIGIOUS = "Religious Concept", "Concepts linked or contained in the Syra."

    def __new__(cls, value, description):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    @classmethod
    def get_llm_definitions(cls) -> str:
        """Returns the definitions in a format MS prompts expect."""
        return "\n".join([f"- {e.value}: {e.description}" for e in cls])

class TextUnit(BaseModel):
    """Représente un fragment de texte enrichi prêt pour l'extraction de graphe."""
    id: str = Field(..., description="Hash unique du contenu")
    text: str
    headings: List[str] = []
    page_numbers: List[int] = []
    tables: List[str] = []  # Markdown
    images_b64: List[str] = []
    metadata: dict = {}