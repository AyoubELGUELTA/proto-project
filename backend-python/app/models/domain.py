from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from enum import Enum
from typing import List, Optional

class EntityCategory(str, Enum):
    GOD = "God"
    HUMAN = "Human"           # Personnes physiques
    GEOGRAPHIC = "Geographic" # Lieux, villes, points GPS
    CONCEPTUAL = "Conceptual" # Idées, groupes, théologie
    EVENT = "Event"           # Moments dans le temps, batailles
    OBJECT = "Object"         # Documents, animaux, objets physiques

class SiraEntityType(str, Enum):
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
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        obj.category = category
        return obj

    @classmethod
    def get_category(cls, type_name: str) -> Optional[EntityCategory]:
        """Récupère la catégorie à partir de la string du LLM."""
        try:
            return cls(type_name).category
        except ValueError:
            return None

class TextUnit(BaseModel):
    """Représente un fragment de texte enrichi prêt pour l'extraction de graphe."""
    id: str = Field(..., description="Hash unique du contenu")
    text: str
    headings: List[str] = []
    page_numbers: List[int] = []
    tables: List[str] = []  # Markdown
    images_b64: List[str] = []
    metadata: dict = {}