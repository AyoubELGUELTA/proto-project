"""
Ontologie Islam pour Graph RAG
Définit les types d'entités et relations spécifiques à la Sira
TODO --> faire en sorte qu'il soit exhaustif a 99.9 prct a l'avenir ig
"""

from typing import List, Dict
from enum import Enum

class EntityType(str, Enum):
    """Types d'entités spécifiques Islam"""
    PROPHET = "Prophet"
    MOTHER_BELIEVER = "MotherBeliever"
    SAHABI = "Sahabi"              # Compagnon
    AHL_BAYT = "AhlBayt"           # Famille proche
    TRIBE = "Tribe"
    PLACE = "Place"
    EVENT = "Event"
    BATTLE = "Battle"
    CONCEPT = "Concept"
    PERIOD = "Period"

class RelationType(str, Enum):
    """Relations spécifiques Sira"""
    # Familiales
    MARRIED_TO = "MARRIED_TO"
    DAUGHTER_OF = "DAUGHTER_OF"
    SON_OF = "SON_OF"
    MOTHER_OF = "MOTHER_OF"
    FATHER_OF = "FATHER_OF"
    SIBLING = "SIBLING"
    
    # Événementielles
    PARTICIPATED_IN = "PARTICIPATED_IN"
    WITNESSED = "WITNESSED"
    ORGANIZED = "ORGANIZED"
    DIED_IN = "DIED_IN"
    
    # Spatiales
    LIVED_IN = "LIVED_IN"
    TRAVELED_TO = "TRAVELED_TO"
    BORN_IN = "BORN_IN"
    
    # Tribales
    MEMBER_OF_TRIBE = "MEMBER_OF_TRIBE"
    
    # Temporelles
    OCCURRED_DURING = "OCCURRED_DURING"
    BEFORE = "BEFORE"
    AFTER = "AFTER"

# Schema pour LlamaIndex
SIRA_SCHEMA: Dict[str, List[str]] = {
    EntityType.PROPHET: [
        RelationType.MARRIED_TO,
        RelationType.FATHER_OF,
        RelationType.LIVED_IN,
        RelationType.ORGANIZED,
    ],
    EntityType.MOTHER_BELIEVER: [
        RelationType.MARRIED_TO,
        RelationType.DAUGHTER_OF,
        RelationType.MOTHER_OF,
        RelationType.MEMBER_OF_TRIBE,
        RelationType.PARTICIPATED_IN,
    ],
    EntityType.SAHABI: [
        RelationType.PARTICIPATED_IN,
        RelationType.WITNESSED,
        RelationType.MEMBER_OF_TRIBE,
    ],
    EntityType.TRIBE: [
        RelationType.LIVED_IN,
    ],
    EntityType.PLACE: [
        RelationType.BORN_IN,
        RelationType.DIED_IN,
    ],
    EntityType.EVENT: [
        RelationType.OCCURRED_DURING,
        RelationType.PARTICIPATED_IN,
    ],
    EntityType.BATTLE: [
        RelationType.OCCURRED_DURING,
        RelationType.PARTICIPATED_IN,
    ],
}

# Validation rules (pour post-processing)
VALIDATION_RULES = {
    "max_prophets": 1,
    "max_mothers_believers": 11,
    "required_honorifics": {
        EntityType.PROPHET: ["SAW", "ﷺ"],
        EntityType.MOTHER_BELIEVER: ["RA", "رضي الله عنها"],
        EntityType.SAHABI: ["RA", "رضي الله عنه"],
    }
}