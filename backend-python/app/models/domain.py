"""
Ontologie Islam pour Graph RAG
Définit les types d'entités et relations spécifiques à la Sira
TODO --> faire en sorte qu'il soit exhaustif a 99.9 prct a l'avenir ig
"""
from enum import Enum
from typing import List,Optional ,Dict, Any
from pydantic import BaseModel, Field

class EntityType(str, Enum):
    """
    Single Source of Truth pour les types d'entités.
    Priorité : Utilisée pour le merging (le type avec le plus haut score gagne).
    """
    PROPHET = "Prophet", 100, "Le Prophète Muhammad ﷺ uniquement.", ["Human", "Man"]
    MOTHER_BELIEVER = "MotherBeliever", 95, "Épouses du Prophète ﷺ (Mères des Croyants).", ["Human", "Woman"]
    AHL_BAYT = "AhlBayt", 90, "Famille proche du Prophète ﷺ (enfants, petits-enfants, oncles proches).", ["Human"]
    SAHABI = "Sahabi", 85, "Compagnons masculins du Prophète ﷺ.", ["Human", "Man"]
    SAHABIYA = "Sahabiya", 85, "Compagnons féminins du Prophète ﷺ.", ["Human", "Woman"]
    OPPONENT = "Opponent", 80, "Adversaires notables de l'Islam durant la période prophétique.", ["Human"]
    CITY = "City", 70, "Villes, cités et agglomérations (ex: La Mecque, Médine, Taif).", ["Location", "Place"]
    PLACE = "Place", 65, "Lieux géographiques précis (montagnes, puits, vallées, grottes).", ["Location"]
    BATTLE = "Battle", 60, "Conflits armés, expéditions (Ghazwa) ou escarmouches (Sariya).", ["Event"]
    EVENT = "Event", 55, "Événements historiques (ex: Hijra, Pacte d'Al-Hudaybiya, Isra wal-Miraj)."
    TRIBE = "Tribe", 50, "Clans, tribus et confédérations (ex: Quraysh, Banu Hashim).", ["Group"]
    CONCEPT = "Concept", 40, "Termes techniques ou groupes sociopolitiques (ex: Ansar, Muhajirun, Munafiqun)."
    DOCUMENT = "Document", 30, "Traités, lettres, ou textes sacrés (ex: Constitution de Médine)."
    ANIMAL = "Animal", 20, "Animaux nommés spécifiquement (ex: la chamelle Al-Qaswa)."
    CHILD = "Child", 15, "Enfant mentionné dont le statut de Sahabi n'est pas encore le trait principal.", ["Human"]
    GROUP = "Group", 12, "Groupe de personnes quelquonques"
    LOCATION = "Location", 1, "Lieu générique non nommé ou incertain."

    def __new__(cls, value, priority, description, parents=None):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.priority = priority
        obj.description = description
        obj.parents = parents or []
        return obj


    @classmethod
    def get_all_labels(cls, type_name: str) -> List[str]:
        """Récupère le label principal + tous les parents définis (si il y en a)."""
        try:
            entity_enum = cls(type_name)
            return [entity_enum.value] + entity_enum.parents
        except ValueError:
            return [type_name]
        
    @classmethod
    def get_priority(cls, type_name: str) -> int:
        try: return cls(type_name).priority
        except ValueError: return 0

    @classmethod
    def get_llm_definitions(cls) -> str:
        return "\n".join([f"- {e.value}: {e.description}" for e in cls])
    
    @classmethod
    def get_taxonomy(cls) -> str:
        """
        Génère une taxonomie propre et triée pour injection directe dans le prompt.
        """
        taxonomy_lines = ["### ENTITY TAXONOMY"]

        # Tri par priorité décroissante
        sorted_types = sorted(cls, key=lambda x: x.priority, reverse=True)

        for e in sorted_types:
            # Labels parents (héritage)
            parents = f" (Inherits: {', '.join(e.parents)})" if e.parents else ""
            
            # Format: - TYPE (Priorité): Description (Héritage)
            line = f"- **{e.value}** ({e.priority}): {e.description}{parents}"
            taxonomy_lines.append(line)

        return "\n".join(taxonomy_lines)

class RelationType(str, Enum):
    """
    Taxonomie des relations.
    Description : Utilisée pour guider le LLM lors de l'extraction.
    """
    # Familiales & Domestiques
    MARRIED_TO = "MARRIED_TO", "Lien de mariage (symétrique)."
    PARENT_OF = "PARENT_OF", "Lien du parent vers l'enfant."
    CHILD_OF = "CHILD_OF", "Lien de l'enfant vers le parent."
    SIBLING_OF = "SIBLING_OF", "Lien entre frères et sœurs (fratrie)."
    MASTER_OF = "MASTER_OF", "Relation de maître à esclave/affranchi (ex: Zayd)."
    BROTHER_WITH = "BROTHER_WITH", "Fraternité spirituelle (Muwakhat) établie à Médine."

    # Événementielles & Militaires
    PARTICIPATED_IN = "PARTICIPATED_IN", "Présence physique à une bataille ou un événement."
    COMMANDED = "COMMANDED", "Direction d'une armée, d'une expédition ou d'un groupe."
    WITNESSED = "WITNESSED", "Témoin oculaire d'un événement sans rôle de combat."
    DIED_IN = "DIED_IN", "Lieu ou événement du décès."
    CONVERTED_AT = "CONVERTED_AT", "Lieu ou moment précis de la conversion à l'Islam."
    OPPOSED_IN = "OPPOSED_IN", "Action d'opposition directe lors d'un conflit ou événement."

    # Spatiales & Tribales
    LIVED_IN = "LIVED_IN", "Résidence habituelle dans une ville ou un lieu."
    MIGRATED_TO = "MIGRATED_TO", "Action de migration (ex: Hijra vers Médine ou Abyssinie)."
    TRAVELED_TO = "TRAVELED_TO", "Voyage temporaire ou expédition de commerce."
    MEMBER_OF = "MEMBER_OF", "Appartenance à une tribu, un clan ou un groupe (Ansar/Muhajirun)."
    ALLIED_WITH = "ALLIED_WITH", "Alliance politique ou militaire entre groupes/individus."

    # Transmission & Chronologie
    NARRATED_BY = "NARRATED_BY", "Source de l'information (Rawi)."
    OCCURRED_DURING = "OCCURRED_DURING", "Lien temporel entre un événement et une période."
    FOUNDED = "FOUNDED", "Création d'une mosquée, d'une ville ou d'une institution."

    def __new__(cls, value, description):
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.description = description
        return obj

    @classmethod
    def get_llm_definitions(cls) -> str:
        return "\n".join([f"- {e.value}: {e.description}" for e in cls])

# # Schéma d'intégrité (pour limiter les erreurs du LLM) TODO, have to add some garde-fous to not have des aberrances dans le graph (ex: Hira went in the battle of Badr..)
# SIRA_SCHEMA_CONSTRAINTS = {
#     EntityType.PROPHET: {
#         "allowed_relations": [RelationType.MARRIED_TO, RelationType.PARENT_OF, RelationType.COMMANDED, RelationType.MIGRATED_TO, RelationType.FOUNDED],
#         "mandatory_honorific": "ﷺ"
#     },
#     EntityType.MOTHER_BELIEVER: {
#         "allowed_relations": [RelationType.MARRIED_TO, RelationType.CHILD_OF, RelationType.MEMBER_OF, RelationType.NARRATED_BY],
#         "mandatory_honorific": "RA"
#     },
#     EntityType.BATTLE: {
#         "allowed_relations": [RelationType.OCCURRED_DURING, RelationType.LIVED_IN], # Lived_in ici pour la localisation
#     }
# }

VALIDATION_RULES = {
    "max_prophets": 1,
    "max_mothers_believers": 11,
}



class TextUnit(BaseModel):
    """Représente un fragment de texte enrichi prêt pour l'extraction de graphe."""
    id: str = Field(..., description="Hash unique du contenu")
    text: str
    headings: List[str] = []
    page_numbers: List[int] = []
    tables: List[str] = []  # Markdown
    images_b64: List[str] = []
    metadata: dict = {}