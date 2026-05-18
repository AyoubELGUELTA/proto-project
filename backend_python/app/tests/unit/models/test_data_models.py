import pytest
import math
import hashlib
from unittest.mock import patch

# Adapte ces imports à la structure exacte de ton projet
from app.core.data_model.base import slugify_entity
from app.models.domain import SiraEntityType # (Assure-toi que cet import est correct pour ton mock)
from app.core.data_model.entity import EntityModel
from app.core.data_model.relationship import RelationshipModel
from app.core.data_model.encyclopedia import EncyclopediaEntry
from app.core.data_model.community import CommunityModel
from app.core.data_model.base import IdentifiedModel,DescriptiveModel

# ==========================================
# 1. TESTS DES UTILITAIRES CORE
# ==========================================

def test_slugify_entity_logic():
    """Valide les règles de transformation strictes des slugs."""
    assert slugify_entity("Maymuna bint al-Harith") == "MAYMUNA_BINT_AL_HARITH"
    assert slugify_entity("Prophet (SAW)") == "PROPHET_SAW"
    assert slugify_entity("  Badr-Battle  ") == "BADR_BATTLE"
    assert slugify_entity(None) == ""
    assert slugify_entity("") == ""


# ==========================================
# 2. TESTS DES MODÈLES DE BASE (Héritage)
# ==========================================

def test_identified_model_auto_uuid():
    """Valide que IdentifiedModel génère bien un UUID4 si aucun ID n'est fourni."""
    model = IdentifiedModel()
    assert model.id is not None
    assert isinstance(model.id, str)
    assert len(model.id) == 36 # Longueur standard UUID4

    # Si un ID est forcé, il ne doit pas être écrasé
    model_custom = IdentifiedModel(id="custom-id-123")
    assert model_custom.id == "custom-id-123"

def test_descriptive_model_auto_slug():
    """Valide que le slug est automatiquement généré à partir du titre s'il manque."""
    model = DescriptiveModel(title="Abu Bakr")
    assert model.slug == "ABU_BAKR"
    
    # Si le slug est forcé, il ne doit pas être écrasé
    model_custom = DescriptiveModel(title="Abu Bakr", slug="CUSTOM_SLUG")
    assert model_custom.slug == "CUSTOM_SLUG"


# ==========================================
# 3. TESTS DU MODÈLE RELATIONSHIP
# ==========================================

def test_relationship_model_slug_normalization():
    """Valide le '@field_validator' de normalization des slugs source/target."""
    rel = RelationshipModel(
        source_slug="Prophet Muhammad", 
        target_slug="Khadija bint Khuwaylid",
        description="Marriage"
    )
    
    # Les slugs doivent avoir été transformés 'before' validation
    assert rel.source_slug == "PROPHET_MUHAMMAD"
    assert rel.target_slug == "KHADIJA_BINT_KHUWAYLID"
    # Vérification des defaults
    assert rel.weight == 1.0
    assert rel.rank == 1
    assert rel.source_ids == []

def test_relationship_model_preserves_none_slugs():
    """Si aucun slug n'est fourni, il doit rester None sans crasher."""
    rel = RelationshipModel(description="Unknown relation")
    assert rel.source_slug is None
    assert rel.target_slug is None


# ==========================================
# 4. TESTS DU MODÈLE ENTITY (Le plus complexe)
# ==========================================

@patch("app.models.domain.SiraEntityType.get_category")
def test_entity_model_deterministic_hashing(mock_get_category):
    """
    Valide que le hash déterministe fonctionne parfaitement :
    Deux entités avec le même (titre, type) doivent avoir le MÊME id.
    """
    mock_get_category.return_value = "HUMAN"
    
    title = "Umar ibn al-Khattab"
    etype = "SAHABI"
    
    entity1 = EntityModel(title=title, type=etype)
    entity2 = EntityModel(title=title, type=etype)
    
    # Validation du hachage sha256 tronqué à 16 caractères
    expected_hash = hashlib.sha256(f"{title}_{etype}".encode()).hexdigest()[:16]
    
    assert entity1.id == expected_hash
    assert entity2.id == expected_hash
    assert entity1.id == entity2.id

@patch("app.models.domain.SiraEntityType.get_category")
def test_entity_model_nan_handling(mock_get_category):
    """
    Simule la sortie d'un DataFrame Pandas où les valeurs vides sont des math.nan.
    Le '@model_validator' handle_nan doit les convertir en None.
    """
    mock_get_category.return_value = "EVENT"
    
    # On passe volontairement un math.nan dans category
    entity = EntityModel(
        title="Battle of Uhud", 
        type="BATTLE",
        canonical_id=math.nan 
    )
    
    assert entity.canonical_id is None # NaN a été purgé avec succès

@patch("app.models.domain.SiraEntityType.get_category")
def test_entity_model_auto_category(mock_get_category):
    """Valide que la catégorie se peuple automatiquement à l'instanciation."""
    mock_get_category.return_value = "GEOGRAPHIC"
    
    entity = EntityModel(title="Medina", type="CITY")
    
    mock_get_category.assert_called_once_with("CITY")
    assert entity.category == "GEOGRAPHIC"
    assert entity.slug == "MEDINA" # Auto-slug test


# ==========================================
# 5. TESTS DE L'ENCYCLOPÉDIE
# ==========================================

@patch("app.models.domain.SiraEntityType.get_category")
def test_encyclopedia_entry_initialization(mock_get_category):
    """Valide la création propre d'une entrée de l'encyclopédie."""
    mock_get_category.return_value = "HUMAN"
    
    entry = EncyclopediaEntry(
        title="Aisha bint Abi Bakr",
        type="SAHABIYYAT",
        core_summary="Wife of the Prophet and prominent scholar.",
        properties={"scholar_rank": "High"}
    )
    
    assert entry.id is not None # UUID généré via IdentifiedModel
    assert entry.slug == "AISHA_BINT_ABI_BAKR"
    assert entry.review_status == "OFFICIAL"
    assert entry.is_verified is True
    assert entry.properties["scholar_rank"] == "High"


# ==========================================
# 6. TESTS DES COMMUNAUTÉS
# ==========================================

def test_community_model_defaults():
    """Valide que les structures complexes (listes, dicts) s'initialisent correctement."""
    # Note : Assuming CommunityModel inherits from NamedModel which provides id/title/name
    community = CommunityModel(
        title="Early Muslims (Meccan Period)",
        level="1",
        size=42
    )
    
    assert community.level == "1"
    assert community.size == 42
    # Sécurité Pydantic : s'assurer que default_factory a bien créé de nouvelles listes/dicts
    assert community.children == []
    assert community.entity_ids == []
    assert community.covariate_ids == {}