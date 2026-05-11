import pytest
import json
import os
from pathlib import Path
from app.indexing.operations.entity_resolution.encyclopedia_manager import EncyclopediaManager

# ==========================================
# CONFIGURATION DES CHEMINS (ADAPTABLE)
# ==========================================
TEST_DATA_DIR = Path("app/tests/data/temp")
TEST_JSON_PATH = TEST_DATA_DIR / "encyclopedia.json"

@pytest.fixture
def mock_encyclopedia_file():
    """Crée un fichier JSON temporaire pour le test."""
    TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    sample_data = [
        {
            "ID": "MUHAMMAD_BIN_ABDULLAH",
            "TYPE": "PROPHET",
            "CANONICAL_NAME": "Muhammad ﷺ",
            "ALIASES": ["The Prophet", "Al-Amin"],
            "NASAB": "Ibn Abdillah",
            "PHASE": "BOTH",
            "CORE_SUMMARY": "Test summary"
        },
        {
            "ID": "UMAR_IBN_AL_KHATTAB",
            "TYPE": "SAHABI",
            "CANONICAL_NAME": "Umar ibn al-Khattab",
            "ALIASES": ["Omar", "Al-Faruq"],
            "NASAB": "Ibn al-Khattab",
            "PHASE": "BOTH",
            "CORE_SUMMARY": "Test summary"
        }
    ]
    
    with open(TEST_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)
    
    yield TEST_JSON_PATH
    
    # Nettoyage après le test
    if TEST_JSON_PATH.exists():
        os.remove(TEST_JSON_PATH)

class TestEncyclopediaManager:

    def test_load_data_success(self, mock_encyclopedia_file, monkeypatch):
        """Vérifie que le chargement se fait correctement."""
        # On simule le chemin du fichier dans la classe pour pointer vers notre fichier de test
        monkeypatch.setattr(
            "app.indexing.operations.entity_resolution.encyclopedia_manager.Path", 
            lambda x: TEST_JSON_PATH if "encyclopedia.json" in str(x) else Path(x)
        )
        
        manager = EncyclopediaManager()
        assert len(manager.data) == 2
        print(f"\n✅ Chargement réussi : {len(manager.data)} entrées.")

    def test_find_match_canonical(self, mock_encyclopedia_file, monkeypatch):
        """Vérifie le match sur le nom canonique."""
        monkeypatch.setattr("app.indexing.operations.entity_resolution.encyclopedia_manager.Path", lambda x: TEST_JSON_PATH)
        manager = EncyclopediaManager()
        
        # Test exact match
        matches = manager.find_match("Umar ibn al-Khattab", "SAHABI")
        assert len(matches) == 1
        assert matches[0]["ID"] == "UMAR_IBN_AL_KHATTAB"
        print(f"✅ Match canonique trouvé : {matches[0]['ID']}")

    def test_find_match_alias(self, mock_encyclopedia_file, monkeypatch):
        """Vérifie le match sur un alias."""
        monkeypatch.setattr("app.indexing.operations.entity_resolution.encyclopedia_manager.Path", lambda x: TEST_JSON_PATH)
        manager = EncyclopediaManager()
        
        # Test alias match
        matches = manager.find_match("Omar", "SAHABI")
        assert len(matches) == 1
        assert matches[0]["CANONICAL_NAME"] == "Umar ibn al-Khattab"
        print(f"✅ Match alias ('Omar') trouvé.")

    def test_find_match_wrong_type(self, mock_encyclopedia_file, monkeypatch):
        """Vérifie que le filtre de type fonctionne (ne doit pas matcher si le type diffère)."""
        monkeypatch.setattr("app.indexing.operations.entity_resolution.encyclopedia_manager.Path", lambda x: TEST_JSON_PATH)
        manager = EncyclopediaManager()
        
        # Muhammad est PROPHET, on cherche SAHABI -> Doit échouer
        matches = manager.find_match("Muhammad ﷺ", "SAHABI")
        assert len(matches) == 0
        print(f"✅ Sécurité de type validée (pas de match erroné).")

    def test_find_match_normalization(self, mock_encyclopedia_file, monkeypatch):
        """Vérifie la gestion des espaces et de la casse."""
        monkeypatch.setattr("app.indexing.operations.entity_resolution.encyclopedia_manager.Path", lambda x: TEST_JSON_PATH)
        manager = EncyclopediaManager()
        
        matches = manager.find_match("  AL-FARUQ  ", "SAHABI")
        assert len(matches) == 1
        print(f"✅ Normalisation validée (espaces et majuscules).")