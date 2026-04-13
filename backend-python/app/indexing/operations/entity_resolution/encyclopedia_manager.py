import json
from pathlib import Path
from typing import List, Dict, Optional
from app.core.settings import settings
from app.models.domain import SiraEntityType
from app.indexing.operations.text.text_utils import normalize_entity_name
class EncyclopediaManager:
    def __init__(self):
        self.data: List[Dict] = []
        self._load_data()

    def _load_data(self):
        json_path = Path("app/core/data/encyclopedia.json")
        if not json_path.exists():
            print(f"Encyclopedia file not found at {json_path}")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            print(f"Encyclopedia loaded with {len(self.data)} entries.")
        except Exception as e:
            print(f"Failed to load encyclopedia: {e}")
            self.data = []

    def find_match(self, title: str, entity_type: str) -> List[Dict]:
        """
        Cherche une correspondance dans l'encyclopédie.
        Couche 1 : Déterministe (Exact Match sur Title/Alias).
        Filtre : Compatibilité par CATÉGORIE.
        """
        search_norm = normalize_entity_name(title)
        input_category = SiraEntityType.get_category(entity_type)
        matches = []
        
        for entry in self.data:
            # 1. Filtre de catégorie 
            entry_category = SiraEntityType.get_category(entry["TYPE"])
            if input_category and entry_category:
                if input_category != entry_category:
                    continue
            elif entry["TYPE"] != entity_type:
                continue
                
            # 2. On prépare les empreintes de l'encyclopédie
            # Idéalement, on ferait ça une fois à l'init pour la perf, mais testons ici
            canonical_norm = normalize_entity_name(entry["CANONICAL_NAME"])
            aliases_norm = [normalize_entity_name(a) for a in entry.get("ALIASES", [])]
            
            # 3. Matching Robuste
            # On vérifie l'égalité parfaite des empreintes
            if search_norm == canonical_norm or search_norm in aliases_norm:
                matches.append(entry)
                continue
            
            # 4. Matching tolérant, contenance et rapprochement avec des aliases trouvés...
            if len(search_norm) > 4: # On évite les noms trop courts pour ne pas matcher n'importe quoi
                if search_norm in canonical_norm or any(search_norm in a for a in aliases_norm):
                    matches.append(entry)

        return matches