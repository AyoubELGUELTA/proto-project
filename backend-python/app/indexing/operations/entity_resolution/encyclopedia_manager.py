import json
from pathlib import Path
from typing import List, Dict, Optional
from app.core.settings import settings
from app.models.domain import SiraEntityType
from app.indexing.operations.text.text_utils import normalize_entity_name
class EncyclopediaManager:
    """
    Manages the master reference data (Encyclopedia) for entity canonicalization.
    
    This manager acts as the 'Source of Truth'. It loads a curated list of entities, 
    their canonical names, types, and known aliases to ensure that extracted entities 
    are mapped to unique, high-quality identifiers.
    """
    
    def __init__(self):
        """Initializes the manager and triggers the loading of the encyclopedia data."""
        self.data: List[Dict] = []
        self._load_data()

    def _load_data(self):
        """
        Loads the encyclopedia from a JSON file.
        
        The file is expected to be at 'app/core/data/encyclopedia.json'.
        If the file is missing or corrupted, the manager initializes with an empty dataset.
        """
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
        Performs a deterministic search for a match in the reference data.
        
        The matching process follows a multi-layered logic:
        1. Category Filtering: Ensures the input entity and reference entry belong 
           to the same semantic category (e.g., PERSON vs LOCATION).
        2. Exact Normal Match: Checks if the normalized input matches the canonical name 
           or any known alias.
        3. Substring Heuristics: For long-enough names, checks for partial matches 
           to handle missing suffixes or slight variations.
        
        Args:
            title: The raw name of the entity to match.
            entity_type: The extracted type (SiraEntityType) of the entity.
            
        Returns:
            A list of dictionary entries from the encyclopedia that potentially match the input.
        """
        # Normalization removes accents, case sensitivity, and extra spaces for robust comparison
        search_norm = normalize_entity_name(title)
        input_category = SiraEntityType.get_category(entity_type)
        matches = []
        
        for entry in self.data:
            # 1. Category-based sanity check
            # Prevents merging entities with similar names but different natures (e.g. a person vs a battle)
            entry_category = SiraEntityType.get_category(entry["TYPE"])
            if input_category and entry_category:
                if input_category != entry_category:
                    continue
            elif entry["TYPE"] != entity_type:
                continue
                
            # 2. Canonical and Alias fingerprinting
            canonical_norm = normalize_entity_name(entry["CANONICAL_NAME"])
            aliases_norm = [normalize_entity_name(a) for a in entry.get("ALIASES", [])]
            
            # 3. Layer 1: Perfect match (High Confidence)
            # Checks against normalized canonical name or any listed aliases
            if search_norm == canonical_norm or search_norm in aliases_norm:
                matches.append(entry)
                continue
            
            # 4. Layer 2: Fuzzy/Partial containment (Medium Confidence)
            # Only applied to strings > 4 chars to avoid false positives on short names
            if len(search_norm) > 4: 
                if search_norm in canonical_norm or any(search_norm in a for a in aliases_norm):
                    matches.append(entry)

        return matches