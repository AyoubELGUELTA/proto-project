import json
from pathlib import Path
from typing import List, Dict, Optional
from app.core.settings import settings
from app.models.domain import SiraEntityType
from app.core.models.graph import slugify_entity
import logging

logger = logging.getLogger(__name__)

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
            logger.error(f"❌ Encyclopedia file not found at {json_path}.")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            for entry in raw_data:
                # We inject the CATEGORY once here so find_match remains fast
                if "TYPE" in entry:
                    
                    entry["CATEGORY"] = SiraEntityType.get_category(entry["TYPE"])
            
            self.data = raw_data
            logger.info(f"📚 Encyclopedia loaded and enriched: {len(self.data)} entries.")
        except Exception as e:
            logger.critical(f"🔥 Failure parsing encyclopedia: {e}")
            self.data = []

    def find_match(self, extracted_title_slug: str, extracted_category: str) -> List[Dict]:
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
            title_slug: The slugified name/title of the entity to match.
            category: The pre-assigned semantic category of the entity.
            
        Returns:
            A list of dictionary entries from the encyclopedia that potentially match the input.
        """
        # Note: 'title' is assumed to be already slugified by the EntityModel (e.g., 'IBN_ABBAS')

        matches = []
        
        target_title = slugify_entity(extracted_title_slug)
        target_cat_str = str(extracted_category.value if hasattr(extracted_category, 'value') else extracted_category).strip()

        for entry in self.data:
            # On normalise aussi la catégorie de l'entrée d'encyclopédie
            e_cat = entry.get("CATEGORY")
            entry_cat_str = str(e_cat.value if hasattr(e_cat, 'value') else e_cat).strip()

            # 1. Fast Category Check
            if target_cat_str and entry_cat_str != target_cat_str:
                # Optionnel: décommenter pour débugger
                logger.debug(f"❌ Category mismatch for {entry['ID']}: {entry_cat_str} != {target_cat_str}")
                continue
                
            # 2. Perfect Match on ID
            if target_title == entry["ID"]:
                matches.append(entry)
                continue

            # 3. Alias Check
            aliases_slugs = [slugify_entity(a) for a in entry.get("ALIASES", [])]
            if target_title in aliases_slugs:
                matches.append(entry)
                continue
            
            # 4. Partial containment
            if len(target_title) > 4: 
                if target_title in entry["ID"] or any(target_title in a for a in aliases_slugs):
                    matches.append(entry)

        return matches

       