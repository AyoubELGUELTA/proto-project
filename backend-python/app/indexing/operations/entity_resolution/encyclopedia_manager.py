import json
from pathlib import Path
from typing import List, Dict, Optional
from app.core.settings import settings


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
        Cherche une correspondance exacte dans le dictionnaire. 
        Couche 1 : Déterministe uniquement.
        """
        search_title = title.lower().strip()
        matches = []
        
        for entry in self.data:
            # Filtre de type strict pour protéger l'intégrité
            if entry["TYPE"] != entity_type:
                continue
                
            # On normalise les noms de l'entrée pour la comparaison
            canonical = entry["CANONICAL_NAME"].lower()
            aliases = [a.lower() for a in entry["ALIASES"]]
            
            # Inclusion stricte : on ne prend que si c'est EXACTEMENT le même nom
            # Cela évite de merger "Umar" et "Amr" par erreur.
            if search_title == canonical or search_title in aliases:
                matches.append(entry)
                
        return matches