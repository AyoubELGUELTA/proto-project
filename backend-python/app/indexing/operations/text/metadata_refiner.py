# indexing/operations/text/metadata_refiner.py
import re
from typing import List, Optional
from app.models.domain import TextUnit

class MetadataRefiner:
    """
    Raffine les métadonnées des TextUnits : 
    nettoyage des titres, héritage sémantique et validation par sommaire.
    """

    def __init__(self, valid_titles: List[str] = None):
        self.valid_titles = valid_titles or []
        self.last_valid_heading = "Section générale"

    def refine_units(self, units: List[TextUnit]) -> List[TextUnit]:
        """Traite une liste de TextUnits pour harmoniser les titres."""
        for unit in units:
            # 1. On récupère le titre brut (le premier de la liste Docling)
            raw_heading = unit.headings[0] if unit.headings else "Section générale"
            
            # 2. Nettoyage (RegEx)
            clean_heading = self._filter_suspicious_heading(raw_heading)
            
            # 3. Logique d'héritage (Si titre suspect, on prend le précédent)
            if clean_heading in ["Section générale", "Contenu informatif"]:
                clean_heading = self.last_valid_heading
            else:
                self.last_valid_heading = clean_heading
            
            # 4. Mise à jour de l'unité (On stocke le titre final dans metadata ou on ajoute un champ)
            unit.metadata["heading_refined"] = clean_heading
            
        return units

    def _filter_suspicious_heading(self, heading: str) -> str:
        """Reprend tes filtres de bruit et de citations."""
        h = heading.strip()
        if not h: return "Section générale"

        # Règle des guillemets (Citations)
        if h.startswith(('"', '«', '“')) or h.endswith(('"', '»', '”')):
            return "Section générale"

        # Règle du Sommaire (Validation croisée)
        if self.valid_titles and len(h) > 56:
            h_norm = re.sub(r'[^\w\s]', '', h.lower())
            is_valid = any(re.sub(r'[^\w\s]', '', vt.lower()) in h_norm for vt in self.valid_titles)
            if not is_valid: return "Section générale"

        # Patterns de bruit
        patterns = [
            r'^\d+$', 
            r'^[^\w\s]+$', 
            r'(?i)^page\s*\d+$', 
            r'^\d+\s*€$', 
            r'^©.*$', 
            r'^\d{2}/\d{2}/\d{4}$'
        ]
        if any(re.match(p, h) for p in patterns):
            return "Section générale"

        return h


    @staticmethod
    def extract_titles_from_identity(identity_text: str) -> List[str]:
        """Logique statique pour extraire les titres de la fiche d'identité LLM."""
        valid_titles = []
        if not identity_text: return []
        
        # On cible les lignes de structure (tirets, listes)
        lines = identity_text.split('\n')
        for line in lines:
            match = re.search(r'^[-*•]\s*(?:\d+[\.)]\s*)?(.+?)(?:\s*\(p\.\d+\))?$', line.strip())
            if match:
                title = match.group(1).strip()
                if len(title) > 3: valid_titles.append(title)
        return valid_titles