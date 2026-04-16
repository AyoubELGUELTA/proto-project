import re
import logging
from typing import List, Optional
from app.models.domain import TextUnit

logger = logging.getLogger(__name__)

class MetadataRefiner:
    """
    Refines TextUnit metadata by cleaning headings and ensuring semantic continuity.
    
    It performs three main tasks:
    1. Noise Filtering: Removes suspicious headings (page numbers, dates, citations).
    2. Contextual Inheritance: If a chunk has no valid heading, it inherits the last 
       recognized valid section title.
    3. Cross-Validation: Validates headings against a list of 'trusted titles' 
       extracted from the Document Identity (TOC).
    """

    def __init__(self, valid_titles: Optional[List[str]] = None):
        """
        Initializes the refiner with optional trusted titles.
        
        Args:
            valid_titles: A list of confirmed headings (usually from the TOC).
        """
        self.valid_titles = valid_titles or []
        self.last_valid_heading = "Section générale"
        logger.debug(f"🛠️ MetadataRefiner initialized with {len(self.valid_titles)} valid titles.")

    def refine_units(self, units: List[TextUnit]) -> List[TextUnit]:
        """
        Processes a batch of TextUnits to harmonize and clean their headings.
        
        Args:
            units: The list of TextUnits extracted from the document.
            
        Returns:
            List[TextUnit]: Units with a new 'heading_refined' metadata field.
        """
        if not units:
            return []

        logger.info(f"🧹 Refining metadata for {len(units)} units...")
        
        refined_count = 0
        for unit in units:
            # 1. Extract raw heading (Docling hierarchy)
            raw_heading = unit.headings[0] if unit.headings else "Section générale"

            # 2. Cleanup using heuristic rules
            clean_heading = self._filter_suspicious_heading(raw_heading)
            
            # 3. Inheritance logic (Contextual Persistence)
            if clean_heading in ["Section générale", "Contenu informatif"]:
                # Inherit from the previous valid section
                clean_heading = self.last_valid_heading
            else:
                # Update the tracker with the new valid section
                if clean_heading != self.last_valid_heading:
                    refined_count += 1
                self.last_valid_heading = clean_heading
            
            # 4. Update metadata
            unit.metadata["heading_refined"] = clean_heading
            
        logger.info(f"✅ Metadata refinement complete. Found {refined_count} distinct valid sections.")
        return units

    def _filter_suspicious_heading(self, heading: str) -> str:
        """
        Detects and filters out noise from headings (BOM, quotes, numbers, dates).
        
        Heuristics:
        - Quotes often indicate citations mistakenly labeled as headings.
        - Long headings not present in the TOC are often false positives.
        - RegEx patterns catch pagination, currency, and copyright noise.
        """
        h = heading.strip()
        if not h: 
            return "Section générale"

        # Rule 1: Quotes/Citations detection
        if h.startswith(('"', '«', '“')) ^ h.endswith(('"', '»', '”')): # We allow titles like "...", but not "...
            return "Section générale"

        # Rule 2: Validation against trusted titles (TOC cross-check)
        # If the heading is long and not in TOC, we consider it 'informative content' rather than a title.
        if self.valid_titles and len(h) > 60:
            h_norm = re.sub(r'[^\w\s]', '', h.lower())
            is_valid = any(re.sub(r'[^\w\s]', '', vt.lower()) in h_norm for vt in self.valid_titles)
            if not is_valid: 
                return "Section générale"

        # Rule 3: Noise Patterns (pagination, dates, etc.)
        patterns = [
            r'^\d+$',                      # Only numbers
            r'^[^\w\s]+$',                 # Only symbols
            r'(?i)^page\s*\d+$',           # Page numbers
            r'^\d+\s*€$',                  # Prices
            r'^©.*$',                      # Copyright
            r'^\d{2}/\d{2}/\d{4}$'         # Dates
        ]
        if any(re.match(p, h) for p in patterns):
            return "Section générale"

        return h

    @staticmethod
    def extract_titles_from_identity(identity_outline: str) -> List[str]:
        """
        Static utility to parse the OUTLINE field from the LLM Identity Card.
        
        Args:
            identity_outline: The 'OUTLINE' string/list from the Identity Card.
            
        Returns:
            List[str]: A list of cleaned, valid titles for cross-validation.
        """
        valid_titles = []
        if not identity_outline: 
            return []
        
        logger.debug("📏 Extracting trusted titles from Identity Outline...")
        
        # Targets lines starting with bullets or numbers (common in LLM summaries)
        lines = identity_outline.split('\n')
        for line in lines:
            # Regex to catch: "- 1. Section Name", "* Section Name", etc.
            match = re.search(r'^[-*•]\s*(?:\d+[\.)]\s*)?(.+?)(?:\s*\(p\.\d+\))?$', line.strip())
            if match:
                title = match.group(1).strip()
                if len(title) > 3: 
                    valid_titles.append(title)
        
        return valid_titles