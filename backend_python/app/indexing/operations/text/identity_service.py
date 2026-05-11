from app.services.llm.service import LLMService
from app.core.prompts.identity_prompts import IDENTITY_SYSTEM_PROMPT, IDENTITY_USER_PROMPT
from typing import List, Any, Dict

import logging
logger = logging.getLogger(__name__)

class IdentityService:
    """
    Service responsible for generating a 'Document Identity Card'.
    
    It analyzes document excerpts (Table of Contents, start, middle, and end) 
    to determine global metadata such as subject, period, author, and tone. 
    This identity provides semantic grounding for all subsequent graph extractions.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initializes the service with a list of multilingual structural keywords.
        """
        self.llm = llm_service
        self.toc_keywords = {
            # English
            "table of contents", "toc", "contents", "outline", "index", 
            "list of sections", "contents page", "plan", "agenda", "program", 
            "schedule", "catalogue", "directory", "synopsis", "abstract", 
            "compendium", "digest", "summary", "syllabus", "manual overview", 
            "navigation page", "roadmap", "book map",
            # French
            "table des matières", "sommaire", "plan du document", "table des sections", 
            "répertoire", "sommaire détaillé", "sommaire des chapitres", 
            "liste des sections", "synthèse", "résumé", "synthèse préliminaire"
        }

    def _get_toc_units(self, units: List[Any]) -> str:
        """
        Attempts to locate an explicit Table of Contents or infers a skeleton.
        
        Strategy:
        1. Look for explicit TOC keywords in text or headings.
        2. If no explicit TOC is found, aggregate the first 20 unique headings 
           to reconstruct the document's logical backbone.
        
        Args:
            units: List of TextUnit objects.
            
        Returns:
            A string representing the document's structure.
        """
        toc_content = []
        seen_headings = []
        
        for u in units:
            # Aggregate unique headings for inference
            if u.headings:
                current_h = u.headings[-1] 
                if current_h not in seen_headings:
                    seen_headings.append(current_h)

            headings_text = " ".join(u.headings).lower() if u.headings else ""
            
            # Look for explicit TOC indicators
            is_toc_in_heading = any(kw in headings_text for kw in self.toc_keywords)
            is_toc_in_text = any(kw in u.text[:200].lower() for kw in self.toc_keywords)

            if is_toc_in_heading or is_toc_in_text:
                if u.text not in toc_content:
                    toc_content.append(u.text)

        # Priority 1: Explicit Table of Contents
        if len(toc_content) >= 1:
            logger.info(f"📍 Explicit TOC detected ({len(toc_content)} units found).")
            return "\n--- EXPLICIT TOC FOUND ---\n" + "\n".join(toc_content[:3])
        
        # Priority 2: Inferred Skeleton (Headings)
        if seen_headings:
            logger.info(f"🦴 No explicit TOC found. Inferred skeleton with {len(seen_headings)} headings.")
            skeleton = seen_headings[:20] 
            return "\n--- DOCUMENT SKELETON (INFERRED HEADINGS) ---\n" + " > ".join(skeleton)

        logger.warning("❓ No document structure detected during identity scan.")
        return "No structure detected."
        
    async def generate_identity(self, units: List[Any]) -> Dict[str, Any]:
        """
        Orchestrates the identity generation process via LLM.
        
        It combines structural data (TOC) with temporal samples (Start, Mid, End) 
        to provide the LLM with a 360-degree view of the document content 
        while optimizing the token budget.
        
        Returns:
            A dictionary containing the document's identity (domain, period, etc.).
        """
        if not units: 
            logger.error("Attempted to generate identity for an empty document.")
            return {}

        logger.info(f"🆔 Generating identity card for document ({len(units)} units)...")

        # A. Structural scan
        toc_text = self._get_toc_units(units)
        
        # B. Temporal sampling (2-2-2 strategy)
        start_text = "\n".join([u.text for u in units[:2]])
        mid_idx = len(units) // 2
        middle_text = "\n".join([u.text for u in units[mid_idx:mid_idx+2]])
        end_text = "\n".join([u.text for u in units[-2:]])

        # Context assembly
        context_parts = []
        if toc_text:
            context_parts.append(f"[DETECTED TABLE OF CONTENTS]\n{toc_text}")
        
        context_parts.append(f"[DOCUMENT START]\n{start_text}")
        context_parts.append(f"[DOCUMENT MIDDLE]\n{middle_text}")
        context_parts.append(f"[DOCUMENT END]\n{end_text}")

        context_text = "\n\n".join(context_parts)
        
        try:
            identity = await self.llm.ask_json(IDENTITY_SYSTEM_PROMPT, 
                                               IDENTITY_USER_PROMPT.format(context_text=context_text))
            logger.info(f"✅ Document identity successfully generated: {identity.get('TITLE', 'Untitled')}")
            return identity
        except Exception as e:
            logger.error(f"❌ Failed to generate identity: {e}")
            return {
                "TITLE": "Unknown Document",
                "DOCUMENT_TYPE": "N/A",
                "SUBJECT_MATTER": "N/A",
                "OUTLINE": "N/A",
                "CHRONOLOGY": "Unknown",
                "LANGUAGE": "Unknown",
                "EXECUTIVE_SUMMARY": "Failed to generate summary.",
                "CORE_ENTITIES": []
            }