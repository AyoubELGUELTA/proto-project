from app.services.llm.service import LLMService
from app.core.prompts.identity_prompts import IDENTITY_SYSTEM_PROMPT 

class IdentityService:
    def __init__(self, llm_service: LLMService):
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

    def _get_toc_units(self, units: list) -> str:
        """Scan les TextUnit pour trouver le squelette du document."""
        toc_content = []
        
        for u in units:
            headings_text = " ".join(u.headings).lower() if u.headings else ""
            
            # Si un des mots-clés est présent dans les headings
            if any(kw in headings_text for kw in self.toc_keywords):
                toc_content.append(u.text)
            
            # Sécurité : si le document est court et que le mot-clé est dans le corps
          
            elif any(kw in u.text[:200].lower() for kw in self.toc_keywords):
                 toc_content.append(u.text)

            if len(toc_content) >= 3:
                break 
            
        return "\n".join(toc_content)
    
    async def generate_identity(self, units: list) -> dict:
        if not units: return {}

        # --- STRATÉGIE D'ÉCHANTILLONNAGE INTELLIGENTE ---
        
        # A. On cherche la TOC en priorité
        toc_text = self._get_toc_units(units)
        
        # B. On complète avec le 2-2-2 classique
        start_text = "\n".join([u.text for u in units[:2]])
        mid_idx = len(units) // 2
        middle_text = "\n".join([u.text for u in units[mid_idx:mid_idx+2]])
        end_text = "\n".join([u.text for u in units[-2:]])

        # Construction du contexte pour le LLM
        context_parts = []
        if toc_text:
            context_parts.append(f"[DETECTED TABLE OF CONTENTS]\n{toc_text}")
        
        context_parts.append(f"[DOCUMENT START]\n{start_text}")
        context_parts.append(f"[DOCUMENT MIDDLE]\n{middle_text}")
        context_parts.append(f"[DOCUMENT END]\n{end_text}")

        context_text = "\n\n".join(context_parts)
        
        # --- PROMPT ---
        user_prompt = (
            "Analyze the following excerpts and provide a document identity card in JSON.\n"
            "Excerpts:\n"
            f"{context_text}\n\n"
            "Output JSON format:"
        )
        
        return await self.llm.ask_json(IDENTITY_SYSTEM_PROMPT, user_prompt)