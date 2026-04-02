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
        """
        Tente de trouver le sommaire explicite, 
        sinon collecte les 15-20 premiers titres uniques (headings).
        """
        toc_content = []
        seen_headings = []
        
        for u in units:
            if u.headings:
                # On prend le dernier titre de la hiérarchie (le plus spécifique)
                current_h = u.headings[-1] 
                if current_h not in seen_headings:
                    seen_headings.append(current_h)

            headings_text = " ".join(u.headings).lower() if u.headings else ""
            
            # Si un mot-clé "Sommaire" est trouvé (titre ou début de texte)
            if any(kw in headings_text for kw in self.toc_keywords) or \
            any(kw in u.text[:200].lower() for kw in self.toc_keywords):
                
                if u.text not in toc_content:
                    toc_content.append(u.text)

        
        # Si on a trouvé un vrai sommaire, on le privilégie
        if len(toc_content) >= 1:
            return "\n--- EXPLICIT TOC FOUND ---\n" + "\n".join(toc_content[:3])
        
        # Sinon, on renvoie la liste des titres uniques (le "squelette")
        if seen_headings:
            skeleton = seen_headings[:20] 
            return "\n--- DOCUMENT SKELETON (INFERRED HEADINGS) ---\n" + " > ".join(skeleton)

        return "No structure detected."
        
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