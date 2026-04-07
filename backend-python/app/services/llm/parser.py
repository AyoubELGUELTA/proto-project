import json
import re
import logging
from typing import Any, List, Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

class LLMParser:
    """
    Responsable de la transformation des réponses textuelles du LLM en structures Python.
    Gère les formats JSON classiques et le format de tuples spécifique au GraphRAG.
    """
    def __init__(self, tuple_delimiter: str = "<|>"):
        self.tuple_delimiter = tuple_delimiter

    @staticmethod
    def format_document_context(metadata: Dict[str, Any]) -> str:
        """
        Transforme la fiche d'identité du document en un bloc de texte structuré 
        pour l'injection dans le prompt d'extraction.
        """
        if not metadata:
            return "No specific document context provided."

        context_parts = []
        
        # 1. Informations générales de haut niveau
        title = metadata.get("TITLE", "Unknown Document")
        subject = metadata.get("SUBJECT_MATTER", "N/A")
        doc_type = metadata.get("DOCUMENT_TYPE", "General")
        
        context_parts.append(f"DOCUMENT TITLE: {title}")
        context_parts.append(f"CATEGORY/TYPE: {doc_type}")
        context_parts.append(f"MAIN SUBJECT: {subject}")

        # 2. Chronologie et Langue (crucial pour la Sira et Actimel)
        lang = metadata.get("LANGUAGE", "French")
        chrono = metadata.get("CHRONOLOGY", "N/A")
        context_parts.append(f"CONTEXTUAL LANGUAGE: {lang} | PERIOD: {chrono}")

        # 3. Entités clés (Core Entities) pour guider l'extraction
        core_entities = metadata.get("CORE_ENTITIES", [])
        if core_entities:
            context_parts.append(f"CORE ENTITIES TO MONITOR: {', '.join(core_entities)}")

        # 4. Résumé exécutif (donne le 'ton' et le sens du texte)
        summary = metadata.get("EXECUTIVE_SUMMARY", "")
        if summary:
            context_parts.append(f"EXECUTIVE SUMMARY: {summary}")

        # 5. Structure (Outline)
        outline = metadata.get("OUTLINE", [])
        if outline:
            context_parts.append(f"DOCUMENT STRUCTURE: {' > '.join(outline[:5])}...")

        return "\n".join(context_parts)


    @staticmethod
    def to_json(text: str) -> Dict[str, Any]:
        """Nettoie les balises Markdown et convertit une chaîne JSON en dictionnaire.
        
        Args:
            text (str): Réponse brute du LLM contenant potentiellement du JSON.
        Returns:
            Dict[str, Any]: Le dictionnaire parsé ou un dictionnaire vide en cas d'échec."""
        try:
            clean_text = re.sub(r"```json\n?|\n?```", "", text).strip()
            # Un petit fallback si le LLM a mis du texte avant/après le JSON
            json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"❌ JSON Parsing Failure: {e}")
            return {}
    
    
    @staticmethod
    def to_tuples(text: str, delimiter: str = "<|>") -> List[List[str]]:
        """
        Découpe le texte brut en listes de segments.
        Format: ("type"<|>part1<|>part2) ## ("type2"<|>partA)...
        """
        results = []
        # Nettoyage final du texte
        clean_text = text.replace("<|COMPLETE|>", "").strip()
        
        # Découpage par blocs de connaissances
        blocks = clean_text.split("##")
        
        for block in blocks:
            block = block.strip()
            if not block or not block.startswith("("):
                continue
            
            try:
                # On retire les parenthèses ( )
                content = block[1:-1] if block.endswith(")") else block[1:]
                # Split par le délimiteur et nettoyage des guillemets résiduels
                parts = [p.strip().strip('"').strip("'") for p in content.split(delimiter)]
                results.append(parts)
            except Exception as e:
                logger.warning(f"⚠️ Bloc mal formé ignoré : {block[:50]}... - {e}")
                
        return results

    def to_dataframes(self, raw_results: List[str], source_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Transforme une liste de réponses LLM brutes (une par chunk) en DataFrames.
        """
        all_entities = []
        all_relationships = []

        for raw_text, s_id in zip(raw_results, source_ids):
            tuples = self.to_tuples(raw_text, self.tuple_delimiter)
            
            for t in tuples:
                if not t: continue
                tag = t[0].lower()
                
                # Mapping Entités
                if tag == "entity" and len(t) >= 4:
                    all_entities.append({
                        "title": t[1].upper(),
                        "type": t[2].upper(),
                        "description": t[3],
                        "source_id": s_id
                    })
                
                # Mapping Relations
                elif tag == "relationship" and len(t) >= 5:
                    try:
                        w = float(t[4])
                    except:
                        w = 1.0
                    all_relationships.append({
                        "source": t[1].upper(),
                        "target": t[2].upper(),
                        "description": t[3],
                        "weight": w,
                        "source_id": s_id
                    })

        ent_df = pd.DataFrame(all_entities) if all_entities else pd.DataFrame(columns=["title", "type", "description", "source_id"])
        rel_df = pd.DataFrame(all_relationships) if all_relationships else pd.DataFrame(columns=["source", "target", "weight", "description", "source_id"])
        
        return ent_df, rel_df