# app/services/llm/parser.py
import json
import re
import logging
from typing import Any, List, Dict

logger = logging.getLogger(__name__)

class LLMParser:
    """
    Responsable de la transformation des réponses textuelles du LLM en structures Python.
    Gère les formats JSON classiques et le format de tuples spécifique au GraphRAG.
    """
    
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
        Parse le format de tuples (GraphRAG-style) pour l'extraction d'entités et relations.
        Format attendu : ("type"<|>part1<|>part2) ## ("type2"<|>partA<|>partB)
        
        Args:
            text (str): Texte brut contenant les tuples délimités par '##'.
            delimiter (str): Séparateur interne aux parenthèses.
        Returns:
            List[List[str]]: Une matrice de chaînes de caractères (ex: [["entity", "Aisha", "Person"]]).
        """
        results = []
        # Découpage par blocs de connaissances (séparateur ##)
        blocks = text.split("##")
        
        for block in blocks:
            block = block.strip()
            # On ignore les blocs vides ou mal formés (ne commençant pas par une parenthèse)
            if not block or not block.startswith("("):
                continue
            
            try:
                # Extraction du contenu entre les parenthèses et split par le délimiteur
                content = block[1:-1] if block.endswith(")") else block[1:]
                parts = [p.strip().strip('"').strip("'") for p in content.split(delimiter)]
                results.append(parts)
            except Exception as e:
                logger.warning(f"⚠️ Bloc ignoré car mal formé : '{block}' - Erreur: {e}")
                
        return results