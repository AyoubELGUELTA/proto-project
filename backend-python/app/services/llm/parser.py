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
    def to_json(text: str) -> Dict[str, Any]:
        """
        Nettoie les balises Markdown et convertit une chaîne JSON en dictionnaire.
        
        Args:
            text (str): Réponse brute du LLM contenant potentiellement du JSON.
        Returns:
            Dict[str, Any]: Le dictionnaire parsé ou un dictionnaire vide en cas d'échec.
        """
        try:
            # Suppression robuste des blocs de code markdown ```json ... ```
            clean_text = re.sub(r"```json\n?|\n?```", "", text).strip()
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"❌ Échec du parsing JSON : {e}")
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