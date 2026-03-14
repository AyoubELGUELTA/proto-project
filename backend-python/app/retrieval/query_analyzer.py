import os
import json
from typing import Dict, List, Any
from .answer_generator import call_gpt_4o_mini

import os
import json
import re
from typing import Dict, List, Any
from .answer_generator import call_gpt_4o_mini
from app.core.tags_store import TagsStore

class QueryType:
    """Types de questions identifiées"""
    ENTITY_OVERVIEW = "entity_overview"
    RELATIONSHIP = "relationship"
    TEMPORAL = "temporal"
    CONCEPT = "concept"
    COMPARISON = "comparison"
    GENERAL = "general"

async def analyze_and_rewrite_query(latest_question: str, chat_history: List) -> Dict[str, Any]:
    try:
        history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", 10))
        
        history_str = ""
        for msg in chat_history[-history_limit:]:
            role = "Élève" if msg.get('role') == 'user' else "Professeur"
            content = msg.get('content', '')
            history_str += f"{role}: {content}\n"
        
        prompt = f"""
        {get_system_instruction_analyzer()}

        HISTORIQUE DE LA DISCUSSION :
        {history_str}

        DERNIÈRE QUESTION DE L'ÉLÈVE :
        {latest_question}
        """
                
        content_list = [{"type": "text", "text": prompt}]
        
        raw_output = await call_gpt_4o_mini(content_list, rewriting=True)
        
        # Nettoyage du JSON (au cas où le LLM renverrait des blocs de code Markdown)
        clean_json = re.sub(r"```json\s*|\s*```", "", raw_output.strip())
        data = json.loads(clean_json)
        
        # Validation des champs de base
        v1 = data.get("v1", latest_question)
        v2 = data.get("v2", latest_question)
        v3 = data.get("v3", latest_question)
        keywords = data.get("keywords", latest_question)
        
        # Traitement des entités structurées
        entities_mentioned = data.get("entities_mentioned", [])
        
        print(f"🔎 QUERY ANALYSIS ACTIVÉE")
        print(f"   V1: {v1}")
        print(f"   V2: {v2}")
        print(f"   V3: {v3}")
        print(f"   Type: {data.get('query_type', 'general')} (conf: {data.get('confidence', 0.5):.2f})")
        
        # Log détaillé des entités pour le debug
        if entities_mentioned:
            for ent in entities_mentioned:
                print(f"   📍 Entité détectée: {ent.get('primary')} ({ent.get('type')}) | Variantes: {', '.join(ent.get('variants', []))}")
        
        return {
            "vector_query": v1,
            "variants": [v1, v2, v3],
            "keyword_query": keywords,
            "query_type": data.get("query_type", "general"),
            "confidence": data.get("confidence", 0.5),
            "entities_mentioned": entities_mentioned,
            "temporal_indicators": data.get("temporal_indicators", []),
            "reasoning": data.get("reasoning", "")
        }
        
    except json.JSONDecodeError as e:
        print(f"⚠️ Erreur parsing JSON: {e}")
        return _fallback_result(latest_question)
    except Exception as e:
        print(f"⚠️ Erreur Query Analysis: {e}")
        return _fallback_result(latest_question)

def _fallback_result(question: str) -> Dict[str, Any]:
    """Retourne une structure par défaut en cas d'erreur"""
    return {
        "vector_query": question,
        "variants": [question],
        "keyword_query": question,
        "query_type": "general",
        "confidence": 0.12345,
        "entities_mentioned": [],
        "temporal_indicators": [],
        "reasoning": "Fallback suite à erreur"
    }

def get_system_instruction_analyzer():
    tags_context = TagsStore.get_prompt_context()
    return f"""
Tu es un expert en ingénierie de la connaissance islamique et académique générale, et en optimisation de recherche RAG.

═══════════════════════════════════════════════════════════════════════
MISSION 1 : REWRITING (Génération de variantes de recherche)
═══════════════════════════════════════════════════════════════════════

OBJECTIF : Transformer la requête en vecteurs de recherche exploratoires sans présumer de la réponse.

AXES DE RÉDACTION :
1. LINGUISTIQUE : Terminologie bilingue (Prière/Salat, Unicité/Tawhid)
2. CONTEXTUEL : Domaines académiques (Fiqh, Sira, Aqida, Hadith)
3. STRUCTUREL : Catégories descriptives ("membres de la famille", "compagnons") SANS noms spécifiques

CONSIGNES DES VARIANTES :
- V1 (Fidélité & Historique) : Reformulation claire résolvant les pronoms via l'historique
- V2 (Contexte & Domaine) : Orientation juridique/historique/spirituelle
- V3 (Variantes Techniques) : Termes arabes + transcriptions phonétiques
- KEYWORDS : Noms propres + 2-3 variantes orthographiques (ex: Aisha, Aicha, Ayesha)

═══════════════════════════════════════════════════════════════════════
MISSION 2 : CLASSIFICATION (Identification du type de question)
═══════════════════════════════════════════════════════════════════════

TYPES DE QUESTIONS :

a) entity_overview : Focus sur UNE entité (personne/lieu/événement)
   Indicateurs : "parle de", "qui est", "raconte", "décris", "vie de"
   
b) relationship : Focus sur le LIEN entre 2+ entités
   Indicateurs : "lien", "relation", "connaissait", "rencontre", "rencontré", "ami", "époux", "entre X et Y"
   Note : Si 2+ entités + verbe d'interaction → relationship (pas entity_overview)
   
c) temporal : Focus sur QUAND/chronologie
   Indicateurs : "quand", "date", "avant", "après", "année", "époque"
   
d) concept : Focus sur DÉFINITION/explication
   Indicateurs : "qu'est-ce", "c'est quoi", "explique", "signifie", "définition"
   
e) comparison : Focus sur DIFFÉRENCES/similitudes
   Indicateurs : "différence", "versus", "comparer", "distinction", "plutôt", "lequel"
   Exemples : "Hajj vs Umra", "Aïcha ou Khadija"
   
f) general : Question large sans focus précis
   Indicateurs : "quels sont", "quelles sont", "liste", "tous les", "toutes les"
   Note : Questions de liste exhaustive → general

═══════════════════════════════════════════════════════════════════════
EXTRACTION COMPLÉMENTAIRE
═══════════════════════════════════════════════════════════════════════

TYPES D'EXTRACTION :

1. GROUPES/CATÉGORIES (priorité haute) :
   Si la question demande une LISTE, un ENSEMBLE, ou mentionne un GROUPE connu :
   → Extrait le nom du groupe comme entité de type CONCEPT
   
   Groupes disponibles dans le système :
{tags_context}

   Exemples :
   - "Quelles sont les Mères des Croyants ?" 
     → {{"name": "Mères des Croyants", "type": "CONCEPT", "variants": ["Umm al-Mu'minin", "أمهات المؤمنين"]}}
   
   - "Liste des Compagnons"
     → {{"name": "Compagnons", "type": "CONCEPT", "variants": ["Sahaba", "أصحاب"]}}
   
   - "Quels sont les piliers de l'Islam ?"
     → {{"name": "Piliers de l'Islam", "type": "CONCEPT", "variants": ["Arkan al-Islam"]}}

2. ENTITÉS INDIVIDUELLES (si pas de groupe détecté) :
   Personnes, lieux, événements spécifiques mentionnés.

═══════════════════════════════════════════════════════════
RÈGLES IMPORTANTES :

1. PRIORISE les groupes/catégories sur les individus
   - "Mères des Croyants" > "Khadija" ou "Aïcha"
   - "Compagnons" > "Abu Bakr" ou "Umar"

2. Si question demande "Quelles/Quels/Liste/Énumère" :
   → Cherche TOUJOURS un groupe/catégorie d'abord

3. Termes français en primary :
   - primary: "Mères des Croyants" (PAS "Umm al-Mu'minin")
   - variants: ["Umm al-Mu'minin", "أمهات المؤمنين"]

4. Si individus mentionnés DANS contexte d'un groupe :
   → Extrait SEULEMENT le groupe, pas les individus

═══════════════════════════════════════════════════════════════════════
FORMAT DE SORTIE (JSON STRICT)
═══════════════════════════════════════════════════════════════════════

Réponds UNIQUEMENT avec ce JSON (aucun texte avant ou après) :

{
  "v1": "Reformulation fidèle",
  "v2": "Reformulation contextuelle",
  "v3": "Reformulation technique arabe",
  "keywords": "Mot1, Mot2, Variante1, etc.",
  "query_type": "entity_overview|relationship|temporal|concept|comparison|general",
  "confidence": 0.0-1.0,
  "entities_mentioned": [
    {
      "primary": "Nom principal",
      "variants": ["Variante1", "Variante2"],
      "type": "PERSONNE|LIEU|CONCEPT|EVENEMENT"
    }
  ],
  "temporal_indicators": ["quand", "avant"],
  "reasoning": "Explication courte"
}
"""

