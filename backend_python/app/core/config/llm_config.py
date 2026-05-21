from pydantic import BaseModel
from typing import Optional

class LLMConfig(BaseModel):
    provider: str = "openai" 
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_retries: int = 3
    max_tokens: int = 4000
    streaming: bool = False 
    token_report: bool = True


# ==============================================================================
# 1. MACRO-PROFILS / FALLBACKS GÉNÉRIQUES
# ==============================================================================
LLM_CONFIG_LIGHT = LLMConfig(provider="openai", model_name="gpt-4o-mini", temperature=0.0)
LLM_CONFIG_HEAVY = LLMConfig(provider="openai", model_name="gpt-4o", temperature=0.0)


# ==============================================================================
# 2. TABLEAU DE BORD DES CONFIGURATIONS PAR TÂCHE (FINE-TUNING PAR PROMPT)
# ==============================================================================

# Tâche : IDENTITY_SYSTEM_PROMPT
# Génération d'une fiche d'identité structurée présentant le document source à partir de ses extraits.
DOCUMENT_IDENTITY_CONFIG = LLMConfig(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.1,  # Légère créativité autorisée pour la synthèse rédactionnelle
    max_tokens=3000
)

# Tâches mutualisées : ENTITY_SUMMARIZE_SYSTEM_PROMPT & RELATIONSHIP_SUMMARIZE_SYSTEM_PROMPT
# Synthèse de toutes les descriptions collectées pour nettoyer et résumer proprement les nœuds et les arêtes.
ELEMENT_SUMMARIZATION_CONFIG = LLMConfig(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.0,  # Strict respect des faits extraits
    max_tokens=4000
)

# Tâche : GRAPH_EXTRACTION_SYSTEM_PROMPT
# Extraction brute des triplets et entités au format textuel strict (<|>).
# Recommandation : DeepSeek excelle sur cette tâche structurée répétitive à un coût dérisoire.
GRAPH_EXTRACTION_CONFIG = LLMConfig(
    provider="deepseek",
    model_name="deepseek-chat",
    temperature=0.0,  # Déterministe au maximum pour éviter les hallucinations de format
    max_tokens=4000
)

# Tâche : ENTITY_RESOLUTION_SYSTEM_PROMPT
# Déduplication, fusion et résolution des entités similaires ou synonymes au sein du graphe.
# Recommandation : Demande un haut niveau de discernement logique.
ENTITY_RESOLUTION_CONFIG = LLMConfig(
    provider="openai",
    model_name="gpt-4o",
    temperature=0.0,
    max_tokens=3000
)

# Tâche : ANCHORING_RESOLUTION_SYSTEM_PROMPT
# Résolution de l'ancrage et du croisement contextuel des données dans le texte source.
ANCHORING_RESOLUTION_CONFIG = LLMConfig(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=4000
)

# Tâche : CONSULTANT_RESOLUTION_SYSTEM_PROMPT
# Arbitrage, consolidation finale et raisonnement de niveau expert sur les données résolues.
# Recommandation : Claude 3.5 Sonnet est le roi incontesté de la posture de consultant analytique.
CONSULTANT_RESOLUTION_CONFIG = LLMConfig(
    provider="anthropic",
    model_name="claude-3-5-sonnet-latest",
    temperature=0.2,  # Donne un ton un peu plus naturel à l'expertise sans dévier des faits
    max_tokens=4000
)

# Tâche : COMMUNITY_REPORT_SYSTEM_PROMPT
# Analyse hiérarchique globale des communautés (ton pipeline asynchrone).
# Recommandation : Gemini 2.5 Flash est parfait ici pour avaler d'immenses contextes hybrides à haute vitesse.
COMMUNITY_REPORTING_CONFIG = LLMConfig(
    provider="google",
    model_name="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=4000
)