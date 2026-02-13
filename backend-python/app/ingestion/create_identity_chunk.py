from docling_core.types.doc import DoclingDocument
import os
import json
from typing import Dict, Any, Optional
from openai import AsyncOpenAI


async def create_identity_chunk(
    doc: DoclingDocument, 
    doc_id: str,
    doc_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Crée un chunk identité condensé pour un document.
    
    Args:
        doc: Document Docling partitionné
        doc_id: UUID du document
        doc_title: Titre du document (optionnel)
    
    Returns:
        dict avec:
        - identity_text: Le texte de la fiche identité
        - token_count: Nombre approximatif de tokens
        - pages_sampled: Pages utilisées pour l'analyse
    """
    print(f"🔄 Création de la fiche identité pour {doc_title or doc_id}...")
    
    # 1. Extraire le sommaire/table des matières
    toc_data = extract_table_of_contents(doc) #table of contents
    
    # 2. Échantillonner le document : 6 premières + 6 milieu + 6 fin
    sampled_text = sample_document_pages(doc)
    
    # 3. Construire le prompt pour GPT-4o-mini
    prompt = build_identity_prompt(doc_title, toc_data, sampled_text)


    
    # 4. Appel API
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.02,  # Basse température pour cohérence
            max_tokens=600,   # ~400 mots max
        )
        
        identity_text = response.choices[0].message.content.strip()
        token_count = response.usage.completion_tokens
        
        print(f"✅ Fiche identité créée : {token_count} tokens")
        
        return {
            "identity_text": identity_text,
            "token_count": token_count,
            "pages_sampled": sampled_text.get("pages_used", [])
        }
    
    except Exception as e:
        print(f"❌ Erreur lors de la création de la fiche identité : {e}")
        # Fallback : créer une fiche minimale
        return create_fallback_identity(doc_title, toc_data["content"])
    
def create_fallback_identity(doc_title: Optional[str], toc: str) -> Dict[str, Any]:
    # On force un nettoyage du sommaire pour s'assurer qu'il y a des retours à la ligne
    formatted_toc = toc.replace(". ", ".\n- ") # Simple hack pour aérer si c'est collé

    identity_text = f"""
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📋 FICHE IDENTITÉ DU DOCUMENT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📚 TITRE: {doc_title or "Titre non détecté"}
    📖 TYPE: Document religieux / éducatif
    🎯 SUJET: Contenu en cours d'analyse

    STRUCTURE DU DOCUMENT:
    - {formatted_toc}

    🔑 THÈMES CLÉS: À déterminer
    🕌 CONTEXTE: Islam / Académique

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """.strip()
    return {"identity_text": identity_text, "token_count": 0}

def extract_table_of_contents(doc: DoclingDocument) -> str:
    """
    Extrait le sommaire du document, qu'il soit au début ou à la fin.
    """
    try:
        markdown = doc.export_to_markdown()
        lines = markdown.split('\n')
        total_lines = len(lines)
        
        # 1. Définir les zones de recherche (Début et Fin)
        # On prend les 200 premières et 200 dernières lignes
        search_limit = 200
        start_chunk = lines[:search_limit]
        end_chunk = lines[-search_limit:] if total_lines > search_limit else []
        
        # On combine pour la recherche de mots-clés
        search_zones = [
            ("DÉBUT", start_chunk),
            ("FIN", end_chunk)
        ]
        
        keywords = ['sommaire', 'table des matières', 'table of contents', 'plan du document']

        for zone_name, zone_lines in search_zones:
            for i, line in enumerate(zone_lines):
                stripped = line.strip()
                if any(kw in stripped.lower() for kw in keywords):
                    start_idx = i
                    end_idx = i + 50
                    toc_lines = zone_lines[start_idx:end_idx]
                    
                    print(f"📍 Sommaire détecté dans la zone : {zone_name}")
                    
                    # --- MODIFICATION ICI : RETOUR STRUCTURED ---
                    return {
                        "type": "OFFICIEL",
                        "content": "\n".join([l.strip() for l in toc_lines if l.strip()]),
                        "label": "TABLE DES MATIÈRES"
                    }

        # 2. Fallback : Si aucun mot-clé n'est trouvé, on récupère tous les titres (Headings)
        # Mais on se limite à un nombre raisonnable pour la fiche d'identité
        headings = []
        for item in doc.iterate_items():
            # Docling marque les titres avec le label 'heading'
            if item.label == 'heading':
                text = item.text.strip()
                if text and len(text) < 120:
                    headings.append(text)
            
            if len(headings) > 60: # Sécurité pour ne pas saturer le prompt
                break
                
        if headings:
            return {
                "type": "ESTIMÉ", 
                "content": "\n".join(headings), 
                "label": "STRUCTURE DÉTECTÉE (Titres principaux)"
            }

    except Exception as e:
        print(f"⚠️ Erreur extraction sommaire : {e}")

    return {"type": "ABSENT", "content": "Non détecté", "label": "STRUCTURE INCONNUE"}

def sample_document_pages(doc: DoclingDocument, max_chars: int = 10000) -> Dict[str, Any]:
    try:
        full_markdown = doc.export_to_markdown()
        
        if len(full_markdown) <= max_chars:
            return {"text": full_markdown, "pages_used": ["Complet"]}

        # On découpe par paragraphes (double saut de ligne) plutôt que par lignes
        # C'est plus sémantique pour le LLM
        paragraphs = full_markdown.split('\n\n')
        total_p = len(paragraphs)
        
        # Échantillonnage : 15 début, 15 milieu, 15 fin
        start_p = paragraphs[:15]
        mid_idx = total_p // 2
        mid_p = paragraphs[mid_idx-2 : mid_idx+7]
        end_p = paragraphs[-15:]
        
        sampled_text = (
            "--- DÉBUT DU DOCUMENT ---\n" + "\n\n".join(start_p) +
            "\n\n... [CONTENU INTERMÉDIAIRE] ...\n\n" + "\n\n".join(mid_p) +
            "\n\n... [CONTENU FINAL] ...\n\n" + "\n\n".join(end_p) +
            "\n--- FIN DU DOCUMENT ---"
        )

        return {
            "text": sampled_text[:max_chars], # Sécurité finale
            "pages_used": [0] #Symbolique, structure linéaire: 15 paragraphes de Début/Milieu/Fin du doc,chunk identité est spécial
        }
    except Exception as e:
        print(f"⚠️ Erreur échantillonnage (fallback lignes) : {e}")
        return {"text": doc.export_to_markdown()[:max_chars], "pages_used": ["Fallback 10k chars"]}


def build_identity_prompt(
    doc_title: Optional[str], 
    toc_data: str, 
    sampled_text_data: Dict[str, Any]
) -> str:
    """
    Construit le prompt pour générer la fiche identité.
    """
    sampled_text = sampled_text_data.get("text", "")
    pages_used = sampled_text_data.get("pages_used", [])

    toc_type = toc_data.get('type')
    toc_content = toc_data.get('content')

    context_instruction = ""
    if toc_type == "OFFICIEL":
        context_instruction = "Utilise le sommaire officiel suivant pour comprendre l'organisation exacte du document."
    else:
        context_instruction = "Attention : Aucun sommaire officiel n'a été trouvé. Voici une liste de titres extraits du corps du texte pour te donner une idée de la structure."
    
    return f"""
Tu es un assistant spécialisé dans la création de FICHES IDENTITÉ ultra-condensées pour des documents religieux et/ou éducatifs.

DOCUMENT ANALYSÉ:
Titre: {doc_title or "Non spécifié"}
Pages échantillonnées: {pages_used}

STRUCTURE FOURNIE ({toc_data.get('label')}) :
    {toc_content}

EXTRAITS DU DOCUMENT:
{sampled_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TÂCHE: Crée une FICHE IDENTITÉ ultra-condensée (MAX 400 mots).
TU DOIS IMPÉRATIVEMENT UTILISER DES RETOURS À LA LIGNE ENTRE CHAQUE ÉLÉMENT.
{context_instruction}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FORMAT STRICT À RESPECTER :

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 FICHE IDENTITÉ DU DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 TITRE: [titre exact]
📖 TYPE: [biographie / cours / essai / etc.]
🎯 SUJET: [résumé en 2,3 phrases de quoi parle le document]

STRUCTURE DU DOCUMENT (SOMMAIRE/LISTE DES TITRES) :
(Chaque chapitre/titre DOIT être sur une nouvelle ligne avec un tiret)
- 1. [Nom Chapitre] (p.[numéro])
- 2. [Nom Chapitre] (p.[numéro])
...

🔑 THÈMES CLÉS: [3-5 mots-clés séparés par virgules]

🕌 CONTEXTE: [époque, lieu, cadre si trouvé dans les pages échantillonnées - 1,2 lignes max]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RÈGLES D'OR DE MISE EN PAGE :
1. INTERDICTION FORMELLE de faire des paragraphes de texte compacts pour le sommaire/. 
2. UN CHAPITRE = UNE LIGNE. C'est crucial pour la distinction sémantique.
3. Ne mélange jamais les noms de personnes ou de sections sur la même ligne.
4. Les numéros de page sont ESSENTIELS.
5. Format ultra-scannable pour un LLM et un Reranker.

COMMENCE DIRECTEMENT PAR "━━━━━..." (pas de préambule).
""".strip()
    


# Helper function (déjà définie dans separate_content_types.py mais répétée ici pour clarté)
def get_item_page(item) -> Optional[int]:
    """Récupère le numéro de page d'un item."""
    if hasattr(item, 'prov') and item.prov:
        for prov in item.prov:
            if hasattr(prov, 'page_no'):
                return prov.page_no
    return None
