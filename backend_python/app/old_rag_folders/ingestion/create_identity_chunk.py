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
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # Basse température pour cohérence
            max_tokens=1000,   # ~650 mots max
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

def extract_table_of_contents(doc: DoclingDocument) -> Dict[str, str]:
    """
    Extrait le sommaire du document de manière robuste.
    Retourne TOUJOURS un dictionnaire avec 'type', 'content', 'label'.
    """
    try:
        # 1. Tentative par Markdown (Recherche explicite de "Sommaire")
        markdown = doc.export_to_markdown()
        lines = markdown.split('\n')
        
        # On cherche dans les 200 premières lignes (début) et 200 dernières (fin)
        search_zones = [
            ("DÉBUT", lines[:200]),
            ("FIN", lines[-200:] if len(lines) > 200 else [])
        ]
        
        keywords = ['sommaire', 'table des matières', 'table of contents', 'plan du document', 'contenu']

        for zone_name, zone_lines in search_zones:
            for i, line in enumerate(zone_lines):
                stripped = line.strip().lower()
                # On vérifie si la ligne contient un mot clé ET fait moins de 50 chars (pour éviter les faux positifs dans le texte)
                if any(kw in stripped for kw in keywords) and len(stripped) < 50:
                    
                    # On prend les 50 lignes suivantes comme sommaire potentiel
                    start_idx = i + 1 
                    end_idx = min(i + 60, len(zone_lines))
                    toc_lines = zone_lines[start_idx:end_idx]
                    
                    # Nettoyage basique
                    clean_toc = [l.strip() for l in toc_lines if l.strip() and len(l) > 3]
                    
                    if len(clean_toc) > 3: # On veut au moins 3 entrées pour valider
                        print(f"📍 Sommaire détecté dans la zone : {zone_name}")
                        return {
                            "type": "OFFICIEL",
                            "content": "\n".join(clean_toc),
                            "label": "TABLE DES MATIÈRES DÉTECTÉE"
                        }

        # Fallback : Extraction des Titres (Headings) via Docling
        headings = []
        
        # Docling v2: iterate_items() retourne un générateur d'items
        for item, _ in doc.iterate_items(): # Le _ capture le level/parent si iterate_items renvoie un tuple
             # Si item est un tuple (ce qui arrive parfois selon la version), on prend le 1er élément
            if isinstance(item, tuple):
                item = item[0]
            
            # Vérification sécurisée du label
            # Docling utilise parfois 'heading', parfois 'section_header' selon les modèles
            if hasattr(item, 'label') and str(item.label).lower() in ['heading', 'title', 'section_header']:
                text = item.text.strip()
                if text and len(text) < 150: # On évite les titres à rallonge qui sont des erreurs
                    headings.append(f"- {text}")
            
            if len(headings) >= 40: # On limite à 40 titres pour le prompt
                break
                
        if headings:
            print(f"📍 Structure reconstruite via {len(headings)} titres.")
            return {
                "type": "ESTIMÉ", 
                "content": "\n".join(headings), 
                "label": "STRUCTURE RECONSTRUITE (TITRES)"
            }

    except Exception as e:
        print(f"⚠️ Erreur non-bloquante extraction sommaire : {e}")

    return {
        "type": "ABSENT", 
        "content": "Aucune structure détectée.", 
        "label": "STRUCTURE INCONNUE"
    }

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
        start_p = paragraphs[:20]
        mid_idx = total_p // 2
        mid_p = paragraphs[mid_idx-10 : mid_idx+10]
        end_p = paragraphs[-20:]
        
        sampled_text = (
            "--- DÉBUT DU DOCUMENT ---\n" + "\n\n".join(start_p) +
            "\n\n... [CONTENU INTERMÉDIAIRE] ...\n\n" + "\n\n".join(mid_p) +
            "\n\n... [CONTENU FINAL] ...\n\n" + "\n\n".join(end_p) +
            "\n--- FIN DU DOCUMENT ---"
        )

        return {
            "text": sampled_text[:max_chars], # Sécurité finale
            "pages_used": [0] #Symbolique, structure linéaire: 20 paragraphes de Début/Milieu/Fin du doc,chunk identité est spécial
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
TÂCHE: Crée une FICHE IDENTITÉ ultra-condensée (MAX 650 mots).
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
