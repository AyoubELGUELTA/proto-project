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
    toc = extract_table_of_contents(doc) #table of contents
    
    # 2. Échantillonner le document : 6 premières + 6 milieu + 6 fin
    sampled_text = sample_document_pages(doc, 
                                          start_pages=6, 
                                          middle_pages=6, 
                                          end_pages=6)
    
    # 3. Construire le prompt pour GPT-4o-mini
    prompt = build_identity_prompt(doc_title, toc, sampled_text)
    
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
        token_count = response.usage.total_tokens
        
        print(f"✅ Fiche identité créée : {token_count} tokens")
        
        return {
            "identity_text": identity_text,
            "token_count": token_count,
            "pages_sampled": sampled_text.get("pages_used", [])
        }
    
    except Exception as e:
        print(f"❌ Erreur lors de la création de la fiche identité : {e}")
        # Fallback : créer une fiche minimale
        return create_fallback_identity(doc_title, toc)

def extract_table_of_contents(doc: DoclingDocument) -> str:
    """
    Extrait le sommaire du document.
    
    L'API Docling utilise doc.iterate_items() ou doc.export_to_markdown()
    """
    toc_text = ""
    try:
        # Stratégie 1 : Utiliser export_to_markdown pour avoir la structure
        # (Docling génère automatiquement les headings en Markdown)
        markdown = doc.export_to_markdown()
        
        # Extraire les lignes qui commencent par # (headings)
        lines = markdown.split('\n')
        headings = []
        
        for line in lines[:100]:  # Limiter aux 100 premières lignes
            stripped = line.strip()
            if stripped.startswith('#'):
                # Nettoyer le heading (enlever les #)
                heading = stripped.lstrip('#').strip()
                
                # Filtrer les headings trop longs (probablement pas un titre)
                if heading and len(heading) < 100:
                    # Détecter si c'est un sommaire
                    if 'sommaire' in heading.lower() or 'table des matières' in heading.lower():
                        # Extraire les 20 prochaines lignes après "Sommaire"
                        idx = lines.index(line)
                        toc_lines = lines[idx:idx+25]
                        toc_text = "\n".join([l.strip() for l in toc_lines if l.strip()])
                        return toc_text
                    
                    headings.append(heading)
        
        # Stratégie 2 : Si pas de sommaire explicite, retourner les headings trouvés
        if headings:
            toc_text = "\n".join(headings)
            return toc_text
        
        # Stratégie 3 : Fallback - Itérer sur les items du document
        if not toc_text and hasattr(doc, 'body') and hasattr(doc.body, 'children'):
            for item in doc.body.children[:50]:
                if hasattr(item, 'label') and 'heading' in str(item.label).lower():
                    text = getattr(item, 'text', '').strip()
                    if text and len(text) < 100:
                        headings.append(text)
            
            if headings:
                toc_text = "\n".join(headings)

    except Exception as e:
        print(f"⚠️ Erreur extraction sommaire : {e}")

    return toc_text or "Sommaire non détecté"

def sample_document_pages(
    doc: DoclingDocument, 
    start_pages: int = 6, 
    middle_pages: int = 6, 
    end_pages: int = 6,
    max_chars: int = 12500
) -> Dict[str, Any]:
    """
    Échantillonne le document pour l'analyse LLM.
    
    Returns:
        dict avec:
        - text: Texte échantillonné
        - pages_used: Liste des pages utilisées
    """
    total_pages = len(doc.pages) if hasattr(doc, 'pages') else 0
    
    if total_pages == 0:
        # Fallback : prendre les premiers N items
        text = "\n".join([
            getattr(item, 'text', '') 
            for item in doc.main_text[:100]
        ])[:max_chars]
        return {"text": text, "pages_used": [1]}
    
    # Calculer les indices des pages
    start_indices = list(range(0, min(start_pages, total_pages)))
    
    middle_start = max(0, (total_pages // 2) - (middle_pages // 2))
    middle_indices = list(range(middle_start, min(middle_start + middle_pages, total_pages)))
    
    end_start = max(0, total_pages - end_pages)
    end_indices = list(range(end_start, total_pages))
    
    # Fusionner et dédupliquer
    page_indices = sorted(set(start_indices + middle_indices + end_indices))
    
    # Extraire le texte des pages sélectionnées
    sampled_texts = []
    for page_idx in page_indices:
        if page_idx < len(doc.pages):
            page = doc.pages[page_idx]
            # Extraire les items de cette page
            page_items = [
                item for item in doc.main_text 
                if get_item_page(item) == page_idx + 1  # Page numbers start at 1
            ]
            page_text = "\n".join([
                getattr(item, 'text', '') 
                for item in page_items
            ])
            sampled_texts.append(f"[Page {page_idx + 1}]\n{page_text}")
    
    full_text = "\n\n".join(sampled_texts)
    
    # Tronquer si trop long
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n...[TRONQUÉ]"
    
    return {
        "text": full_text,
        "pages_used": [idx + 1 for idx in page_indices]
    }


def build_identity_prompt(
    doc_title: Optional[str], 
    toc: str, 
    sampled_text_data: Dict[str, Any]
) -> str:
    """
    Construit le prompt pour générer la fiche identité.
    """
    sampled_text = sampled_text_data.get("text", "")
    pages_used = sampled_text_data.get("pages_used", [])
    
    return f"""
Tu es un assistant spécialisé dans la création de FICHES IDENTITÉ ultra-condensées pour des documents religieux et/ou éducatifs.

DOCUMENT ANALYSÉ:
Titre: {doc_title or "Non spécifié"}
Pages échantillonnées: {pages_used}

TABLE DES MATIÈRES:
{toc}

EXTRAITS DU DOCUMENT:
{sampled_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TÂCHE: Crée une FICHE IDENTITÉ ultra-condensée (MAX 400 mots).

FORMAT STRICT À RESPECTER:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 FICHE IDENTITÉ DU DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 TITRE: [titre exact]
📖 TYPE: [biographie / cours / essai / etc.]
🎯 SUJET: [résumé en 2,3 phrases de quoi parle le document]

STRUCTURE DU DOCUMENT:
[Liste numérotée des chapitres/sections AVEC numéros de page]
Exemple:
1. Chapitre 1 (p.15-32)
2. Chapitre 2 (p.33-46)
...
A défault de ne pas avoir des chapitres/sections, donne la structure du document, comment c'est organisé.

🔑 THÈMES CLÉS: [3-5 mots-clés séparés par virgules]
[OPTIONNEL] 🕌 CONTEXTE: [époque, lieu, cadre si pertinent - 1,2 lignes max], si tu trouves du contexte dans les pages échantillonées.ß

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONTRAINTES CRITIQUES:
- MAX 400 mots (compte-les !)
- Pas de détails narratifs ou anecdotes
- Juste la structure + thèmes + index
- Format ultra-scannable pour un LLM
- Les numéros de page sont ESSENTIELS

COMMENCE DIRECTEMENT PAR "━━━━━..." (pas de préambule).
""".strip()


def create_fallback_identity(doc_title: Optional[str], toc: str) -> Dict[str, Any]:
    """
    Crée une fiche identité minimale en cas d'échec de l'API.
    """
    identity_text = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📋 FICHE IDENTITÉ DU DOCUMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📚 TITRE: {doc_title or "Titre non détecté"}
📖 TYPE: Document religieux et/ou éducatif/scolaire
🎯 SUJET: Contenu en cours d'analyse

STRUCTURE DU DOCUMENT:
{toc}

🔑 THÈMES CLÉS: À déterminer
🕌 CONTEXTE: Islam et/ou Académique

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""".strip()
    
    return {
        "identity_text": identity_text,
        "token_count": len(identity_text.split()) * 1.3,  # Approximation
        "pages_sampled": []
    }


# Helper function (déjà définie dans separate_content_types.py mais répétée ici pour clarté)
def get_item_page(item) -> Optional[int]:
    """Récupère le numéro de page d'un item."""
    if hasattr(item, 'prov') and item.prov:
        for prov in item.prov:
            if hasattr(prov, 'page_no'):
                return prov.page_no
    return None
