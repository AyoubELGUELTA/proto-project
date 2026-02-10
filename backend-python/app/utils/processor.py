from typing import List, Dict, Any
from .s3_storage import storage  
from ..ingestion.separate_content_types import separate_content_types 
import re


from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_enriched_chunks(enriched_chunks: List[Dict], max_tokens=1100) -> List[Dict]:
    final_list = []
    title_counters = {} 

    # 1. On ne boucle sur le splitter QUE si max_tokens est défini
    for original in enriched_chunks:
        text = original['text']
        base_title = original['heading_full']
        
        if base_title not in title_counters:
            title_counters[base_title] = 0
            
        # LOGIQUE DE BYPASS : si max_tokens est None, on simule un seul "sous-texte"
        if max_tokens is None:
            sub_texts = [text]
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=max_tokens * 4,
                chunk_overlap=200, 
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            sub_texts = splitter.split_text(text)
        
        for i, sub_text in enumerate(sub_texts):
            sub_chunk = original.copy()
            sub_chunk['text'] = sub_text
            
            # Gestion des titres (Suite X) : reste active même sans split pour l'indexation globale
            if title_counters[base_title] == 0:
                sub_chunk['heading_full'] = base_title
            else:
                sub_chunk['heading_full'] = f"{base_title} (Suite {title_counters[base_title]})"
            
            # Nettoyage des visuels sur les segments suivants
            if i > 0:
                sub_chunk['tables'] = []
                sub_chunk['images_urls'] = []
            
            title_counters[base_title] += 1
            final_list.append(sub_chunk)

    # 2. Réindexation finale (crucial pour le tri dans ton RAG) 
    for i in range(len(final_list)):
        final_list[i]['chunk_index'] = i 
            
    return final_list

def process_enriched_chunks(doc, chunks, identity_metadata=None) -> List[Dict[str, Any]]:
    """
    Transforme les chunks bruts de Docling en chunks enrichis (tables + images uniques).
    """
    uploaded_images = {}
    enriched_chunks = []

    valid_titles = []
    if identity_metadata and 'chunk_text' in identity_metadata:
        # On suppose que ton LLM a mis les titres dans une clé 'structure' ou 'chapters'
        valid_titles = extract_valid_titles_from_identity(identity_metadata.get('chunk_text', ''))

    for i, chunk in enumerate(chunks):
        # On suppose que separate_content_types est importé ou défini
        content = separate_content_types(chunk, doc)

        # 1. Nettoyage initial du titre
        raw_heading = content.get("chunk_heading_full", "Sans titre")
        clean_heading = filter_suspicious_heading(raw_heading, valid_titles)
        
        # 2. LOGIQUE D'HÉRITAGE (Si le titre est suspect ou générique)
        if clean_heading == "Section générale" or clean_heading == "Contenu informatif":
            # On remonte la liste des chunks déjà traités pour trouver le dernier titre valide
            for previous_chunk in reversed(enriched_chunks):
                prev_h = previous_chunk['heading_full']
                if prev_h and prev_h not in ["Section générale", "Contenu informatif"]:
                    clean_heading = prev_h
                    break
        
        chunk_images_urls = []
        chunk_pages = content.get("chunk_page_numbers", [])

        # Calcul de la zone verticale du chunk
        chunk_tops = [item.prov[0].bbox.t for item in chunk.meta.doc_items if item.prov]
        chunk_bottoms = [item.prov[0].bbox.b for item in chunk.meta.doc_items if item.prov]
        c_min = min(chunk_tops) if chunk_tops else 0
        c_max = max(chunk_bottoms) if chunk_bottoms else 1000

        if hasattr(doc, 'pictures') and doc.pictures:
            for pic in doc.pictures:
                if not pic.prov: continue
                
                pic_page = pic.prov[0].page_no
                pic_bbox = pic.prov[0].bbox
                img_id = f"pg_{pic_page}_{pic_bbox.l}_{pic_bbox.t}_{pic_bbox.r}_{pic_bbox.b}"

                # Déjà traitée ?
                if img_id in uploaded_images:
                    continue

                if pic_page in chunk_pages:
                    pic_top = pic_bbox.t
                    
                    # Logique de capture robuste
                    is_near = (c_min - 100) <= pic_top <= (c_max + 100)
                    is_sole = len(chunk_pages) == 1 and chunk_pages[0] == pic_page

                    if is_near or is_sole:
                        try:
                            image_obj = pic.get_image(doc)
                            if image_obj:
                                # Filtre taille (200px)
                                if image_obj.size[0] < 200 or image_obj.size[1] < 200:
                                    continue
                                
                                url = storage.upload_image(image_obj)
                                if url:
                                    chunk_images_urls.append(url)
                                    uploaded_images[img_id] = url
                                    print(f"   🎯 IMAGE UNIQUE CAPTURÉE (Page {pic_page}) pour chunk {i}")
                        except Exception as e:
                            print(f"   ⚠️ Erreur extraction pic: {e}")

        enriched_chunks.append({
            'chunk_index': i,
            'text': content['chunk_text'],
            'headings': content['chunk_headings'],
            'heading_full': clean_heading,
            'page_numbers': chunk_pages,
            'tables': content.get('chunk_tables', []),
            'images_urls': chunk_images_urls 
        })
        print(f"📦 Traitement chunk {i+1}/{len(chunks)} terminé.")

    return enriched_chunks



def filter_suspicious_heading(heading: str, valid_titles: List[str] = None) -> str:
    """
    Filtrage avancé : Bruit + Citations + Validation par Sommaire.
    """
    if not heading:
        return "Section générale"
        
    h = heading.strip()

    # 1. RÈGLE DES GUILLEMETS (Citations suspectes)
    # Si ça commence et finit par des guillemets, ou finit par », c'est une citation.
    if (h.startswith(('"', '«', '“')) or h.endswith(('"', '»', '”'))):
        return "Section générale"

    # 2. RÈGLE DE LA FICHE D'IDENTITÉ (Si fournie)
    if len(valid_titles) > 3:
        # On normalise pour comparer (minuscules, sans ponctuation)
        h_norm = re.sub(r'[^\w\s]', '', h.lower())
        
        is_in_summary = False
        for vt in valid_titles:
            vt_norm = re.sub(r'[^\w\s]', '', vt.lower())
            # Match si le titre extrait est inclus dans le sommaire ou inversement
            if vt_norm in h_norm or h_norm in vt_norm:
                is_in_summary = True
                break
        
        # Si le titre est long (>30 car) ET absent du sommaire -> Suspect (probablement un paragraphe)
        if not is_in_summary and len(h) > 56:
            return "Section générale"

    # 3. TES RÈGLES EXISTANTES (RegEx)
    patterns_bruit = [
        r'^\d+$',                      # Uniquement chiffres
        r'^[^\w\s]+$',                 # Uniquement ponctuation
        r'^(?i)page\s*\d+$',           # "Page 12"
        r'^\d+\s*€$',                  # "15 €"
        r'^©.*$',                      # Copyright
        r'^\d{2}/\d{2}/\d{4}$'         # Dates
    ]
    for pattern in patterns_bruit:
        if re.match(pattern, h):
            return "Section générale"

    return h

def extract_valid_titles_from_identity(identity_text: str) -> List[str]:
    """
    Extrait les titres du sommaire depuis le bloc de texte de la fiche d'identité.
    """
    valid_titles = []
    # On cherche la section Sommaire/Structure
    if "SOMMAIRE" in identity_text.upper() or "STRUCTURE" in identity_text.upper() or "TABLE DES MATIERES" in identity_text.upper():
        # On récupère les lignes qui commencent par un tiret ou un numéro
        lines = identity_text.split('\n')
        for line in lines:
            line = line.strip()
            # Regex pour capturer : "- 1. Titre (p.0)" ou "- Titre"
            match = re.search(r'^[-*•]\s*(?:\d+[\.)]\s*)?(.+?)(?:\s*\(p\.\d+\))?$', line)
            if match:
                title = match.group(1).strip()
                if len(title) > 3: # Éviter les débris
                    valid_titles.append(title)
    
    return valid_titles


