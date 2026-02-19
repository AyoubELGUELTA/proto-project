from .base import get_connection, release_connection
import uuid
from typing import List, Dict, Any, Optional
import unicodedata
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

def normalize_entity_name(name: str) -> str:
    """
    Normalise un nom d'entité pour matching robuste.
    Version ULTRA-STRICTE pour éviter les doublons.
    """
    if not name:
        return ""
    
    # 1. Strip whitespace
    normalized = name.strip()
    
    # 2. Lowercase
    normalized = normalized.lower()
    
    # 3. Supprime accents (é → e, ï → i)
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(c) != 'Mn'
    )
    
    # 4. Supprime ALL apostrophes et quotes
    normalized = re.sub(r"[''`´]", "", normalized)
    
    # 5. Supprime parenthèses et leur contenu
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    
    # 6. Supprime tirets et underscores
    normalized = re.sub(r'[-_]', ' ', normalized)
    
    # 7. Normalise "ibn", "bint", "al", "as"
    normalized = re.sub(r'\bbin\b', 'ibn', normalized)
    normalized = re.sub(r'\bal[\s-]', 'al ', normalized)
    normalized = re.sub(r'\bas[\s-]', 'as ', normalized)
    
    # 8. Supprime espaces multiples
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


# TESTS
assert normalize_entity_name("Umar ibn Al Khattab") == "umar ibn al khattab"
assert normalize_entity_name("Umar ibn al-Khattab") == "umar ibn al khattab"
assert normalize_entity_name("'Aisha bint Abi Bakr") == "aisha bint abi bakr"
assert normalize_entity_name("Aïcha bint Abi Bakr") == "aicha bint abi bakr"
assert normalize_entity_name("Prophète Muhammad (saw)") == "prophete muhammad"

# Exemples de résultats
# "Aïcha bint Abi Bakr" → "aicha bint abi bakr"
# "'Aisha (ra)" → "aisha"
# "Wudu" → "wudu"
# "Woudou" → "woudou" (⚠️ problème de variante linguistique)

async def resolve_entity(extracted_name: str, extracted_aliases: List[str], conn=None):
    should_release = False
    if conn is None:
        conn = await get_connection()
        should_release = True

    try:
        # Normalise en Python
        normalized_main = normalize_entity_name(extracted_name)
        
        all_normalized = set([normalized_main])
        for alias in extracted_aliases:
            if alias:
                all_normalized.add(normalize_entity_name(alias))
        
        # 1. Cherche par normalized_name exact
        exact_match = await conn.fetchrow("""
            SELECT entity_id, name, aliases, normalized_name 
            FROM entities 
            WHERE normalized_name = $1
        """, normalized_main)
        
        if exact_match:
            print(f"✅ Match exact : {extracted_name} → {exact_match['name']}")
            return exact_match
        
        # 2. Cherche dans les aliases (utilise la fonction SQL normalize_entity_name)
        candidates = await conn.fetch("""
            SELECT entity_id, name, aliases, normalized_name 
            FROM entities 
            WHERE EXISTS (
                SELECT 1 
                FROM unnest(aliases) AS alias
                WHERE normalize_entity_name(alias) = ANY($1::text[])
            )
        """, list(all_normalized))
        
        if not candidates:
            print(f"🆕 Nouvelle entité : {extracted_name}")
            return None
        
        # 3. Si plusieurs candidats, prend celui avec le plus d'overlap
        best_cand = None
        max_score = -1
        
        for cand in candidates:
            # Normalise en Python pour comparaison
            cand_normalized = set([normalize_entity_name(cand['name'])])
            for alias in (cand['aliases'] or []):
                cand_normalized.add(normalize_entity_name(alias))
            
            score = len(all_normalized.intersection(cand_normalized))
            if score > max_score:
                max_score = score
                best_cand = cand
        
        print(f"✅ Match par alias : {extracted_name} → {best_cand['name']} (score: {max_score})")
        return best_cand
        
    finally:
        if should_release:
            await release_connection(conn)

async def link_entity_to_chunk(chunk_id: str, extracted_entity: Dict[str, Any]):
    """
    Lie une entité extraite à un chunk.
    Gère la création/mise à jour d'entité + liaison aux tags système.
    """
    conn = await get_connection()
    name = extracted_entity['name']
    aliases = extracted_entity.get('aliases', [])
    entity_type = extracted_entity.get('type', 'CONCEPT')
    themes = extracted_entity.get('themes', [])

    try:
        async with conn.transaction():
            entity = await resolve_entity(name, aliases, conn=conn)

            if entity:
                # Entité existe, mise à jour aliases
                entity_id = entity['entity_id']
                existing_aliases = set(entity['aliases'] or [])
                new_aliases = existing_aliases.union(set(aliases))
                
                if len(new_aliases) > len(existing_aliases):
                    await conn.execute("""
                        UPDATE entities 
                        SET aliases = $1, 
                            chunk_count = chunk_count + 1,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE entity_id = $2
                    """, list(new_aliases), entity_id)
                else:
                    # Incrémente juste le compteur
                    await conn.execute("""
                        UPDATE entities 
                        SET chunk_count = chunk_count + 1,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE entity_id = $1
                    """, entity_id)
            else:
                # Création nouvelle entité
                normalized = normalize_entity_name(name)
                
                # Double-check pour éviter race condition
                existing = await conn.fetchrow("""
                    SELECT entity_id FROM entities WHERE normalized_name = $1
                """, normalized)
                
                if existing:
                    entity_id = existing['entity_id']
                else:
                    entity_id = await conn.fetchval("""
                        INSERT INTO entities (name, normalized_name, aliases, entity_type, chunk_count)
                        VALUES ($1, $2, $3, $4, 1)
                        RETURNING entity_id;
                    """, name, normalized, aliases, entity_type)
            
            # Liaison entité ↔ chunk
            await conn.execute("""
                INSERT INTO entity_links (entity_id, chunk_id, relevance_score)
                VALUES ($1, $2, $3)
                ON CONFLICT (entity_id, chunk_id) DO NOTHING;
            """, entity_id, uuid.UUID(chunk_id), extracted_entity.get('relevance', 1.0))
            
            # Liaison entité ↔ tags système
            await link_entity_to_system_tags(entity_id, themes, conn)
            
    except Exception as e:  
        print(f"❌ Erreur link_entity [{name}]: {e}")
        raise
    finally:
        await release_connection(conn)


async def link_entity_to_system_tags(entity_id: str, extracted_themes: List[str], conn):
    """
    Lie une entité aux tags système via matching textuel.
    """
    if not extracted_themes:
        return
    
    system_tags = await conn.fetch("""
        SELECT tag_id, label FROM tags WHERE is_system = TRUE
    """)
    
    for theme in extracted_themes:
        normalized_theme = normalize_entity_name(theme)
        
        best_match = None
        max_similarity = 0
        
        for tag in system_tags:
            normalized_tag = normalize_entity_name(tag['label'])
            
            # Matching simple par mots communs
            if normalized_theme in normalized_tag or normalized_tag in normalized_theme:
                theme_words = set(normalized_theme.split())
                tag_words = set(normalized_tag.split())
                similarity = len(theme_words & tag_words)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = tag['tag_id']
        
        if best_match:
            await conn.execute("""
                INSERT INTO entity_tags (entity_id, tag_id, confidence)
                VALUES ($1, $2, $3)
                ON CONFLICT (entity_id, tag_id) DO NOTHING
            """, entity_id, best_match, 0.8)

async def generate_global_summaries_for_document(doc_id: str):
    """
    Génère les résumés globaux (master chunks) pour les entités importantes.
    Critère : entités avec 5+ chunks dans ce document.
    """
    conn = await get_connection()
    
    try:
        print(f"📝 Génération des résumés globaux...")
        
        # Trouve les entités importantes (5+ chunks)
        important_entities = await conn.fetch("""
            SELECT 
                e.entity_id,
                e.name,
                COUNT(el.chunk_id) as chunk_count
            FROM entities e
            JOIN entity_links el ON e.entity_id = el.entity_id
            JOIN chunks c ON el.chunk_id = c.chunk_id
            WHERE 
                c.doc_id = $1
                AND e.global_summary IS NULL  -- Pas déjà généré
            GROUP BY e.entity_id, e.name
            HAVING COUNT(el.chunk_id) >= 5
            ORDER BY chunk_count DESC
        """, uuid.UUID(doc_id))
        
        if not important_entities:
            print("ℹ️ Aucune entité nécessitant un résumé global")
            return 0
        
        print(f"📊 {len(important_entities)} entités à résumer")     
        
        llm = ChatOpenAI(
            model=os.getenv("SUMMARIZER_MODEL_NAME", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )
        
        summaries_generated = 0
        
        for entity in important_entities:
            # Récupère tous les chunks de cette entité
            chunks = await conn.fetch("""
                SELECT c.chunk_text, c.chunk_heading_full
                FROM chunks c
                JOIN entity_links el ON c.chunk_id = el.chunk_id
                WHERE el.entity_id = $1
                ORDER BY c.chunk_index
            """, entity['entity_id'])
            
            # Prépare le contexte
            context = "\n\n---\n\n".join([
                f"[{ch['chunk_heading_full']}]\n{ch['chunk_text']}"
                for ch in chunks
            ])
            
            # Génère le résumé
            prompt = f"""
            Tu es un expert en synthèse de connaissances islamiques.

            ENTITÉ : {entity['name']}
            NOMBRE DE PASSAGES : {len(chunks)}

            MISSION : Crée un résumé global structuré et complet sur cette entité.

            CONTENU À SYNTHÉTISER :
            {context[:10000]}  # Limite à ~8k caractères pour éviter dépassement tokens

            STRUCTURE DU RÉSUMÉ :
            1. **Identité** : Qui est cette personne/concept ? (2-3 phrases)
            2. **Éléments clés** : Points importants mentionnés dans les passages
            3. **Contexte** : Relations avec d'autres personnes/événements si pertinents

            RÈGLES :
            - Synthèse factuelle, pas de phrases d'introduction
            - 200-400 mots maximum
            - Préserve les détails importants (dates, lieux, relations)
            """
            
            try:
                response = await llm.ainvoke([HumanMessage(content=prompt)])
                summary = response.content.strip()
                
                # Stocke le résumé
                await conn.execute("""
                    UPDATE entities 
                    SET global_summary = $1, last_updated = CURRENT_TIMESTAMP
                    WHERE entity_id = $2
                """, summary, entity['entity_id'])
                
                summaries_generated += 1
                print(f"  ✅ Résumé généré pour : {entity['name']}")
                
            except Exception as e:
                print(f"  ⚠️ Erreur résumé {entity['name']} : {e}")
        
        print(f"✅ {summaries_generated} résumés globaux générés")
        return summaries_generated
        
    except Exception as e:
        print(f"❌ Erreur génération summaries : {e}")
        raise
    finally:
        await release_connection(conn)

        

async def compute_cooccurrences_for_document(doc_id: str):
    """
    Calcule les co-occurrences entre entités pour un document donné.
    À appeler après ingestion complète d'un document.
    
    Logique :
    - Pour chaque chunk du document
    - Trouve toutes les paires d'entités qui apparaissent ensemble
    - Incrémente leur compteur de co-occurrence
    """
    conn = await get_connection()
    
    try:
        print(f"📊 Calcul des co-occurrences pour document {doc_id}...")
        
        # Calcule et insère les co-occurrences en une seule requête optimisée
        result = await conn.execute("""
            INSERT INTO entity_cooccurrences (entity_a_id, entity_b_id, co_occurrence_count, shared_chunks)
            SELECT 
                LEAST(el1.entity_id, el2.entity_id) as entity_a_id,
                GREATEST(el1.entity_id, el2.entity_id) as entity_b_id,
                COUNT(*) as co_occurrence_count,
                ARRAY_AGG(DISTINCT el1.chunk_id) as shared_chunks
            FROM entity_links el1
            JOIN entity_links el2 ON el1.chunk_id = el2.chunk_id
            JOIN chunks c ON el1.chunk_id = c.chunk_id
            WHERE 
                c.doc_id = $1
                AND el1.entity_id < el2.entity_id  -- Évite les doublons et auto-liens
            GROUP BY entity_a_id, entity_b_id
            ON CONFLICT (entity_a_id, entity_b_id) 
            DO UPDATE SET 
                co_occurrence_count = entity_cooccurrences.co_occurrence_count + EXCLUDED.co_occurrence_count,
                shared_chunks = (
                    SELECT ARRAY(
                        SELECT DISTINCT unnest(
                            entity_cooccurrences.shared_chunks || EXCLUDED.shared_chunks
                        )
                    )
                ),
                last_updated = CURRENT_TIMESTAMP;
        """, uuid.UUID(doc_id))
        
        # Compte combien de relations ont été créées/mises à jour
        count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM entity_cooccurrences ec
            WHERE EXISTS (
                SELECT 1 FROM entity_links el
                JOIN chunks c ON el.chunk_id = c.chunk_id
                WHERE c.doc_id = $1 
                AND (el.entity_id = ec.entity_a_id OR el.entity_id = ec.entity_b_id)
            )
        """, uuid.UUID(doc_id))
        
        print(f"✅ {count} relations de co-occurrence calculées/mises à jour")
        
        return count
        
    except Exception as e:
        print(f"❌ Erreur calcul co-occurrences : {e}")
        raise
    finally:
        await release_connection(conn)


async def get_top_cooccurrences(entity_id: str, limit: int = 10):
    """
    Récupère les entités les plus co-occurrentes avec une entité donnée.
    Utile pour le query router (questions type "lien entre X et Y"). 
    (Reproducion artificielle d'un graphe, avec les k voisins plus proches...)
    """
    conn = await get_connection()
    
    try:
        results = await conn.fetch("""
            SELECT 
                CASE 
                    WHEN ec.entity_a_id = $1 THEN e.name
                    ELSE e.name
                END as related_entity_name,
                CASE 
                    WHEN ec.entity_a_id = $1 THEN ec.entity_b_id
                    ELSE ec.entity_a_id
                END as related_entity_id,
                ec.co_occurrence_count,
                array_length(ec.shared_chunks, 1) as nb_shared_chunks
            FROM entity_cooccurrences ec
            JOIN entities e ON (
                e.entity_id = CASE 
                    WHEN ec.entity_a_id = $1 THEN ec.entity_b_id 
                    ELSE ec.entity_a_id 
                END
            )
            WHERE ec.entity_a_id = $1 OR ec.entity_b_id = $1
            ORDER BY ec.co_occurrence_count DESC
            LIMIT $2
        """, uuid.UUID(entity_id), limit)
        
        return [dict(r) for r in results]
        
    finally:
        await release_connection(conn)

async def finalize_entity_graph(doc_id: str):
    """
    Finalise le graphe d'entités après ingestion d'un document.
    
    Étapes :
    1. Calcule les co-occurrences
    2. Génère les résumés globaux pour entités importantes
    
    À appeler à la FIN de l'ingestion d'un document.
    """
    print("\n" + "="*70)
    print("🧠 Finalisation du graphe d'entités...")
    print("="*70)
    
    try:
        # 1. Co-occurrences
        nb_cooccurrences = await compute_cooccurrences_for_document(doc_id)
        
        # 2. Résumés globaux
        nb_summaries = await generate_global_summaries_for_document(doc_id)
        
        print("\n" + "="*70)
        print(f"✅ Graphe finalisé : {nb_cooccurrences} relations, {nb_summaries} résumés")
        print("="*70 + "\n")
        
        return {
            "cooccurrences": nb_cooccurrences,
            "summaries": nb_summaries
        }
        
    except Exception as e:
        print(f"❌ Erreur finalisation graphe : {e}")
        raise