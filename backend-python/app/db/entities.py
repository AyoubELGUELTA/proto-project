from .base import get_connection, release_connection
import uuid
from typing import List, Dict, Any, Optional
import unicodedata
import re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from difflib import SequenceMatcher

# ENV VARIABLES FOR LINK_ENTITY_TO_CHUNK FUNCTION

TRUST_THRESHOLD_ALIASES = int(os.getenv("ENTITY_TRUST_MAX_ALIASES", "5"))
TRUST_THRESHOLD_CHUNKS = int(os.getenv("ENTITY_TRUST_MAX_CHUNKS", "2"))
SIMILARITY_THRESHOLD = float(os.getenv("ENTITY_SIMILARITY_THRESHOLD", "0.7"))


def similarity(a: str, b: str) -> float:
    """Calcule similarité entre 2 strings (0.0 à 1.0)"""
    return SequenceMatcher(None, a, b).ratio()

def normalize_entity_name(name: str) -> str:
    """
    Normalise un nom d'entité pour matching robuste.
    Version ULTRA-STRICTE v2.
    """
    if not name:
        return ""
    
    # 1. Strip + lowercase
    normalized = name.strip().lower()
    
    # 2. Supprime accents
    normalized = ''.join(
        c for c in unicodedata.normalize('NFD', normalized)
        if unicodedata.category(c) != 'Mn'
    )
    
    # 3. Supprime apostrophes et quotes
    normalized = re.sub(r"[''`´']", "", normalized)
    
    # 4. Supprime parenthèses
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized)
    
    # 5. Supprime tirets et underscores
    normalized = re.sub(r'[-_]', ' ', normalized)
    
    # 6. Normalise "ibn", "bint", "al", "as"
    normalized = re.sub(r'\bbin\b', 'ibn', normalized)
    normalized = re.sub(r'\bal\s', '', normalized)  # Supprime "al " complètement
    normalized = re.sub(r'\bas\s', '', normalized)  # Supprime "as " complètement
    
    # 7. Normalise les doubles consonnes (yy → y, ss → s)
    normalized = normalized.replace('yy', 'y')
    normalized = normalized.replace('ss', 's')
    
    # 8. ch → sh (variante arabe)
    normalized = re.sub(r'ch\b', 'sh', normalized)
    
    # 9. Supprime espaces multiples
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

async def resolve_entity(extracted_name: str, extracted_aliases: List[str], conn=None):
    should_release = False
    if conn is None:
        conn = await get_connection()
        should_release = True

    try:
        normalized_main = normalize_entity_name(extracted_name)
        all_normalized = set([normalized_main])
        for alias in extracted_aliases:
            if alias:
                all_normalized.add(normalize_entity_name(alias))
        
        # Match exact
        exact_match = await conn.fetchrow("""
            SELECT entity_id, name, aliases, normalized_name, chunk_count, normalized_aliases
            FROM entities 
            WHERE normalized_name = $1
        """, normalized_main)
        
        if exact_match:
            return exact_match
        
        #  Match via index GIN (très rapide même avec 10k entités)
        candidates = await conn.fetch("""
            SELECT entity_id, name, aliases, normalized_name, chunk_count, normalized_aliases
            FROM entities
            WHERE normalized_aliases && $1::text[]
        """, list(all_normalized))
        
        if not candidates:
            return None
        
        # Meilleur score
        best_cand = None
        max_score = -1
        
        for cand in candidates:
            cand_normalized = set(cand['normalized_aliases'] or [])
            score = len(all_normalized.intersection(cand_normalized))
            if score > max_score:
                max_score = score
                best_cand = cand
        
        return best_cand
        
    finally:
        if should_release:
            await release_connection(conn)
            
async def link_entity_to_chunk(chunk_id: str, extracted_entity: Dict[str, Any]):
    """
    Lie une entité extraite à un chunk avec normalisation automatique des alias.
    """
    conn = await get_connection()
    name = extracted_entity['name']
    aliases = extracted_entity.get('aliases', [])
    entity_type = extracted_entity.get('type', 'CONCEPT')
    try:
        async with conn.transaction():
            # 1. Tentative de résolution
            entity = await resolve_entity(name, aliases, conn=conn)

            if entity:
                # Entité existe déjà
                entity_id = entity['entity_id']
                existing_aliases = set(entity['aliases'] or [])
                current_chunk_count = entity['chunk_count']
                
                # Stratégie de confiance progressive
                is_early_stage = (len(existing_aliases) < 5 and current_chunk_count <= 2)
                
                filtered_new_aliases = set()
                entity_normalized = normalize_entity_name(entity['name'])
                
                # Normalise tous les aliases existants pour comparaison
                existing_normalized = set([entity_normalized])
                for existing_alias in existing_aliases:
                    existing_normalized.add(normalize_entity_name(existing_alias))
                
                for alias in aliases:
                    alias_normalized = normalize_entity_name(alias)
                    is_valid = False
                    match_reason = ""
                    
                    # Logique de validation (Check 1 à 4 identique à ton code actuel)
                    if alias_normalized == entity_normalized or alias_normalized in entity_normalized or entity_normalized in alias_normalized:
                        is_valid = True
                        match_reason = "match nom principal"
                    
                    if not is_valid:
                        for existing_norm in existing_normalized:
                            if alias_normalized == existing_norm or alias_normalized in existing_norm or existing_norm in alias_normalized:
                                is_valid = True
                                match_reason = "match alias existant (exact)"
                                break
                            sim = similarity(alias_normalized, existing_norm)
                            if sim > 0.7:
                                is_valid = True
                                match_reason = f"match alias existant (sim={sim:.2f})"
                                break
                    
                    if not is_valid:
                        sim = similarity(alias_normalized, entity_normalized)
                        if sim > 0.7:
                            is_valid = True
                            match_reason = f"similarité nom principal (sim={sim:.2f})"
                    
                    if not is_valid and is_early_stage:
                        is_valid = True
                        match_reason = "confiance LLM"
                    
                    if is_valid:
                        filtered_new_aliases.add(alias)

                # --- MISE À JOUR AVEC NORMALIZED_ALIASES ---
                new_aliases_list = list(existing_aliases.union(filtered_new_aliases))
                
                # On prépare la liste normalisée pour la nouvelle colonne
                # On inclut le nom principal + tous les alias pour un matching exhaustif
                all_to_normalize = [entity['name']] + new_aliases_list
                new_normalized_aliases = list(set([normalize_entity_name(a) for a in all_to_normalize if a]))

                if len(new_aliases_list) > len(existing_aliases):
                    await conn.execute("""
                        UPDATE entities 
                        SET aliases = $1,
                            normalized_aliases = $2,
                            chunk_count = chunk_count + 1,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE entity_id = $3
                    """, new_aliases_list, new_normalized_aliases, entity_id)
                    print(f"  📝 {len(filtered_new_aliases)} nouveaux aliases (et leurs versions normalisées) ajoutés")
                else:
                    await conn.execute("""
                        UPDATE entities 
                        SET chunk_count = chunk_count + 1,
                            last_updated = CURRENT_TIMESTAMP
                        WHERE entity_id = $1
                    """, entity_id)
                    
            else:
                # 2. Création nouvelle entité avec NORMALIZED_ALIASES
                main_normalized = normalize_entity_name(name)
                unique_aliases = list(set(aliases))
                
                # Préparation du tableau de recherche normalisé
                all_to_normalize = [name] + unique_aliases
                normalized_aliases_list = list(set([normalize_entity_name(a) for a in all_to_normalize if a]))
                
                print(f"  🆕 Nouvelle entité : '{name}' avec {len(unique_aliases)} aliases")
                
                try:
                    entity_id = await conn.fetchval("""
                        INSERT INTO entities (name, normalized_name, aliases, normalized_aliases, entity_type, chunk_count)
                        VALUES ($1, $2, $3, $4, $5, 1)
                        RETURNING entity_id;
                    """, name, main_normalized, unique_aliases, normalized_aliases_list, entity_type)
                except Exception as insert_error:
                    if "duplicate key" in str(insert_error):
                        print(f"  ⚠️ Race condition détectée")
                        existing = await conn.fetchrow("SELECT entity_id FROM entities WHERE normalized_name = $1", main_normalized)
                        entity_id = existing['entity_id'] if existing else None
                    else:
                        raise  
            
            # 3 & 4 (Liaison chunk et tags identique...)
            if entity_id:
                await conn.execute("""
                    INSERT INTO entity_links (entity_id, chunk_id, relevance_score)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (entity_id, chunk_id) DO NOTHING;
                """, entity_id, uuid.UUID(chunk_id), extracted_entity.get('relevance', 1.0))
                
                # Récupération du tag suggéré par le LLM
                suggested = extracted_entity.get('suggested_tag')
                await link_entity_to_system_tags(entity_id, suggested, conn)

            await conn.execute(""" 
            UPDATE chunks
            SET processed_for_entities = TRUE
            WHERE chunk_id = $1
        """, uuid.UUID(chunk_id)) # Sert de monitoring, voir sur tableplus si tous les chunks ont été link a une entité (ce qui en général devrait etre le cas)
            
    except Exception as e:  
        print(f"❌ Erreur link_entity [{name}]: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await release_connection(conn)


async def link_entity_to_system_tags(entity_id: int, suggested_tag_label: str, conn):
    """
    LE GATEKEEPER : Lie l'entité au tag système suggéré par le LLM
    après vérification de l'existence du tag.
    """
    if not suggested_tag_label:
        return

    # On cherche le tag par son label exact (puisque le LLM choisit dans la liste)
    tag = await conn.fetchrow("""
        SELECT tag_id FROM tags 
        WHERE label = $1 AND is_system = TRUE
    """, suggested_tag_label)

    if tag:
        await conn.execute("""
            INSERT INTO entity_tags (entity_id, tag_id, confidence)
            VALUES ($1, $2, 0.9)
            ON CONFLICT (entity_id, tag_id) DO NOTHING
        """, entity_id, tag['tag_id'])
        print(f"    🔗 Tag lié : {suggested_tag_label}")

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

