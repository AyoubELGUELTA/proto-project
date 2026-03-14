import os
import uuid
import time
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Importation de mes modules nettoyés

from .ingestion.pdf_loader import partition_document
from .ingestion.chunker import create_chunks
from .ingestion.create_identity_chunk import create_identity_chunk
from .db import (init_db, seed_system_tags, get_documents, get_or_create_document, store_chunks_batch, store_identity_chunk, 
                fetch_identities_by_doc_ids, get_chunk_with_metadata, 
                update_chunks_with_ai_data, link_entity_to_chunk, resolve_entity, finalize_entity_graph)
from app.db.base import get_connection, release_connection
from .embeddings.embedder import vectorize_documents
from .vector_store.qdrant_service import store_vectors_incrementally
from .retrieval.retriever import retrieve_chunks
from .retrieval.answer_generator import generate_answer_with_history
from .retrieval.entity_resolver import resolve_entities_in_query
from .retrieval.query_analyzer import analyze_and_rewrite_query, QueryType
from .retrieval.strategies import select_strategy, VectorOnlyStrategy
from .utils.chunks_ingest_processor import process_enriched_chunks, split_enriched_chunks
from .utils.summarize_and_extract_entities import summarise_and_extract_entities
from app.core.tags_store import TagsStore
from contextlib import asynccontextmanager

from .benchmark_test import get_benchmark_config_rag, get_ingest_benchmark_config
import aiofiles 
import asyncio

app = FastAPI(title="Dawask RAG Prototype")
# Indispensable pour que l'UI (frontend) puisse appeler Docker (backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # À restreindre en prod (ex: ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


os.environ["PYTHONUTF8"] = "1"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)



@app.get("/") #allows to check if nodejs commuicate or not with fastapi, health check nothing more
def read_root():
    return {"status": "ok", "message": "FastAPI is hungry for PDFs"}
# app/main.py



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie unique (Boot & Shutdown).
    Remplace avantageusement les anciens @app.on_event.
    """
    print("🚀 [BOOT] Initialisation du système...")
    
    try:
        # 1. Initialisation des tables (si nécessaire)
        await init_db()
        print("✅ DB: Tables synchronisées.")

        # 2. Seeding des tags système (Jour 1-2 de ton plan)
        await seed_system_tags()
        print("✅ DB: Tags système vérifiés/insérés.")

        # 3. Chargement du cache de Tags pour le Query Analyzer
        conn = await get_connection()
        try:
            tags = await conn.fetch("""
                SELECT label, description 
                FROM tags 
                WHERE is_system = TRUE
            """)
            TagsStore.set_tags(tags)
            print(f"🚀 CACHE: {len(tags)} tags chargés en RAM.")
        finally:
            await release_connection(conn)

    except Exception as e:
        print(f"❌ [CRITICAL] Échec du boot : {e}")
        # En production, on pourrait lever une erreur ici pour empêcher le serveur de démarrer mal configuré
    
    yield  # L'application tourne et accepte des requêtes

    # --- PHASE SHUTDOWN ---
    print("🛑 [SHUTDOWN] Nettoyage des ressources...")


@app.get("/ingested-documents")
async def list_documents():
    try:
        docs = await get_documents()
        print ("DEBUG PYTHON : ", docs)
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest-bulk")
async def ingest_bulk(files: List[UploadFile] = File(...), config_id: str = "01", background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Route pour uploader et ingérer plusieurs PDFs à la fois.
    Utilise la configuration de benchmark spécifiée pour l'ensemble du lot.
    """
    overall_start = time.perf_counter()
    results = []

    print(f"📦 BULK INGESTION STARTED - {len(files)} files with Config: {config_id}")

    for file in files:
        try:
            # ICI : tu passes background_tasks en argument
            file_result = await ingest_single_file(file, config_id, background_tasks)
            results.append(file_result)
            
        except Exception as e:
            print(f"❌ Error ingesting {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "status": "error",
                "detail": str(e)
            })
        finally:
            # Nettoyage agressif entre chaque fichier pour libérer la RAM du Mac [cite: 2026-01-08]
            gc.collect()

    duration = round(time.perf_counter() - overall_start, 2)
    return {
        "overall_status": "completed",
        "total_files": len(files),
        "total_duration": duration,
        "results": results
    }

async def ingest_single_file(file: UploadFile, config_id: str, background_tasks: BackgroundTasks):
    """
    Pipeline d'ingestion optimisé : Partition -> Identity -> Store -> AI Enrichment & Entity Graph -> Vectorize
    """
    start_time = time.perf_counter()
    config = get_ingest_benchmark_config(config_id)
    target_tokens = config["chunk_size"]
    overlap = config["overlap"]

    # 1. Gestion du document physique et BDD
    doc_uuid = await get_or_create_document(file.filename)
    file_path = os.path.join(UPLOAD_DIR, f"{doc_uuid}.pdf")

    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # 2. Parsing et Chunking initial
    doc = partition_document(file_path)
    chunks = create_chunks(doc, max_tokens=target_tokens)
    
    # 3. Identity Chunk (Le contexte global du document)
    identity_data = await create_identity_chunk(doc=doc, doc_id=doc_uuid, doc_title=file.filename)
    await store_identity_chunk(
        doc_id=doc_uuid, 
        identity_text=identity_data["identity_text"], 
        pages_sampled=identity_data.get("pages_sampled", [])
    )
    
    # 4. Enrichissement structurel (Headings, Tables, Images)
    enriched_chunks_raw = process_enriched_chunks(doc, chunks)          
    enriched_chunks = split_enriched_chunks(enriched_chunks_raw, max_tokens=target_tokens, overlap=overlap)
    
    # 5. Stockage des chunks bruts pour obtenir les UUIDs
    chunk_ids = await store_chunks_batch(enriched_chunks, doc_uuid)

    # 6. IA : Synthèse visuelle ET Extraction d'entités (GraphRAG)
    # On utilise ta nouvelle fonction groupée
    enriched_results = await summarise_and_extract_entities(enriched_chunks, chunk_ids)
    
    # 7. Vectorisation (basée sur le texte enrichi par l'IA)
    vectorised_chunks = await vectorize_documents(enriched_results) 
    
    # 8. Stockage Vectoriel
    await store_vectors_incrementally(vectorized_docs=vectorised_chunks, collection_name="dev_collection")

    #9. FINALISATION DU GRAPHE D'ENTITÉS (Etabli les cooccurences + les refresh/make les global summaries si + de 5 chunks)
    background_tasks.add_task(finalize_entity_graph, doc_uuid)

    duration = round(time.perf_counter() - start_time, 2)
    print(f"📡 Vectorisation terminée en {duration}s. Finalisation du graphe lancée en tâche de fond.")
    return {
        "status": "success",
        "doc_id": doc_uuid,
        "filename": file.filename,
        "chunks_count": len(chunk_ids),
        "message": "Le document est prêt pour la recherche. Les résumés d'entités sont en cours de génération.",
        "duration_to_vector": duration
    }
chat_history = []


@app.get("/query")
async def query_rag(question: str, limit: int = 20, config_id: str = "01"):
    global chat_history
    
    try:
        config = get_benchmark_config_rag(config_id) if config_id else {
            "top_k": 50,
            "top_n": limit,
            "prompt_style": "verbose"
        }
        
        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 1 : Analyse (rewriting + classification)
        # ═══════════════════════════════════════════════════════════
        query_analysis = await analyze_and_rewrite_query(question, chat_history)
        
        standalone_query = query_analysis["vector_query"]
        query_type = query_analysis["query_type"]
        confidence = query_analysis["confidence"]
        entities_mentioned = query_analysis["entities_mentioned"]
        
        print(f"📊 Type: {query_type} | Conf: {confidence:.2f}")
        
        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 2 : Entity Resolution
        # ═══════════════════════════════════════════════════════════
        detected_entities = []
        if entities_mentioned:
            detected_entities = await resolve_entities_in_query(entities_mentioned)
        
        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 3 : Strategy Selection
        # ═══════════════════════════════════════════════════════════
        strategy = select_strategy(query_type, detected_entities, retrieve_chunks)
        
        print(f"🎯 Stratégie : {strategy.__class__.__name__}")
        
        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 4 : Retrieval via strategy
        # ═══════════════════════════════════════════════════════════
        entity_context = await strategy.retrieve({
        **query_analysis,  # Inclut variants, keyword_query, vector_query, etc.
        "question": question,
        "entities": detected_entities
    })
        
        print(f"📦 Chunks strategy : {len(entity_context)}")
        
        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 5 : Fallback vector si insuffisant 
        # ═══════════════════════════════════════════════════════════
        if len(entity_context) < config["top_n"]:
            needed = config["top_n"] - len(entity_context)
            print(f"⚠️ Complète avec vector search ({needed} chunks)")
            
            # On crée explicitement la stratégie vectorielle pour le fallback
            vector_strategy = VectorOnlyStrategy(retrieve_chunks)
            vector_context = await vector_strategy.retrieve(query_analysis)
            
            # Déduplique
            entity_chunk_ids = {c.get('chunk_id') for c in entity_context if c.get('chunk_id')}
            vector_context = [c for c in vector_context if c.get('chunk_id') not in entity_chunk_ids]
            
            final_context = entity_context + vector_context[:needed]
        else:
            # Cas où la stratégie a déjà ramené assez de matière
            final_context = entity_context[:config["top_n"]]
        
        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 6 : Generation
        # ═══════════════════════════════════════════════════════════
        answer = await generate_answer_with_history(
            question,
            final_context,
            chat_history,
            style=config["prompt_style"]
        )
        
        # ═══════════════════════════════════════════════════════════
        # ÉTAPE 7 : Retour enrichi
        # ═══════════════════════════════════════════════════════════
        return {
            "answer": answer,
            "standalone_query": standalone_query,
            "query_type": query_type,
            "confidence": confidence,
            "strategy_used": strategy.__class__.__name__,
            "entities_detected": entities_mentioned,
            "entities_resolved": detected_entities,
            "config_applied": config_id or "default",
            "chunks_count": len(final_context),
            "sources": final_context
        }
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/clear-history") #to clear history context of the user, every day, every new chat, ...
async def reset_chat():
    global chat_history
    chat_history = []
    return {"message": "Discussion reset successfully"}




