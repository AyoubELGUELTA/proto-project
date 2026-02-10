import os
import uuid
import time
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Importation de tes modules nettoyés
from .ingestion.pdf_loader import partition_document
from .ingestion.chunker import create_chunks
from .ingestion.create_identity_chunk import create_identity_chunk
from .db.postgres import store_chunks_batch, get_documents, get_or_create_document, init_db, store_identity_chunk
from .embeddings.embedder import vectorize_documents
from .embeddings.summarizing import summarise_chunks
from .vector_store.qdrant_service import store_vectors_incrementally
from .rag.retriever import retrieve_chunks
from .rag.answer_generator import generate_answer_with_history
from .rag.query_rewriter import rewrite_query
from .utils.processor import process_enriched_chunks, split_enriched_chunks

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

@app.on_event("startup")
async def startup_event():
    print("🚀 Starting up FastAPI...")
    try:
        await init_db()
        print("✅ Database tables are ready.")
    except Exception as e:
        print(f"❌ Failed to initialize database on startup: {e}")

@app.get("/") #allows to check if nodejs commuicate or not with fastapi, health check nothing more
def read_root():
    return {"status": "ok", "message": "FastAPI is hungry for PDFs"}

@app.get("/ingested-documents")
async def list_documents():
    try:
        docs = await get_documents()
        print ("DEBUG PYTHON : ", docs)
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...), config_id: str = "01"):
    """
    Upload d'un PDF et ingestion complète dans la base avec support Benchmark.
    """
    try:
        print(f"🔥 INGEST_PDF ROUTE EXECUTED - Config: {config_id} 🔥")
        start_time = time.perf_counter()

        # 0. Récupération de la config de benchmark pour l'ingestion [cite: 2026-02-10]
        config = get_ingest_benchmark_config(config_id)
        # On définit la taille cible des tokens selon la config (ex: 1000, 1500, 2500) [cite: 2026-02-10]
        target_tokens = config["chunk_size"] if config["chunk_size"] else None

        # creation of a unique document id
        doc_uuid = await get_or_create_document(file.filename)
        file_path = os.path.join(UPLOAD_DIR, f"{doc_uuid}.pdf")

        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        #  Partition of PDF
        t0 = time.perf_counter()
        doc = partition_document(file_path)
        t1 = time.perf_counter()
        print("🔥 PDF PARTIIONNED 🔥")

        #  Chunking : Utilisation de la taille dynamique 
        chunks = create_chunks(doc, max_tokens=target_tokens)
        t2 = time.perf_counter()
        print(f"🔥 Chunks done, 📄 {len(chunks)} chunks créés avec target: {target_tokens}🔥")

        # Fiche identité
        print("\n🔄 Création de la fiche identité du document...")
        identity_data = await create_identity_chunk(
            doc=doc,
            doc_id=doc_uuid,
            doc_title=file.filename
        )
        
        await store_identity_chunk(
            doc_id=doc_uuid,
            identity_text=identity_data["identity_text"],
            pages_sampled=identity_data.get("pages_sampled", [])
        )
        print(f"✅ Fiche identité créée : {identity_data['token_count']} tokens")
        
        # Enrichissement et Split final
        enriched_chunks_raw = process_enriched_chunks(doc, chunks)          
        enriched_chunks = split_enriched_chunks(enriched_chunks_raw, max_tokens=target_tokens)

        print(f"📊 Après découpage : {len(enriched_chunks)} chunks finaux prêts pour stockage.")
        chunk_ids = await store_chunks_batch(enriched_chunks, doc_uuid)
        t3 = time.perf_counter()
        print("🔥 Chunks stored 🔥")

        # Summarization, Vectorization et Qdrant
        summarised_chunks = await summarise_chunks(enriched_chunks, chunk_ids)
        print("🔥 Chunks summarized 🔥")
        
        vectorised_chunks = await vectorize_documents(summarised_chunks) 
        t4 = time.perf_counter()
        print("🔥 Chunks vectorized 🔥")

        # Stockage dans la collection dédiée au benchmark [cite: 2026-02-10]
        collection_name = f"dev_collection"
        await store_vectors_incrementally(
            vectorized_docs=vectorised_chunks,
            collection_name=collection_name
        )
        print(f"🔥 vectored chunks stored 🔥")

        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)

        # Nettoyage
        del doc, chunks, enriched_chunks, summarised_chunks
        gc.collect()

        return {
            "status": "success",
            "doc_id": doc_uuid,
            "config_id": config_id,
            "collection": collection_name,
            "filename": file.filename if file.filename else "unknown_file",
            "chunks_stored": len(chunk_ids),
            "timings": {
                "partition": round(t1 - t0, 2),
                "chunking": round(t2 - t1, 2),
                "storage_postgres": round(t3 - t2, 2), 
                "vectorize": round(t4 - t3, 2),
                "qdrant": round(end_time - t4, 2),
                "total": duration
            }
        }

    except Exception as e:
        print(f"❌ Erreur Ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

chat_history = []

@app.get("/query")
async def query_rag(question: str, limit: int = 20, config_id: str = None):
    global chat_history

    try:
        # 0. Gestion de la config Benchmark
        # Si un config_id est passé, on écrase les paramètres par défaut
        config = get_benchmark_config_rag(config_id) if config_id else {
            "top_k": 50, # Retrieval élargi par défaut [cite: 2026-02-10]
            "top_n": limit, 
            "prompt_style": "verbose"
        }

        # 1. Rewrite : Génération des 3 variantes [cite: 2026-02-10]
        standalone_query = await rewrite_query(question, chat_history)

        # 2. Retrieve + Rerank (On utilise les limites de la config)
        # On passe config["top_k"] pour Qdrant et config["top_n"] pour le Reranker [cite: 2026-02-10]
        final_context = await retrieve_chunks(
            standalone_query, 
            limit=config["top_k"], 
            rerank_limit=config["top_n"]
        )

        if not final_context:
            return {"answer": "Pas d'infos trouvées.", "sources": []}

        # 3. Generation avec le style de prompt choisi [cite: 2026-02-10]
        answer = await generate_answer_with_history(
            question, 
            final_context, 
            chat_history, 
            style=config["prompt_style"]
        )

        # 4. Retour enrichi pour ton analyse
        return {
            "answer": answer,
            "standalone_query": standalone_query,
            "config_applied": config_id or "default",
            "chunks_count": len([c for c in final_context if not c.get("is_identity")]),
            "sources": final_context # Ici tu verras tes textes, tableaux et scores de reranking [cite: 2026-02-10]
        }

    except Exception as e:
        print(f"❌ Erreur: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 
    
    
@app.post("/clear-history") #to clear history context of the user, every day, every new chat, ...
async def reset_chat():
    global chat_history
    chat_history = []
    return {"message": "Discussion reset successfully"}




