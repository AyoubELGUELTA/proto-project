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


@app.post("/ingest-bulk")
async def ingest_bulk(files: List[UploadFile] = File(...), config_id: str = "01"):
    """
    Route pour uploader et ingérer plusieurs PDFs à la fois.
    Utilise la configuration de benchmark spécifiée pour l'ensemble du lot.
    """
    overall_start = time.perf_counter()
    results = []

    print(f"📦 BULK INGESTION STARTED - {len(files)} files with Config: {config_id}")

    for file in files:
        try:
            # On appelle directement la logique de ingest_pdf pour chaque fichier
            # Note : on réutilise la logique interne pour garder la cohérence
            file_result = await ingest_single_file(file, config_id)
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

async def ingest_single_file(file: UploadFile, config_id: str):
    """
    Version interne de la logique d'ingestion (extraite de ta route ingest_pdf)
    """
    start_time = time.perf_counter()
    config = get_ingest_benchmark_config(config_id)
    target_tokens = config["chunk_size"]

    doc_uuid = await get_or_create_document(file.filename)
    file_path = os.path.join(UPLOAD_DIR, f"{doc_uuid}.pdf")

    # Écriture asynchrone [cite: 2026-02-11]
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)

    # Pipeline Ingestion (Partition -> Chunk -> Identity -> Store)
    doc = partition_document(file_path)
    chunks = create_chunks(doc, max_tokens=target_tokens)
    
    identity_data = await create_identity_chunk(doc=doc, doc_id=doc_uuid, doc_title=file.filename)
    print(f"✅ Fiche identité créée du fichier '{file.filename}'.")
    await store_identity_chunk(doc_id=doc_uuid, identity_text=identity_data["identity_text"], pages_sampled=identity_data.get("pages_sampled", []))
    
    enriched_chunks_raw = process_enriched_chunks(doc, chunks)          
    enriched_chunks = split_enriched_chunks(enriched_chunks_raw, max_tokens=target_tokens)
    
    chunk_ids = await store_chunks_batch(enriched_chunks, doc_uuid)
    summarised_chunks = await summarise_chunks(enriched_chunks, chunk_ids)
    vectorised_chunks = await vectorize_documents(summarised_chunks) 
    
    await store_vectors_incrementally(vectorized_docs=vectorised_chunks, collection_name="dev_collection")
    print(f"✅ Vecteurs storés du fichier '{file.filename}'.")


    return {
        "status": "success",
        "doc_id": doc_uuid,
        "filename": file.filename,
        "chunks_count": len(chunk_ids),
        "duration": round(time.perf_counter() - start_time, 2)
    }
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




