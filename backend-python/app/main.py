import os
import uuid
import time
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Importation de tes modules nettoyés
from .ingestion.pdf_loader import partition_document
from .ingestion.chunker import create_chunks
from .ingestion.analyze_content import separate_content_types
from .db.postgres import store_chunks_batch
from .embeddings.embedder import vectorize_documents
from .embeddings.summarizing import summarise_chunks
from .vector_store.qdrant_service import store_vectors_incrementally
from .rag.retriever import search_top_k
from .rag.answer_generator import generate_answer_with_history
from .rag.reranker import rerank_results
from .rag.query_rewriter import rewrite_query

app = FastAPI(title="Dawask RAG Prototype")
# Indispensable pour que l'UI (frontend) puisse appeler Docker (backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # À restreindre en prod (ex: ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/") #allows to check if nodejs commuicate or not with fastapi, health check nothing more
def read_root():
    return {"status": "ok", "message": "FastAPI is hungry for PDFs"}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
        Upload d'un PDF et ingestion complète dans la base
        """

    try:
        print("🔥 INGEST_PDF ROUTE EXECUTED 🔥")
        start_time = time.perf_counter()

        

        # creation of a unique document id
        doc_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
        
        # local save of the PDF TO DELETE IN PROD


        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        #  Partition of PDF
        t0 = time.perf_counter()
        elements = partition_document(file_path)
        t1 = time.perf_counter()
        print("🔥 PDF PARTIIONNED 🔥")


        #  Chunking
        chunks = create_chunks(elements)
        t2 = time.perf_counter()
        print("🔥 Chunks done 🔥")


        # Seperate the content type of each chunk

        analyzed_chunks = []
        for chunk in chunks:
            analyzed_chunks.append(separate_content_types(chunk))
        
        # we store them in postgress, with the proper doc_id
        store_chunks_batch(analyzed_chunks, doc_id)
        t3 = time.perf_counter()
        print("🔥 Chunks stored 🔥")



        # # we clean their format to have a better summarising
        # cleaned_chunks = export_chunks_to_json(analyzed_chunks) IF DEBUG

        # we summarise them to prepare the embedding
        summarised_chunks = summarise_chunks(analyzed_chunks)
        print("🔥 Chunks smmarized 🔥")


        vectorised_chunks = vectorize_documents(summarised_chunks)
        t4 = time.perf_counter()
        print("🔥 Chunks vectorized 🔥")



        
        original_filename = file.filename if file.filename else "unknown_file"

        store_vectors_incrementally(vectorized_docs=vectorised_chunks,doc_id = doc_id)
        print("🔥 vectored chunks stored 🔥")

        
        
        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)

        # On sécurise l'affichage pour éviter l'erreur ASCII au cas où
        first_chunk_preview = str(summarised_chunks[0])[:200] # Un aperçu court

        return {
            "status": "success",
            "document": doc_id,
            "filename": original_filename,
            "chunks_stored": len(chunks),
            "first_chunk_summarized": first_chunk_preview,
            "timings": {
                "partition": round(t1 - t0, 2),
                "chunking": round(t2 - t1, 2),
                "clean_chunks + storage Postgres": round(t3-t2),
                "summarize + vectorize chunks": round(t4-t3),
                "qdrant vector db storage": round(end_time-t4),
                "total": duration
                }

        }

    except Exception as e:
        print(f"❌ Erreur Ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))    

chat_history = []

@app.get("/query")
async def query_rag(question: str, limit: int = 8):
    """
    Endpoint to process RAG queries with a Retrieve-then-Rerank pipeline.    
    """
    global chat_history

    try:
        # 1. Rewrite the question BEFORE retrieval

        standalone_query = rewrite_query(question, chat_history)

        # 2. Retrieve
                
        initial_chunks = search_top_k(standalone_query, limit=16)

        if not initial_chunks:
            return {"answer": "Je n'ai pas trouvé de documents pertinents pour répondre a ta demande. :/", "sources": []}
        
        # 3. Rerank the results
        # This will re-order the 10 chunks and return the top 'limit' (default 8)
        # Based on deep semantic understanding

        refined_chunks = rerank_results(standalone_query, initial_chunks, top_n=limit)

        # 4. Generate Answer using the refined context
        # The LLM now receives only the most pertinent information
        answer = generate_answer_with_history(question, refined_chunks, chat_history)

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "standalone_query": standalone_query,
            "sources": [c for c in refined_chunks]
        }
    except Exception as e:
        print(f"❌ Erreur Query: {e}")
        raise HTTPException(status_code=500, detail=str(e))    
    
    
@app.post("/clear-history") #to clear history context of the user, every day, every new chat, ...
async def reset_chat():
    global chat_history
    chat_history = []
    return {"message": "Discussion reset successfully"}




