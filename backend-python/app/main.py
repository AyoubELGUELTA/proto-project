import os
import uuid
import time
import shutil
import gc
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Importation de tes modules nettoyés
from .ingestion.pdf_loader import partition_document
from .ingestion.chunker import create_chunks, extract_single_image_base64
from .ingestion.separate_content_types import separate_content_types
from .ingestion.create_identity_chunk import create_identity_chunk
from .db.postgres import store_chunks_batch, get_documents, get_or_create_document, init_db, store_identity_chunk
from .embeddings.embedder import vectorize_documents
from .embeddings.summarizing import summarise_chunks
from .vector_store.qdrant_service import store_vectors_incrementally
from .rag.retriever import retrieve_chunks
from .rag.answer_generator import generate_answer_with_history
from .rag.reranker import rerank_results
from .rag.query_rewriter import rewrite_query
from .utils.s3_storage import storage
from .utils.processor import process_enriched_chunks

from docling_core.types.doc import PictureItem

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
async def ingest_pdf(file: UploadFile = File(...)):
    """
        Upload d'un PDF et ingestion complète dans la base
        """

    try:
        print("🔥 INGEST_PDF ROUTE EXECUTED 🔥")
        start_time = time.perf_counter()

        # creation of a unique document id
        doc_uuid = await get_or_create_document(file.filename)
        file_path = os.path.join(UPLOAD_DIR, f"{doc_uuid}.pdf")

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        #  Partition of PDF
        t0 = time.perf_counter()

        doc = partition_document(file_path)
        # --- DEBUG IMAGES DANS LE DOC ---
        print(f"🔍 DEBUG: Nombre de pages dans le doc: {len(doc.pages)}")
        # Vérifier les images globales
        num_pictures = len(list(doc.iterate_items())) # On va compter les items de type Picture
        pictures_found = [item for item, _level in doc.iterate_items() if "PictureItem" in str(type(item))]
        print(f"🔍 DEBUG: Nombre d'items 'PictureItem' détectés dans le doc entier: {len(pictures_found)}")

        if hasattr(doc, 'pictures'):
            print(f"🔍 DEBUG: Nombre d'entrées dans doc.pictures: {len(doc.pictures)}")
        # --------------------------------

        t1 = time.perf_counter()
        print("🔥 PDF PARTIIONNED 🔥")


        #  Chunking
        chunks = create_chunks(doc)
        t2 = time.perf_counter()
        print(f"🔥 Chunks done, 📄 {len(chunks)} chunks créés🔥")

        print("\n🔄 Création de la fiche identité du document...")
        identity_data = await create_identity_chunk(
            doc=doc,
            doc_id=doc_uuid,
            doc_title=file.filename
        )
        
        # Stocker le chunk identité

        await store_identity_chunk(
            doc_id=doc_uuid,
            identity_text=identity_data["identity_text"],
            pages_sampled=identity_data.get("pages_sampled", [])
        )

        print(f"✅ Fiche identité créée : {identity_data['token_count']} tokens")
        print(f"   Pages échantillonnées : {identity_data.get('pages_sampled', [])}")

        
        enriched_chunks = process_enriched_chunks(doc, chunks)          
            
        # we store them in postgress, with the proper doc_id + keep the chunk ids in order to keep the same ids for qdrant
        chunk_ids = await store_chunks_batch(enriched_chunks, doc_uuid)
        t3 = time.perf_counter()
        
        print("🔥 Chunks stored 🔥")

        # we summarise them to prepare the embedding
        summarised_chunks = await summarise_chunks(enriched_chunks, chunk_ids)
        print("🔥 Chunks smmarized 🔥")
        
        vectorised_chunks = vectorize_documents(summarised_chunks)

        t4 = time.perf_counter()
        print("🔥 Chunks vectorized 🔥")

        original_filename = file.filename if file.filename else "unknown_file"

        await store_vectors_incrementally(vectorized_docs=vectorised_chunks)
        print("🔥 vectored chunks stored 🔥")

        
        
        end_time = time.perf_counter()
        duration = round(end_time - start_time, 2)

        # On sécurise l'affichage pour éviter l'erreur ASCII au cas où
        first_chunk_preview = str(summarised_chunks[0])[:200] # Un aperçu court

        del doc, chunks
        gc.collect() # Force la libération de la RAM sur le Mac
        return {
            "status": "success",
            "doc_id": doc_uuid,
            "filename": original_filename,
            "chunks_stored": len(chunk_ids),
            "first_chunk_summarized": first_chunk_preview,
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
async def query_rag(question: str, limit: int = 20):
    global chat_history

    try:
        # 1. Rewrite : On transforme la question "élève" en question "autonome"
        standalone_query = await rewrite_query(question, chat_history)

        # 2. Retrieve + Rerank + Group (Tout est packagé dans retrieve_chunks maintenant)
        # On récupère directement la liste finale : [Identité, Chunk1, Chunk2, Identité2, ...]
        print("🚀 Step: Retrieving, Reranking and Grouping...")
        final_context = await retrieve_chunks(standalone_query, limit=limit)

        if not final_context:
            return {
                "answer": "Je n'ai pas trouvé d'informations dans mes cours pour répondre à cette question.", 
                "sources": []
            }

        # 3. Generation : On passe le contexte groupé au Professeur
        print("🧠 Step: Generating Answer...")
        answer = await generate_answer_with_history(question, final_context, chat_history)

        # 4. Retour au Frontend
        # 'final_context' contient déjà 'visual_summary', 'text', 'images_urls', etc.
        return {
            "answer": answer,
            "standalone_query": standalone_query,
            "sources": final_context 
        }

    except Exception as e:
        print(f"❌ Erreur critique dans l'endpoint Query: {e}")
        raise HTTPException(status_code=500, detail=str(e))   
    
    
@app.post("/clear-history") #to clear history context of the user, every day, every new chat, ...
async def reset_chat():
    global chat_history
    chat_history = []
    return {"message": "Discussion reset successfully"}




