from fastapi import FastAPI, UploadFile, File
import os
import shutil
import uuid

from .ingestion.pdf_loader import partition_document
from .ingestion.chunker import create_chunks
from .ingestion.analyze_content import separate_content_types
from .db.postgres import store_chunks_batch
from .embeddings.cleaning_chunks_format import export_chunks_to_json
from .embeddings.embedder import vectorize_documents
from .embeddings.summarizing import summarise_chunks

app = FastAPI(title="Dawask RAG Prototype")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload d'un PDF et ingestion complète dans la base
    """
    # local save of the PDF

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    #  Partition of PDF
    elements = partition_document(file_path)

    #  Chunking
    chunks = create_chunks(elements)

    # Seperate the content type of each chunk

    analyzed_chunks = []
    for chunk in chunks:
        analyzed_chunks.append(separate_content_types(chunk))
    
    # we store them in postgress, with the proper doc_id
    store_chunks_batch(analyzed_chunks, doc_id)

    # we summarise them to prepare the embedding
    summarised_chunks = summarise_chunks(analyzed_chunks)

    vectorised_chunks = vectorize_documents(summarised_chunks)



    
    
    
    return {
        "status": "success",
        "document": doc_id,
        "chunks_stored": len(chunks)
    }




