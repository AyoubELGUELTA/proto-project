from fastapi import FastAPI, UploadFile, File
import os
import shutil
import uuid

from .ingestion.pdf_loader import partition_document
from .ingestion.chunker import create_chunks
from .ingestion.analyze_content import separate_content_types
from .db.postgres import store_chunk

app = FastAPI(title="Dawask RAG Prototype")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload d'un PDF et ingestion complète dans la base
    """
    # 1️⃣ Sauvegarde locale du PDF

    doc_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2️⃣ Partition du PDF
    elements = partition_document(file_path)

    # 3️⃣ Chunking
    chunks = create_chunks(elements)

    # 4️⃣ Analyse + stockage
    for chunk in chunks:
        analyzed_chunk = separate_content_types(chunk)
        store_chunk(analyzed_chunk, doc_id)

    return {
        "status": "success",
        "document": doc_id,
        "chunks_stored": len(chunks)
    }




