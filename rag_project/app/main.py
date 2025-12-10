from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.ingestion.pdf_loader import load_pdf_elements
from app.ingestion.ocr import ocr_fallback
from app.ingestion.chunker import chunk_elements
import os

app = FastAPI(title="RAG Multi-Modal Prototype")

# Dossier temporaire pour stocker les fichiers uploadés
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Endpoint pour ingérer un PDF :
    1. Sauvegarde temporaire
    2. Extraction multimodale via unstructured + OCR si nécessaire
    3. Chunking
    4. Stockage dans Postgres
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")
    
    # Utiliser le nom du fichier comme doc_id
    doc_id = file.filename
    temp_path = os.path.join(UPLOAD_DIR, doc_id)
    
    # Sauvegarder le fichier temporairement
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    
    # Étape 1 : extraire les éléments atomiques (texte, tables, images)
    elements = load_pdf_elements(temp_path)

    # Étape 2 : fallback OCR si texte vide
    for el in elements:
        if el.type == "Text" and not el.text.strip():
            el.text = ocr_fallback(el)  # ocr_fallback doit retourner une string
    
    # Étape 3 : chunker et stocker dans Postgres
    chunks = chunk_elements(elements, doc_id=doc_id)
    
    # Supprimer le fichier temporaire
    os.remove(temp_path)
    
    return JSONResponse(
        content={
            "doc_id": doc_id,
            "num_elements": len(elements),
            "num_chunks": len(chunks),
            "message": "PDF ingéré avec succès"
        }
    )
