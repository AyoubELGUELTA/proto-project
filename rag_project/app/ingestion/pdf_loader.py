# app/ingestion/pdf_loader.py
# app/ingestion/pdf_loader.py
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image

def partition_document(file_path: str, output_image_dir="images"):
    """Extract elements from PDF using unstructured and save images locally"""
    
    # Crée le dossier pour les images s'il n'existe pas
    os.makedirs(output_image_dir, exist_ok=True)
    
    elements = partition_pdf(
    filename=file_path,
    strategy="hi_res",
    infer_table_structure=True,
    extract_image_block_types=["Image"],
    extract_image_block_to_payload=True,
    extract_image_block_output_dir=output_image_dir
)
    
    # Parcours les éléments pour sauvegarder les images localement
    for el in elements:
        if isinstance(el, Image):
            # ajoute un attribut file_path pour le chemin local
            imageId = el._element_id

            el.metadata.image_path = os.path.join(output_image_dir, str(imageId)+".jpg")
    
    return elements



