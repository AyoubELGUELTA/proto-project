import os
from docling.datamodel.base_models import InputFormat   
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def get_docling_converter():
    """
    Configure le convertisseur avec les options de vision.
    C'est ici que Docling utilise le GPU/MPS de ton Mac.
    """
    pipeline_options = PdfPipelineOptions()
    
    # 1. Capture des images pour GPT-4.1 Nano
    pipeline_options.images_scale = 2.0 
    pipeline_options.generate_page_images = True  # Permet d'extraire les crops
    
    # 2. Activation de la compréhension des tableaux
    pipeline_options.do_table_structure = True
    
    # 3. OCR (si tes notes sont parfois des scans de qualité moyenne)
    pipeline_options.do_ocr = True 

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    return converter

def partition_document(file_path: str):
    """
    Renvoie l'objet 'doc' complet.
    """
    try:
        converter = get_docling_converter()
        result = converter.convert(file_path)
        
        print(f"✅ Document partitionné avec Docling : {file_path}")
        return result.document
        
    except Exception as e:
        print(f"❌ Erreur lors du partitionnement Docling: {e}")
        raise