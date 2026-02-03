import os
import fitz  # PyMuPDF
from docling.datamodel.base_models import InputFormat   
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def is_scanned_pdf(file_path):
    """
    Détecte si un PDF est scanné ou contient du texte natif.
    Vérifie les 3 premières pages pour déterminer le besoin d'OCR.
    """
    try:
        doc = fitz.open(file_path)
        
        for page_num in range(min(3, len(doc))):
            page = doc[page_num]
            text = page.get_text().strip()
            images = page.get_images()
            
            # Si presque pas de texte mais des images → PDF scanné
            if len(text) < 50 and len(images) > 0:
                doc.close()
                return True
        
        doc.close()
        return False
    
    except Exception as e:
        print(f"⚠️ Erreur détection PDF scanné : {e}")
        # Par défaut, supposer qu'OCR n'est pas nécessaire
        return False


def get_docling_converter(file_path: str):
    """
    Configure le convertisseur Docling avec détection automatique OCR.
    Optimise les performances en activant l'OCR uniquement si nécessaire.
    
    Args:
        file_path: Chemin vers le fichier PDF à analyser
    
    Returns:
        DocumentConverter configuré
    """
    # Détection automatique du besoin d'OCR
    needs_ocr = is_scanned_pdf(file_path)
    
    print(f"📄 PDF {'scanné' if needs_ocr else 'natif'} détecté")
    print(f"   → OCR {'activé' if needs_ocr else 'désactivé'} (gain de temps estimé: {'0s' if needs_ocr else '3-5min'})")
    
    # Configuration du pipeline
    pipeline_options = PdfPipelineOptions()
    
    # Images : extraction pour GPT-4o vision
    pipeline_options.images_scale = 1.75
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    # Tableaux : extraction et conversion en Markdown
    pipeline_options.do_table_structure = True
    
    # OCR : activé uniquement si PDF scanné
    pipeline_options.do_ocr = needs_ocr
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    return converter


def partition_document(file_path: str):
    """
    Partitionne un PDF en utilisant Docling avec détection automatique OCR.
    
    Args:
        file_path: Chemin vers le fichier PDF
    
    Returns:
        DoclingDocument: Document partitionné avec texte, images et tableaux extraits
    
    Raises:
        Exception: Si le partitionnement échoue
    """
    try:
        print(f"🔄 Démarrage du partitionnement : {file_path}")
        
        # Obtenir le converter configuré selon le type de PDF
        converter = get_docling_converter(file_path)
        
        # Conversion du PDF
        result = converter.convert(file_path)
        
        print(f"✅ Document partitionné avec succès : {file_path}")
        print(f"   → Pages traitées : {len(result.document.pages) if hasattr(result.document, 'pages') else 'N/A'}")
        
        return result.document
        
    except Exception as e:
        print(f"❌ Erreur lors du partitionnement Docling : {e}")
        import traceback
        traceback.print_exc()
        raise
