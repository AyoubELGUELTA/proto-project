import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

class DocumentLoader:
    """Responsable de la conversion technique du fichier en document structuré."""
    
    @staticmethod
    def needs_ocr(file_path: str) -> bool:
        """Détecte si le PDF est scanné (manque de texte natif)."""
        try:
            with fitz.open(file_path) as doc:
                for page in list(doc)[:3]: # Analyse les 3 premières pages
                    if len(page.get_text().strip()) > 50:
                        return False
            return True
        except Exception:
            return False

    @classmethod
    def get_converter(cls, file_path: str) -> DocumentConverter:
        ocr_enabled = cls.needs_ocr(file_path)
        
        options = PdfPipelineOptions()
        options.do_ocr = ocr_enabled
        options.do_table_structure = True
        options.generate_page_images = True # Nécessaire pour l'extraction d'images
        
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )