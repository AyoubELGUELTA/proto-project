import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

import logging
logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    Handles the technical conversion of raw files into structured Docling documents.
    
    This loader acts as the ingestion gateway, determining the necessary 
    processing depth (Standard vs. OCR) based on the document's internal structure.
    """
    
    @staticmethod
    def needs_ocr(file_path: str) -> bool:
        """
        Detects if a PDF is likely a scan by analyzing its native text content.
        
        It inspects the first 3 pages. If the average text density is below 
        a certain threshold, it assumes OCR is required to extract meaning.
        
        Args:
            file_path: Path to the PDF file.
        Returns:
            bool: True if OCR should be enabled, False otherwise.
        """
        try:
            with fitz.open(file_path) as doc:
                # We analyze the first few pages to form a representative sample
                for page in list(doc)[:3]: 
                    text_content = page.get_text().strip()
                    if len(text_content) > 50:
                        logger.debug(f"📄 Native text found on page {page.number}. OCR not strictly required.")
                        return False
            
            logger.warning(f"📸 No significant native text found in {file_path}. Enabling OCR mode.")
            return True
        except Exception as e:
            logger.error(f"❌ Error during OCR detection for {file_path}: {e}")
            return False

    @classmethod
    def get_converter(cls, file_path: str) -> DocumentConverter:
        """
        Configures and returns a DocumentConverter tailored to the specific file.
        
        The configuration includes table structure recognition and image 
        generation, while dynamically toggling the OCR engine based on 
        the file's nature.
        
        Args:
            file_path: Path to the document to be converted.
        Returns:
            A configured DocumentConverter instance.
        """
        ocr_enabled = cls.needs_ocr(file_path)
        
        logger.info(f"⚙️ Configuring Docling pipeline (OCR: {ocr_enabled}, Tables: True)")
        
        options = PdfPipelineOptions()
        options.do_ocr = ocr_enabled
        options.do_table_structure = True
        options.generate_page_images = True # Required for downstream image/figure extraction
        
        # Note: OCR might significantly increase processing time
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )