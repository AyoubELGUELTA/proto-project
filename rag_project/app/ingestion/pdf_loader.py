# app/ingestion/pdf_loader.py
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Table, Image
import os

class PDFElement:
    def __init__(self, type, text=None, filename=None):
        self.type = type  # "Text", "Table", "Image"
        self.text = text
        self.filename = filename

def load_pdf_elements(pdf_path):
    """
    Retourne une liste d'éléments atomiques (texte, tables, images)
    """
    elements = []

    # Partitionne le PDF en objets par unstructured
    partitions = partition_pdf(filename=pdf_path)

    for part in partitions:
        if isinstance(part, Text):
            if part.text.strip():
                elements.append(PDFElement(type="Text", text=part.text.strip()))
        elif isinstance(part, Table):
            elements.append(PDFElement(type="Table", text=part.to_html()))
        elif isinstance(part, Image):
            # Sauvegarde temporaire de l'image pour OCR
            base_dir = os.path.dirname(pdf_path)
            image_filename = os.path.join(base_dir, f"{os.path.basename(pdf_path)}_img_{len(elements)}.png")
            with open(image_filename, "wb") as img_f:
                img_f.write(part.data)
            elements.append(PDFElement(type="Image", filename=image_filename))

    return elements
