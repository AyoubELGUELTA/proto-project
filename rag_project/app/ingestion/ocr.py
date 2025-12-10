import pytesseract
from PIL import Image

def ocr_fallback(element):
    """
    Applique OCR sur un élément de type Image pour récupérer le texte
    """
    if element.type != "Image" or not element.filename:
        return ""

    # Ouvre l'image
    img = Image.open(element.filename)
    # OCR
    text = pytesseract.image_to_string(img, lang="fra")  # français, à adapter
    return text.strip()
