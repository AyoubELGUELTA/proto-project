# 1. Image de base
FROM python:3.11-slim

# Force l'UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Installer les outils système essentiels
# Docling a besoin de libgl1 et libglib2 pour la vision (OpenCV)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libmagic-dev \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Pré-téléchargement des modèles Docling (Optionnel mais recommandé)
# Cela évite que le premier PDF soit très lent à cause du téléchargement
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# 5. Copie du code et préparation
COPY . .
RUN mkdir -p uploads

EXPOSE 8000

ENV PYTHONPATH=/app

# Lancement
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]