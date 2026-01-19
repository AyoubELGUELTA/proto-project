# 1. Utiliser une version de Python légère mais complète
FROM python:3.11-slim

# Force l'utilisation de l'UTF-8 pour éviter les erreurs d'accents
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8

# 2. Installer les outils tiers (Système) dont ton code a besoin
# On installe Tesseract, les langues, et les outils pour les PDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic-dev \
    libgl1 \
    libglib2.0-0 \
    gcc \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 4. Copier d'abord le fichier requirements pour optimiser le cache Docker
COPY requirements.txt .

# 5. Installer les bibliothèques Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y pdfminer pdfminer.six && \
    pip install --no-cache-dir -r requirements.txt

# 5.1 Pré-télécharger les modèles NLTK (Tokenizers)
RUN python -m nltk.downloader punkt averaged_perceptron_tagger averaged_perceptron_tagger_eng punkt_tab
# 5.2 Pré-télécharger les modèles NLTK et Unstructured
RUN python -m nltk.downloader punkt averaged_perceptron_tagger punkt_tab
RUN python -c "from unstructured_inference.models.base import get_model; get_model('yolox')"

# 6. Copier tout ton code (le dossier src, ingestion, rag, etc.)
COPY . .

# 7. Créer le dossier uploads (que main.py utilise)
RUN mkdir -p uploads

# 8. Exposer le port que FastAPI va utiliser
EXPOSE 8000

# 9. Commande pour démarrer fastApi
# On s'assure que le dossier racine est dans le chemin de recherche de Python
ENV PYTHONPATH=/app
# On lance uvicorn via le module python pour garantir la résolution des imports
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]