# 1. Utiliser une version de Python légère mais complète
FROM python:3.11-slim

# 2. Installer les outils tiers (Système) dont ton code a besoin
# On installe Tesseract, les langues, et les outils pour les PDF
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    poppler-utils \
    libmagic-dev \
    gcc \
    python3-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 3. Définir le dossier de travail dans le conteneur
WORKDIR /app

# 4. Copier d'abord le fichier requirements pour optimiser le cache Docker
COPY requirements.txt .

# 5. Installer les bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier tout ton code (le dossier src, ingestion, rag, etc.)
COPY . .

# 7. Créer le dossier uploads (que ton main.py utilise)
RUN mkdir -p uploads

# 8. Exposer le port que FastAPI va utiliser
EXPOSE 8000

# 9. Commande pour démarrer ton "Chef d'Orchestre"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]