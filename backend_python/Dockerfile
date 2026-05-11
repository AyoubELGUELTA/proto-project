# 1. Utilise l'image Python qui est déjà passée
FROM python:3.11-slim

# Force l'UTF-8
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 2. Installer les outils système + CURL pour installer UV
RUN apt-get update && apt-get install -y \
    curl \
    tesseract-ocr \
    tesseract-ocr-fra \
    libmagic-dev \
    libgl1 \
    libglib2.0-0 \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Installer UV via le script (souvent mieux toléré par les proxies que le pull Docker)
# On utilise le flag -k (insecure) si vraiment le certificat de l'université bloque
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# 4. Installation des dépendances
COPY requirements.txt .
# On ajoute --insecure ici au cas où le proxy bloque les certificats PyPI
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -r requirements.txt

# 5. Docling & Models (baked in)
RUN python -c "from docling.document_converter import DocumentConverter; DocumentConverter()"

# 6. Finalisation
COPY . .
RUN mkdir -p uploads app/test/output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]