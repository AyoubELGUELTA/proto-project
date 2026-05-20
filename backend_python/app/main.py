# app/main.py
from contextlib import asynccontextmanager

from app.services.startup_service import StartupService
from app.services.database.encyclopedia_repository import EncyclopediaRepository
from app.infrastructure.database.postgres_client import PostgresClient

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- IMPORTATION DES ROUTEURS ---
# On importe les routeurs de tes fichiers v1
from app.api.v1.ingest import router as ingest_router
from app.api.v1.graph.communities import router as communities_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup: Initialisation de la DB et de l'encyclopédie au démarrage
    db = PostgresClient()
    await db.connect()
    repo = EncyclopediaRepository(db)
    startup = StartupService(db, repo)
    
    await startup.initialize_encyclopedia() # Migration faite UNE FOIS au lancement
    
    yield

    await db.disconnect()
    # Cleanup si besoin


app = FastAPI(
    title="Dawask RAG Prototype",
    lifespan=lifespan  
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DECLARATION DES ROUTES ---

# 1. Route d'ingestion principale (gère le endpoint configuré dans ingest.py, ex: /ingest)
app.include_router(ingest_router)

# 2. Nouvelle route analytique pour les rapports de communautés
# En ajoutant prefix="/graph", l'URL finale devient : /graph/communities/refresh-reports
app.include_router(communities_router, prefix="/graph")