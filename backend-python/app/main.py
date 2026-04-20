
from contextlib import asynccontextmanager

from app.services.startup_service import StartupService
from app.services.database.encyclopedia_repository import EncyclopediaRepository

from app.infrastructure.database.postgres_client import PostgresClient

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



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
