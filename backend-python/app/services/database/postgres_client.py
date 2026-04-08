import asyncpg
from typing import Optional
from app.core.settings import settings  

class PostgresClient:
    def __init__(self):
        self.host = settings.db_host
        self.port = settings.db_port
        self.database = settings.db_name
        self.user = settings.db_user
        self.password = settings.db_password
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        if not self._pool:
            try:
                # DEBUG : On affiche l'utilisateur qui tente de se connecter
                print(f"📡 Connexion DB : user={self.user} sur {self.host}:{self.port}")
                
                self._pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    min_size=5,
                    max_size=20
                )
            except Exception as e:
                print(f"❌ Échec de création du pool : {e}")
                raise e

    async def disconnect(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def fetch(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchval(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute(self, query: str, *args):
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)