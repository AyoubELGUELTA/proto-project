import asyncpg
import os

_pool = None

async def init_pool():
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            min_size=5, max_size=20
        )
    return _pool

async def get_connection():
    pool = await init_pool()
    return await pool.acquire()

async def release_connection(conn):
    pool = await init_pool()
    await pool.release(conn)