import asyncio
from app.infrastructure.database.postgres_client import PostgresClient


async def main():
    client = PostgresClient()
    await client.connect()
    await client.initialize_schema()
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())