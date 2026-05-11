import asyncio
from app.db.neo4j.connection import test_connection
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

async def main():
    await test_connection()

if __name__ == "__main__":
    asyncio.run(main())