import psycopg2
import json
import os 

def get_connection():
    """
    Retourne une connexion PostgreSQL.
    Changez les paramètres selon votre setup.
    """

    host = os.getenv("DB_HOST") # "postgres" is the name of our service in docker compose
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT", "5432")

    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password,
            port=port
        )
        conn.set_client_encoding("UTF8")

        return conn
    except Exception as e:
        print(f"❌ Erreur de connexion à la base de données : {e}")
        raise

def init_db():
    """
    Initialise la base de données : crée la table 'chunks' si elle n'existe pas.
    """
    conn = get_connection()
    cur = conn.cursor()
        
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

    create_table_query = """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        doc_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_tables JSONB,
        chunk_images_base64 JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_doc_chunk
    ON chunks(doc_id, chunk_index);
    """
    
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized (table 'chunks' checked/created).")



def store_chunks_batch(analyzed_chunks, doc_id):
    init_db()  # ideally move this to app startup later

    insert_query = """
        INSERT INTO chunks (
            doc_id,
            chunk_index,
            chunk_text,
            chunk_tables,
            chunk_images_base64
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING chunk_id;
    """

    chunk_ids = []

    with get_connection() as conn:
        with conn.cursor() as cur:

            # Get starting index
            cur.execute("""
                SELECT COALESCE(MAX(chunk_index), 0)
                FROM chunks
                WHERE doc_id = %s
            """, (doc_id,))
            start_index = cur.fetchone()[0]

            # Insert chunks ONE BY ONE (safe + ordered)
            for i, chunk in enumerate(analyzed_chunks):
                cur.execute(
                    insert_query,
                    (
                        doc_id,
                        start_index + i + 1,
                        chunk["text"],
                        json.dumps(chunk["tables"]),
                        json.dumps(chunk["images_base64"]),
                    )
                )
                chunk_ids.append(cur.fetchone()[0])

        conn.commit()

    return chunk_ids


