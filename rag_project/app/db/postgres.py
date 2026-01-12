import psycopg2
import json
from psycopg2.extras import execute_values

def get_connection():
    """
    Retourne une connexion PostgreSQL.
    Changez les paramètres selon votre setup.
    """
    conn = psycopg2.connect(
        host="localhost",
        database="ragdb",
        user="admin",
        password="mysecretpassword"
    )
    return conn

def init_db():
    """
    Initialise la base de données : crée la table 'chunks' si elle n'existe pas.
    """
    conn = get_connection()
    cur = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS chunks (
        id SERIAL PRIMARY KEY,
        doc_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT,
        chunk_tables JSONB,
        chunk_images_base64 JSONB,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
    """
    
    cur.execute(create_table_query)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized (table 'chunks' checked/created).")



def store_chunks_batch(analyzed_chunks, doc_id):

    init_db()
    
    conn = get_connection()
    cur = conn.cursor()

    # Get the starting index
    cur.execute("""
        SELECT COALESCE(MAX(chunk_index), 0)
        FROM chunks
        WHERE doc_id = %s
    """, (doc_id,))    

    row = cur.fetchone()
    start_index = row[0] if row is not None else 1 

    #Prepare values to insert
    values = []
    for i, chunk in enumerate(analyzed_chunks):
        values.append((
            doc_id,
            start_index + i,
            chunk["text"],
            json.dumps(chunk["tables"]),
            json.dumps(chunk["images_base64"]),
        ))

    # INSERT batch
    cur.executemany("""
        INSERT INTO chunks (
            doc_id,
            chunk_index,
            chunk_text,
            chunk_tables,
            chunk_images_base64
        )
        VALUES (%s, %s, %s, %s, %s)
    """, values)

    conn.commit()
    cur.close()
    conn.close()
