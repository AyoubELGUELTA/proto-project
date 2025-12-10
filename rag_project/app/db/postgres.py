import psycopg2
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

def create_chunks_table():
    """
    Crée la table chunks si elle n'existe pas encore.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id SERIAL PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INT NOT NULL,
            chunk_type TEXT,
            chunk_text TEXT
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def store_chunks(chunks, doc_id):
    """
    Stocke une liste de chunks dans la base.
    Chaque chunk peut être texte, table ou image (type dans chunk_type)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    # Préparer les données pour insertion
    data = [(doc_id, i, "Text", chunk) for i, chunk in enumerate(chunks)]
    
    # Insertion multiple optimisée
    execute_values(cur,
        "INSERT INTO chunks (doc_id, chunk_index, chunk_type, chunk_text) VALUES %s",
        data
    )
    
    conn.commit()
    cur.close()
    conn.close()
