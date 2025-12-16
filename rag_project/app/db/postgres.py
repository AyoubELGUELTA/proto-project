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



def store_chunk(analyzed_chunk, doc_id):
    conn = get_connection()
    cur = conn.cursor()

    # Calcul du prochain index pour ce document
    cur.execute("""
        SELECT COALESCE(MAX(chunk_index), 0) + 1 
        FROM chunks
        WHERE doc_id = %s
    """, (doc_id,)) #COALESCE récupere la premiere valeure non nulle de (a,b,c...)
    
    row = cur.fetchone()
    chunk_index = row[0] if row is not None else 1 

    cur.execute("""
        INSERT INTO chunks (
            doc_id,
            chunk_index,
            chunk_type,
            chunk_text,
            chunk_tables,
            chunk_images_base64
        )
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        doc_id,
        chunk_index,
        analyzed_chunk["types"],
        analyzed_chunk["text"],
        json.dumps(analyzed_chunk["tables"]),
        json.dumps(analyzed_chunk["images_base64"])
    ))

    conn.commit()
    cur.close()
    conn.close()

