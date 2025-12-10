from langchain_text_splitters import RecursiveCharacterTextSplitter
from ..db.postgres import create_chunks_table, store_chunks

def chunk_elements(elements, doc_id='not_found.pdf', chunk_size=500, overlap=50):
    """
    Découpe les éléments atomiques en chunks intelligents :
    - Texte : découpe en utilisant sauts de ligne ou points pour ne pas couper les phrases
    - Tables / Images : un chunk par élément
    """
    # Crée la table si elle n'existe pas encore
    create_chunks_table()
    
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?"],
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    for el in elements:
        if el.type == "Text":
            text = el.text.strip()
            if text:
                text_chunks = splitter.split_text(text)
                chunks.extend(text_chunks)
        elif el.type == "Table":
            chunks.append(el.to_html())
        elif el.type == "Image":
            chunks.append(f"[IMAGE: {el.filename}]")

    # Stocke les chunks dans Postgres
    store_chunks(chunks, doc_id)

    return chunks
