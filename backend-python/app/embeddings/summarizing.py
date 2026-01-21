import os 
from langchain_core.documents import Document
from .create_enhanced_ai_summary import create_ai_enhanced_summary

def summarise_chunks(organized_chunks, chunk_ids):
    langchain_documents = []

    for chunk, chunk_id in zip(organized_chunks, chunk_ids):
        text = chunk["text"]
        tables = chunk["tables"]
        images = chunk["images_base64"]

        if tables or images:
            try:
                enhanced_content = create_ai_enhanced_summary(
                    text,
                    tables,
                    images
                )
            except Exception:
                enhanced_content = text
        else:
            enhanced_content = text

        doc = Document(
            page_content=enhanced_content,
            metadata={
                "chunk_id": str(chunk_id)
            }
        )

        langchain_documents.append(doc)

    print('that is the first summarized chunk page content + metadata ' + str(langchain_documents[0]))
    return langchain_documents
