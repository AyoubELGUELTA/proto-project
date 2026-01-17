import os 
from langchain_core.documents import Document
from .create_enhanced_ai_summary import create_ai_enhanced_summary

def summarise_chunks(organized_chunks):
    langchain_documents = []

    for chunk in organized_chunks:
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
                "original_content": {
                    "raw_text": text,
                    "tables_html": tables,
                    "images_base64": images
                }
            }
        )

        langchain_documents.append(doc)

    print('that is the first summarized chunk page content + metadata ' + str(langchain_documents[0]))
    return langchain_documents
