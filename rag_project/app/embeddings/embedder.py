
from langchain_openai import OpenAIEmbeddings

def vectorize_documents(docs, embeddings_model=OpenAIEmbeddings(model="text-embedding-3-large")):
    """Convert LangChain Document objects into embedding vectors"""
    vectorized_docs = []
    
    for doc in docs:
        vector = embeddings_model.embed_query(doc.enhanced_content)  # Returns a list[float]
        vectorized_docs.append({
            "metadata": doc.metadata, #basically the original elements: "rawtext", "tables_html", and "images_base64"
            "vector": vector #float numbers
        })
    
    return vectorized_docs

