from langchain_core.documents import Document
from dotenv import load_dotenv
from embeddings.create_enhanced_ai_summary import create_ai_enhanced_summary
import json
load_dotenv()

def summarise_chunks(organized_chunks): #chunks already organized != clean format
    """Process all chunks with AI Summaries"""
    
    langchain_documents = []
    total_chunks = len(organized_chunks)
    
    for i in range(total_chunks):

        # Create AI-enhanced summary if chunk has tables/images
        if organized_chunks['tables'] or organized_chunks['images']:
            try:
                enhanced_content = create_ai_enhanced_summary(
                    organized_chunks['text'],
                    organized_chunks['tables'], 
                    organized_chunks['images']
                )
            except Exception as e:
                enhanced_content = organized_chunks['text']
        else:
            enhanced_content = organized_chunks['text'] #in case there is only text, no need to summarize
        
        # Create LangChain Document with rich metadata
        doc = Document(
            page_content=enhanced_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": organized_chunks['text'],
                    "tables_html": organized_chunks['tables'],
                    "images_base64": organized_chunks['images']
                })
            }
        )
        
    
        langchain_documents.append(doc)
    
    print(total_chunks)
    return langchain_documents