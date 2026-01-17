
import json
def export_chunks_to_json(chunks, filename="chunks_export.json"):
    """Export processed chunks to clean JSON format"""
    export_data = []
    
    for i, doc in enumerate(chunks):
        chunk_data = {
            "chunk_id": i + 1,
            "enhanced_content": doc.page_content,
            "metadata": {
                "original_content": json.loads(doc.metadata.get("original_content", "{}"))
            }
        }
        export_data.append(chunk_data)
    
    # Save to file for now, to test TO DELETE in production
    with open(filename, 'w', encoding='utf-8') as f: 
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    return export_data