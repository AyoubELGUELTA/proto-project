import asyncio
from app.db.base import get_connection, release_connection
from app.db.entities import normalize_entity_name

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


async def normalize_all_tag_aliases():
    """
    Pour chaque tag avec aliases, génère normalized_aliases.
    """
    conn = await get_connection()
    
    try:
        # Récupère tous les tags avec aliases
        tags = await conn.fetch("""
            SELECT tag_id, label, aliases
            FROM tags
            WHERE aliases IS NOT NULL AND array_length(aliases, 1) > 0
        """)
        
        print(f"🔄 Normalisation de {len(tags)} tags...\n")
        
        for tag in tags:
            tag_id = tag['tag_id']
            label = tag['label']
            aliases = tag['aliases']
            
            # Normalise chaque alias
            normalized = []
            for alias in aliases:
                norm = normalize_entity_name(alias)
                if norm:  # Skip si vide après normalization
                    normalized.append(norm)
            
            # Ajoute aussi le label normalisé pour matching complet
            label_norm = normalize_entity_name(label)
            if label_norm and label_norm not in normalized:
                normalized.insert(0, label_norm)  # En premier
            
            # Update BDD
            await conn.execute("""
                UPDATE tags
                SET normalized_aliases = $1
                WHERE tag_id = $2
            """, normalized, tag_id)
            
            print(f"✅ {label}")
            print(f"   Aliases : {aliases}")
            print(f"   Normalized : {normalized}\n")
        
        print(f"✅ {len(tags)} tags normalisés")
        
    finally:
        await release_connection(conn)

if __name__ == "__main__":
    asyncio.run(normalize_all_tag_aliases())