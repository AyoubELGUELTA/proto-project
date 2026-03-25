# import asyncio
# import os
# import uuid
# from pathlib import Path
# from dotenv import load_dotenv

# # --- CONFIGURATION ENV ---
# env_path = Path(__file__).parent.parent.parent / '.env' # a ajuste selon la structure
# load_dotenv(dotenv_path=env_path)

# from app.utils.summarize_and_extract_entities import process_single_chunk
# from app.db.entities import link_entity_to_chunk
# from app.db.base import get_connection, release_connection

# async def test_ingestion_flow():
#     print("🚀 Démarrage du test d'ingestion avec Tagging...")

#     # 1. Préparation d'un chunk de test
#     test_chunk_id = str(uuid.uuid4())
#     test_chunk_data = {
#         "text": "Aïcha bint Abi Bakr, qu'Allah l'agrée, était connue pour sa grande science. En tant que Mère des Croyants, elle a transmis des milliers de hadiths.",
#         "heading_full": "Histoire > Les épouses du Prophète",
#         "tables": [],
#         "images_urls": []
#     }

#     print(f"\n📝 Analyse du chunk par le LLM...")
#     # 2. Test de l'analyse LLM
#     analysis_result = await process_single_chunk(test_chunk_data, test_chunk_id)
    
#     entities = analysis_result.get("entities", [])
#     print(f"🔍 Entités extraites : {len(entities)}")
    
#     for ent in entities:
#         print(f"   - Nom: {ent['name']}")
#         print(f"   - Type: {ent['type']}")
#         print(f"   - Tag suggéré: {ent.get('suggested_tag')}")
#         print(f"   - Aliases: {ent.get('aliases')}")

#     # 3. Test du Linking et du Gatekeeper en BDD
#     print(f"\n💾 Insertion en base de données et validation des tags...")
#     try:
#         for entity in entities:
#             await link_entity_to_chunk(test_chunk_id, entity)
        
#         # 4. Vérification finale en BDD
#         conn = await get_connection()
        
#         # Vérifions si l'entité a bien reçu son tag système
#         report = await conn.fetch("""
#             SELECT e.name, t.label as tag_label, et.confidence
#             FROM entities e
#             JOIN entity_tags et ON e.entity_id = et.entity_id
#             JOIN tags t ON et.tag_id = t.tag_id
#             WHERE e.normalized_name = 'aisha ibn abi bakr' 
#                OR e.name ILIKE '%Aïcha%'
#         """)
        
#         if report:
#             print(f"\n✅ SUCCÈS : L'entité a été liée aux tags suivants :")
#             for row in report:
#                 print(f"   🏆 Tag: {row['tag_label']} (Confiance: {row['confidence']})")
#         else:
#             print(f"\n❌ ÉCHEC : Aucun tag système n'a été lié à l'entité.")
            
#         await release_connection(conn)

#     except Exception as e:
#         print(f"❌ Erreur durant le linking : {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     asyncio.run(test_ingestion_flow())

USELESS_SCRIPT_AFTER_THE_DB_ARCHITECURAL_MODIFICATIONS