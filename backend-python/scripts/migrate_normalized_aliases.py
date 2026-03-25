# import asyncio
# from app.db.base import get_connection, release_connection
# from app.db.entities import normalize_entity_name
# from pathlib import Path
# from dotenv import load_dotenv

# env_path = Path(__file__).parent.parent.parent / '.env'
# load_dotenv(dotenv_path=env_path)


# async def migrate_aliases():
#     print("🔄 Démarrage de la migration des alias normalisés...")
#     conn = await get_connection()
    
#     try:
#         # 1. Récupérer toutes les entités qui ont des alias mais pas encore de version normalisée
#         # Ou simplement toutes les entités pour être sûr
#         entities = await conn.fetch("""
#             SELECT entity_id, name, aliases 
#             FROM entities 
#             WHERE aliases IS NOT NULL AND array_length(aliases, 1) > 0
#         """)
        
#         print(f"Il y a {len(entities)} entités à traiter.")
        
#         updated_count = 0
#         for record in entities:
#             e_id = record['entity_id']
#             aliases = record['aliases']
            
#             # 2. Normaliser chaque alias
#             # On ajoute aussi le nom principal dans les alias normalisés pour plus de sécurité
#             all_to_norm = [record['name']] + aliases
#             normalized_list = list(set([normalize_entity_name(a) for a in all_to_norm if a]))
            
#             # 3. Mettre à jour la ligne en BDD
#             await conn.execute("""
#                 UPDATE entities 
#                 SET normalized_aliases = $1 
#                 WHERE entity_id = $2
#             """, normalized_list, e_id)
            
#             updated_count += 1
#             if updated_count % 10 == 0:
#                 print(f"  {updated_count}/{len(entities)} entités mises à jour...")

#         print(f"✅ Migration terminée ! {updated_count} entités mises à jour.")

#     except Exception as e:
#         print(f"❌ Erreur pendant la migration : {e}")
#     finally:
#         await release_connection(conn)

# if __name__ == "__main__":
#     asyncio.run(migrate_aliases())

USELESS_SCRIPT_AFTER_THE_DB_ARCHITECURAL_MODIFICATIONS