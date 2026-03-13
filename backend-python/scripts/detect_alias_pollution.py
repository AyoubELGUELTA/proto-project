import asyncio
from app.db.base import get_connection, release_connection
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

async def detect_alias_pollution():
    conn = await get_connection()
    
    try:
        entities = await conn.fetch("""
            SELECT entity_id, name, normalized_name, normalized_aliases, chunk_count
            FROM entities
            WHERE entity_type = 'PERSONNE'
        """)
        
        print(f"\n{'='*80}")
        print(f"🔍 DÉTECTION POLLUTION ALIASES (avec logique diminutifs)")
        print(f"{'='*80}\n")
        
        pollutions = []
        
        for entity_a in entities:
            entity_a_id = str(entity_a['entity_id'])
            entity_a_name = entity_a['name']
            entity_a_normalized = entity_a['normalized_name']
            aliases = entity_a['normalized_aliases'] or []
            
            for alias in aliases:
                # ═══════════════════════════════════════════════════════════
                # RÈGLE 1 : Si alias ⊂ nom de A → Diminutif légitime, SKIP
                # ═══════════════════════════════════════════════════════════
                if alias in entity_a_normalized:
                    continue  # ✅ Diminutif légitime
                
                # ═══════════════════════════════════════════════════════════
                # RÈGLE 2 : Si alias ⊂ nom d'une AUTRE entité → Pollution
                # ═══════════════════════════════════════════════════════════
                for entity_b in entities:
                    entity_b_id = str(entity_b['entity_id'])
                    entity_b_normalized = entity_b['normalized_name']
                    
                    # Skip si même entité
                    if entity_a_id == entity_b_id:
                        continue
                    
                    # Si alias contenu dans le nom de B
                    if alias in entity_b_normalized:
                        
                        # ⚠️ EXCEPTION : Homonymes (les 2 commencent par l'alias)
                        if entity_a_normalized.startswith(alias) and entity_b_normalized.startswith(alias):
                            # Exemple : "ali" légitime pour "Ali ibn Abi Talib" ET "Ali ibn al-Husayn"
                            continue
                        
                        # ❌ POLLUTION détectée
                        pollutions.append({
                            'entity': entity_a_name,
                            'entity_id': entity_a_id,
                            'polluted_alias': alias,
                            'conflicts_with': entity_b['name'],
                            'conflict_type': 'substring_cross_entity'
                        })
        
        # Affichage...
        if not pollutions:
            print("✅ Aucune pollution détectée !\n")
        else:
            print(f"❌ {len(pollutions)} pollution(s) détectée(s)\n")
            
            # Déduplique (même alias peut apparaître plusieurs fois)
            unique_pollutions = {}
            for p in pollutions:
                key = (p['entity_id'], p['polluted_alias'])
                if key not in unique_pollutions:
                    unique_pollutions[key] = p
            
            for p in unique_pollutions.values():
                print(f"Entity: {p['entity']}")
                print(f"  → alias '{p['polluted_alias']}'")
                print(f"  → conflit avec '{p['conflicts_with']}'")
                print()
            
            # SQL cleanup
            pollution_by_entity = {}
            for p in unique_pollutions.values():
                eid = p['entity_id']
                if eid not in pollution_by_entity:
                    pollution_by_entity[eid] = {
                        'name': p['entity'],
                        'aliases_to_remove': []
                    }
                pollution_by_entity[eid]['aliases_to_remove'].append(p['polluted_alias'])
            
            print(f"\n{'='*80}")
            print("🔧 SQL CLEANUP")
            print(f"{'='*80}\n")
            
            for eid, data in pollution_by_entity.items():
                aliases = list(set(data['aliases_to_remove']))
                print(f"-- {data['name']}")
                for alias in aliases:
                    print(f"UPDATE entities SET normalized_aliases = array_remove(normalized_aliases, '{alias}') WHERE entity_id = '{eid}';")
                print()
        
        return pollutions
        
    finally:
        await release_connection(conn)
if __name__ == "__main__":
    asyncio.run(detect_alias_pollution())