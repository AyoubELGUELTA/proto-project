from app.infrastructure.neo4j.client import Neo4jClient

import asyncio
import logging
import json

logger = logging.getLogger(__name__)

async def debug_print_community_composition(client: Neo4jClient):
    """
    Query de diagnostic pour inspecter quelles entités 
    composent chaque communauté dans Neo4j.
    """
    query = """
    MATCH (e:Entity)-[:IN_COMMUNITY]->(c:Community)
    RETURN c.id AS community_id, 
           c.level AS level, 
           collect(e.title) AS entities, 
           count(e) AS size
    ORDER BY level ASC, community_id ASC
    """
    try:
        records = await client.execute_query(query)
        print("\n🔎 ========================================================")
        print("📦 COMPOSITION STRUCTURELLE DES COMMUNAUTÉS")
        print("==========================================================")
        for r in records:
            print(f"\n🔹 [{r['community_id']}] (Level {r['level']}) - Contient {r['size']} entités :")
            print(f"   👉 {', '.join(r['entities'])}")
    except Exception as e:
        logger.error(f"❌ Erreur diagnostic composition : {e}")


async def debug_print_community_reports(client: Neo4jClient):
    """
    Query pour extraire, décoder et afficher les rapports réels
    sauvegardés dans Neo4j (Titres, Résumés, Ratings et Findings).
    """
    # Alignement parfait sur les clés réelles de ta base
    query = """
    MATCH (c:Community)
    RETURN c.id AS community_id,
           c.level AS level,
           c.title AS title,
           c.summary AS summary,
           c.findings_json AS findings_raw,
           c.last_report_rating AS rating,
           c.last_report_rating_explanation AS rating_explanation,
           c.last_report_hash AS state_hash
    ORDER BY level ASC, community_id ASC
    """
    try:
        records = await client.execute_query(query)
        print("\n🔎 ========================================================")
        print("📝 PIPELINE GRAPH-RAG : RAPPORT DES COMMUNAUTÉS")
        print("==========================================================")
        
        for r in records:
            print(f"\n╔════════════════════════════════════════════════════════════════════════")
            print(f"║ 🌐 COMMUNAUTÉ : {r['community_id']} (Level {r['level']})")
            print(f"║ 🔑 Hash de Synchro : {r['state_hash'] or 'N/A'}")
            print(f"╚════════════════════════════════════════════════════════════════════════")
            
            # Gestion du cas particulier où la communauté est vide (comme ton ID 3)
            if r['title'] == "No Data Available" or not r['title']:
                print("   📭 [NO DATA] Cette communauté ne contient aucune entité exploitable.")
                print(f"   📝 Résumé : {r['summary']}")
                continue
                
            print(f"   👑 Titre   : {r['title']}")
            print(f"   ⭐️ Note (Rating) : {r['rating']}/10")
            print(f"   💡 Justification : {r['rating_explanation']}")
            print(f"   📝 Résumé Sémantique :")
            print(f"      {r['summary']}\n")
            
            # Décodage chirurgical du champ findings_json
            findings_raw = r['findings_raw']
            if findings_raw:
                print("   🎯 Key Findings (Points Clés) :")
                
                # C'est stocké en string JSON dans Neo4j, on le retransforme en liste Python
                if isinstance(findings_raw, str):
                    try:
                        findings = json.loads(findings_raw)
                    except Exception as parse_err:
                        print(f"      ⚠️ Impossible de parser le JSON des findings : {parse_err}")
                        findings = []
                else:
                    findings = findings_raw
                        
                if isinstance(findings, list):
                    for idx, f in enumerate(findings, start=1):
                        summary = f.get('summary') or f.get('explanation_by_key') or "Sans titre"
                        explanation = f.get('explanation') or f.get('description') or ""
                        print(f"      {idx}. 📌 [{summary}]")
                        if explanation:
                            # Petit retrait pour aligner l'explication proprement
                            wrapped_exp = explanation.replace("\n", "\n         ")
                            print(f"         ↳ {wrapped_exp}")
                else:
                    print(f"      {findings}")
            else:
                print("   🎯 Key Findings : Aucun point clé enregistré.")
                
        print("\n==========================================================")
    except Exception as e:
        logger.error(f"❌ Erreur lors de la lecture des rapports alignés : {e}", exc_info=True)


async def main():
    # Petit trick pour éviter les logs intempestifs de connexion pendant le print
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    
    client = Neo4jClient()
    await client.connect()
    try:
        # 1. On affiche la structure
        await debug_print_community_composition(client)
        
        # 2. On affiche le contenu textuel généré
        await debug_print_community_reports(client)
        
    except Exception as e:
        logger.error(f"❌ Erreur au lancement du script : {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())