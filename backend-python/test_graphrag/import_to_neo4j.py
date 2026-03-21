import pandas as pd
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "dawa pour toujours"
ROOT_DIR = "./test_graphrag/output"

class Neo4jImporter:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_all(self, nodes_path, rels_path):
        df_nodes = pd.read_parquet(nodes_path)
        df_rels = pd.read_parquet(rels_path)

        with self.driver.session() as session:
            # 1. Nettoyage total
            print("🧹 Nettoyage de la base...")
            session.run("MATCH (n) DETACH DELETE n")
            
            # 2. Import des NOEUDS (on utilise 'title' comme clé de recherche)
            # On stocke l'ID original dans une propriété pour ne rien perdre
            node_query = """
            UNWIND $rows AS row
            MERGE (e:Entity {name: row.title})
            SET e.id = row.id,
                e.type = row.type,
                e.description = row.description,
                e.degree = row.degree
            """
            session.run(node_query, rows=df_nodes.to_dict('records'))
            print(f"✅ {len(df_nodes)} nœuds importés (Indexés sur le nom).")

            # 3. Import des RELATIONS (Liaison via le nom)
            rel_query = """
            UNWIND $rows AS row
            MATCH (source:Entity {name: row.source})
            MATCH (target:Entity {name: row.target})
            MERGE (source)-[r:RELATED]->(target)
            SET r.description = row.description,
                r.weight = row.weight
            """
            session.run(rel_query, rows=df_rels.to_dict('records'))
            print(f"✅ Relations créées entre les entités.")

# --- EXÉCUTION ---
importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
try:
    importer.import_all(
        f"{ROOT_DIR}/entities.parquet", 
        f"{ROOT_DIR}/relationships.parquet"
    )
finally:
    importer.close()