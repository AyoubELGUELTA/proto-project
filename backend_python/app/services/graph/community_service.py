import logging
import pandas as pd
import json
from typing import Dict, Any, List
from app.infrastructure.neo4j.client import Neo4jClient
from app.indexing.operations.communities.community_state_evaluator import CommunityStateEvaluator

logger = logging.getLogger(__name__)

class CommunityService:
    """
    Service responsible for Neo4j read/write operations related to graph communities.
    """

    def __init__(self, client: Neo4jClient):
        """
        Initializes the CommunityService with a Neo4j client.
        """
        self.client = client

    async def get_relationships_for_clustering(self) -> pd.DataFrame:
        """
        Extracts all relationships from the graph to feed the clustering algorithm.
        """
        query = """
        MATCH (s:Entity)-[r:RELATED]->(t:Entity)
        RETURN s.id AS source, t.id AS target, r.weight AS weight
        """
        try:
            records = await self.client.execute_query(query)
            if not records:
                logger.warning("⚠️ No relationships found in Neo4j for clustering.")
                return pd.DataFrame(columns=['source', 'target', 'weight'])
            
            df = pd.DataFrame(records)
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0)
            logger.info(f"📊 Extracted {len(df)} relationships for community detection.")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to extract relationships for clustering: {e}")
            raise

    async def get_all_entity_titles(self) -> Dict[str, str]:
        """
        Retrieves a mapping of Entity IDs to their Titles.
        """
        query = "MATCH (e:Entity) RETURN e.id AS id, e.title AS title"
        try:
            records = await self.client.execute_query(query)
            mapping = {r['id']: r['title'] for r in records}
            logger.debug(f"📑 Retrieved title mapping for {len(mapping)} entities.")
            return mapping
        except Exception as e:
            logger.error(f"❌ Failed to retrieve entity titles: {e}")
            raise

    async def save_community_assignments(self, clusters_df: pd.DataFrame):
        """
        Persists community assignments into Neo4j and builds the hierarchy.
        """
        if clusters_df.empty:
            logger.warning("⚠️ No clusters to save.")
            return

        clusters_df['community_uid'] = clusters_df.apply(
            lambda x: f"level_{x['level']}_id_{x['community']}", axis=1
        )
        clusters_df['parent_uid'] = clusters_df.apply(
            lambda x: f"level_{int(x['level'])-1}_id_{x['parent']}" if x['parent'] != -1 else None, 
            axis=1
        )

        entity_link_query = """
        UNWIND $data AS row
        MERGE (c:Community {id: row.community_uid})
        SET c.level = row.level,
            c.id_in_level = row.community,
            c.title = "Community " + row.community_uid
        
        WITH c, row
        MATCH (e:Entity {id: row.node})
        MERGE (e)-[:IN_COMMUNITY]->(c)
        """

        hierarchy_query = """
        UNWIND $data AS row
        WITH row WHERE row.parent_uid IS NOT NULL
        MATCH (child:Community {id: row.community_uid})
        MATCH (parent:Community {id: row.parent_uid})
        MERGE (child)-[:CHILD_OF]->(parent)
        """

        try:
            data_payload = clusters_df.to_dict(orient='records')
            await self.client.execute_query(entity_link_query, {"data": data_payload})
            await self.client.execute_query(hierarchy_query, {"data": data_payload})
            logger.info(f"💾 Successfully persisted {len(clusters_df)} community assignments to Neo4j.")
        except Exception as e:
            logger.error(f"❌ Failed to persist communities: {e}")
            raise

    async def get_communities_analysis_manifest(self, drift_threshold: float = 0.15) -> List[Dict[str, Any]]:
        """
        Scans all graph communities, evaluates their live structural states,
        and computes divergence strategies using the CommunityStateEvaluator.
        
        Returns:
            List[Dict]: A list of actionable targets filtering out skipped workloads.
        """
        # Cypher captures description text lengths as raw semantic mass
        query = """
        MATCH (c:Community)
        OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
        WITH c, collect(distinct e.id) AS entity_ids, 
            count(distinct e) AS current_ent_count, 
            sum(size(coalesce(e.description, ""))) AS current_sem_mass

        OPTIONAL MATCH (src)-[r:RELATED]->(tgt)
        WHERE src.id IN entity_ids AND tgt.id IN entity_ids
        WITH c, current_ent_count, current_sem_mass, count(distinct r) AS current_rel_count

        RETURN c.id AS id,
            c.level AS level,  
            current_ent_count AS current_entity_count,
            current_rel_count AS current_relationship_count,
            current_sem_mass AS current_semantic_mass,
            c.last_report_hash AS last_report_hash,
            c.last_report_entity_count AS last_report_entity_count,
            c.last_report_relationship_count AS last_report_relationship_count,
            c.last_report_semantic_mass AS last_report_semantic_mass
        """
        try:
            records = await self.client.execute_query(query)
            actionable_manifest = []

            for r in records:
                current_hash = CommunityStateEvaluator.generate_fingerprint(
                    entity_count=r["current_entity_count"],
                    relationship_count=r["current_relationship_count"],
                    semantic_mass=r["current_semantic_mass"]
                )
                
                state_map = {**r, "current_hash": current_hash}
                strategy, score = CommunityStateEvaluator.evaluate_divergence(state_map, threshold=drift_threshold)
                
                if strategy != "SKIP":
                    actionable_manifest.append({
                        "id": r["id"],
                        "level": r["level"] or 0,  
                        "strategy": strategy,
                        "score": score,  # On injecte le score pour ton idée de tri !
                        "target_hash": current_hash,
                        "entity_count": r["current_entity_count"],
                        "relationship_count": r["current_relationship_count"],
                        "semantic_mass": r["current_semantic_mass"]
                    })
            
            # Application directe de ton idée : Tri par score décroissant (les plus gros changements d'abord)
            actionable_manifest.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"📋 Manifest prioritised. {len(actionable_manifest)} targets sorted by urgency.")
            return actionable_manifest
        except Exception as e:
            logger.error(f"❌ Failed to construct prioritised community manifest: {e}")
            raise

    async def get_community_context(self, community_id: str) -> Dict[str, Any]:
        """
        Extracts all raw components of a community to form the baseline text context for the LLM.
        """
        query = """
        MATCH (c:Community {id: $community_id})
        
        // 1. Capture all entities in this community
        MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
        WITH c, collect(e) AS nodes
        
        // 2. Capture internal relationships exclusively spanning these entities
        OPTIONAL MATCH (src)-[r:RELATED]->(tgt)
        WHERE src IN nodes AND tgt IN nodes
        
        // 3. Format structured payload output
        RETURN 
            [n IN nodes | {
                id: n.id, 
                title: n.title, 
                type: n.type, 
                description: coalesce(n.description, "No description available")
            }] AS entities,
            collect(distinct {
                source: src.title, 
                target: tgt.title, 
                description: coalesce(r.description, "No details available")
            }) AS relationships
        """
        try:
            records = await self.client.execute_query(query, {"community_id": community_id})
            if not records:
                return {"entities": [], "relationships": []}
            
            context_data = records[0]
            if context_data["relationships"] == [{"source": None, "target": None, "description": "No details available"}]:
                context_data["relationships"] = []
                
            return context_data
        except Exception as e:
            logger.error(f"❌ Failed to build community context for {community_id}: {e}")
            raise

    async def save_community_report(self, community_id: str, title: str, summary: str, findings: List[Dict[str, Any]], state_metadata: Dict[str, Any]):
        """
        Updates a :Community node with its generated semantic report and freezes 
        its current structural fingerprint to track future delta drifts.
        
        Args:
            community_id (str): Target community ID.
            title (str): LLM generated high-level title.
            summary (str): Executive summary text.
            findings (List[Dict]): Structured list of key findings.
            state_metadata (Dict): Contains 'hash', 'entity_count', 'relationship_count', 'semantic_mass', 'rate', and 'rating_explanation'.
        """
        query = """
        MATCH (c:Community {id: $community_id})
        SET c.title = $title,
            c.summary = $summary,
            c.findings_json = $findings_json,
            c.last_report_hash = $last_report_hash,
            c.last_report_entity_count = $last_report_entity_count,
            c.last_report_relationship_count = $last_report_relationship_count,
            c.last_report_semantic_mass = $last_report_semantic_mass,
            c.last_report_rating = $last_report_rating,
            c.last_report_rating_explanation = $last_report_rating_explanation,
            c.updated_at = timestamp()
        """
        try:
            payload = {
                "community_id": community_id,
                "title": title,
                "summary": summary,
                "findings_json": json.dumps(findings, ensure_ascii=False),
                "last_report_hash": state_metadata["hash"],
                "last_report_entity_count": state_metadata["entity_count"],
                "last_report_relationship_count": state_metadata["relationship_count"],
                "last_report_semantic_mass": state_metadata["semantic_mass"],
                "last_report_rating" : state_metadata["rating"],
                "last_report_rating_explanation" : state_metadata["rating_explanation"],
            }
            await self.client.execute_query(query, payload)
            logger.info(f"📝 Saved validated report for [{community_id}] (State frozen via hash lock).")
        except Exception as e:
            logger.error(f"❌ Failed to save community report for {community_id}: {e}")
            raise

    

    async def get_raw_community_payload(self, community_id: str) -> Dict[str, Any]:
        """
        Fetches both raw direct elements AND child sub-reports for a community.
        Pure data provider, no token logic here.
        """
        query = """
        MATCH (c:Community {id: $community_id})

        // 1. Extraction des entités directes et leur degré d'importance
        OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
        WITH c, collect(distinct e) AS entities, collect(distinct e.id) AS entity_ids

        // 2. Extraction des relations internes (on map directement source et target via les IDs des nœuds)
        OPTIONAL MATCH (src:Entity)-[r:RELATED]->(tgt:Entity)
        WHERE src.id IN entity_ids AND tgt.id IN entity_ids
        WITH c, entities, collect(distinct {source: src.id, target: tgt.id, description: coalesce(r.description, "")}) AS relationships

        // 3. Extraction et aggregation immédiate des rapports des sous-communautés enfants
        OPTIONAL MATCH (child:Community)-[:CHILD_OF]->(c)
        WHERE child.report_title IS NOT NULL
        WITH c, entities, relationships, 
            collect(distinct {id: child.id, title: child.report_title, summary: child.report_summary, findings: child.report_findings}) AS sub_reports

        // 4. Assemblage final (Aucune agrégation ici, Neo4j est content !)
        RETURN {
            entities: [ent IN entities | {id: ent.id, title: ent.title, type: ent.type, description: ent.description, degree: coalesce(ent.degree, 0)}],
            relationships: relationships,
            sub_reports: sub_reports
        } AS payload
        """
        records = await self.client.execute_query(query, {"community_id": community_id})
        return records[0]["payload"] if records else {"entities": [], "relationships": [], "sub_reports": []}
    
        #TODO the step 3 of the query makes me 