import logging
import pandas as pd
import networkx as nx
from graspologic.partition import hierarchical_leiden

logger = logging.getLogger(__name__)

def run_clustering(
    relationships: pd.DataFrame, 
    max_cluster_size: int = 15, #TODO centralize
    random_seed: int = 42 #TODO centralize + clarify what does it really quantify??
) -> pd.DataFrame:
    """
    Performs hierarchical community detection using the Leiden algorithm.

    This operation transforms a relationship edge list into a hierarchical 
    structure of communities. It utilizes Graspologic's implementation of 
    Hierarchical Leiden, which is the core engine used in Microsoft's GraphRAG.

    Args:
        relationships (pd.DataFrame): DataFrame containing the graph edges.
            Expected columns: ['source', 'target', 'weight'].
        max_cluster_size (int): The maximum number of nodes allowed in a single 
            community before it attempts further partitioning. Defaults to 15.
        random_seed (int): Seed for reproducibility of the clustering. Defaults to 42.

    Returns:
        pd.DataFrame: A mapping of nodes to their respective communities.
            Columns:
                - 'level': Hierarchy depth (0 is the root/broadest).
                - 'community': The unique ID of the cluster at that level.
                - 'parent': The ID of the parent cluster in the level above.
                - 'node': The unique ID of the entity (source/target).

    Raises:
        ValueError: If the input DataFrame is missing required columns.
        Exception: For underlying clustering algorithm failures.
    """
    # Validate input structure
    required_cols = {'source', 'target', 'weight'}
    if not required_cols.issubset(relationships.columns):
        raise ValueError(f"Relationships DataFrame must contain: {required_cols}")

    if relationships.empty:
        logger.warning("Empty relationships DataFrame provided to clustering.")
        return pd.DataFrame(columns=['level', 'community', 'parent', 'node'])

    try:
        # 1. Build the NetworkX graph from the edge list
        # We use a Graph (undirected) as Leiden usually operates on modularity
        g = nx.from_pandas_edgelist(
            relationships,
            source='source',
            target='target',
            edge_attr='weight'
        )

        logger.debug(f"Graph built with {g.number_of_nodes()} nodes.")

        # 2. Execute Hierarchical Leiden
        # This returns a list of HierarchicalCluster objects
        logger.info("Starting Hierarchical Leiden partitioning...")
        hierarchical_clusters = hierarchical_leiden(
            g,
            max_cluster_size=max_cluster_size,
            random_seed=random_seed
        )

        # 3. Flatten the hierarchical structure into a MC-compliant DataFrame
        cluster_data = []
        for cluster in hierarchical_clusters:
            cluster_data.append({
                "level": cluster.level,
                "community": cluster.cluster,
                "parent": cluster.parent,
                "node": cluster.node
            })

        result_df = pd.DataFrame(cluster_data)
        
        logger.info(
            f"Clustering successful. Created {len(result_df)} node-community assignments."
        )
        
        return result_df

    except Exception as e:
        logger.error(f"Failed to compute hierarchical communities: {e}")
        raise