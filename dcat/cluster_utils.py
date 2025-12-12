"""
Disjoint clustering utilities for navigating and computing cluster membership
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import random


class DisjointClustering:
    """Parse and navigate disjoint clustering structure"""

    def __init__(self, clustering_df: pd.DataFrame):
        """
        Load clustering from DataFrame

        Args:
            clustering_df: DataFrame with columns [node_id, cluster_id]
                          where node_id is the node/document ID
                          and cluster_id is the cluster it belongs to
        """
        # Ensure both columns are strings for consistent lookups
        self.clustering_df = clustering_df.copy()
        self.clustering_df['node_id'] = self.clustering_df['node_id'].astype(str)
        self.clustering_df['cluster_id'] = self.clustering_df['cluster_id'].astype(str)

        # Build lookup: node_id -> cluster_id
        self.node_to_cluster = dict(
            zip(self.clustering_df['node_id'], self.clustering_df['cluster_id'])
        )

        # Build reverse lookup: cluster_id -> list of node_ids
        self.cluster_to_nodes = {}
        for node_id, cluster_id in self.node_to_cluster.items():
            if cluster_id not in self.cluster_to_nodes:
                self.cluster_to_nodes[cluster_id] = []
            self.cluster_to_nodes[cluster_id].append(node_id)

        # Store all unique node IDs and cluster IDs
        self.all_nodes = list(self.node_to_cluster.keys())
        self.all_clusters = list(self.cluster_to_nodes.keys())

        print(f"Loaded clustering:")
        print(f"  Total nodes: {len(self.all_nodes)}")
        print(f"  Total clusters: {len(self.all_clusters)}")
        print(f"  Avg cluster size: {len(self.all_nodes) / len(self.all_clusters):.1f}")

    def get_cluster_id(self, node_id: str) -> Optional[str]:
        """Get cluster ID for a given node"""
        return self.node_to_cluster.get(str(node_id))

    def get_nodes_in_cluster(self, cluster_id: str) -> List[str]:
        """Get all nodes in a given cluster"""
        return self.cluster_to_nodes.get(str(cluster_id), [])

    def are_in_same_cluster(self, node_id1: str, node_id2: str) -> bool:
        """Check if two nodes are in the same cluster"""
        cluster1 = self.get_cluster_id(str(node_id1))
        cluster2 = self.get_cluster_id(str(node_id2))

        if cluster1 is None or cluster2 is None:
            return False

        return cluster1 == cluster2

    def cluster_distance(self, node_id1: str, node_id2: str) -> float:
        """
        Compute distance between two nodes based on cluster membership.

        Returns:
            0.0 if nodes are in the same cluster
            1.0 if nodes are in different clusters
        """
        return 0.0 if self.are_in_same_cluster(node_id1, node_id2) else 1.0

    def sample_triplet_with_distances(
        self,
        anchor_id: str,
    ) -> Tuple[str, str, float, float]:
        """
        Sample positive and negative with their cluster distances.

        For disjoint clustering:
        - Positive: same cluster as anchor (distance = 0.0)
        - Negative: different cluster from anchor (distance = 1.0)

        Args:
            anchor_id: ID of the anchor node

        Returns:
            (positive_id, negative_id, distance_to_positive, distance_to_negative)
        """
        anchor_id = str(anchor_id)
        anchor_cluster = self.get_cluster_id(anchor_id)

        if anchor_cluster is None:
            raise ValueError(f"Node {anchor_id} not found in clustering")

        # Get nodes in same cluster (excluding anchor itself)
        same_cluster_nodes = [
            n for n in self.get_nodes_in_cluster(anchor_cluster)
            if n != anchor_id
        ]

        # Get nodes in different clusters
        different_cluster_nodes = [
            n for n in self.all_nodes
            if n != anchor_id and self.get_cluster_id(n) != anchor_cluster
        ]

        # Sample positive from same cluster
        if len(same_cluster_nodes) == 0:
            # Edge case: anchor is alone in its cluster
            # Sample any other node as positive (will have distance 1.0)
            positive_id = random.choice([n for n in self.all_nodes if n != anchor_id])
        else:
            positive_id = random.choice(same_cluster_nodes)

        # Sample negative from different cluster
        if len(different_cluster_nodes) == 0:
            # Edge case: only one cluster exists
            # Sample any other node as negative
            negative_id = random.choice([n for n in self.all_nodes if n != anchor_id and n != positive_id])
        else:
            negative_id = random.choice(different_cluster_nodes)

        # Compute distances
        dist_pos = self.cluster_distance(anchor_id, positive_id)
        dist_neg = self.cluster_distance(anchor_id, negative_id)

        return positive_id, negative_id, dist_pos, dist_neg
