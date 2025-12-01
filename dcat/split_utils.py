"""
Utilities for creating train/test splits for disjoint clustering
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


def create_cluster_based_split(
    cluster_df: pd.DataFrame,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """
    Create train/test split at the CLUSTER level.
    
    This ensures no information leakage - nodes in test clusters are
    completely held out during training.
    
    Args:
        cluster_df: DataFrame with columns [node, cluster]
        test_ratio: Fraction of clusters to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_cluster_ids: List of cluster IDs for training
        test_cluster_ids: List of cluster IDs for testing
        train_node_ids: List of node IDs in train clusters
        test_node_ids: List of node IDs in test clusters
    """
    np.random.seed(seed)
    
    # Get all unique clusters
    all_clusters = cluster_df['cluster'].unique()
    
    # Filter out single-node clusters (can't form triplets)
    cluster_sizes = cluster_df.groupby('cluster').size()
    valid_clusters = cluster_sizes[cluster_sizes >= 2].index.tolist()
    
    print(f"Total clusters: {len(all_clusters)}")
    print(f"Valid clusters (size >= 2): {len(valid_clusters)}")
    
    # Shuffle and split
    np.random.shuffle(valid_clusters)
    n_test = int(test_ratio * len(valid_clusters))
    
    test_cluster_ids = valid_clusters[:n_test]
    train_cluster_ids = valid_clusters[n_test:]
    
    # Get node IDs
    train_node_ids = cluster_df[cluster_df['cluster'].isin(train_cluster_ids)]['node'].astype(str).tolist()
    test_node_ids = cluster_df[cluster_df['cluster'].isin(test_cluster_ids)]['node'].astype(str).tolist()
    
    print(f"\nTrain/Test Split:")
    print(f"  Train clusters: {len(train_cluster_ids)}")
    print(f"  Test clusters: {len(test_cluster_ids)}")
    print(f"  Train nodes: {len(train_node_ids)}")
    print(f"  Test nodes: {len(test_node_ids)}")
    
    return train_cluster_ids, test_cluster_ids, train_node_ids, test_node_ids


def create_node_based_split(
    cluster_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Create train/test split at the NODE level within each cluster.
    
    This ensures that each cluster has both train and test nodes,
    which is essential when you have only a few clusters.
    Test nodes can be used to evaluate if the model can predict
    their cluster membership based on learned embeddings.
    
    Args:
        cluster_df: DataFrame with columns [node, cluster]
        metadata_df: DataFrame with columns [id, title, abstract]
        test_ratio: Fraction of nodes to use for testing FROM EACH CLUSTER
        seed: Random seed for reproducibility
    
    Returns:
        train_node_ids: List of node IDs for training
        test_node_ids: List of node IDs for testing
    """
    np.random.seed(seed)
    
    train_node_ids = []
    test_node_ids = []
    
    # Get unique clusters
    unique_clusters = cluster_df['cluster'].unique()
    
    print(f"Creating node-based split from {len(unique_clusters)} clusters...")
    
    # Split nodes within each cluster
    for cluster_id in unique_clusters:
        # Get all nodes in this cluster that have metadata
        cluster_nodes = cluster_df[cluster_df['cluster'] == cluster_id]['node'].values
        cluster_nodes = [str(n) for n in cluster_nodes if int(n) in metadata_df['id'].values]
        
        # Skip if cluster is too small
        if len(cluster_nodes) < 2:
            print(f"  Warning: Cluster {cluster_id} has only {len(cluster_nodes)} nodes, skipping")
            continue
        
        # Shuffle and split this cluster's nodes
        np.random.shuffle(cluster_nodes)
        n_test = max(1, int(test_ratio * len(cluster_nodes)))  # At least 1 test node
        
        cluster_test = cluster_nodes[:n_test]
        cluster_train = cluster_nodes[n_test:]
        
        train_node_ids.extend(cluster_train)
        test_node_ids.extend(cluster_test)
        
        print(f"  Cluster {cluster_id}: {len(cluster_nodes)} nodes -> {len(cluster_train)} train, {len(cluster_test)} test")
    
    print(f"\nNode-based Split Summary:")
    print(f"  Train nodes: {len(train_node_ids)} (from all clusters)")
    print(f"  Test nodes: {len(test_node_ids)} (from all clusters)")
    print(f"  Total: {len(train_node_ids) + len(test_node_ids)} nodes")
    
    return train_node_ids, test_node_ids


def get_cluster_statistics(cluster_df: pd.DataFrame) -> Dict:
    """Get statistics about cluster distribution"""
    cluster_sizes = cluster_df.groupby('cluster').size()
    
    stats = {
        'n_clusters': len(cluster_sizes),
        'n_nodes': len(cluster_df),
        'mean_size': cluster_sizes.mean(),
        'median_size': cluster_sizes.median(),
        'min_size': cluster_sizes.min(),
        'max_size': cluster_sizes.max(),
        'single_node_clusters': (cluster_sizes == 1).sum(),
        'valid_clusters': (cluster_sizes >= 2).sum()
    }
    
    return stats


def print_split_info(
    train_node_ids: List[str],
    test_node_ids: List[str],
    cluster_df: pd.DataFrame
):
    """Print detailed information about train/test split"""
    
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT SUMMARY")
    print("="*60)
    
    # Convert to int for matching
    train_nodes_int = [int(n) for n in train_node_ids]
    test_nodes_int = [int(n) for n in test_node_ids]
    
    # Train statistics
    train_df = cluster_df[cluster_df['node'].isin(train_nodes_int)]
    train_cluster_sizes = train_df.groupby('cluster').size()
    
    print("\nTRAIN SET:")
    print(f"  Nodes: {len(train_df)}")
    print(f"  Clusters represented: {len(train_cluster_sizes)}")
    print(f"  Avg nodes per cluster: {train_cluster_sizes.mean():.2f}")
    print(f"  Median nodes per cluster: {train_cluster_sizes.median():.0f}")
    print(f"  Range: [{train_cluster_sizes.min()}, {train_cluster_sizes.max()}]")
    
    # Test statistics
    test_df = cluster_df[cluster_df['node'].isin(test_nodes_int)]
    test_cluster_sizes = test_df.groupby('cluster').size()
    
    print("\nTEST SET:")
    print(f"  Nodes: {len(test_df)}")
    print(f"  Clusters represented: {len(test_cluster_sizes)}")
    print(f"  Avg nodes per cluster: {test_cluster_sizes.mean():.2f}")
    print(f"  Median nodes per cluster: {test_cluster_sizes.median():.0f}")
    print(f"  Range: [{test_cluster_sizes.min()}, {test_cluster_sizes.max()}]")
    
    print("\nCLUSTER DISTRIBUTION:")
    for cluster_id in sorted(cluster_df['cluster'].unique()):
        train_count = len(train_df[train_df['cluster'] == cluster_id])
        test_count = len(test_df[test_df['cluster'] == cluster_id])
        total = train_count + test_count
        print(f"  Cluster {cluster_id}: {total} total -> {train_count} train ({train_count/total*100:.1f}%), {test_count} test ({test_count/total*100:.1f}%)")
    
    print("\n" + "="*60)
