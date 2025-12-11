"""
Utilities for creating train/test splits
"""

import numpy as np
from typing import List, Tuple


def create_cluster_based_split(
    cluster_df,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Create cluster-based train/test split.

    Args:
        cluster_df: DataFrame with 'cluster' column
        test_ratio: Fraction of clusters for test
        seed: Random seed

    Returns:
        (train_clusters, test_clusters) as lists
    """
    np.random.seed(seed)

    unique_clusters = cluster_df['cluster'].unique()
    num_test = int(len(unique_clusters) * test_ratio)

    test_clusters = np.random.choice(unique_clusters, size=num_test, replace=False)
    train_clusters = [c for c in unique_clusters if c not in test_clusters]

    return train_clusters, test_clusters.tolist()


def create_node_based_split(
    node_ids: List[str],
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """
    Create node-based train/test split.

    Args:
        node_ids: List of all node IDs
        test_ratio: Fraction of nodes for test
        seed: Random seed

    Returns:
        (train_nodes, test_nodes) as lists
    """
    np.random.seed(seed)

    num_test = int(len(node_ids) * test_ratio)
    test_indices = np.random.choice(len(node_ids), size=num_test, replace=False)

    test_nodes = [node_ids[i] for i in test_indices]
    train_nodes = [node_ids[i] for i in range(len(node_ids)) if i not in test_indices]

    return train_nodes, test_nodes


def print_split_info(train_items, test_items, item_type="items"):
    """
    Print split information.

    Args:
        train_items: List of training items
        test_items: List of test items
        item_type: Type of items (e.g., "clusters", "nodes")
    """
    total = len(train_items) + len(test_items)
    train_pct = len(train_items) / total * 100
    test_pct = len(test_items) / total * 100

    print(f"\nSplit Information:")
    print(f"  Train {item_type}: {len(train_items)} ({train_pct:.1f}%)")
    print(f"  Test {item_type}: {len(test_items)} ({test_pct:.1f}%)")
    print(f"  Total {item_type}: {total}")
