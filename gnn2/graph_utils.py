"""
Graph utilities for creating induced subgraphs and preparing GNN data
"""

import pandas as pd
import torch
import numpy as np
from typing import List, Set, Tuple, Dict


def create_induced_subgraph(
    edgelist_path: str,
    train_node_ids: List[str],
    metadata_df: pd.DataFrame
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Create induced subgraph containing only edges between training nodes.

    This is the key function for your baseline: test nodes will have NO edges
    in this graph, so the GNN won't be able to propagate information to them.

    Args:
        edgelist_path: Path to CSV with columns [source, target]
        train_node_ids: List of training node IDs (as strings)
        metadata_df: Metadata DataFrame with 'id' column

    Returns:
        edge_index: Tensor of edges [2, num_edges]
        node_to_idx: Mapping from node_id (str) -> index (int)
        idx_to_node: Mapping from index (int) -> node_id (str)
    """
    # Load full edgelist
    print(f"Loading edgelist from {edgelist_path}...")
    edgelist_df = pd.read_csv(edgelist_path)
    print(f"  Full graph: {len(edgelist_df)} edges")

    # Convert to string for consistency
    edgelist_df['source'] = edgelist_df['source'].astype(str)
    edgelist_df['target'] = edgelist_df['target'].astype(str)

    # Create set of training nodes for fast lookup
    train_nodes_set = set(train_node_ids)

    # Filter to only edges between training nodes (INDUCED SUBGRAPH)
    print(f"\nFiltering to induced subgraph of {len(train_node_ids)} training nodes...")
    filtered_edges = edgelist_df[
        edgelist_df['source'].isin(train_nodes_set) &
        edgelist_df['target'].isin(train_nodes_set)
    ]
    print(f"  Induced subgraph: {len(filtered_edges)} edges")
    print(f"  Removed {len(edgelist_df) - len(filtered_edges)} edges involving test nodes")

    # Create node index mapping
    # Include ALL nodes (train + test) but only edges for train nodes
    all_node_ids = [str(nid) for nid in metadata_df['id'].values]
    node_to_idx = {node_id: idx for idx, node_id in enumerate(all_node_ids)}
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

    print(f"\nNode mapping:")
    print(f"  Total nodes: {len(all_node_ids)}")
    print(f"  Train nodes: {len(train_node_ids)}")
    print(f"  Test nodes: {len(all_node_ids) - len(train_node_ids)}")

    # Convert edges to indices
    edge_list = []
    for _, row in filtered_edges.iterrows():
        src_idx = node_to_idx[row['source']]
        tgt_idx = node_to_idx[row['target']]

        # Add both directions (undirected graph)
        edge_list.append([src_idx, tgt_idx])
        edge_list.append([tgt_idx, src_idx])

    # Convert to tensor
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        # Empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    print(f"\nFinal edge_index shape: {edge_index.shape}")
    print(f"  Directed edges: {edge_index.shape[1]}")

    # Verify test nodes have no edges
    test_nodes = set(all_node_ids) - train_nodes_set
    test_indices = {node_to_idx[nid] for nid in test_nodes}

    edges_involving_test = 0
    if edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
            if src in test_indices or tgt in test_indices:
                edges_involving_test += 1

    print(f"\n✓ Verification: {edges_involving_test} edges involve test nodes (should be 0)")

    return edge_index, node_to_idx, idx_to_node


def create_train_only_graph(
    edgelist_path: str,
    train_node_ids: List[str]
) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    """
    Create graph containing ONLY training nodes with remapped indices.

    Unlike create_induced_subgraph (which maps all nodes but only has train edges),
    this creates a smaller graph with only train nodes and remapped indices 0..N-1.

    Args:
        edgelist_path: Path to CSV with columns [source, target]
        train_node_ids: List of training node IDs (as strings)

    Returns:
        edge_index: Tensor of edges [2, num_edges] with indices 0..N-1
        node_to_idx: Mapping from node_id (str) -> index (int) for TRAIN nodes only
        idx_to_node: Mapping from index (int) -> node_id (str) for TRAIN nodes only
    """
    # Load full edgelist
    print(f"Loading edgelist from {edgelist_path}...")
    edgelist_df = pd.read_csv(edgelist_path)
    print(f"  Full graph: {len(edgelist_df)} edges")

    # Convert to string for consistency
    edgelist_df['source'] = edgelist_df['source'].astype(str)
    edgelist_df['target'] = edgelist_df['target'].astype(str)

    # Create set of training nodes for fast lookup
    train_nodes_set = set(train_node_ids)

    # Filter to only edges between training nodes
    print(f"\nFiltering to edges between {len(train_node_ids)} training nodes...")
    filtered_edges = edgelist_df[
        edgelist_df['source'].isin(train_nodes_set) &
        edgelist_df['target'].isin(train_nodes_set)
    ]
    print(f"  Filtered edges: {len(filtered_edges)}")

    # Create node index mapping for TRAIN NODES ONLY (0 to N-1)
    train_node_list = sorted(train_node_ids)
    node_to_idx = {node_id: idx for idx, node_id in enumerate(train_node_list)}
    idx_to_node = {idx: node_id for node_id, idx in node_to_idx.items()}

    print(f"\nNode mapping (train only):")
    print(f"  Train nodes: {len(train_node_list)}")
    print(f"  Index range: 0 to {len(train_node_list) - 1}")

    # Convert edges to indices
    edge_list = []
    for _, row in filtered_edges.iterrows():
        src_idx = node_to_idx[row['source']]
        tgt_idx = node_to_idx[row['target']]

        # Add both directions (undirected graph)
        edge_list.append([src_idx, tgt_idx])
        edge_list.append([tgt_idx, src_idx])

    # Convert to tensor
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    print(f"\nFinal edge_index shape: {edge_index.shape}")
    print(f"  Directed edges: {edge_index.shape[1]}")

    return edge_index, node_to_idx, idx_to_node


def analyze_graph_statistics(
    edge_index: torch.Tensor,
    train_node_ids: List[str],
    node_to_idx: Dict[str, int],
    metadata_df: pd.DataFrame
):
    """
    Analyze graph statistics for train vs test nodes.

    This will show that test nodes have degree 0 in the induced subgraph.

    Args:
        edge_index: Graph edges [2, num_edges]
        train_node_ids: List of training node IDs
        node_to_idx: Mapping from node_id -> index
        metadata_df: Metadata DataFrame
    """
    print("\n" + "="*70)
    print("GRAPH STATISTICS")
    print("="*70)

    all_node_ids = [str(nid) for nid in metadata_df['id'].values]
    num_nodes = len(all_node_ids)
    train_set = set(train_node_ids)

    # Compute node degrees
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.shape[1] > 0:
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i].item()
            degrees[src] += 1

    # Separate train and test statistics
    train_degrees = []
    test_degrees = []

    for node_id in all_node_ids:
        idx = node_to_idx[node_id]
        degree = degrees[idx].item()

        if node_id in train_set:
            train_degrees.append(degree)
        else:
            test_degrees.append(degree)

    # Print statistics
    print(f"\nTraining nodes ({len(train_degrees)} nodes):")
    if len(train_degrees) > 0:
        print(f"  Mean degree: {np.mean(train_degrees):.2f}")
        print(f"  Median degree: {np.median(train_degrees):.0f}")
        print(f"  Max degree: {np.max(train_degrees)}")
        print(f"  Isolated nodes: {sum(1 for d in train_degrees if d == 0)}")

    print(f"\nTest nodes ({len(test_degrees)} nodes):")
    if len(test_degrees) > 0:
        print(f"  Mean degree: {np.mean(test_degrees):.2f}")
        print(f"  Median degree: {np.median(test_degrees):.0f}")
        print(f"  Max degree: {np.max(test_degrees)}")
        print(f"  Isolated nodes: {sum(1 for d in test_degrees if d == 0)} (should be ALL)")

    print("\n" + "="*70)


def create_test_split(node_ids: List[str], test_ratio: float = 0.1, seed: int = 42) -> List[str]:
    """
    Create test split from node IDs.

    Args:
        node_ids: List of all node IDs
        test_ratio: Fraction of nodes to use for test
        seed: Random seed

    Returns:
        List of test node IDs
    """
    np.random.seed(seed)
    num_test = int(len(node_ids) * test_ratio)

    test_indices = np.random.choice(len(node_ids), size=num_test, replace=False)
    test_nodes = [node_ids[i] for i in test_indices]

    return test_nodes


def get_node_texts(node_ids: List[str], metadata_df: pd.DataFrame) -> List[str]:
    """
    Get text content for list of node IDs.

    Args:
        node_ids: List of node IDs (as strings)
        metadata_df: Metadata DataFrame

    Returns:
        List of text strings (title + abstract)
    """
    texts = []

    for node_id in node_ids:
        row = metadata_df[metadata_df['id'] == int(node_id)].iloc[0]
        title = str(row['title']) if pd.notna(row['title']) else ""
        abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
        text = f"{title} {abstract}".strip()
        texts.append(text)

    return texts
