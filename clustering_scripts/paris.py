#!/usr/bin/env python3
"""
Paris Algorithm for Hierarchical Clustering
Reads an edgelist (source, target columns) and outputs hierarchical clustering as JSON
"""

import pandas as pd
import numpy as np
from sknetwork.hierarchy import Paris
from sknetwork.data import from_edge_list
import json
import argparse
from scipy.cluster.hierarchy import to_tree


def edgelist_to_adjacency(edgelist_df):
    """Convert edgelist dataframe to adjacency matrix using sknetwork"""
    from sklearn.preprocessing import LabelEncoder
    
    # Get all unique nodes
    all_nodes = pd.concat([edgelist_df['source'], edgelist_df['target']])
    
    # Create label encoder to map node names to indices
    le = LabelEncoder()
    le.fit(all_nodes)
    
    # Convert edges to numeric indices
    source_indices = le.transform(edgelist_df['source'])
    target_indices = le.transform(edgelist_df['target'])
    edges_numeric = np.column_stack([source_indices, target_indices])
    
    # Create adjacency matrix
    adjacency = from_edge_list(edges_numeric, directed=False)
    
    # Return adjacency and the ordered node names
    return adjacency, list(le.classes_)


def dendrogram_to_json(dendrogram, node_names=None):
    """
    Convert hierarchical clustering dendrogram to JSON format
    
    Args:
        dendrogram: scipy linkage matrix from Paris algorithm
        node_names: list of node names (optional)
    
    Returns:
        JSON-serializable dictionary representing the hierarchy
    """
    # Convert linkage matrix to tree structure
    tree = to_tree(dendrogram)
    
    def node_to_dict(node, node_names=None):
        """Recursively convert tree nodes to dictionary"""
        if node.is_leaf():
            # Leaf node - represents an original data point
            node_id = int(node.id)  # Convert to Python int
            result = {
                "id": node_id,
                "name": str(node_names[node_id]) if node_names else str(node_id),
                "type": "leaf",
                "count": 1
            }
        else:
            # Internal node - represents a cluster
            left_child = node_to_dict(node.left, node_names)
            right_child = node_to_dict(node.right, node_names)
            
            result = {
                "id": int(node.id),
                "type": "cluster",
                "distance": float(node.dist),
                "count": int(node.count),
                "children": [left_child, right_child]
            }
        
        return result
    
    return node_to_dict(tree, node_names)


def run_paris_algorithm(input_file, output_file=None):
    """
    Run Paris algorithm on edgelist and save hierarchical clustering as JSON
    
    Args:
        input_file: path to CSV file with 'source' and 'target' columns
        output_file: path to output JSON file (optional)
    """
    # Read edgelist
    print(f"Reading edgelist from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Validate columns
    if 'source' not in df.columns or 'target' not in df.columns:
        raise ValueError("Input file must have 'source' and 'target' columns")
    
    print(f"Found {len(df)} edges")
    
    # Convert to adjacency matrix (also returns node names)
    print("Converting to adjacency matrix...")
    adjacency, node_names = edgelist_to_adjacency(df)
    
    print(f"Found {len(node_names)} unique nodes")
    
    # Run Paris algorithm
    print("Running Paris algorithm...")
    paris = Paris()
    dendrogram = paris.fit_transform(adjacency)
    
    print("Converting to JSON format...")
    # Convert to JSON structure
    hierarchy_json = dendrogram_to_json(dendrogram, node_names)
    
    # Add metadata
    result = {
        "algorithm": "Paris",
        "num_nodes": len(node_names),
        "num_edges": len(df),
        "hierarchy": hierarchy_json
    }
    
    # Save to file or print
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Hierarchical clustering saved to {output_file}")
    else:
        print(json.dumps(result, indent=2))
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run Paris algorithm on edgelist and output hierarchical clustering as JSON'
    )
    parser.add_argument(
        'input_file',
        help='Path to CSV file with source and target columns'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to output JSON file (if not specified, prints to stdout)',
        default=None
    )
    
    args = parser.parse_args()
    
    try:
        run_paris_algorithm(args.input_file, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
