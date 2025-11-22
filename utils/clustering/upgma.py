import json
import pandas as pd
import numpy as np
import networkx as nx
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm

def load_edgelist(filepath):
    """Load edgelist from CSV file."""
    df = pd.read_csv(filepath)
    G = nx.from_pandas_edgelist(df, source='source', target='target')
    return G

def get_distance_matrix(G, method='shortest_path'):
    """
    Compute distance matrix from graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    method : str
        'shortest_path': geodesic distance (default)
        'resistance': effective resistance distance (for connected graphs)
    
    Returns:
    --------
    distance_matrix : numpy.ndarray
        Symmetric distance matrix
    nodes : list
        Ordered list of node labels
    """
    nodes = list(G.nodes())
    n = len(nodes)
    distance_matrix = np.zeros((n, n))
    
    if method == 'shortest_path':
        # Compute all-pairs shortest paths
        print("Computing shortest paths...")
        shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        
        for i, node_i in enumerate(tqdm(nodes, desc="Building distance matrix")):
            for j, node_j in enumerate(nodes):
                if i != j:
                    # If no path exists, use a large distance
                    distance_matrix[i, j] = shortest_paths[node_i].get(node_j, n * 10)
                    
    elif method == 'resistance':
        # Effective resistance distance (requires connected graph)
        if not nx.is_connected(G):
            print("Warning: Graph is not connected. Using largest component.")
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
            nodes = list(G.nodes())
            n = len(nodes)
            distance_matrix = np.zeros((n, n))
        
        L = nx.laplacian_matrix(G).toarray()
        # Compute pseudo-inverse of Laplacian
        L_pinv = np.linalg.pinv(L)
        
        for i in tqdm(range(n), desc="Computing resistance distances"):
            for j in range(i+1, n):
                # Resistance distance formula
                distance_matrix[i, j] = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
                distance_matrix[j, i] = distance_matrix[i, j]
    
    return distance_matrix, nodes

def hierarchical_clustering_distance(G, method='shortest_path', linkage_method='average'):
    """
    Perform hierarchical clustering using distance matrix.
    
    Parameters:
    -----------
    G : networkx.Graph
        Input graph
    method : str
        Distance metric ('shortest_path' or 'resistance')
    linkage_method : str
        Linkage method for hierarchical clustering
        'average' (UPGMA), 'single', 'complete', 'ward', etc.
    
    Returns:
    --------
    linkage_matrix : numpy.ndarray
        Linkage matrix for dendrogram
    nodes : list
        Node labels
    """
    distance_matrix, nodes = get_distance_matrix(G, method=method)
    
    # Convert to condensed distance matrix for scipy
    print("Converting to condensed distance matrix...")
    condensed_dist = squareform(distance_matrix)
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    linkage_matrix = linkage(condensed_dist, method=linkage_method)
    
    return linkage_matrix, nodes

def get_clusters(linkage_matrix, n_clusters=None, threshold=None):
    """
    Extract flat clusters from hierarchical clustering.
    
    Parameters:
    -----------
    linkage_matrix : numpy.ndarray
        Linkage matrix
    n_clusters : int, optional
        Number of clusters to extract
    threshold : float, optional
        Distance threshold for clustering
    
    Returns:
    --------
    clusters : numpy.ndarray
        Cluster assignments for each node
    """
    if n_clusters is not None:
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    elif threshold is not None:
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
    else:
        # Default to 5 clusters
        clusters = fcluster(linkage_matrix, 5, criterion='maxclust')
    
    return clusters

def hierarchical2json(linkage_matrix, nodes):
    """
    Convert hierarchical clustering to JSON format.
    
    Parameters:
    -----------
    linkage_matrix : numpy.ndarray
        Linkage matrix
    nodes : list
        Node labels
    
    Returns:
    --------
    json_dict : dict
        JSON representation of hierarchical clustering
    """
    from scipy.cluster.hierarchy import to_tree
    
    def build_json(node):
        if node.is_leaf():
            return {"name": nodes[node.id]}
        else:
            return {
                "children": [build_json(node.get_left()), build_json(node.get_right())],
                "distance": node.dist
            }
    
    root_node, nodelist = to_tree(linkage_matrix, rd=True)
    json_dict = build_json(root_node)
    
    return json_dict

# Example usage
if __name__ == "__main__":
    # Load graph
    G = load_edgelist('oc_mini/network/oc_mini_edgelist.csv')
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Method 1: Hierarchical clustering with distance matrix
    print("\n=== Distance-based Hierarchical Clustering ===")
    linkage_matrix, nodes = hierarchical_clustering_distance(
        G, 
        method='shortest_path',  # or 'resistance'
        linkage_method='average'  # UPGMA
    )

    # Convert hierarchical clustering to JSON
    print("\nConverting to JSON...")
    json_result = hierarchical2json(linkage_matrix, nodes)
    print("\n=== Hierarchical Clustering JSON ===")
    
    with open('hierarchical_clustering.json', 'w') as f:
        json.dump(json_result, f, indent=2)
    print("Hierarchical clustering saved to 'hierarchical_clustering.json'")
    