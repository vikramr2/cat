"""
Graph Distance Correlation Evaluation
Tests how well embedding distances correlate with shortest path distances in the graph
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import networkit as nk


def load_network_edges(edgelist_path: str) -> pd.DataFrame:
    """Load edgelist from CSV"""
    df = pd.read_csv(edgelist_path)
    # Convert IDs to strings for consistency with tree leaf names
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
    return df


def build_graph(edgelist_df: pd.DataFrame) -> Tuple[nk.Graph, Dict[str, int], Dict[int, str]]:
    """
    Build NetworkKit graph from edgelist.

    Returns:
        G: NetworkKit graph
        node_to_id: Dict mapping node name (str) -> networkit node id (int)
        id_to_node: Dict mapping networkit node id (int) -> node name (str)
    """
    # Create mapping from node names to integer IDs
    unique_nodes = sorted(set(edgelist_df['source'].unique()) | set(edgelist_df['target'].unique()))
    node_to_id = {node: i for i, node in enumerate(unique_nodes)}
    id_to_node = {i: node for node, i in node_to_id.items()}

    # Build networkit graph
    G = nk.Graph(len(unique_nodes), weighted=False, directed=False)

    for _, row in edgelist_df.iterrows():
        source_id = node_to_id[row['source']]
        target_id = node_to_id[row['target']]
        G.addEdge(source_id, target_id)

    return G, node_to_id, id_to_node


def compute_shortest_path_distances(
    G: nk.Graph,
    node_to_id: Dict[str, int],
    node_pairs: List[Tuple[str, str]],
    max_distance: int = None
) -> Dict[Tuple[str, str], float]:
    """
    Compute shortest path distances for a list of node pairs using NetworkKit.

    Args:
        G: NetworkKit graph
        node_to_id: Dict mapping node name -> networkit node id
        node_pairs: List of (source, target) node pairs (string names)
        max_distance: Maximum distance to compute (None = no limit)

    Returns:
        Dict mapping (source, target) -> shortest path distance
        Returns infinity if no path exists
    """
    distances = {}

    # Use BFS for each source node to get distances
    # Group pairs by source for efficiency
    pairs_by_source = defaultdict(list)
    for source, target in node_pairs:
        if source in node_to_id and target in node_to_id:
            pairs_by_source[source].append(target)

    for source, targets in tqdm(pairs_by_source.items(), desc="Computing shortest paths"):
        source_id = node_to_id[source]

        # Run BFS from source
        bfs = nk.distance.BFS(G, source_id, storePaths=False)
        bfs.run()

        for target in targets:
            if source == target:
                distances[(source, target)] = 0.0
            else:
                target_id = node_to_id[target]
                dist = bfs.distance(target_id)

                if dist == float('inf'):
                    distances[(source, target)] = float('inf')
                elif max_distance is None or dist <= max_distance:
                    distances[(source, target)] = float(dist)
                else:
                    distances[(source, target)] = float('inf')

    return distances


def compute_embedding_distances(
    embeddings_dict: Dict[str, np.ndarray],
    node_pairs: List[Tuple[str, str]],
    distance_metric: str = 'cosine'
) -> Dict[Tuple[str, str], float]:
    """
    Compute embedding distances for a list of node pairs.

    Args:
        embeddings_dict: Dict mapping node_id -> embedding vector
        node_pairs: List of (source, target) node pairs
        distance_metric: 'cosine', 'euclidean', or 'dot_product'

    Returns:
        Dict mapping (source, target) -> embedding distance
    """
    distances = {}

    for source, target in node_pairs:
        if source not in embeddings_dict or target not in embeddings_dict:
            continue

        emb1 = embeddings_dict[source]
        emb2 = embeddings_dict[target]

        if distance_metric == 'cosine':
            # Cosine distance (1 - cosine similarity)
            dist = cosine(emb1, emb2)
        elif distance_metric == 'euclidean':
            dist = euclidean(emb1, emb2)
        elif distance_metric == 'dot_product':
            # Negative dot product (so lower = more similar)
            dist = -np.dot(emb1, emb2)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        distances[(source, target)] = dist

    return distances


def sample_node_pairs(
    G: nk.Graph,
    node_to_id: Dict[str, int],
    id_to_node: Dict[int, str],
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str],
    num_samples_per_node: int = 100,
    max_distance: int = None,
    strategy: str = 'random'
) -> List[Tuple[str, str]]:
    """
    Sample node pairs for evaluation from test nodes.

    Args:
        G: NetworkKit graph
        node_to_id: Dict mapping node name -> networkit node id
        id_to_node: Dict mapping networkit node id -> node name
        embeddings_dict: Node embeddings (for filtering)
        test_nodes: List of test node IDs to evaluate
        num_samples_per_node: Number of pairs to sample per test node
        max_distance: Maximum graph distance to consider
        strategy: 'random' or 'stratified'

    Returns:
        List of (source, target) node pairs (string names)
    """
    # Filter test nodes to those with embeddings and in the graph
    test_nodes = [n for n in test_nodes if n in embeddings_dict and n in node_to_id]
    all_nodes = [id_to_node[i] for i in G.iterNodes() if id_to_node[i] in embeddings_dict]

    print(f"Sampling from {len(test_nodes)} test nodes...")

    pairs = []

    if strategy == 'random':
        # Random sampling: for each test node, sample random other nodes
        for test_node in test_nodes:
            other_nodes = [n for n in all_nodes if n != test_node]
            sample_size = min(num_samples_per_node, len(other_nodes))
            sampled_targets = np.random.choice(other_nodes, sample_size, replace=False)

            for target in sampled_targets:
                pairs.append((test_node, target))

        return pairs

    elif strategy == 'stratified':
        # Stratified sampling: sample evenly across different distance levels
        for test_node in tqdm(test_nodes, desc="Stratified sampling"):
            # Get distances to all other nodes using BFS
            try:
                source_id = node_to_id[test_node]
                bfs = nk.distance.BFS(G, source_id, storePaths=False)
                bfs.run()

                # Group by distance
                pairs_by_distance = defaultdict(list)
                for target_id in G.iterNodes():
                    target = id_to_node[target_id]
                    if target in embeddings_dict and target != test_node:
                        dist = bfs.distance(target_id)
                        if dist != float('inf'):
                            if max_distance is None or dist <= max_distance:
                                pairs_by_distance[int(dist)].append((test_node, target))

                if not pairs_by_distance:
                    continue

                # Sample evenly from each distance level
                distances = sorted(pairs_by_distance.keys())
                samples_per_level = max(1, num_samples_per_node // len(distances))

                for dist in distances:
                    available = pairs_by_distance[dist]
                    sample_size = min(samples_per_level, len(available))
                    sampled = np.random.choice(len(available), sample_size, replace=False)
                    pairs.extend([available[i] for i in sampled])

            except Exception as e:
                print(f"Error processing {test_node}: {e}")
                continue

        return pairs

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def evaluate_distance_correlation(
    edgelist_path: str,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str],
    num_samples_per_node: int = 100,
    max_graph_distance: int = 10,
    embedding_distance_metric: str = 'cosine',
    sampling_strategy: str = 'stratified'
) -> Dict:
    """
    Evaluate correlation between graph distances and embedding distances.

    Args:
        edgelist_path: Path to CSV with 'source' and 'target' columns
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node IDs to evaluate
        num_samples_per_node: Number of pairs to sample per test node
        max_graph_distance: Maximum graph distance to consider
        embedding_distance_metric: 'cosine', 'euclidean', or 'dot_product'
        sampling_strategy: 'random' or 'stratified'

    Returns:
        Dict with correlation metrics and distance data
    """
    print("\n" + "="*60)
    print("GRAPH DISTANCE CORRELATION EVALUATION")
    print("="*60)

    # Load graph
    print(f"\nLoading edgelist from: {edgelist_path}")
    edgelist_df = load_network_edges(edgelist_path)
    print(f"Loaded {len(edgelist_df)} edges")

    # Build graph
    print("Building NetworkKit graph...")
    G, node_to_id, id_to_node = build_graph(edgelist_df)
    print(f"Graph has {G.numberOfNodes()} nodes and {G.numberOfEdges()} edges")

    # Check connectivity
    cc = nk.components.ConnectedComponents(G)
    cc.run()
    num_components = cc.numberOfComponents()

    if num_components > 1:
        print(f"WARNING: Graph is not fully connected!")
        print(f"Number of connected components: {num_components}")

        # Use largest connected component
        component_sizes = cc.getComponentSizes()
        largest_component_id = max(component_sizes.items(), key=lambda x: x[1])[0]

        # Get nodes in largest component
        nodes_in_largest = [id_to_node[node_id] for node_id in G.iterNodes()
                           if cc.componentOfNode(node_id) == largest_component_id]

        # Rebuild graph with only largest component
        mask = (edgelist_df['source'].isin(nodes_in_largest) &
                edgelist_df['target'].isin(nodes_in_largest))
        largest_cc_df = edgelist_df[mask].copy()
        G, node_to_id, id_to_node = build_graph(largest_cc_df)
        print(f"Using largest connected component: {G.numberOfNodes()} nodes")

    # Filter test nodes
    test_nodes_in_graph = [n for n in test_nodes if n in node_to_id and n in embeddings_dict]
    print(f"Test nodes with embeddings and in graph: {len(test_nodes_in_graph)}")

    # Sample node pairs
    print(f"\nSampling node pairs (strategy: {sampling_strategy})...")
    node_pairs = sample_node_pairs(
        G,
        node_to_id,
        id_to_node,
        embeddings_dict,
        test_nodes_in_graph,
        num_samples_per_node=num_samples_per_node,
        max_distance=max_graph_distance,
        strategy=sampling_strategy
    )
    print(f"Sampled {len(node_pairs)} pairs")

    # Compute graph distances
    print(f"\nComputing shortest path distances...")
    graph_distances = compute_shortest_path_distances(G, node_to_id, node_pairs, max_graph_distance)

    # Filter out infinite distances (disconnected pairs)
    finite_pairs = [(s, t) for (s, t) in node_pairs
                    if graph_distances[(s, t)] != float('inf')]
    print(f"Pairs with finite graph distance: {len(finite_pairs)}")

    # Compute embedding distances
    print(f"\nComputing embedding distances (metric: {embedding_distance_metric})...")
    embedding_distances = compute_embedding_distances(
        embeddings_dict,
        finite_pairs,
        distance_metric=embedding_distance_metric
    )

    # Align distances for correlation
    graph_dist_list = []
    emb_dist_list = []

    for pair in finite_pairs:
        if pair in embedding_distances:
            graph_dist_list.append(graph_distances[pair])
            emb_dist_list.append(embedding_distances[pair])

    graph_dist_array = np.array(graph_dist_list)
    emb_dist_array = np.array(emb_dist_list)

    print(f"\nFinal pair count: {len(graph_dist_array)}")

    # Compute correlations
    print("\nComputing correlations...")

    spearman_corr, spearman_pval = spearmanr(graph_dist_array, emb_dist_array)
    pearson_corr, pearson_pval = pearsonr(graph_dist_array, emb_dist_array)

    # Compute correlation by distance level
    distance_levels = defaultdict(lambda: {'graph': [], 'emb': []})
    for gd, ed in zip(graph_dist_array, emb_dist_array):
        distance_levels[int(gd)]['graph'].append(gd)
        distance_levels[int(gd)]['emb'].append(ed)

    distance_stats = {}
    for dist, data in sorted(distance_levels.items()):
        distance_stats[dist] = {
            'count': len(data['graph']),
            'mean_emb_dist': np.mean(data['emb']),
            'std_emb_dist': np.std(data['emb']),
            'median_emb_dist': np.median(data['emb'])
        }

    results = {
        'spearman_corr': spearman_corr,
        'spearman_pval': spearman_pval,
        'pearson_corr': pearson_corr,
        'pearson_pval': pearson_pval,
        'num_pairs': len(graph_dist_array),
        'graph_distances': graph_dist_array,
        'embedding_distances': emb_dist_array,
        'distance_stats': distance_stats,
        'embedding_metric': embedding_distance_metric,
        'max_graph_distance': max_graph_distance
    }

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\nCorrelation Metrics:")
    print(f"  Spearman correlation: {spearman_corr:.4f} (p={spearman_pval:.4e})")
    print(f"  Pearson correlation: {pearson_corr:.4f} (p={pearson_pval:.4e})")

    print(f"\nDistance Distribution:")
    for dist in sorted(distance_stats.keys()):
        stats = distance_stats[dist]
        print(f"  Distance {dist}: {stats['count']} pairs, "
              f"mean emb dist = {stats['mean_emb_dist']:.4f} ± {stats['std_emb_dist']:.4f}")

    print("\n" + "="*60)

    return results


def plot_distance_correlation(results: Dict):
    """Plot graph distance vs embedding distance correlation"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    graph_dist = results['graph_distances']
    emb_dist = results['embedding_distances']

    # 1. Scatter plot
    ax = axes[0, 0]
    ax.scatter(graph_dist, emb_dist, alpha=0.3, s=20, color='#3498db')
    ax.set_xlabel('Graph Distance (Shortest Path)', fontsize=12)
    ax.set_ylabel(f"Embedding Distance ({results['embedding_metric']})", fontsize=12)
    ax.set_title(f"Distance Correlation (Spearman: {results['spearman_corr']:.3f})",
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(graph_dist, emb_dist, 1)
    p = np.poly1d(z)
    x_line = np.linspace(graph_dist.min(), graph_dist.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Linear fit')
    ax.legend()

    # 2. Box plot by distance level
    ax = axes[0, 1]
    distance_stats = results['distance_stats']
    distances = sorted(distance_stats.keys())

    box_data = []
    labels = []
    for dist in distances:
        # Get all embedding distances for this graph distance
        mask = graph_dist == dist
        box_data.append(emb_dist[mask])
        labels.append(str(dist))

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#2ecc71')
        patch.set_alpha(0.7)

    ax.set_xlabel('Graph Distance', fontsize=12)
    ax.set_ylabel('Embedding Distance', fontsize=12)
    ax.set_title('Embedding Distance Distribution by Graph Distance',
                 fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # 3. Histogram of graph distances
    ax = axes[1, 0]
    ax.hist(graph_dist, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Graph Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Graph Distances', fontsize=13, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)

    # 4. Mean embedding distance by graph distance
    ax = axes[1, 1]
    distances = sorted(distance_stats.keys())
    means = [distance_stats[d]['mean_emb_dist'] for d in distances]
    stds = [distance_stats[d]['std_emb_dist'] for d in distances]

    ax.errorbar(distances, means, yerr=stds, marker='o', linewidth=2,
                markersize=8, capsize=5, color='#9b59b6',
                ecolor='#95a5a6', alpha=0.8)
    ax.set_xlabel('Graph Distance', fontsize=12)
    ax.set_ylabel('Mean Embedding Distance', fontsize=12)
    ax.set_title('Mean Embedding Distance vs Graph Distance',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Create a second figure for additional analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # 5. Hexbin plot (density)
    ax = axes2[0]
    hb = ax.hexbin(graph_dist, emb_dist, gridsize=30, cmap='YlOrRd', mincnt=1)
    ax.set_xlabel('Graph Distance', fontsize=12)
    ax.set_ylabel('Embedding Distance', fontsize=12)
    ax.set_title('Distance Correlation Density', fontsize=13, fontweight='bold')
    plt.colorbar(hb, ax=ax, label='Count')

    # 6. Summary statistics table
    ax = axes2[1]
    ax.axis('off')

    summary_text = "Distance Correlation Summary\n" + "="*50 + "\n\n"
    summary_text += f"Spearman Correlation: {results['spearman_corr']:.4f}\n"
    summary_text += f"  p-value: {results['spearman_pval']:.4e}\n\n"
    summary_text += f"Pearson Correlation: {results['pearson_corr']:.4f}\n"
    summary_text += f"  p-value: {results['pearson_pval']:.4e}\n\n"
    summary_text += f"Number of pairs: {results['num_pairs']}\n"
    summary_text += f"Embedding metric: {results['embedding_metric']}\n"
    summary_text += f"Max graph distance: {results['max_graph_distance']}\n\n"
    summary_text += "Distance Level Statistics:\n"
    summary_text += "-" * 50 + "\n"

    for dist in sorted(distance_stats.keys()):
        stats = distance_stats[dist]
        summary_text += f"Distance {dist}: {stats['count']} pairs\n"
        summary_text += f"  Mean: {stats['mean_emb_dist']:.4f}\n"
        summary_text += f"  Std:  {stats['std_emb_dist']:.4f}\n"
        summary_text += f"  Median: {stats['median_emb_dist']:.4f}\n\n"

    ax.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.show()


def compare_distance_metrics(
    edgelist_path: str,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str],
    num_samples_per_node: int = 100,
    max_graph_distance: int = 10,
    metrics: List[str] = None
) -> Dict:
    """
    Compare different embedding distance metrics.

    Args:
        edgelist_path: Path to edgelist CSV
        embeddings_dict: Node embeddings
        test_nodes: List of test node IDs to evaluate
        num_samples_per_node: Number of pairs to sample per test node
        max_graph_distance: Maximum graph distance
        metrics: List of metrics to compare (default: all)

    Returns:
        Dict with results for each metric
    """
    if metrics is None:
        metrics = ['cosine', 'euclidean', 'dot_product']

    print("\n" + "="*60)
    print("COMPARING DISTANCE METRICS")
    print("="*60)

    results = {}

    for metric in metrics:
        print(f"\n\nEvaluating metric: {metric}")
        print("-" * 60)

        results[metric] = evaluate_distance_correlation(
            edgelist_path=edgelist_path,
            embeddings_dict=embeddings_dict,
            test_nodes=test_nodes,
            num_samples_per_node=num_samples_per_node,
            max_graph_distance=max_graph_distance,
            embedding_distance_metric=metric,
            sampling_strategy='stratified'
        )

    # Summary comparison
    print("\n" + "="*60)
    print("METRIC COMPARISON SUMMARY")
    print("="*60)

    for metric in metrics:
        res = results[metric]
        print(f"\n{metric}:")
        print(f"  Spearman: {res['spearman_corr']:.4f}")
        print(f"  Pearson:  {res['pearson_corr']:.4f}")

    return results
