"""
Disjoint Clustering Structure Evaluation
Tests if embeddings capture disjoint clustering structure

Works with DisjointClustering class from dcat/cluster_utils.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import random
import sys
from pathlib import Path

# Import DisjointClustering
sys.path.append(str(Path(__file__).parent.parent.parent / "dcat"))
from cluster_utils import DisjointClustering


def evaluate_cluster_classification(
    clustering: DisjointClustering,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Classify nodes into their correct clusters.

    Tests whether embeddings can predict which cluster a node belongs to.
    This measures how well the embedding space captures clustering structure.

    Args:
        clustering: DisjointClustering object
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node IDs to evaluate (default: all nodes with embeddings)
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dict with classification accuracy and metrics
    """
    print("\n" + "="*60)
    print("CLUSTER CLASSIFICATION EVALUATION")
    print("="*60)
    print(f"Cross-validation folds: {cv_folds}")

    # Determine which nodes to evaluate
    if test_nodes is None:
        eval_nodes = [n for n in clustering.all_nodes if n in embeddings_dict]
    else:
        eval_nodes = [n for n in test_nodes if n in embeddings_dict and n in clustering.all_nodes]

    print(f"Evaluating {len(eval_nodes)} test nodes")

    # Create labels: which cluster does each node belong to?
    node_labels = {}
    for node_id in eval_nodes:
        cluster_id = clustering.get_cluster_id(node_id)
        if cluster_id is not None:
            node_labels[node_id] = cluster_id

    # Filter to nodes that have labels
    nodes = [n for n in eval_nodes if n in node_labels]

    if len(nodes) < cv_folds:
        print(f"  Error: only {len(nodes)} nodes with embeddings (need at least {cv_folds})")
        return {}

    X = np.array([embeddings_dict[n] for n in nodes])

    # Convert cluster IDs to integer labels
    unique_clusters = sorted(set(node_labels.values()))
    cluster_to_label = {cluster_id: i for i, cluster_id in enumerate(unique_clusters)}
    y = np.array([cluster_to_label[node_labels[n]] for n in nodes])

    # Check if we have enough samples per class
    unique, counts = np.unique(y, return_counts=True)
    min_class_size = counts.min()
    max_class_size = counts.max()

    print(f"  Clusters: {len(unique_clusters)}")
    print(f"  Samples: {len(nodes)}")
    print(f"  Min cluster size: {min_class_size}")
    print(f"  Max cluster size: {max_class_size}")
    print(f"  Mean cluster size: {counts.mean():.1f}")

    if min_class_size < 2:
        print(f"  Warning: smallest cluster has only {min_class_size} sample(s)")
        print(f"  Falling back to evaluation on top 5 largest clusters...")

        # Get cluster sizes
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            cluster_sizes[cluster_id] = sum(1 for label in y if label == cluster_to_label[cluster_id])

        # Get top 5 largest clusters
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
        top_cluster_ids = {cluster_id for cluster_id, _ in top_clusters}

        print(f"  Top 5 clusters: {[f'{cid} (n={size})' for cid, size in top_clusters]}")

        # Filter to only nodes in top clusters
        filtered_nodes = []
        for node_id in nodes:
            cluster_id = node_labels[node_id]
            if cluster_id in top_cluster_ids:
                filtered_nodes.append(node_id)

        if len(filtered_nodes) < cv_folds:
            print(f"  Error: only {len(filtered_nodes)} nodes in top clusters (need at least {cv_folds})")
            return {}

        # Re-build X and y with only top clusters
        nodes = filtered_nodes
        X = np.array([embeddings_dict[n] for n in nodes])
        y = np.array([cluster_to_label[node_labels[n]] for n in nodes])

        # Re-compute class counts
        unique, counts = np.unique(y, return_counts=True)
        min_class_size = counts.min()
        max_class_size = counts.max()

        print(f"  Filtered to {len(nodes)} nodes across {len(unique)} clusters")
        print(f"  Min cluster size: {min_class_size}, Max: {max_class_size}")

    # Use stratified k-fold if possible
    actual_folds = min(cv_folds, min_class_size)

    # Cross-validated classification
    clf = LogisticRegression(max_iter=1000, random_state=random_state)

    try:
        cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')

        # Compute baseline (random guessing)
        class_proportions = counts / counts.sum()
        random_baseline = (class_proportions ** 2).sum()  # Expected accuracy of random classifier

        results = {
            'accuracy': float(scores.mean()),
            'std': float(scores.std()),
            'num_clusters': len(unique_clusters),
            'num_samples': len(nodes),
            'min_cluster_size': int(min_class_size),
            'max_cluster_size': int(max_class_size),
            'mean_cluster_size': float(counts.mean()),
            'random_baseline': float(random_baseline),
            'cv_folds': actual_folds
        }

        print(f"\n  Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  Random baseline: {random_baseline:.4f}")
        print(f"  Lift over random: {scores.mean() / random_baseline:.2f}x")

        return results

    except Exception as e:
        print(f"  Error during classification: {e}")
        return {}


def evaluate_cluster_distance_correlation(
    clustering: DisjointClustering,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str] = None,
    sample_size: int = 2000,
    random_state: int = 42
) -> Dict:
    """
    Measure correlation between embedding distance and cluster membership.

    For disjoint clustering:
    - Same cluster pairs should have smaller embedding distances
    - Different cluster pairs should have larger embedding distances

    Args:
        clustering: DisjointClustering object
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node IDs (default: all nodes with embeddings)
        sample_size: Number of node pairs to sample
        random_state: Random seed for reproducibility

    Returns:
        Dict with correlation metrics and distance statistics
    """
    print("\n" + "="*60)
    print("CLUSTER DISTANCE EVALUATION")
    print("="*60)

    random.seed(random_state)
    np.random.seed(random_state)

    # Determine which nodes to evaluate
    if test_nodes is None:
        node_ids = [n for n in clustering.all_nodes if n in embeddings_dict]
    else:
        node_ids = [n for n in test_nodes if n in embeddings_dict and n in clustering.all_nodes]

    print(f"Evaluating {len(node_ids)} test nodes")
    print(f"Sampling {sample_size} pairs...")

    # Sample pairs
    all_pairs = [(i, j) for i in range(len(node_ids)) for j in range(i+1, len(node_ids))]
    max_pairs = min(sample_size, len(all_pairs))
    pairs = random.sample(all_pairs, max_pairs)

    same_cluster_dists = []
    diff_cluster_dists = []

    embedding_dists = []
    cluster_dists = []

    for i, j in tqdm(pairs, desc="Computing distances"):
        node_i_id, node_j_id = node_ids[i], node_ids[j]

        # Embedding distance (cosine distance)
        emb_dist = cosine_distances(
            embeddings_dict[node_i_id].reshape(1, -1),
            embeddings_dict[node_j_id].reshape(1, -1)
        )[0, 0]

        # Cluster distance (binary: 0 or 1)
        cluster_dist = clustering.cluster_distance(node_i_id, node_j_id)

        embedding_dists.append(emb_dist)
        cluster_dists.append(cluster_dist)

        if cluster_dist == 0.0:
            same_cluster_dists.append(emb_dist)
        else:
            diff_cluster_dists.append(emb_dist)

    # Convert to arrays
    embedding_dists = np.array(embedding_dists)
    cluster_dists = np.array(cluster_dists)

    # Compute statistics
    same_cluster_mean = np.mean(same_cluster_dists) if same_cluster_dists else 0.0
    same_cluster_std = np.std(same_cluster_dists) if same_cluster_dists else 0.0
    diff_cluster_mean = np.mean(diff_cluster_dists) if diff_cluster_dists else 0.0
    diff_cluster_std = np.std(diff_cluster_dists) if diff_cluster_dists else 0.0

    # Spearman correlation (though it's binary, still useful)
    correlation, p_value = spearmanr(embedding_dists, cluster_dists)

    # Point-biserial correlation (more appropriate for binary variables)
    # This measures how well cluster membership (binary) correlates with embedding distance (continuous)
    from scipy.stats import pointbiserialr
    pb_correlation, pb_p_value = pointbiserialr(cluster_dists, embedding_dists)

    print(f"\nSpearman correlation: {correlation:.4f} (p={p_value:.2e})")
    print(f"Point-biserial correlation: {pb_correlation:.4f} (p={pb_p_value:.2e})")
    print(f"Number of pairs evaluated: {len(pairs)}")

    print(f"\nEmbedding distance statistics:")
    print(f"  Same cluster: {same_cluster_mean:.4f} ± {same_cluster_std:.4f} ({len(same_cluster_dists)} pairs)")
    print(f"  Diff cluster: {diff_cluster_mean:.4f} ± {diff_cluster_std:.4f} ({len(diff_cluster_dists)} pairs)")

    if same_cluster_mean > 0:
        separation = (diff_cluster_mean - same_cluster_mean) / same_cluster_mean
        print(f"  Separation ratio: {separation:.2f}x")

    return {
        'spearman_correlation': float(correlation),
        'spearman_p_value': float(p_value),
        'pointbiserial_correlation': float(pb_correlation),
        'pointbiserial_p_value': float(pb_p_value),
        'num_pairs': len(pairs),
        'same_cluster_pairs': len(same_cluster_dists),
        'diff_cluster_pairs': len(diff_cluster_dists),
        'same_cluster_mean': float(same_cluster_mean),
        'same_cluster_std': float(same_cluster_std),
        'diff_cluster_mean': float(diff_cluster_mean),
        'diff_cluster_std': float(diff_cluster_std),
        'embedding_distances': embedding_dists.tolist(),
        'cluster_distances': cluster_dists.tolist()
    }


def evaluate_clustering_structure(
    clustering: DisjointClustering,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str] = None,
    distance_sample_size: int = 2000,
    random_state: int = 42
) -> Dict:
    """
    Comprehensive evaluation of how well embeddings capture clustering structure.

    Runs two evaluations:
    1. Cluster classification (accuracy)
    2. Cluster distance analysis

    Args:
        clustering: DisjointClustering object
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node IDs (default: all nodes with embeddings)
        distance_sample_size: Sample size for distance analysis
        random_state: Random seed

    Returns:
        Dict with all evaluation results
    """
    print("\n" + "="*70)
    print(" " * 15 + "CLUSTERING STRUCTURE EVALUATION")
    print("="*70)

    results = {}

    # 1. Cluster classification
    results['cluster_classification'] = evaluate_cluster_classification(
        clustering, embeddings_dict, test_nodes=test_nodes,
        random_state=random_state
    )

    # 2. Cluster distance analysis
    results['distance_analysis'] = evaluate_cluster_distance_correlation(
        clustering, embeddings_dict, test_nodes=test_nodes,
        sample_size=distance_sample_size, random_state=random_state
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if results['cluster_classification']:
        cc = results['cluster_classification']
        print(f"\nCluster Classification:")
        print(f"  Accuracy: {cc['accuracy']:.4f} ± {cc['std']:.4f}")
        print(f"  Random baseline: {cc['random_baseline']:.4f}")
        print(f"  Lift: {cc['accuracy'] / cc['random_baseline']:.2f}x")
        print(f"  Clusters: {cc['num_clusters']}")

    if results['distance_analysis']:
        da = results['distance_analysis']
        print(f"\nCluster Distance Analysis:")
        print(f"  Point-biserial ρ: {da['pointbiserial_correlation']:.4f}")
        print(f"  Same cluster distance: {da['same_cluster_mean']:.4f}")
        print(f"  Diff cluster distance: {da['diff_cluster_mean']:.4f}")

    print("\n" + "="*70)

    return results
