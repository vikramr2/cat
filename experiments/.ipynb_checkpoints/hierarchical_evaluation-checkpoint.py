"""
Hierarchical Structure Evaluation
Evaluates how well fine-tuned embeddings preserve tree structure
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import networkx as nx


# ============================================================================
# 1. TREE STRUCTURE PRESERVATION METRICS
# ============================================================================

def evaluate_tree_distance_correlation(
    tree,
    embeddings_dict,
    sample_size=1000
):
    """
    Evaluate correlation between tree distance and embedding distance.

    Args:
        tree: HierarchicalTree instance
        embeddings_dict: Dict mapping leaf_name -> embedding vector
        sample_size: Number of leaf pairs to sample

    Returns:
        Dict with correlation metrics
    """
    print("Evaluating tree distance correlation...")

    leaf_names = list(tree.leaves.keys())
    leaf_names = [l for l in leaf_names if l in embeddings_dict]

    # Sample pairs
    pairs = []
    for _ in range(sample_size):
        l1, l2 = np.random.choice(leaf_names, 2, replace=False)
        pairs.append((l1, l2))

    tree_distances = []
    embedding_distances = []

    for l1, l2 in tqdm(pairs, desc="Computing distances"):
        # Tree distance
        node1 = tree.leaves[l1]
        node2 = tree.leaves[l2]
        tree_dist = tree.tree_distance(node1, node2)
        tree_distances.append(tree_dist)

        # Embedding distance (1 - cosine similarity)
        emb1 = embeddings_dict[l1]
        emb2 = embeddings_dict[l2]
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        emb_dist = 1 - cos_sim
        embedding_distances.append(emb_dist)

    # Compute correlations
    spearman_corr, spearman_p = spearmanr(tree_distances, embedding_distances)
    pearson_corr = np.corrcoef(tree_distances, embedding_distances)[0, 1]

    return {
        'spearman_correlation': spearman_corr,
        'spearman_pvalue': spearman_p,
        'pearson_correlation': pearson_corr,
        'tree_distances': tree_distances,
        'embedding_distances': embedding_distances
    }


# ============================================================================
# 2. CLUSTER MEMBERSHIP PREDICTION
# ============================================================================

def evaluate_cluster_membership(
    tree,
    embeddings_dict,
    test_leaves,
    k_neighbors=10
):
    """
    Evaluate how well embeddings predict cluster membership.
    For each test leaf, find k nearest neighbors in embedding space
    and check if they share the same parent cluster.

    Args:
        tree: HierarchicalTree instance
        embeddings_dict: Dict mapping leaf_name -> embedding
        test_leaves: List of test leaf names
        k_neighbors: Number of neighbors to retrieve

    Returns:
        Dict with cluster prediction metrics
    """
    print(f"\nEvaluating cluster membership prediction (k={k_neighbors})...")

    # Get all training leaves (not in test set)
    all_leaves = [l for l in tree.leaves.keys() if l in embeddings_dict]
    train_leaves = [l for l in all_leaves if l not in test_leaves]

    # Build embedding matrix
    train_embeddings = np.array([embeddings_dict[l] for l in train_leaves])

    precisions = []
    recalls = []

    for test_leaf in tqdm(test_leaves, desc="Testing cluster membership"):
        if test_leaf not in embeddings_dict:
            continue

        test_emb = embeddings_dict[test_leaf].reshape(1, -1)

        # Find k nearest neighbors
        similarities = cosine_similarity(test_emb, train_embeddings)[0]
        top_k_indices = np.argsort(similarities)[::-1][:k_neighbors]
        top_k_neighbors = [train_leaves[i] for i in top_k_indices]

        # Get true siblings (same parent)
        test_node = tree.leaves[test_leaf]
        if test_node.parent:
            true_siblings = set([
                l.name for l in tree.get_leaves_in_subtree(test_node.parent)
                if l.name != test_leaf and l.name in train_leaves
            ])
        else:
            true_siblings = set()

        if len(true_siblings) == 0:
            continue

        # Compute precision and recall
        predicted_siblings = set(top_k_neighbors)
        tp = len(predicted_siblings & true_siblings)

        precision = tp / len(predicted_siblings) if len(predicted_siblings) > 0 else 0
        recall = tp / len(true_siblings) if len(true_siblings) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    # Compute F1
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p + r > 0:
            f1_scores.append(2 * p * r / (p + r))
        else:
            f1_scores.append(0)

    return {
        'mean_precision': np.mean(precisions),
        'mean_recall': np.mean(recalls),
        'mean_f1': np.mean(f1_scores),
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores
    }


# ============================================================================
# 3. LINK PREDICTION
# ============================================================================

def get_tree_edges(tree, max_depth_diff=2):
    """
    Extract edges from tree where nodes are within max_depth_diff levels.
    Returns list of (leaf1, leaf2) tuples that should be linked.
    """
    edges = []
    leaf_names = list(tree.leaves.keys())

    for i, l1 in enumerate(leaf_names):
        for l2 in leaf_names[i+1:]:
            node1 = tree.leaves[l1]
            node2 = tree.leaves[l2]

            # Check if they share a close ancestor
            lca = tree.find_lca(node1, node2)
            depth_diff = min(node1.depth, node2.depth) - lca.depth

            if depth_diff <= max_depth_diff:
                edges.append((l1, l2))

    return edges


def evaluate_link_prediction(
    tree,
    embeddings_dict,
    test_leaves,
    max_depth_diff=2,
    num_negative_samples=None
):
    """
    Evaluate link prediction: predict which pairs of leaves should be connected
    based on their proximity in the tree.

    Args:
        tree: HierarchicalTree instance
        embeddings_dict: Dict mapping leaf_name -> embedding
        test_leaves: List of test leaf names
        max_depth_diff: Maximum depth difference for considering leaves "linked"
        num_negative_samples: Number of negative samples (default: 2x positive samples)

    Returns:
        Dict with link prediction metrics (AUC-ROC, AUC-PR)
    """
    print(f"\nEvaluating link prediction (max_depth_diff={max_depth_diff})...")

    test_leaves = [l for l in test_leaves if l in embeddings_dict]

    # Generate positive samples (true links in tree)
    positive_pairs = []
    for i, l1 in enumerate(test_leaves):
        for l2 in test_leaves[i+1:]:
            node1 = tree.leaves[l1]
            node2 = tree.leaves[l2]

            lca = tree.find_lca(node1, node2)
            depth_diff = min(node1.depth, node2.depth) - lca.depth

            if depth_diff <= max_depth_diff:
                positive_pairs.append((l1, l2))

    # Generate negative samples (no link in tree)
    if num_negative_samples is None:
        num_negative_samples = len(positive_pairs) * 2

    negative_pairs = []
    attempts = 0
    max_attempts = num_negative_samples * 10

    while len(negative_pairs) < num_negative_samples and attempts < max_attempts:
        l1, l2 = np.random.choice(test_leaves, 2, replace=False)

        node1 = tree.leaves[l1]
        node2 = tree.leaves[l2]
        lca = tree.find_lca(node1, node2)
        depth_diff = min(node1.depth, node2.depth) - lca.depth

        if depth_diff > max_depth_diff and (l1, l2) not in negative_pairs and (l2, l1) not in negative_pairs:
            negative_pairs.append((l1, l2))

        attempts += 1

    print(f"  Positive pairs: {len(positive_pairs)}")
    print(f"  Negative pairs: {len(negative_pairs)}")

    # Compute similarity scores
    y_true = []
    y_scores = []

    for l1, l2 in tqdm(positive_pairs + negative_pairs, desc="Computing scores"):
        emb1 = embeddings_dict[l1]
        emb2 = embeddings_dict[l2]

        # Cosine similarity as score
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        is_positive = (l1, l2) in positive_pairs
        y_true.append(1 if is_positive else 0)
        y_scores.append(cos_sim)

    # Compute metrics
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'num_positive': len(positive_pairs),
        'num_negative': len(negative_pairs),
        'y_true': y_true,
        'y_scores': y_scores
    }


# ============================================================================
# 4. HIERARCHICAL CLUSTERING QUALITY
# ============================================================================

def evaluate_dendrogram_reconstruction(
    tree,
    embeddings_dict,
    test_leaves,
    linkage_method='average'
):
    """
    Build hierarchical clustering from embeddings and compare to true tree.

    Args:
        tree: HierarchicalTree instance
        embeddings_dict: Dict mapping leaf_name -> embedding
        test_leaves: List of test leaf names
        linkage_method: 'average', 'complete', 'single', 'ward'

    Returns:
        Dict with dendrogram comparison metrics
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
    from scipy.spatial.distance import pdist

    print(f"\nEvaluating dendrogram reconstruction (linkage={linkage_method})...")

    test_leaves = [l for l in test_leaves if l in embeddings_dict]

    # Build embedding matrix
    embeddings_matrix = np.array([embeddings_dict[l] for l in test_leaves])

    # Compute distance matrix
    distances = pdist(embeddings_matrix, metric='cosine')

    # Build dendrogram
    Z = linkage(distances, method=linkage_method)

    # Compute cophenetic correlation
    c, coph_dists = cophenet(Z, distances)

    return {
        'cophenetic_correlation': c,
        'linkage_matrix': Z,
        'test_leaves': test_leaves
    }


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_evaluation_results(results):
    """Plot comprehensive evaluation results"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Tree distance vs Embedding distance
    ax = axes[0, 0]
    if 'tree_distance_correlation' in results:
        corr_results = results['tree_distance_correlation']
        ax.scatter(
            corr_results['tree_distances'],
            corr_results['embedding_distances'],
            alpha=0.5,
            s=20
        )
        ax.set_xlabel('Tree Distance', fontsize=12)
        ax.set_ylabel('Embedding Distance (1 - cosine sim)', fontsize=12)
        ax.set_title(
            f"Tree vs Embedding Distance\nSpearman ρ = {corr_results['spearman_correlation']:.3f}",
            fontsize=13
        )
        ax.grid(True, alpha=0.3)

    # 2. Link prediction ROC
    ax = axes[0, 1]
    if 'link_prediction' in results:
        link_results = results['link_prediction']
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(link_results['y_true'], link_results['y_scores'])
        ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {link_results['auc_roc']:.3f}")
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Link Prediction ROC Curve', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    # 3. Cluster membership metrics
    ax = axes[1, 0]
    if 'cluster_membership' in results:
        cluster_results = results['cluster_membership']
        metrics = ['Precision', 'Recall', 'F1']
        values = [
            cluster_results['mean_precision'],
            cluster_results['mean_recall'],
            cluster_results['mean_f1']
        ]
        bars = ax.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Cluster Membership Prediction', fontsize=13)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontsize=11)

    # 4. Summary metrics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = "Evaluation Summary\n" + "="*40 + "\n\n"

    if 'tree_distance_correlation' in results:
        corr = results['tree_distance_correlation']['spearman_correlation']
        summary_text += f"Tree Distance Correlation:\n"
        summary_text += f"  Spearman ρ = {corr:.4f}\n\n"

    if 'link_prediction' in results:
        auc_roc = results['link_prediction']['auc_roc']
        auc_pr = results['link_prediction']['auc_pr']
        summary_text += f"Link Prediction:\n"
        summary_text += f"  AUC-ROC = {auc_roc:.4f}\n"
        summary_text += f"  AUC-PR = {auc_pr:.4f}\n\n"

    if 'cluster_membership' in results:
        f1 = results['cluster_membership']['mean_f1']
        summary_text += f"Cluster Membership:\n"
        summary_text += f"  Mean F1 = {f1:.4f}\n\n"

    if 'dendrogram' in results:
        coph = results['dendrogram']['cophenetic_correlation']
        summary_text += f"Dendrogram Quality:\n"
        summary_text += f"  Cophenetic corr = {coph:.4f}\n"

    ax.text(0.1, 0.9, summary_text, fontsize=12, verticalalignment='top',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_hierarchical_embeddings(
    tree,
    embeddings_dict,
    test_leaves,
    k_neighbors=10,
    max_depth_diff=2,
    correlation_samples=1000
):
    """
    Run comprehensive evaluation of hierarchical embeddings.

    Args:
        tree: HierarchicalTree instance
        embeddings_dict: Dict mapping leaf_name -> embedding vector
        test_leaves: List of test leaf names
        k_neighbors: Number of neighbors for cluster membership
        max_depth_diff: Depth threshold for link prediction
        correlation_samples: Number of pairs for correlation analysis

    Returns:
        Dict with all evaluation results
    """
    print("\n" + "="*60)
    print("HIERARCHICAL EMBEDDING EVALUATION")
    print("="*60)

    results = {}

    # 1. Tree distance correlation
    results['tree_distance_correlation'] = evaluate_tree_distance_correlation(
        tree, embeddings_dict, sample_size=correlation_samples
    )

    # 2. Cluster membership prediction
    results['cluster_membership'] = evaluate_cluster_membership(
        tree, embeddings_dict, test_leaves, k_neighbors=k_neighbors
    )

    # 3. Link prediction
    results['link_prediction'] = evaluate_link_prediction(
        tree, embeddings_dict, test_leaves, max_depth_diff=max_depth_diff
    )

    # 4. Dendrogram reconstruction
    results['dendrogram'] = evaluate_dendrogram_reconstruction(
        tree, embeddings_dict, test_leaves
    )

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"\n1. Tree Distance Correlation:")
    print(f"   Spearman ρ = {results['tree_distance_correlation']['spearman_correlation']:.4f}")
    print(f"   p-value = {results['tree_distance_correlation']['spearman_pvalue']:.4e}")

    print(f"\n2. Cluster Membership Prediction (k={k_neighbors}):")
    print(f"   Precision = {results['cluster_membership']['mean_precision']:.4f}")
    print(f"   Recall = {results['cluster_membership']['mean_recall']:.4f}")
    print(f"   F1 = {results['cluster_membership']['mean_f1']:.4f}")

    print(f"\n3. Link Prediction:")
    print(f"   AUC-ROC = {results['link_prediction']['auc_roc']:.4f}")
    print(f"   AUC-PR = {results['link_prediction']['auc_pr']:.4f}")

    print(f"\n4. Dendrogram Reconstruction:")
    print(f"   Cophenetic correlation = {results['dendrogram']['cophenetic_correlation']:.4f}")

    print("\n" + "="*60)

    return results
