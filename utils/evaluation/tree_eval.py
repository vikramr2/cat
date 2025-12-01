"""
Hierarchical Tree Structure Evaluation
Tests if embeddings capture hierarchical clustering tree structure

Works with HierarchicalTree class from hcat/tree_utils.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.stats import spearmanr
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Optional
import random
import sys
from pathlib import Path

# Import HierarchicalTree
sys.path.append(str(Path(__file__).parent.parent.parent / "hcat"))
from tree_utils import HierarchicalTree, TreeNode


def get_subtrees_at_depth(tree: HierarchicalTree, depth: int) -> List[TreeNode]:
    """
    Get all subtrees at a specific depth from root.

    Args:
        tree: HierarchicalTree object
        depth: Depth level (0 = root, 1 = children of root, etc.)

    Returns:
        List of TreeNode objects at the specified depth
    """
    if depth == 0:
        return [tree.root]

    current_level = [tree.root]

    for _ in range(depth):
        next_level = []
        for node in current_level:
            if node.type != 'leaf':
                next_level.extend(node.children)
        current_level = next_level

        if not current_level:
            break

    return current_level


def evaluate_subtree_classification(
    tree,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str] = None,
    depth_levels: List[int] = None,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict:
    """
    Classify nodes into subtrees at different depths.

    Tests whether embeddings can predict which subtree/clade a node belongs to.
    This measures how well the embedding space captures hierarchical structure.

    Args:
        tree: HierarchicalTree object
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node names to evaluate (default: all leaves with embeddings)
        depth_levels: List of tree depths to evaluate (default: [2, 3, 4, 5])
        cv_folds: Number of cross-validation folds
        random_state: Random seed for reproducibility

    Returns:
        Dict with classification accuracy at each depth level
    """
    if depth_levels is None:
        depth_levels = [2, 3, 4, 5]

    print("\n" + "="*60)
    print("SUBTREE CLASSIFICATION EVALUATION")
    print("="*60)
    print(f"Evaluating at depths: {depth_levels}")
    print(f"Cross-validation folds: {cv_folds}")

    # Determine which nodes to evaluate
    if test_nodes is None:
        eval_nodes = [n for n in tree.leaves.keys() if n in embeddings_dict]
    else:
        eval_nodes = [n for n in test_nodes if n in embeddings_dict and n in tree.leaves]

    print(f"Evaluating {len(eval_nodes)} test nodes")

    results = {}

    for depth in depth_levels:
        print(f"\n--- Depth {depth} ---")

        # Get all subtrees at this depth
        subtrees = get_subtrees_at_depth(tree, depth)

        if len(subtrees) < 2:
            print(f"  Skipping: only {len(subtrees)} subtree(s) at this depth")
            continue

        # Create labels: which subtree does each leaf belong to?
        node_labels = {}
        valid_subtrees = 0

        for subtree_id, subtree in enumerate(subtrees):
            # Get all leaves under this subtree
            leaves = tree.get_leaves_in_subtree(subtree)
            leaf_names = [leaf.name for leaf in leaves]
            if len(leaf_names) > 0:
                valid_subtrees += 1
                for leaf_name in leaf_names:
                    node_labels[leaf_name] = subtree_id

        # Filter to test nodes that have labels
        nodes = [n for n in eval_nodes if n in node_labels]

        if len(nodes) < cv_folds:
            print(f"  Skipping: only {len(nodes)} nodes with embeddings (need at least {cv_folds})")
            continue

        X = np.array([embeddings_dict[n] for n in nodes])
        y = np.array([node_labels[n] for n in nodes])

        # Check if we have enough samples per class
        unique, counts = np.unique(y, return_counts=True)
        min_class_size = counts.min()

        if min_class_size < 2:
            print(f"  Skipping: smallest class has only {min_class_size} sample(s)")
            continue

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

            results[depth] = {
                'accuracy': float(scores.mean()),
                'std': float(scores.std()),
                'num_classes': len(subtrees),
                'valid_subtrees': valid_subtrees,
                'num_samples': len(nodes),
                'min_class_size': int(min_class_size),
                'random_baseline': float(random_baseline),
                'cv_folds': actual_folds
            }

            print(f"  Subtrees: {valid_subtrees}")
            print(f"  Samples: {len(nodes)}")
            print(f"  Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
            print(f"  Random baseline: {random_baseline:.4f}")
            print(f"  Lift over random: {scores.mean() / random_baseline:.2f}x")

        except Exception as e:
            print(f"  Error during classification: {e}")
            continue

    return {
        'depth_results': results,
        'depth_levels': depth_levels
    }


def evaluate_tree_distance_correlation(
    tree,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str] = None,
    sample_size: int = 2000,
    random_state: int = 42
) -> Dict:
    """
    Measure correlation between embedding distance and tree distance.

    Tree distance = number of edges between two leaves (topological distance).
    Tests if nodes closer in embedding space are also closer in the tree.

    Args:
        tree: HierarchicalTree object
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node names (default: all leaves with embeddings)
        sample_size: Number of node pairs to sample
        random_state: Random seed for reproducibility

    Returns:
        Dict with Spearman correlation and other metrics
    """
    print("\n" + "="*60)
    print("TREE DISTANCE CORRELATION EVALUATION")
    print("="*60)

    random.seed(random_state)
    np.random.seed(random_state)

    # Determine which nodes to evaluate
    if test_nodes is None:
        leaf_names = [name for name in tree.leaves.keys() if name in embeddings_dict]
    else:
        leaf_names = [name for name in test_nodes if name in embeddings_dict and name in tree.leaves]

    print(f"Evaluating {len(leaf_names)} test nodes")
    print(f"Sampling {sample_size} pairs...")

    # Sample pairs
    all_pairs = [(i, j) for i in range(len(leaf_names)) for j in range(i+1, len(leaf_names))]
    max_pairs = min(sample_size, len(all_pairs))
    pairs = random.sample(all_pairs, max_pairs)

    embedding_dists = []
    tree_dists = []

    for i, j in tqdm(pairs, desc="Computing distances"):
        node_i_name, node_j_name = leaf_names[i], leaf_names[j]

        # Embedding distance (cosine distance)
        emb_dist = cosine_distances(
            embeddings_dict[node_i_name].reshape(1, -1),
            embeddings_dict[node_j_name].reshape(1, -1)
        )[0, 0]

        # Tree distance (use tree_distance method)
        node_i = tree.leaves[node_i_name]
        node_j = tree.leaves[node_j_name]
        tree_dist = tree.tree_distance(node_i, node_j)

        embedding_dists.append(emb_dist)
        tree_dists.append(tree_dist)

    # Spearman correlation (robust to outliers)
    correlation, p_value = spearmanr(embedding_dists, tree_dists)

    # Compute per-depth statistics
    tree_dist_unique = sorted(set(tree_dists))
    depth_stats = {}

    for td in tree_dist_unique[:10]:  # Only first 10 distances
        mask = np.array(tree_dists) == td
        if mask.sum() >= 10:  # Only if we have enough samples
            depth_stats[int(td)] = {
                'mean_emb_dist': float(np.mean(np.array(embedding_dists)[mask])),
                'std_emb_dist': float(np.std(np.array(embedding_dists)[mask])),
                'count': int(mask.sum())
            }

    print(f"\nSpearman correlation: {correlation:.4f} (p={p_value:.2e})")
    print(f"Number of pairs evaluated: {len(pairs)}")

    if depth_stats:
        print("\nMean embedding distance by tree distance:")
        for td in sorted(depth_stats.keys())[:5]:
            stats = depth_stats[td]
            print(f"  Tree dist {td}: {stats['mean_emb_dist']:.4f} ± {stats['std_emb_dist']:.4f} ({stats['count']} pairs)")

    return {
        'spearman_correlation': float(correlation),
        'p_value': float(p_value),
        'num_pairs': len(pairs),
        'embedding_distances': embedding_dists,
        'tree_distances': tree_dists,
        'depth_stats': depth_stats
    }


def evaluate_tree_structure(
    tree,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str] = None,
    depth_levels: List[int] = None,
    distance_sample_size: int = 2000,
    random_state: int = 42
) -> Dict:
    """
    Comprehensive evaluation of how well embeddings capture tree structure.

    Runs two evaluations:
    1. Subtree classification at multiple depths
    2. Tree distance correlation

    Args:
        tree: HierarchicalTree object
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node names (default: all leaves with embeddings)
        depth_levels: List of depths for subtree classification
        distance_sample_size: Sample size for distance correlation
        random_state: Random seed

    Returns:
        Dict with all evaluation results
    """
    print("\n" + "="*70)
    print(" " * 15 + "TREE STRUCTURE EVALUATION")
    print("="*70)

    results = {}

    # 1. Subtree classification
    results['subtree_classification'] = evaluate_subtree_classification(
        tree, embeddings_dict, test_nodes=test_nodes,
        depth_levels=depth_levels, random_state=random_state
    )

    # 2. Tree distance correlation
    results['distance_correlation'] = evaluate_tree_distance_correlation(
        tree, embeddings_dict, test_nodes=test_nodes,
        sample_size=distance_sample_size, random_state=random_state
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    if 'depth_results' in results['subtree_classification']:
        print("\nSubtree Classification:")
        for depth, metrics in results['subtree_classification']['depth_results'].items():
            print(f"  Depth {depth}: {metrics['accuracy']:.4f} "
                  f"(baseline: {metrics['random_baseline']:.4f}, "
                  f"{metrics['num_classes']} classes)")

    print(f"\nTree Distance Correlation:")
    print(f"  Spearman ρ: {results['distance_correlation']['spearman_correlation']:.4f}")

    print("\n" + "="*70)

    return results