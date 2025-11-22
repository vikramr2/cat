"""
Network Link Prediction Evaluation
Tests if embedding similarity can recover actual network edges
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def load_network_edges(edgelist_path: str) -> pd.DataFrame:
    """Load edgelist from CSV"""
    df = pd.read_csv(edgelist_path)
    # Convert IDs to strings for consistency with tree leaf names
    df['source'] = df['source'].astype(str)
    df['target'] = df['target'].astype(str)
    return df


def get_node_degree(edgelist_df: pd.DataFrame) -> Dict[str, int]:
    """Get degree (number of neighbors) for each node"""
    degrees = defaultdict(int)

    for _, row in edgelist_df.iterrows():
        degrees[row['source']] += 1
        degrees[row['target']] += 1

    return dict(degrees)


def get_neighbors(node: str, edgelist_df: pd.DataFrame) -> Set[str]:
    """Get all neighbors of a node"""
    neighbors = set()

    # Find edges where node is source
    neighbors.update(edgelist_df[edgelist_df['source'] == node]['target'].values)
    # Find edges where node is target
    neighbors.update(edgelist_df[edgelist_df['target'] == node]['source'].values)

    return neighbors


def evaluate_link_prediction_topk(
    edgelist_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str],
    k_values: List[int] = None
) -> Dict:
    """
    Evaluate link prediction using top-k retrieval.

    For each test node v:
    1. Get true neighbors from edgelist
    2. Rank all other nodes by cosine similarity
    3. Check if true neighbors appear in top-k

    Args:
        edgelist_df: DataFrame with 'source' and 'target' columns
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node IDs
        k_values: List of k values to evaluate (default: [10, 20, 50, 100])

    Returns:
        Dict with precision@k, recall@k, and other metrics
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]

    print(f"\nEvaluating link prediction with top-k retrieval...")
    print(f"Test nodes: {len(test_nodes)}")
    print(f"K values: {k_values}")

    # Filter test nodes to those with embeddings
    test_nodes = [n for n in test_nodes if n in embeddings_dict]

    # Get all nodes with embeddings
    all_nodes = list(embeddings_dict.keys())
    all_embeddings = np.array([embeddings_dict[n] for n in all_nodes])

    # Get node degrees
    node_degrees = get_node_degree(edgelist_df)

    results = {k: {'precision': [], 'recall': [], 'hit_rate': []} for k in k_values}

    for test_node in tqdm(test_nodes, desc="Evaluating nodes"):
        # Get true neighbors
        true_neighbors = get_neighbors(test_node, edgelist_df)
        true_neighbors = [n for n in true_neighbors if n in embeddings_dict and n != test_node]

        if len(true_neighbors) == 0:
            continue

        # Get test node embedding
        test_emb = embeddings_dict[test_node].reshape(1, -1)

        # Compute similarities to all nodes
        similarities = cosine_similarity(test_emb, all_embeddings)[0]

        # Get top-k predictions (excluding self)
        node_similarities = [(all_nodes[i], similarities[i]) for i in range(len(all_nodes)) if all_nodes[i] != test_node]
        node_similarities.sort(key=lambda x: x[1], reverse=True)

        # Evaluate for each k
        for k in k_values:
            top_k_nodes = [node for node, _ in node_similarities[:k]]

            # Compute metrics
            true_positives = len(set(top_k_nodes) & set(true_neighbors))

            precision = true_positives / k if k > 0 else 0
            recall = true_positives / len(true_neighbors) if len(true_neighbors) > 0 else 0
            hit_rate = 1.0 if true_positives > 0 else 0.0

            results[k]['precision'].append(precision)
            results[k]['recall'].append(recall)
            results[k]['hit_rate'].append(hit_rate)

    # Compute averages
    summary = {}
    for k in k_values:
        summary[k] = {
            'precision@k': np.mean(results[k]['precision']),
            'recall@k': np.mean(results[k]['recall']),
            'hit_rate@k': np.mean(results[k]['hit_rate']),
            'num_nodes': len(results[k]['precision'])
        }

    return {
        'summary': summary,
        'detailed': results,
        'k_values': k_values
    }


def evaluate_link_prediction_auc(
    edgelist_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str],
    num_negative_samples: int = 10
) -> Dict:
    """
    Evaluate link prediction using AUC metrics.

    For each test node:
    1. Positive samples: actual neighbors
    2. Negative samples: randomly sampled non-neighbors
    3. Score: cosine similarity
    4. Compute AUC-ROC and AUC-PR

    Args:
        edgelist_df: DataFrame with 'source' and 'target' columns
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node IDs
        num_negative_samples: Number of negative samples per positive sample

    Returns:
        Dict with AUC metrics
    """
    print(f"\nEvaluating link prediction with AUC metrics...")
    print(f"Test nodes: {len(test_nodes)}")
    print(f"Negative samples per positive: {num_negative_samples}")

    test_nodes = [n for n in test_nodes if n in embeddings_dict]
    all_nodes = list(embeddings_dict.keys())

    y_true = []
    y_scores = []

    for test_node in tqdm(test_nodes, desc="Sampling edges"):
        # Get true neighbors
        true_neighbors = get_neighbors(test_node, edgelist_df)
        true_neighbors = [n for n in true_neighbors if n in embeddings_dict and n != test_node]

        if len(true_neighbors) == 0:
            continue

        test_emb = embeddings_dict[test_node]

        # Positive samples
        for neighbor in true_neighbors:
            neighbor_emb = embeddings_dict[neighbor]
            similarity = np.dot(test_emb, neighbor_emb)
            y_true.append(1)
            y_scores.append(similarity)

        # Negative samples
        non_neighbors = [n for n in all_nodes if n not in true_neighbors and n != test_node]
        num_negatives = min(num_negative_samples * len(true_neighbors), len(non_neighbors))

        negative_samples = np.random.choice(non_neighbors, num_negatives, replace=False)

        for non_neighbor in negative_samples:
            non_neighbor_emb = embeddings_dict[non_neighbor]
            similarity = np.dot(test_emb, non_neighbor_emb)
            y_true.append(0)
            y_scores.append(similarity)

    # Compute metrics
    auc_roc = roc_auc_score(y_true, y_scores)
    auc_pr = average_precision_score(y_true, y_scores)

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'num_samples': len(y_true),
        'num_positive': sum(y_true),
        'num_negative': len(y_true) - sum(y_true),
        'y_true': y_true,
        'y_scores': y_scores
    }


def evaluate_neighbor_overlap(
    edgelist_df: pd.DataFrame,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str],
    k: int = 20
) -> Dict:
    """
    Compute Jaccard similarity between true neighbors and predicted neighbors.

    Args:
        edgelist_df: DataFrame with edges
        embeddings_dict: Node embeddings
        test_nodes: Test node IDs
        k: Number of neighbors to retrieve

    Returns:
        Dict with overlap metrics
    """
    print(f"\nEvaluating neighbor overlap (k={k})...")

    test_nodes = [n for n in test_nodes if n in embeddings_dict]
    all_nodes = list(embeddings_dict.keys())
    all_embeddings = np.array([embeddings_dict[n] for n in all_nodes])

    jaccard_scores = []

    for test_node in tqdm(test_nodes, desc="Computing overlaps"):
        # True neighbors
        true_neighbors = get_neighbors(test_node, edgelist_df)
        true_neighbors = set([n for n in true_neighbors if n in embeddings_dict])

        if len(true_neighbors) == 0:
            continue

        # Predicted neighbors (top-k by similarity)
        test_emb = embeddings_dict[test_node].reshape(1, -1)
        similarities = cosine_similarity(test_emb, all_embeddings)[0]

        top_k_indices = np.argsort(similarities)[::-1][1:k+1]  # Exclude self
        predicted_neighbors = set([all_nodes[i] for i in top_k_indices])

        # Jaccard similarity
        intersection = len(true_neighbors & predicted_neighbors)
        union = len(true_neighbors | predicted_neighbors)
        jaccard = intersection / union if union > 0 else 0

        jaccard_scores.append(jaccard)

    return {
        'mean_jaccard': np.mean(jaccard_scores),
        'median_jaccard': np.median(jaccard_scores),
        'std_jaccard': np.std(jaccard_scores),
        'jaccard_scores': jaccard_scores
    }


def plot_link_prediction_results(topk_results, auc_results=None):
    """Plot link prediction evaluation results"""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Precision@K and Recall@K
    ax = axes[0]
    k_values = topk_results['k_values']
    precisions = [topk_results['summary'][k]['precision@k'] for k in k_values]
    recalls = [topk_results['summary'][k]['recall@k'] for k in k_values]

    ax.plot(k_values, precisions, marker='o', linewidth=2, label='Precision@K', color='#2ecc71')
    ax.plot(k_values, recalls, marker='s', linewidth=2, label='Recall@K', color='#3498db')
    ax.set_xlabel('K (Number of retrieved neighbors)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision@K and Recall@K', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 2. Hit Rate@K
    ax = axes[1]
    hit_rates = [topk_results['summary'][k]['hit_rate@k'] for k in k_values]

    bars = ax.bar(range(len(k_values)), hit_rates, color='#e74c3c', alpha=0.7)
    ax.set_xlabel('K', fontsize=12)
    ax.set_ylabel('Hit Rate@K', fontsize=12)
    ax.set_title('Hit Rate@K (% nodes with ≥1 true neighbor in top-K)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels(k_values)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, hit_rates)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.3f}',
               ha='center', va='bottom', fontsize=10)

    # 3. AUC metrics or summary
    ax = axes[2]
    ax.axis('off')

    if auc_results:
        from sklearn.metrics import roc_curve, precision_recall_curve

        # Create small subplots
        fig2, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

        # ROC curve
        fpr, tpr, _ = roc_curve(auc_results['y_true'], auc_results['y_scores'])
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"AUC-ROC = {auc_results['auc_roc']:.3f}")
        ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax_roc.set_xlabel('False Positive Rate', fontsize=12)
        ax_roc.set_ylabel('True Positive Rate', fontsize=12)
        ax_roc.set_title('ROC Curve', fontsize=13, fontweight='bold')
        ax_roc.legend(fontsize=11)
        ax_roc.grid(True, alpha=0.3)

        # PR curve
        precision, recall, _ = precision_recall_curve(auc_results['y_true'], auc_results['y_scores'])
        ax_pr.plot(recall, precision, linewidth=2, label=f"AUC-PR = {auc_results['auc_pr']:.3f}")
        ax_pr.set_xlabel('Recall', fontsize=12)
        ax_pr.set_ylabel('Precision', fontsize=12)
        ax_pr.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
        ax_pr.legend(fontsize=11)
        ax_pr.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # Summary text in original plot
    summary_text = "Link Prediction Summary\n" + "="*40 + "\n\n"

    for k in k_values:
        summary_text += f"K={k}:\n"
        summary_text += f"  Precision: {topk_results['summary'][k]['precision@k']:.4f}\n"
        summary_text += f"  Recall: {topk_results['summary'][k]['recall@k']:.4f}\n"
        summary_text += f"  Hit Rate: {topk_results['summary'][k]['hit_rate@k']:.4f}\n\n"

    if auc_results:
        summary_text += f"\nAUC Metrics:\n"
        summary_text += f"  AUC-ROC: {auc_results['auc_roc']:.4f}\n"
        summary_text += f"  AUC-PR: {auc_results['auc_pr']:.4f}\n"

    axes[2].text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_network_link_prediction(
    edgelist_path: str,
    embeddings_dict: Dict[str, np.ndarray],
    test_nodes: List[str],
    k_values: List[int] = None,
    compute_auc: bool = True,
    num_negative_samples: int = 10
) -> Dict:
    """
    Comprehensive link prediction evaluation on network edges.

    Args:
        edgelist_path: Path to CSV with 'source' and 'target' columns
        embeddings_dict: Dict mapping node_id -> embedding vector
        test_nodes: List of test node IDs
        k_values: K values for top-k evaluation
        compute_auc: Whether to compute AUC metrics
        num_negative_samples: Negatives per positive for AUC

    Returns:
        Dict with all evaluation results
    """
    print("\n" + "="*60)
    print("NETWORK LINK PREDICTION EVALUATION")
    print("="*60)

    # Load edgelist
    print(f"\nLoading edgelist from: {edgelist_path}")
    edgelist_df = load_network_edges(edgelist_path)
    print(f"Loaded {len(edgelist_df)} edges")

    # Get unique nodes
    all_edge_nodes = set(edgelist_df['source'].unique()) | set(edgelist_df['target'].unique())
    print(f"Network has {len(all_edge_nodes)} unique nodes")

    # Filter test nodes
    test_nodes_in_network = [n for n in test_nodes if n in all_edge_nodes and n in embeddings_dict]
    print(f"Test nodes with embeddings and edges: {len(test_nodes_in_network)}")

    results = {}

    # 1. Top-K evaluation
    results['topk'] = evaluate_link_prediction_topk(
        edgelist_df,
        embeddings_dict,
        test_nodes_in_network,
        k_values=k_values
    )

    # 2. AUC evaluation
    if compute_auc:
        results['auc'] = evaluate_link_prediction_auc(
            edgelist_df,
            embeddings_dict,
            test_nodes_in_network,
            num_negative_samples=num_negative_samples
        )

    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\nTop-K Metrics:")
    for k in results['topk']['k_values']:
        summary = results['topk']['summary'][k]
        print(f"  K={k}:")
        print(f"    Precision@{k}: {summary['precision@k']:.4f}")
        print(f"    Recall@{k}: {summary['recall@k']:.4f}")
        print(f"    Hit Rate@{k}: {summary['hit_rate@k']:.4f}")

    if 'auc' in results:
        print(f"\nAUC Metrics:")
        print(f"  AUC-ROC: {results['auc']['auc_roc']:.4f}")
        print(f"  AUC-PR: {results['auc']['auc_pr']:.4f}")

    print("\n" + "="*60)

    return results
