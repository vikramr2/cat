"""
Evaluation utilities for disjoint clustering models
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_cluster_quality(
    embeddings_dict: Dict[str, np.ndarray],
    cluster_df: pd.DataFrame,
    test_node_ids: List[str] = None
) -> Dict:
    """
    Evaluate how well embeddings preserve cluster structure.
    
    Uses clustering quality metrics like silhouette score.
    
    Args:
        embeddings_dict: Dict mapping node_id -> embedding
        cluster_df: DataFrame with columns [node, cluster]
        test_node_ids: List of test node IDs (if None, use all nodes with embeddings)
    
    Returns:
        Dict with clustering metrics
    """
    print("\n" + "="*60)
    print("CLUSTER QUALITY EVALUATION")
    print("="*60)
    
    # Filter to test nodes if specified
    if test_node_ids is not None:
        eval_cluster_df = cluster_df[cluster_df['node'].astype(str).isin(test_node_ids)]
    else:
        eval_cluster_df = cluster_df
    
    # Get embeddings and labels
    embeddings = []
    labels = []
    valid_nodes = []
    
    for _, row in eval_cluster_df.iterrows():
        node_id = str(row['node'])
        if node_id in embeddings_dict:
            embeddings.append(embeddings_dict[node_id])
            labels.append(row['cluster'])
            valid_nodes.append(node_id)
    
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    
    print(f"\nEvaluating {len(embeddings)} nodes from {len(np.unique(labels))} clusters...")
    
    # Compute metrics
    silhouette = silhouette_score(embeddings, labels, metric='cosine')
    davies_bouldin = davies_bouldin_score(embeddings, labels)
    calinski_harabasz = calinski_harabasz_score(embeddings, labels)
    
    results = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'calinski_harabasz_score': calinski_harabasz,
        'n_nodes': len(embeddings),
        'n_clusters': len(np.unique(labels))
    }
    
    print("\nResults:")
    print(f"  Silhouette Score: {silhouette:.4f} (higher is better, range [-1, 1])")
    print(f"  Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")
    
    return results


def evaluate_intra_cluster_similarity(
    embeddings_dict: Dict[str, np.ndarray],
    cluster_df: pd.DataFrame,
    test_node_ids: List[str] = None,
    n_samples: int = 1000
) -> Dict:
    """
    Evaluate average similarity within clusters vs. across clusters.
    
    Args:
        embeddings_dict: Dict mapping node_id -> embedding
        cluster_df: DataFrame with columns [node, cluster]
        test_node_ids: List of test node IDs (if None, use all clusters)
        n_samples: Number of random pairs to sample
    
    Returns:
        Dict with intra/inter cluster similarity stats
    """
    print("\n" + "="*60)
    print("INTRA/INTER CLUSTER SIMILARITY EVALUATION")
    print("="*60)
    
    # Filter to test nodes if specified
    if test_node_ids is not None:
        eval_cluster_df = cluster_df[cluster_df['node'].astype(str).isin(test_node_ids)]
        eval_cluster_ids = eval_cluster_df['cluster'].unique().tolist()
    else:
        eval_cluster_df = cluster_df
        eval_cluster_ids = cluster_df['cluster'].unique().tolist()
    
    # Build cluster to nodes mapping
    cluster_to_nodes = {}
    for cluster_id in eval_cluster_ids:
        nodes = eval_cluster_df[eval_cluster_df['cluster'] == cluster_id]['node'].astype(str).tolist()
        nodes = [n for n in nodes if n in embeddings_dict]
        if len(nodes) >= 2:
            cluster_to_nodes[cluster_id] = nodes
    
    print(f"\nSampling from {len(cluster_to_nodes)} clusters...")
    
    intra_similarities = []
    inter_similarities = []
    
    # Sample intra-cluster pairs
    for _ in tqdm(range(n_samples), desc="Sampling intra-cluster"):
        cluster_id = np.random.choice(list(cluster_to_nodes.keys()))
        nodes = cluster_to_nodes[cluster_id]
        
        if len(nodes) >= 2:
            node1, node2 = np.random.choice(nodes, size=2, replace=False)
            emb1 = embeddings_dict[node1]
            emb2 = embeddings_dict[node2]
            sim = np.dot(emb1, emb2)
            intra_similarities.append(sim)
    
    # Sample inter-cluster pairs
    cluster_ids = list(cluster_to_nodes.keys())
    for _ in tqdm(range(n_samples), desc="Sampling inter-cluster"):
        if len(cluster_ids) >= 2:
            cluster1, cluster2 = np.random.choice(cluster_ids, size=2, replace=False)
            node1 = np.random.choice(cluster_to_nodes[cluster1])
            node2 = np.random.choice(cluster_to_nodes[cluster2])
            emb1 = embeddings_dict[node1]
            emb2 = embeddings_dict[node2]
            sim = np.dot(emb1, emb2)
            inter_similarities.append(sim)
    
    results = {
        'intra_mean': np.mean(intra_similarities),
        'intra_std': np.std(intra_similarities),
        'inter_mean': np.mean(inter_similarities),
        'inter_std': np.std(inter_similarities),
        'separation': np.mean(intra_similarities) - np.mean(inter_similarities),
        'intra_similarities': intra_similarities,
        'inter_similarities': inter_similarities
    }
    
    print("\nResults:")
    print(f"  Intra-cluster similarity: {results['intra_mean']:.4f} ± {results['intra_std']:.4f}")
    print(f"  Inter-cluster similarity: {results['inter_mean']:.4f} ± {results['inter_std']:.4f}")
    print(f"  Separation: {results['separation']:.4f} (higher is better)")
    
    return results


def evaluate_retrieval(
    embeddings_dict: Dict[str, np.ndarray],
    cluster_df: pd.DataFrame,
    test_node_ids: List[str],
    k_values: List[int] = None
) -> Dict:
    """
    Evaluate retrieval quality: given a query node, retrieve similar nodes
    and check if they're from the same cluster.
    
    Args:
        embeddings_dict: Dict mapping node_id -> embedding
        cluster_df: DataFrame with columns [node, cluster]
        test_node_ids: List of test node IDs
        k_values: List of k values for top-k retrieval
    
    Returns:
        Dict with retrieval metrics
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]
    
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION (CLUSTER MEMBERSHIP)")
    print("="*60)
    
    # Filter test nodes
    test_node_ids = [n for n in test_node_ids if n in embeddings_dict]
    
    # Get all nodes and embeddings
    all_nodes = list(embeddings_dict.keys())
    all_embeddings = np.array([embeddings_dict[n] for n in all_nodes])
    
    # Build node to cluster mapping
    node_to_cluster = {}
    for _, row in cluster_df.iterrows():
        node_to_cluster[str(row['node'])] = row['cluster']
    
    print(f"\nEvaluating {len(test_node_ids)} test nodes...")
    print(f"K values: {k_values}")
    
    results = {k: {'precision': [], 'recall': []} for k in k_values}
    
    for test_node in tqdm(test_node_ids, desc="Evaluating retrieval"):
        if test_node not in node_to_cluster:
            continue
        
        true_cluster = node_to_cluster[test_node]
        
        # Get same-cluster nodes
        same_cluster_nodes = [
            n for n in all_nodes 
            if n != test_node and node_to_cluster.get(n) == true_cluster
        ]
        
        if len(same_cluster_nodes) == 0:
            continue
        
        # Compute similarities
        test_emb = embeddings_dict[test_node].reshape(1, -1)
        similarities = cosine_similarity(test_emb, all_embeddings)[0]
        
        # Get top-k (excluding self)
        node_sims = [(all_nodes[i], similarities[i]) for i in range(len(all_nodes)) if all_nodes[i] != test_node]
        node_sims.sort(key=lambda x: x[1], reverse=True)
        
        # Evaluate for each k
        for k in k_values:
            top_k_nodes = [node for node, _ in node_sims[:k]]
            
            # Count how many are from same cluster
            true_positives = len([n for n in top_k_nodes if node_to_cluster.get(n) == true_cluster])
            
            precision = true_positives / k if k > 0 else 0
            recall = true_positives / len(same_cluster_nodes) if len(same_cluster_nodes) > 0 else 0
            
            results[k]['precision'].append(precision)
            results[k]['recall'].append(recall)
    
    # Compute averages
    summary = {}
    for k in k_values:
        summary[k] = {
            'precision@k': np.mean(results[k]['precision']),
            'recall@k': np.mean(results[k]['recall']),
            'num_queries': len(results[k]['precision'])
        }
    
    print("\nResults:")
    for k in k_values:
        print(f"  K={k}:")
        print(f"    Precision@{k}: {summary[k]['precision@k']:.4f}")
        print(f"    Recall@{k}: {summary[k]['recall@k']:.4f}")
    
    return {
        'summary': summary,
        'detailed': results,
        'k_values': k_values
    }


def plot_similarity_distributions(intra_inter_results):
    """Plot intra-cluster vs inter-cluster similarity distributions"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(intra_inter_results['intra_similarities'], bins=50, alpha=0.7, 
            label='Intra-cluster', color='#2ecc71', edgecolor='black')
    ax.hist(intra_inter_results['inter_similarities'], bins=50, alpha=0.7, 
            label='Inter-cluster', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Similarity Distributions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Box plot
    ax = axes[1]
    data = [intra_inter_results['intra_similarities'], 
            intra_inter_results['inter_similarities']]
    bp = ax.boxplot(data, labels=['Intra-cluster', 'Inter-cluster'],
                    patch_artist=True, widths=0.6)
    
    # Color boxes
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Similarity Comparison', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def plot_retrieval_results(retrieval_results):
    """Plot retrieval evaluation results"""
    
    k_values = retrieval_results['k_values']
    precisions = [retrieval_results['summary'][k]['precision@k'] for k in k_values]
    recalls = [retrieval_results['summary'][k]['recall@k'] for k in k_values]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(k_values, precisions, marker='o', linewidth=2, label='Precision@K', color='#3498db')
    ax.plot(k_values, recalls, marker='s', linewidth=2, label='Recall@K', color='#e67e22')
    
    ax.set_xlabel('K (Number of retrieved items)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Retrieval Performance (Same Cluster)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
