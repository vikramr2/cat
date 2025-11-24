"""
Content-Based Evaluation
Tests if embeddings capture text similarity using BM25 as reference

Measures correlation between embedding similarity and BM25 text similarity
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.stats import spearmanr, pearsonr
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from typing import Dict, List, Optional
import random


def compute_bm25_similarity(
    doc1_tokens: List[str],
    doc2_tokens: List[str],
    corpus_tokens: List[List[str]]
) -> float:
    """
    Compute BM25 similarity between two documents.

    Args:
        doc1_tokens: Tokenized document 1
        doc2_tokens: Tokenized document 2
        corpus_tokens: List of all tokenized documents in corpus

    Returns:
        BM25 score of doc2 with respect to doc1 as query
    """
    # Create BM25 model from corpus
    bm25 = BM25Okapi(corpus_tokens)

    # Use doc1 as query, score doc2
    # We'll use a symmetric approach: average of both directions
    score_1to2 = bm25.get_scores(doc1_tokens)[corpus_tokens.index(doc2_tokens)]
    score_2to1 = bm25.get_scores(doc2_tokens)[corpus_tokens.index(doc1_tokens)]

    return (score_1to2 + score_2to1) / 2


def evaluate_embedding_bm25_correlation(
    embeddings_dict: Dict[str, np.ndarray],
    content_dict: Dict[str, str],
    test_nodes: Optional[List[str]] = None,
    sample_size: int = 2000,
    random_state: int = 42,
    tokenizer_fn = None
) -> Dict:
    """
    Measure correlation between embedding similarity and BM25 text similarity.

    Tests if nodes with similar embeddings also have similar text content.
    Higher correlation suggests embeddings preserve text information.

    Args:
        embeddings_dict: Dict mapping node_id -> embedding vector
        content_dict: Dict mapping node_id -> text content
        test_nodes: List of test node names (default: all nodes with both embeddings and content)
        sample_size: Number of node pairs to sample
        random_state: Random seed for reproducibility
        tokenizer_fn: Optional tokenizer function (default: simple whitespace split + lowercase)

    Returns:
        Dict with correlation metrics and statistics
    """
    print("\n" + "="*60)
    print("EMBEDDING-BM25 CORRELATION EVALUATION")
    print("="*60)

    random.seed(random_state)
    np.random.seed(random_state)

    # Default tokenizer: simple whitespace + lowercase
    if tokenizer_fn is None:
        tokenizer_fn = lambda text: text.lower().split()

    # Determine which nodes to evaluate
    if test_nodes is None:
        node_names = [name for name in embeddings_dict.keys() if name in content_dict]
    else:
        node_names = [name for name in test_nodes if name in embeddings_dict and name in content_dict]

    print(f"Evaluating {len(node_names)} nodes with both embeddings and content")

    # Tokenize all documents
    print("Tokenizing documents...")
    tokenized_docs = {}
    corpus_tokens = []

    for name in tqdm(node_names, desc="Tokenizing"):
        tokens = tokenizer_fn(content_dict[name])
        tokenized_docs[name] = tokens
        corpus_tokens.append(tokens)

    # Create BM25 model once for the entire corpus
    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_tokens)

    # Sample pairs
    print(f"Sampling {sample_size} pairs...")
    all_pairs = [(i, j) for i in range(len(node_names)) for j in range(i+1, len(node_names))]
    max_pairs = min(sample_size, len(all_pairs))
    pairs = random.sample(all_pairs, max_pairs)

    embedding_similarities = []
    bm25_scores = []

    print("Computing similarities...")
    for i, j in tqdm(pairs, desc="Computing pairs"):
        node_i_name, node_j_name = node_names[i], node_names[j]

        # Embedding similarity (cosine similarity)
        emb_sim = cosine_similarity(
            embeddings_dict[node_i_name].reshape(1, -1),
            embeddings_dict[node_j_name].reshape(1, -1)
        )[0, 0]

        # BM25 similarity (symmetric)
        tokens_i = tokenized_docs[node_i_name]
        tokens_j = tokenized_docs[node_j_name]

        # Use queries from both directions and average
        score_i_to_j = bm25.get_scores(tokens_i)[j]
        score_j_to_i = bm25.get_scores(tokens_j)[i]
        bm25_score = (score_i_to_j + score_j_to_i) / 2

        embedding_similarities.append(emb_sim)
        bm25_scores.append(bm25_score)

    # Compute correlations
    spearman_corr, spearman_p = spearmanr(embedding_similarities, bm25_scores)
    pearson_corr, pearson_p = pearsonr(embedding_similarities, bm25_scores)

    # Compute quantile statistics
    bm25_array = np.array(bm25_scores)
    emb_sim_array = np.array(embedding_similarities)

    # Divide BM25 scores into quantiles and compute mean embedding similarity
    quantiles = [0, 25, 50, 75, 90, 95, 100]
    quantile_stats = {}

    for i in range(len(quantiles) - 1):
        lower = np.percentile(bm25_array, quantiles[i])
        upper = np.percentile(bm25_array, quantiles[i+1])

        if i == len(quantiles) - 2:  # Last quantile, include upper bound
            mask = (bm25_array >= lower) & (bm25_array <= upper)
        else:
            mask = (bm25_array >= lower) & (bm25_array < upper)

        if mask.sum() > 0:
            quantile_stats[f"Q{quantiles[i]}-{quantiles[i+1]}"] = {
                'bm25_range': (float(lower), float(upper)),
                'mean_emb_sim': float(emb_sim_array[mask].mean()),
                'std_emb_sim': float(emb_sim_array[mask].std()),
                'count': int(mask.sum())
            }

    print(f"\nSpearman correlation: {spearman_corr:.4f} (p={spearman_p:.2e})")
    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.2e})")
    print(f"Number of pairs evaluated: {len(pairs)}")

    print("\nMean embedding similarity by BM25 score quantile:")
    for q_name, stats in quantile_stats.items():
        print(f"  {q_name}: emb_sim={stats['mean_emb_sim']:.4f} ± {stats['std_emb_sim']:.4f}, "
              f"bm25=[{stats['bm25_range'][0]:.2f}, {stats['bm25_range'][1]:.2f}] "
              f"({stats['count']} pairs)")

    return {
        'spearman_correlation': float(spearman_corr),
        'spearman_p_value': float(spearman_p),
        'pearson_correlation': float(pearson_corr),
        'pearson_p_value': float(pearson_p),
        'num_pairs': len(pairs),
        'embedding_similarities': embedding_similarities,
        'bm25_scores': bm25_scores,
        'quantile_stats': quantile_stats,
        'num_nodes': len(node_names)
    }


def evaluate_content_preservation(
    embeddings_dict: Dict[str, np.ndarray],
    content_dict: Dict[str, str],
    baseline_embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
    test_nodes: Optional[List[str]] = None,
    sample_size: int = 2000,
    random_state: int = 42,
    tokenizer_fn = None
) -> Dict:
    """
    Comprehensive evaluation of content preservation in embeddings.

    Evaluates both fine-tuned and baseline embeddings using the same node pairs
    to ensure fair comparison.

    Args:
        embeddings_dict: Dict mapping node_id -> embedding vector (fine-tuned model)
        content_dict: Dict mapping node_id -> text content
        baseline_embeddings_dict: Optional dict for baseline model embeddings
        test_nodes: List of test node names
        sample_size: Sample size for correlation evaluation
        random_state: Random seed (ensures same pairs for both models)
        tokenizer_fn: Optional custom tokenizer

    Returns:
        Dict with all evaluation results
    """
    print("\n" + "="*70)
    print(" " * 15 + "CONTENT PRESERVATION EVALUATION")
    print("="*70)

    results = {}

    # BM25 correlation for fine-tuned model
    print("\nEvaluating fine-tuned embeddings...")
    results['finetuned'] = evaluate_embedding_bm25_correlation(
        embeddings_dict, content_dict,
        test_nodes=test_nodes,
        sample_size=sample_size,
        random_state=random_state,
        tokenizer_fn=tokenizer_fn
    )

    # BM25 correlation for baseline model (if provided)
    if baseline_embeddings_dict is not None:
        print("\nEvaluating baseline embeddings...")
        results['baseline'] = evaluate_embedding_bm25_correlation(
            baseline_embeddings_dict, content_dict,
            test_nodes=test_nodes,
            sample_size=sample_size,
            random_state=random_state,  # Same random seed ensures same pairs
            tokenizer_fn=tokenizer_fn
        )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nFine-tuned Model BM25 Correlation:")
    print(f"  Spearman ρ: {results['finetuned']['spearman_correlation']:.4f}")
    print(f"  Pearson r: {results['finetuned']['pearson_correlation']:.4f}")
    print(f"  Nodes evaluated: {results['finetuned']['num_nodes']}")
    print(f"  Pairs sampled: {results['finetuned']['num_pairs']}")

    if baseline_embeddings_dict is not None:
        print(f"\nBaseline Model BM25 Correlation:")
        print(f"  Spearman ρ: {results['baseline']['spearman_correlation']:.4f}")
        print(f"  Pearson r: {results['baseline']['pearson_correlation']:.4f}")
        print(f"  Nodes evaluated: {results['baseline']['num_nodes']}")
        print(f"  Pairs sampled: {results['baseline']['num_pairs']}")

        # Compute differences
        spearman_diff = results['finetuned']['spearman_correlation'] - results['baseline']['spearman_correlation']
        pearson_diff = results['finetuned']['pearson_correlation'] - results['baseline']['pearson_correlation']

        print(f"\nDifference (Fine-tuned - Baseline):")
        print(f"  Δ Spearman ρ: {spearman_diff:+.4f}")
        print(f"  Δ Pearson r: {pearson_diff:+.4f}")

    print("\n" + "="*70)

    return results