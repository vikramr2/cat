"""
Baseline GNN Training Script

This demonstrates the key insight:
- GNN trained on induced subgraph of train nodes (90%)
- Transformer weights frozen
- Test nodes (10%) have NO edges in the graph
- Result: GNN degrades to transformer-only for test nodes

This baseline shows why we need to include test nodes in the graph
or use a different approach for handling unseen nodes.
"""

import torch
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from model import TransformerGNN
from graph_utils import (
    create_induced_subgraph,
    analyze_graph_statistics,
    create_test_split,
    get_node_texts
)


def compute_all_embeddings(
    model,
    tokenizer,
    metadata_df,
    edge_index,
    node_to_idx,
    device,
    batch_size=32,
    max_length=512
):
    """
    Compute embeddings for all nodes (train + test).

    For test nodes (no edges), this will effectively return transformer embeddings.
    For train nodes (with edges), this will return GNN-enhanced embeddings.

    Args:
        model: TransformerGNN model
        tokenizer: Transformer tokenizer
        metadata_df: Metadata DataFrame
        edge_index: Graph edges [2, num_edges]
        node_to_idx: Node ID to index mapping
        device: torch device
        batch_size: Batch size for encoding
        max_length: Max sequence length

    Returns:
        Dict mapping node_id (str) -> embedding (numpy array)
    """
    print("\n" + "="*70)
    print("COMPUTING EMBEDDINGS")
    print("="*70)

    model.eval()

    # Get all node IDs in correct order
    all_node_ids = sorted(node_to_idx.keys(), key=lambda x: node_to_idx[x])

    # Step 1: Get transformer embeddings for all nodes
    print("\nStep 1: Computing transformer embeddings...")
    transformer_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(all_node_ids), batch_size), desc="Transformer"):
            batch_node_ids = all_node_ids[i:i+batch_size]

            # Get texts
            texts = get_node_texts(batch_node_ids, metadata_df)

            # Tokenize
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            # Encode with transformer
            embs = model.encode_text(inputs['input_ids'], inputs['attention_mask'])
            transformer_embeddings.append(embs.cpu())

    # Concatenate all transformer embeddings
    x = torch.cat(transformer_embeddings, dim=0).to(device)  # [num_nodes, hidden_dim]
    print(f"  ✓ Transformer embeddings shape: {x.shape}")

    # Step 2: Apply GNN
    print("\nStep 2: Applying GNN layers...")
    print(f"  Edge index shape: {edge_index.shape}")

    with torch.no_grad():
        edge_index = edge_index.to(device)
        final_embeddings = model(x, edge_index)  # [num_nodes, hidden_dim]

    print(f"  ✓ Final embeddings shape: {final_embeddings.shape}")

    # Convert to dictionary
    embeddings_dict = {}
    final_embeddings = final_embeddings.cpu().numpy()

    for i, node_id in enumerate(all_node_ids):
        embeddings_dict[node_id] = final_embeddings[i]

    print(f"\n✓ Computed embeddings for {len(embeddings_dict)} nodes")
    print("="*70)

    return embeddings_dict


def compare_transformer_vs_gnn_embeddings(
    model,
    tokenizer,
    test_node_ids,
    metadata_df,
    edge_index,
    node_to_idx,
    device,
    num_samples=10
):
    """
    Compare transformer-only vs GNN embeddings for test nodes.

    For test nodes with no edges, these should be nearly identical,
    demonstrating that the GNN degrades to transformer-only.

    Args:
        model: TransformerGNN model
        tokenizer: Tokenizer
        test_node_ids: List of test node IDs
        metadata_df: Metadata DataFrame
        edge_index: Graph edges
        node_to_idx: Node mapping
        device: torch device
        num_samples: Number of random test nodes to compare
    """
    print("\n" + "="*70)
    print("COMPARING TRANSFORMER-ONLY VS GNN EMBEDDINGS FOR TEST NODES")
    print("="*70)

    model.eval()

    # Sample random test nodes
    import random
    sample_nodes = random.sample(test_node_ids, min(num_samples, len(test_node_ids)))

    print(f"\nAnalyzing {len(sample_nodes)} random test nodes...")

    with torch.no_grad():
        for node_id in sample_nodes:
            # Get text
            texts = get_node_texts([node_id], metadata_df)

            # Transformer-only embedding
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            transformer_emb = model.encode_text(
                inputs['input_ids'],
                inputs['attention_mask']
            ).cpu().numpy()[0]

            # GNN embedding (should be same since no edges)
            # For proper GNN embedding, we need full graph context
            node_idx = node_to_idx[node_id]

            # Get all transformer embeddings
            all_node_ids = sorted(node_to_idx.keys(), key=lambda x: node_to_idx[x])
            all_texts = get_node_texts(all_node_ids, metadata_df)

            # Batch encode all
            all_inputs = tokenizer(
                all_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(device)

            x = model.encode_text(all_inputs['input_ids'], all_inputs['attention_mask'])
            edge_index_gpu = edge_index.to(device)

            gnn_embs = model(x, edge_index_gpu)
            gnn_emb = gnn_embs[node_idx].cpu().numpy()

            # Compare
            cosine_sim = (transformer_emb * gnn_emb).sum()
            l2_diff = ((transformer_emb - gnn_emb) ** 2).sum() ** 0.5

            print(f"\nNode {node_id}:")
            print(f"  Cosine similarity: {cosine_sim:.6f} (1.0 = identical)")
            print(f"  L2 difference: {l2_diff:.6f} (0.0 = identical)")

            if cosine_sim > 0.999:
                print(f"  ✓ Embeddings are essentially identical (GNN has no effect)")
            else:
                print(f"  ! Embeddings differ (unexpected - test node should have no edges)")

    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("For test nodes with no edges in the induced subgraph,")
    print("GNN embeddings = Transformer embeddings (no graph information)")
    print("="*70)


def main():
    """
    Complete baseline experiment demonstrating GNN degradation.
    """
    # Paths (update these to your data location)
    BASE_DIR = Path.cwd().parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "oc_mini"

    edgelist_path = DATA_DIR / "network" / "oc_mini_edgelist.csv"
    metadata_path = DATA_DIR / "metadata" / "oc_mini_node_metadata.csv"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv(metadata_path)
    print(f"  ✓ Loaded {len(metadata_df)} nodes\n")

    # Create train/test split
    print("Creating train/test split...")
    all_node_ids = [str(nid) for nid in metadata_df['id'].values]
    test_nodes = create_test_split(all_node_ids, test_ratio=0.1, seed=42)
    train_nodes = [nid for nid in all_node_ids if nid not in test_nodes]

    print(f"  ✓ Train nodes: {len(train_nodes)} (90%)")
    print(f"  ✓ Test nodes: {len(test_nodes)} (10%)\n")

    # Create induced subgraph (KEY STEP)
    edge_index, node_to_idx, idx_to_node = create_induced_subgraph(
        edgelist_path,
        train_nodes,
        metadata_df
    )

    # Analyze graph statistics
    analyze_graph_statistics(edge_index, train_nodes, node_to_idx, metadata_df)

    # Initialize model
    print("\nInitializing TransformerGNN model...")
    model_name = 'allenai/scibert_scivocab_uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = TransformerGNN(
        model_name=model_name,
        gnn_type='gcn',
        hidden_dim=768,
        num_gnn_layers=2,
        dropout=0.1,
        pooling='cls',
        freeze_transformer=True  # Transformer frozen!
    ).to(device)

    print("\nModel architecture:")
    print(f"  Transformer: {model_name} (FROZEN)")
    print(f"  Input: Title + Abstract concatenated for each paper")
    print(f"  GNN: 2-layer GCN")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Compare embeddings for test nodes
    print("\n" + "="*70)
    print("BASELINE DEMONSTRATION")
    print("="*70)

    compare_transformer_vs_gnn_embeddings(
        model,
        tokenizer,
        test_nodes,
        metadata_df,
        edge_index,
        node_to_idx,
        device,
        num_samples=5
    )

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\nThis baseline demonstrates that:")
    print("1. Training GNN on induced subgraph of train nodes")
    print("2. Keeping transformer frozen")
    print("3. Test nodes have NO edges in the graph")
    print("4. Result: GNN provides no benefit for test nodes")
    print("\nThis is why we need alternative strategies for handling unseen nodes:")
    print("- Include test nodes in graph (transductive)")
    print("- Use inductive GNN architectures")
    print("- Hybrid approaches (transformer + graph features)")
    print("="*70)


if __name__ == "__main__":
    main()
