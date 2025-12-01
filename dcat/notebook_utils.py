"""
Utility functions for notebook usage
Clean, reusable code to avoid repetition in cells
"""

import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from dataset import DisjointClusterTripletDataset
from model import TripletEmbeddingModel
from losses import TripletMarginLoss
from trainer import train_epoch, evaluate
from split_utils import create_cluster_based_split, create_node_based_split, print_split_info


def load_cluster_and_metadata(cluster_path, metadata_path):
    """Load cluster assignments and metadata"""
    print(f"Loading cluster data from {cluster_path}...")
    cluster_df = pd.read_csv(cluster_path)
    print(f"  ✓ Loaded: {cluster_df.shape[0]} node-cluster assignments")
    
    print(f"\nLoading metadata from {metadata_path}...")
    metadata_df = pd.read_csv(metadata_path)
    print(f"  ✓ Loaded: {len(metadata_df)} entries")
    
    # Sanity check
    print(f"\nCluster Statistics:")
    print(f"  Unique nodes: {cluster_df['node'].nunique()}")
    print(f"  Unique clusters: {cluster_df['cluster'].nunique()}")
    
    cluster_sizes = cluster_df['cluster'].value_counts()
    print(f"  Mean cluster size: {cluster_sizes.mean():.2f}")
    print(f"  Median cluster size: {cluster_sizes.median():.0f}")
    print(f"  Largest cluster: {cluster_sizes.max()} nodes")
    
    return cluster_df, metadata_df


def compute_embeddings(model, tokenizer, metadata_df, device, batch_size=32, max_length=512):
    """
    Compute embeddings for all nodes in metadata using BASE MODEL.
    This is used for baseline evaluation with pre-trained models.
    
    Returns dict mapping node_id (str) -> embedding (numpy array)
    """
    embeddings_dict = {}

    # Prepare texts
    node_ids = metadata_df['id'].astype(str).values
    texts = []
    valid_ids = []

    print(f"Preparing {len(node_ids)} documents...")
    for node_id in tqdm(node_ids, desc="Preparing"):
        row = metadata_df[metadata_df['id'] == int(node_id)].iloc[0]
        title = str(row['title']) if pd.notna(row['title']) else ""
        abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
        text = f"{title} {abstract}".strip()

        if text:
            texts.append(text)
            valid_ids.append(node_id)

    # Compute embeddings in batches
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = valid_ids[i:i+batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            # Get model outputs
            outputs = model(**inputs)
            
            # Extract CLS token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :]
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy
            embeddings = embeddings.cpu().numpy()

            for node_id, emb in zip(batch_ids, embeddings):
                embeddings_dict[node_id] = emb

    print(f"✓ Computed embeddings for {len(embeddings_dict)} nodes")
    return embeddings_dict


def compute_finetuned_embeddings(model, tokenizer, metadata_df, device, batch_size=32, max_length=512):
    """
    Compute embeddings for all nodes in metadata using FINE-TUNED MODEL.
    This handles TripletEmbeddingModel which has custom forward pass.
    
    Returns dict mapping node_id (str) -> embedding (numpy array)
    """
    embeddings_dict = {}

    # Prepare texts
    node_ids = metadata_df['id'].astype(str).values
    texts = []
    valid_ids = []

    print(f"Preparing {len(node_ids)} documents...")
    for node_id in tqdm(node_ids, desc="Preparing"):
        row = metadata_df[metadata_df['id'] == int(node_id)].iloc[0]
        title = str(row['title']) if pd.notna(row['title']) else ""
        abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
        text = f"{title} {abstract}".strip()

        if text:
            texts.append(text)
            valid_ids.append(node_id)

    # Compute embeddings in batches
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_ids = valid_ids[i:i+batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            # TripletEmbeddingModel forward pass
            # Already returns normalized embeddings
            embeddings = model(inputs['input_ids'], inputs['attention_mask'])
            
            # Convert to numpy
            embeddings = embeddings.cpu().numpy()

            for node_id, emb in zip(batch_ids, embeddings):
                embeddings_dict[node_id] = emb

    print(f"✓ Computed embeddings for {len(embeddings_dict)} nodes")
    return embeddings_dict


def train_disjoint_model(
    cluster_df,
    metadata_df,
    train_node_ids,
    model_name='ncbi/MedCPT-Article-Encoder',
    device='cuda',
    batch_size=16,
    epochs=3,
    lr=2e-5,
    margin=1.0,
    samples_per_cluster=5,
    pooling='cls',
    max_length=512
):
    """
    Train disjoint cluster triplet loss model.
    Returns: (trained_model, tokenizer, history)
    
    Args:
        train_node_ids: List of node IDs (as strings) to use for training
    """
    print("\n" + "="*60)
    print("DISJOINT CLUSTER TRIPLET LOSS TRAINING")
    print("="*60)

    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TripletEmbeddingModel(model_name, pooling=pooling).to(device)

    # Create dataset - pass train_node_ids instead of train_clusters
    print("\nCreating training dataset...")
    train_dataset = DisjointClusterTripletDataset(
        cluster_df=cluster_df,
        metadata_df=metadata_df,
        tokenizer=tokenizer,
        max_length=max_length,
        samples_per_cluster=samples_per_cluster,
        train_nodes=train_node_ids  # Changed from train_clusters
    )
    print(f"  Generated {len(train_dataset)} training triplets")

    # Create validation split (10% of training data)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                           shuffle=False, num_workers=0)
    
    print(f"  Train samples: {len(train_subset)}")
    print(f"  Val samples: {len(val_subset)}")

    # Loss and optimizer
    loss_fn = TripletMarginLoss(margin=margin)
    print(f"\nTriplet Margin Loss:")
    print(f"  Margin: {margin}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print('='*60)

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Val loss: {val_loss:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"✓ New best validation loss!")

    print(f"\n{'='*60}")
    print(f"Training complete! Best val loss: {best_val_loss:.4f}")
    print('='*60)

    return model, tokenizer, history


def plot_training_history(history):
    """Plot training curves"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], marker='o', label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], marker='s', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Disjoint Cluster Triplet Loss Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_baseline_finetuned(baseline_results, finetuned_results):
    """
    Compare baseline vs fine-tuned results and create comparison table.
    Returns DataFrame with comparison.
    """
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: BASELINE VS FINE-TUNED")
    print("="*80)

    # Create comparison table
    comparison_data = {
        'K': [],
        'Baseline Precision@K': [],
        'Fine-tuned Precision@K': [],
        'Improvement': []
    }

    k_values = baseline_results['topk']['k_values']
    for k in k_values:
        bl_prec = baseline_results['topk']['summary'][k]['precision@k']
        ft_prec = finetuned_results['topk']['summary'][k]['precision@k']
        improvement = ((ft_prec - bl_prec) / bl_prec) * 100 if bl_prec != 0 else 0
        
        comparison_data['K'].append(k)
        comparison_data['Baseline Precision@K'].append(f"{bl_prec:.4f}")
        comparison_data['Fine-tuned Precision@K'].append(f"{ft_prec:.4f}")
        comparison_data['Improvement'].append(f"{improvement:+.2f}%")

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # AUC comparison
    if 'auc' in baseline_results and 'auc' in finetuned_results:
        bl_auc = baseline_results['auc']['auc_roc']
        ft_auc = finetuned_results['auc']['auc_roc']
        auc_improvement = ((ft_auc - bl_auc) / bl_auc) * 100
        
        print(f"\nAUC-ROC:")
        print(f"  Baseline: {bl_auc:.4f}")
        print(f"  Fine-tuned: {ft_auc:.4f}")
        print(f"  Improvement: {auc_improvement:+.2f}%")

    print("="*80)
    
    return comparison_df


def plot_comparison(baseline_results, finetuned_results):
    """Plot comparison between baseline and fine-tuned results"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    k_values = baseline_results['topk']['k_values']
    bl_precs = [baseline_results['topk']['summary'][k]['precision@k'] for k in k_values]
    ft_precs = [finetuned_results['topk']['summary'][k]['precision@k'] for k in k_values]

    # Precision@K comparison
    ax1.plot(k_values, bl_precs, marker='o', linewidth=2, label='Baseline', color='#e74c3c')
    ax1.plot(k_values, ft_precs, marker='s', linewidth=2, label='Fine-tuned', color='#2ecc71')
    ax1.set_xlabel('K', fontsize=12)
    ax1.set_ylabel('Precision@K', fontsize=12)
    ax1.set_title('Link Prediction Performance', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Improvement bars
    improvements = [(ft - bl) for bl, ft in zip(bl_precs, ft_precs)]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    ax2.bar(range(len(k_values)), improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('K', fontsize=12)
    ax2.set_ylabel('Δ Precision@K', fontsize=12)
    ax2.set_title('Performance Improvement', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels(k_values)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
