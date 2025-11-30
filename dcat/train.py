"""
Core training functions for disjoint clustering

This module provides the main training logic without CLI argument parsing,
making it easy to use from Jupyter notebooks.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import DisjointClusterTripletDataset
from model import TripletEmbeddingModel
from losses import TripletMarginLoss
from trainer import train_epoch, evaluate
from split_utils import create_cluster_based_split, print_split_info


def train_model(
    cluster_csv_path,
    metadata_csv_path,
    output_dir='./finetuned_model',
    model_name='ncbi/MedCPT-Article-Encoder',
    device='cuda',
    batch_size=16,
    epochs=3,
    lr=2e-5,
    margin=1.0,
    max_length=512,
    samples_per_cluster=5,
    pooling='cls',
    train_split=0.9,
    test_ratio=0.1,
    seed=42
):
    """
    Train disjoint cluster triplet loss model with proper train/test split.

    Args:
        cluster_csv_path: Path to cluster CSV (columns: node, cluster)
        metadata_csv_path: Path to metadata CSV (columns: id, title, abstract)
        output_dir: Where to save model
        model_name: HuggingFace model name
        device: 'cuda' or 'cpu'
        batch_size: Training batch size
        epochs: Number of epochs
        lr: Learning rate
        margin: Triplet loss margin
        max_length: Max token length
        samples_per_cluster: Triplets per cluster
        pooling: 'cls' or 'mean'
        train_split: Train/val split ratio (within training clusters)
        test_ratio: Ratio of clusters to hold out for testing
        seed: Random seed

    Returns:
        (model, tokenizer, history, train_cluster_ids, test_cluster_ids, train_node_ids, test_node_ids)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load cluster and metadata
    print(f"\nLoading cluster data from {cluster_csv_path}...")
    cluster_df = pd.read_csv(cluster_csv_path)
    print(f"  Nodes: {len(cluster_df)}")
    print(f"  Clusters: {cluster_df['cluster'].nunique()}")

    print(f"\nLoading metadata from {metadata_csv_path}...")
    metadata_df = pd.read_csv(metadata_csv_path)
    print(f"  Entries: {len(metadata_df)}")

    # Create cluster-based train/test split (CRITICAL: no data leakage!)
    print(f"\n{'='*60}")
    print("CREATING TRAIN/TEST SPLIT")
    print('='*60)
    print(f"Test ratio: {test_ratio} (cluster-level split)")
    
    train_cluster_ids, test_cluster_ids, train_node_ids, test_node_ids = \
        create_cluster_based_split(
            cluster_df,
            test_ratio=test_ratio,
            seed=seed
        )
    
    print_split_info(train_cluster_ids, test_cluster_ids, cluster_df)

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TripletEmbeddingModel(model_name, pooling=pooling).to(device)

    # Create dataset with ONLY train clusters
    print(f"\n{'='*60}")
    print("CREATING TRAINING DATASET")
    print('='*60)
    print(f"Using ONLY {len(train_cluster_ids)} training clusters")
    print(f"Samples per cluster: {samples_per_cluster}")
    
    dataset = DisjointClusterTripletDataset(
        cluster_df=cluster_df,
        metadata_df=metadata_df,
        tokenizer=tokenizer,
        max_length=max_length,
        samples_per_cluster=samples_per_cluster,
        train_clusters=train_cluster_ids,  # CRITICAL: Only train clusters!
        seed=seed
    )
    print(f"Generated {len(dataset)} triplets")

    # Split train/val (within training data)
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    print(f"Train: {len(train_dataset)} triplets | Val: {len(val_dataset)} triplets")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # 0 for notebooks to avoid multiprocessing issues
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Loss function and optimizer
    loss_fn = TripletMarginLoss(margin=margin)
    print(f"\nLoss: TripletMarginLoss")
    print(f"  Margin: {margin}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print(f"\n{'='*60}")
    print(f"STARTING TRAINING FOR {epochs} EPOCHS")
    print('='*60)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print('-' * 60)

        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"  Train loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"  Val loss: {val_loss:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best! Saving to {output_dir}...")
            model.encoder.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir}")
    print('='*60)

    return model, tokenizer, history, train_cluster_ids, test_cluster_ids, train_node_ids, test_node_ids


def compute_all_embeddings(
    model,
    tokenizer,
    metadata_df: pd.DataFrame,
    device,
    batch_size: int = 32,
    max_length: int = 512
):
    """
    Compute embeddings for all nodes in metadata.

    Args:
        model: Trained TripletEmbeddingModel
        tokenizer: Tokenizer
        metadata_df: DataFrame with columns [id, title, abstract]
        device: torch device
        batch_size: Batch size for encoding
        max_length: Max token length

    Returns:
        Dict mapping node_id (str) -> embedding (numpy array)
    """
    from tqdm.auto import tqdm
    
    print(f"\n{'='*60}")
    print("COMPUTING EMBEDDINGS FOR ALL NODES")
    print('='*60)
    
    embeddings_dict = {}
    
    # Get all node IDs from metadata
    node_ids = metadata_df['id'].astype(str).values
    texts = []
    valid_ids = []
    
    print(f"Preparing texts for {len(node_ids)} nodes...")
    for node_id in tqdm(node_ids, desc="Preparing"):
        row = metadata_df[metadata_df['id'] == int(node_id)].iloc[0]
        title = str(row['title']) if pd.notna(row['title']) else ""
        abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
        text = f"{title} {abstract}".strip()
        
        if text:  # Only add if we have text
            texts.append(text)
            valid_ids.append(node_id)
    
    print(f"Computing embeddings for {len(texts)} nodes...")
    
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
            
            # Use model's forward method which handles pooling
            embeddings = model(inputs['input_ids'], inputs['attention_mask'])
            embeddings = embeddings.cpu().numpy()
            
            for node_id, emb in zip(batch_ids, embeddings):
                embeddings_dict[node_id] = emb
    
    print(f"✓ Computed embeddings for {len(embeddings_dict)} nodes")
    
    return embeddings_dict
