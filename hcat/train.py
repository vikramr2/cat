"""
Core training functions for use in notebooks or scripts

This module provides the main training logic without CLI argument parsing,
making it easy to use from Jupyter notebooks.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tree_utils import HierarchicalTree
from dataset import HierarchicalTripletDataset
from model import TripletEmbeddingModel
from losses import (
    AdaptiveMarginTripletLoss,
    SoftAdaptiveTripletLoss,
    HierarchicalContrastiveLoss,
    HybridHierarchicalLoss
)
from trainer import train_epoch, evaluate


def train_model(
    tree_json_path,
    metadata_csv_path,
    output_dir='./finetuned_model',
    model_name='allenai/scibert_scivocab_uncased',
    device='cuda',
    batch_size=16,
    epochs=3,
    lr=2e-5,
    base_margin=0.5,
    distance_scale=0.1,
    max_length=512,
    samples_per_leaf=3,
    sampling_strategy='hierarchical',
    pooling='cls',
    train_split=0.9,
    loss_type='adaptive_triplet',
    temperature=0.07,
    triplet_weight=0.7,
    regression_weight=0.3,
    max_tree_distance=None,
    val_nodes=None
):
    """
    Train hierarchical triplet loss model

    Args:
        tree_json_path: Path to tree JSON file
        metadata_csv_path: Path to metadata CSV
        output_dir: Where to save model
        model_name: HuggingFace model name
        device: 'cuda' or 'cpu'
        batch_size: Training batch size
        epochs: Number of epochs
        lr: Learning rate
        base_margin: Base triplet margin
        distance_scale: Adaptive margin scaling factor
        max_length: Max token length
        samples_per_leaf: Triplets per leaf
        sampling_strategy: 'hierarchical' or 'sibling'
        pooling: 'cls' or 'mean'
        train_split: Train/val split ratio (ignored if val_nodes provided)
        loss_type: Loss function to use. Options:
                   - 'adaptive_triplet': AdaptiveMarginTripletLoss (default, hard margin)
                   - 'soft_adaptive': SoftAdaptiveTripletLoss (smooth margin, helps with overfitting)
                   - 'contrastive': HierarchicalContrastiveLoss (cross-entropy based)
                   - 'hybrid': HybridHierarchicalLoss (combines triplet + regression)
        temperature: Temperature parameter for soft_adaptive and contrastive losses
        triplet_weight: Weight for triplet component in hybrid loss (default: 0.7)
        regression_weight: Weight for regression component in hybrid loss (default: 0.3)
        max_tree_distance: Maximum tree distance for hybrid loss normalization.
                          If None, computed from tree max_depth
        val_nodes: Optional list of node IDs to use for validation.
                   If provided, ensures validation uses these specific nodes.

    Returns:
        (model, tokenizer, history)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tree and metadata
    print(f"\nLoading tree from {tree_json_path}...")
    tree = HierarchicalTree(tree_json_path)
    print(f"  Leaves: {len(tree.leaves)}")
    print(f"  Total nodes: {len(tree.all_nodes)}")
    print(f"  Max depth: {tree.max_depth}")

    print(f"\nLoading metadata from {metadata_csv_path}...")
    metadata_df = pd.read_csv(metadata_csv_path)
    print(f"  Entries: {len(metadata_df)}")

    # Load tokenizer and model
    print(f"\nLoading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TripletEmbeddingModel(model_name, pooling=pooling).to(device)

    # Create dataset
    print("\nCreating dataset...")
    dataset = HierarchicalTripletDataset(
        tree=tree,
        metadata_df=metadata_df,
        tokenizer=tokenizer,
        max_length=max_length,
        samples_per_leaf=samples_per_leaf,
        sampling_strategy=sampling_strategy
    )
    print(f"  Generated {len(dataset)} triplets")

    # Split train/val
    if val_nodes is not None:
        # Node-level split: use provided validation nodes
        print(f"  Using {len(val_nodes)} nodes for validation (node-level split)")

        # Convert val_nodes to set for faster lookup
        val_nodes_set = set(str(n) for n in val_nodes)

        # Split triplets based on anchor node
        train_indices = []
        val_indices = []

        for idx, triplet in enumerate(dataset.triplets):
            # triplet is (anchor_text, pos_text, neg_text, dist_pos, dist_neg, anchor_id)
            anchor_id = triplet[5]  # Last element is anchor node ID

            if anchor_id in val_nodes_set:
                val_indices.append(idx)
            else:
                train_indices.append(idx)

        # Create subset datasets
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)

        print(f"  Train: {len(train_dataset)} triplets | Val: {len(val_dataset)} triplets")
        print(f"  Train nodes: {len(set(str(n) for n in val_nodes)) - len(val_nodes_set)} (approx)")
        print(f"  Val nodes: {len(val_nodes_set)}")
    else:
        # Random triplet-level split (original behavior)
        print("  Using random triplet split for train/val")
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

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
    print(f"\nConfiguring loss function...")

    if loss_type == 'soft_adaptive':
        loss_fn = SoftAdaptiveTripletLoss(
            base_margin=base_margin,
            distance_scale=distance_scale,
            temperature=temperature
        )
        print(f"Loss: SoftAdaptiveTripletLoss")
        print(f"  Base margin: {base_margin}")
        print(f"  Distance scale: {distance_scale}")
        print(f"  Temperature: {temperature}")

    elif loss_type == 'contrastive':
        loss_fn = HierarchicalContrastiveLoss(
            temperature=temperature,
            distance_scale=distance_scale
        )
        print(f"Loss: HierarchicalContrastiveLoss")
        print(f"  Temperature: {temperature}")
        print(f"  Distance scale: {distance_scale}")

    elif loss_type == 'hybrid':
        # Compute max_tree_distance if not provided
        if max_tree_distance is None:
            max_tree_distance = float(tree.max_depth)
            print(f"  Auto-detected max_tree_distance: {max_tree_distance}")

        loss_fn = HybridHierarchicalLoss(
            base_margin=base_margin,
            distance_scale=distance_scale,
            max_tree_distance=max_tree_distance,
            triplet_weight=triplet_weight,
            regression_weight=regression_weight
        )
        print(f"Loss: HybridHierarchicalLoss")
        print(f"  Base margin: {base_margin}")
        print(f"  Distance scale: {distance_scale}")
        print(f"  Max tree distance: {max_tree_distance}")
        print(f"  Triplet weight: {triplet_weight}")
        print(f"  Regression weight: {regression_weight}")

    else:  # adaptive_triplet (default)
        loss_fn = AdaptiveMarginTripletLoss(
            base_margin=base_margin,
            distance_scale=distance_scale
        )
        print(f"Loss: AdaptiveMarginTripletLoss")
        print(f"  Base margin: {base_margin}")
        print(f"  Distance scale: {distance_scale}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
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
    print(f"Training complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {output_dir}")
    print('='*60)

    return model, tokenizer, history