"""
Utility functions for notebook usage
Clean, reusable code to avoid repetition in cells
"""

import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

from tree_utils import HierarchicalTree
from dataset import HierarchicalTripletDataset
from model import TripletEmbeddingModel
from losses import AdaptiveMarginTripletLoss
from trainer import train_epoch, evaluate


def load_tree_and_metadata(tree_path, metadata_path):
    """Load tree and metadata"""
    print(f"Loading tree from {tree_path}...")
    tree = HierarchicalTree(tree_path)
    print(f"  ✓ Loaded: {len(tree.leaves)} leaves, max depth {tree.max_depth}")

    print(f"\nLoading metadata from {metadata_path}...")
    metadata_df = pd.read_csv(metadata_path)
    print(f"  ✓ Loaded: {len(metadata_df)} entries")

    return tree, metadata_df


def compute_embeddings(model, tokenizer, metadata_df, device, batch_size=32):
    """
    Compute embeddings for all nodes in metadata.
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
                max_length=512,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)

            # Use CLS token
            embeddings = outputs.last_hidden_state[:, 0, :]
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.cpu().numpy()

            for node_id, emb in zip(batch_ids, embeddings):
                embeddings_dict[node_id] = emb

    print(f"✓ Computed embeddings for {len(embeddings_dict)} nodes")
    return embeddings_dict


def train_hierarchical_model(
    tree,
    metadata_df,
    model_name='allenai/scibert_scivocab_uncased',
    device='cuda',
    batch_size=16,
    epochs=3,
    lr=2e-5,
    base_margin=0.5,
    distance_scale=0.1,
    samples_per_leaf=3,
    pooling='cls',
    sampling_strategy='hierarchical'
):
    """
    Train hierarchical triplet loss model.
    Returns: (trained_model, tokenizer, history)
    """
    print("\n" + "="*60)
    print("HIERARCHICAL TRIPLET LOSS TRAINING")
    print("="*60)

    # Load model
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TripletEmbeddingModel(model_name, pooling=pooling).to(device)

    # Create dataset
    print("\nCreating dataset...")
    dataset = HierarchicalTripletDataset(
        tree=tree,
        metadata_df=metadata_df,
        tokenizer=tokenizer,
        max_length=512,
        samples_per_leaf=samples_per_leaf,
        sampling_strategy=sampling_strategy
    )
    print(f"  Generated {len(dataset)} triplets")

    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=0)

    # Loss and optimizer
    loss_fn = AdaptiveMarginTripletLoss(
        base_margin=base_margin,
        distance_scale=distance_scale
    )
    print(f"\nAdaptive Margin Loss:")
    print(f"  Base margin: {base_margin}")
    print(f"  Distance scale: {distance_scale}")

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


def create_test_split(all_node_ids, test_ratio=0.1, seed=42):
    """Create reproducible test split"""
    np.random.seed(seed)
    n_test = int(test_ratio * len(all_node_ids))
    test_indices = np.random.choice(len(all_node_ids), n_test, replace=False)
    test_nodes = [all_node_ids[i] for i in test_indices]
    return test_nodes


def plot_training_history(history):
    """Plot training curves"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], marker='o', label='Train Loss')
    plt.plot(history['val_loss'], marker='s', label='Val Loss')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Hierarchical Triplet Loss Training', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.show()