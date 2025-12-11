"""
Training and evaluation utilities for GNN models
"""

import torch
from tqdm.auto import tqdm


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: torch device

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move batch to device
        anchor = batch['anchor'].to(device)
        positive = batch['positive'].to(device)
        negative = batch['negative'].to(device)
        edge_index = batch['edge_index'].to(device)

        # Forward pass
        optimizer.zero_grad()

        # Get embeddings through GNN
        anchor_emb = model(anchor, edge_index)
        positive_emb = model(positive, edge_index)
        negative_emb = model(negative, edge_index)

        # Compute loss
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def evaluate(model, val_loader, loss_fn, device):
    """
    Evaluate model on validation set.

    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        device: torch device

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)
            negative = batch['negative'].to(device)
            edge_index = batch['edge_index'].to(device)

            # Forward pass
            anchor_emb = model(anchor, edge_index)
            positive_emb = model(positive, edge_index)
            negative_emb = model(negative, edge_index)

            # Compute loss
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_epoch_transformer_only(model, train_loader, loss_fn, optimizer, device):
    """
    Train transformer-only model for one epoch (no GNN).

    Used for baseline comparison.

    Args:
        model: TripletEmbeddingModel (transformer only)
        train_loader: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer
        device: torch device

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move batch to device
        anchor_ids = batch['anchor_input_ids'].to(device)
        anchor_mask = batch['anchor_attention_mask'].to(device)
        positive_ids = batch['positive_input_ids'].to(device)
        positive_mask = batch['positive_attention_mask'].to(device)
        negative_ids = batch['negative_input_ids'].to(device)
        negative_mask = batch['negative_attention_mask'].to(device)

        # Forward pass
        optimizer.zero_grad()

        anchor_emb = model(anchor_ids, anchor_mask)
        positive_emb = model(positive_ids, positive_mask)
        negative_emb = model(negative_ids, negative_mask)

        # Compute loss
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def evaluate_transformer_only(model, val_loader, loss_fn, device):
    """
    Evaluate transformer-only model on validation set.

    Args:
        model: TripletEmbeddingModel (transformer only)
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        device: torch device

    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move batch to device
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            positive_ids = batch['positive_input_ids'].to(device)
            positive_mask = batch['positive_attention_mask'].to(device)
            negative_ids = batch['negative_input_ids'].to(device)
            negative_mask = batch['negative_attention_mask'].to(device)

            # Forward pass
            anchor_emb = model(anchor_ids, anchor_mask)
            positive_emb = model(positive_ids, positive_mask)
            negative_emb = model(negative_ids, negative_mask)

            # Compute loss
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches
