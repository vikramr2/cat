"""
Training and evaluation utilities
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model,
    dataloader: DataLoader,
    loss_fn,
    optimizer: torch.optim.Optimizer,
    device: torch.device
):
    """Train for one epoch with adaptive margin loss"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move to device
        anchor_ids = batch['anchor_input_ids'].to(device)
        anchor_mask = batch['anchor_attention_mask'].to(device)
        pos_ids = batch['positive_input_ids'].to(device)
        pos_mask = batch['positive_attention_mask'].to(device)
        neg_ids = batch['negative_input_ids'].to(device)
        neg_mask = batch['negative_attention_mask'].to(device)
        tree_dist_pos = batch['tree_dist_pos'].to(device)
        tree_dist_neg = batch['tree_dist_neg'].to(device)

        # Forward pass
        anchor_emb = model(anchor_ids, anchor_mask)
        pos_emb = model(pos_ids, pos_mask)
        neg_emb = model(neg_ids, neg_mask)

        # Compute loss with tree distances
        loss = loss_fn(anchor_emb, pos_emb, neg_emb, tree_dist_pos, tree_dist_neg)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate(
    model,
    dataloader: DataLoader,
    loss_fn,
    device: torch.device
):
    """Evaluate model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            pos_ids = batch['positive_input_ids'].to(device)
            pos_mask = batch['positive_attention_mask'].to(device)
            neg_ids = batch['negative_input_ids'].to(device)
            neg_mask = batch['negative_attention_mask'].to(device)
            tree_dist_pos = batch['tree_dist_pos'].to(device)
            tree_dist_neg = batch['tree_dist_neg'].to(device)

            anchor_emb = model(anchor_ids, anchor_mask)
            pos_emb = model(pos_ids, pos_mask)
            neg_emb = model(neg_ids, neg_mask)

            loss = loss_fn(anchor_emb, pos_emb, neg_emb, tree_dist_pos, tree_dist_neg)
            total_loss += loss.item()

    return total_loss / len(dataloader)