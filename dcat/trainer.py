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
    """Train for one epoch with triplet loss"""
    model.train()
    total_loss = 0
    total_d_pos = 0
    total_d_neg = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move to device
        anchor_ids = batch['anchor_input_ids'].to(device)
        anchor_mask = batch['anchor_attention_mask'].to(device)
        pos_ids = batch['positive_input_ids'].to(device)
        pos_mask = batch['positive_attention_mask'].to(device)
        neg_ids = batch['negative_input_ids'].to(device)
        neg_mask = batch['negative_attention_mask'].to(device)
        cluster_dist_pos = batch['cluster_dist_pos'].to(device)
        cluster_dist_neg = batch['cluster_dist_neg'].to(device)

        # Forward pass
        anchor_emb = model(anchor_ids, anchor_mask)
        pos_emb = model(pos_ids, pos_mask)
        neg_emb = model(neg_ids, neg_mask)

        # Compute distances for monitoring
        d_pos = torch.norm(anchor_emb - pos_emb, dim=1).mean()
        d_neg = torch.norm(anchor_emb - neg_emb, dim=1).mean()
        total_d_pos += d_pos.item()
        total_d_neg += d_neg.item()

        # Compute loss with cluster distances
        loss = loss_fn(anchor_emb, pos_emb, neg_emb, cluster_dist_pos, cluster_dist_neg)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'd_pos': f'{d_pos.item():.3f}',
            'd_neg': f'{d_neg.item():.3f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_d_pos = total_d_pos / len(dataloader)
    avg_d_neg = total_d_neg / len(dataloader)

    print(f"  Avg d_pos: {avg_d_pos:.4f}, Avg d_neg: {avg_d_neg:.4f}, Margin: {avg_d_neg - avg_d_pos:.4f}")

    return avg_loss


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
            cluster_dist_pos = batch['cluster_dist_pos'].to(device)
            cluster_dist_neg = batch['cluster_dist_neg'].to(device)

            anchor_emb = model(anchor_ids, anchor_mask)
            pos_emb = model(pos_ids, pos_mask)
            neg_emb = model(neg_ids, neg_mask)

            loss = loss_fn(anchor_emb, pos_emb, neg_emb, cluster_dist_pos, cluster_dist_neg)
            total_loss += loss.item()

    return total_loss / len(dataloader)
