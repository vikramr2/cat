"""
Loss functions for GNN training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    """
    Triplet margin loss for metric learning.

    Loss = max(0, margin + d(anchor, positive) - d(anchor, negative))

    where d is the distance function (1 - cosine similarity for normalized embeddings)
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin value for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]

        Returns:
            Scalar loss value
        """
        # Cosine similarity (embeddings should be normalized)
        pos_sim = F.cosine_similarity(anchor, positive, dim=1)
        neg_sim = F.cosine_similarity(anchor, negative, dim=1)

        # Convert to distances (for normalized vectors: distance = 1 - similarity)
        pos_dist = 1 - pos_sim
        neg_dist = 1 - neg_sim

        # Triplet loss
        losses = F.relu(self.margin + pos_dist - neg_dist)

        return losses.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pairs of embeddings.

    For positive pairs: loss = distance^2
    For negative pairs: loss = max(0, margin - distance)^2
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for negative pairs
        """
        super().__init__()
        self.margin = margin

    def forward(self, embedding1, embedding2, labels):
        """
        Compute contrastive loss.

        Args:
            embedding1: First embeddings [batch_size, embedding_dim]
            embedding2: Second embeddings [batch_size, embedding_dim]
            labels: Binary labels (1 for positive pairs, 0 for negative) [batch_size]

        Returns:
            Scalar loss value
        """
        # Euclidean distance
        distances = F.pairwise_distance(embedding1, embedding2)

        # Positive pairs: minimize distance
        pos_loss = labels * (distances ** 2)

        # Negative pairs: maximize distance (up to margin)
        neg_loss = (1 - labels) * F.relu(self.margin - distances) ** 2

        # Average loss
        loss = (pos_loss + neg_loss).mean()

        return loss


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    Used in SimCLR and other contrastive learning methods.
    """

    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss.

        Args:
            z_i: Embeddings for first augmentation [batch_size, embedding_dim]
            z_j: Embeddings for second augmentation [batch_size, embedding_dim]

        Returns:
            Scalar loss value
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        # [2*batch_size, 2*batch_size]
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )

        # Create mask for positive pairs
        # Positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels])

        # Mask out diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss
