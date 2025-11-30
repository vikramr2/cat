"""
Triplet loss functions for disjoint clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletMarginLoss(nn.Module):
    """
    Standard triplet margin loss for disjoint clusters.
    
    Enforces that d(anchor, positive) + margin < d(anchor, negative)
    """

    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet margin loss.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negative: Negative embeddings [batch_size, embed_dim]

        Returns:
            Scalar loss
        """
        # Compute distances
        d_pos = torch.norm(anchor - positive, dim=1)
        d_neg = torch.norm(anchor - negative, dim=1)

        # Triplet loss
        loss = F.relu(d_pos - d_neg + self.margin)

        return loss.mean()


class SoftTripletLoss(nn.Module):
    """
    Soft triplet loss using log-exp formulation.
    
    Provides smoother gradients compared to hard margin.
    """

    def __init__(self, margin: float = 1.0, temperature: float = 1.0):
        """
        Args:
            margin: Target margin
            temperature: Temperature for softmax-like behavior (lower = harder)
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft triplet loss"""
        # Compute distances
        d_pos = torch.norm(anchor - positive, dim=1)
        d_neg = torch.norm(anchor - negative, dim=1)

        # Soft margin using log-sum-exp
        loss = torch.log(1 + torch.exp(
            (d_pos - d_neg + self.margin) / self.temperature
        ))

        return loss.mean()


class CosineTripletLoss(nn.Module):
    """
    Triplet loss using cosine similarity instead of L2 distance.
    
    Good for normalized embeddings.
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss with cosine similarity.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negative: Negative embeddings [batch_size, embed_dim]

        Returns:
            Scalar loss
        """
        # Cosine similarities (higher = more similar)
        sim_pos = F.cosine_similarity(anchor, positive, dim=1)
        sim_neg = F.cosine_similarity(anchor, negative, dim=1)

        # Loss: we want sim_pos > sim_neg + margin
        # Equivalent to: (1 - sim_pos) < (1 - sim_neg) - margin
        # Or: sim_neg - sim_pos + margin > 0
        loss = F.relu(sim_neg - sim_pos + self.margin)

        return loss.mean()
