"""
Triplet loss functions for disjoint clustering

Uses standard triplet loss (not adaptive margin) since cluster distances are binary:
- Same cluster: distance = 0.0
- Different cluster: distance = 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Standard triplet loss with fixed margin.

    For disjoint clustering, we use a simple fixed margin since:
    - Positives are always from the same cluster (distance = 0.0)
    - Negatives are always from different clusters (distance = 1.0)

    Loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
    """

    def __init__(self, margin: float = 0.5):
        """
        Args:
            margin: Fixed margin for triplet loss
        """
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        cluster_dist_pos: torch.Tensor = None,
        cluster_dist_neg: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negative: Negative embeddings [batch_size, embed_dim]
            cluster_dist_pos: Cluster distance to positive [batch_size] (not used, for compatibility)
            cluster_dist_neg: Cluster distance to negative [batch_size] (not used, for compatibility)

        Returns:
            Scalar loss
        """
        # Compute embedding distances (L2 norm)
        d_pos = torch.norm(anchor - positive, dim=1)
        d_neg = torch.norm(anchor - negative, dim=1)

        # Triplet loss with fixed margin
        loss = F.relu(d_pos - d_neg + self.margin)

        return loss.mean()


class SoftTripletLoss(nn.Module):
    """
    Soft version of triplet loss using log-exp formulation.

    Provides smoother gradients compared to hard margin.
    """

    def __init__(self, margin: float = 0.5, temperature: float = 1.0):
        """
        Args:
            margin: Target separation between positive and negative
            temperature: Temperature for softmax-like behavior (lower = harder)
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        cluster_dist_pos: torch.Tensor = None,
        cluster_dist_neg: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute soft triplet loss"""
        # Embedding distances
        d_pos = torch.norm(anchor - positive, dim=1)
        d_neg = torch.norm(anchor - negative, dim=1)

        # Soft margin using log-sum-exp
        loss = torch.log(1 + torch.exp(
            (d_pos - d_neg + self.margin) / self.temperature
        ))

        return loss.mean()


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for binary clustering.

    Uses InfoNCE-style loss where we classify positive vs negative.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for contrastive loss
                        Lower = harder (more peaked), Higher = softer
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        cluster_dist_pos: torch.Tensor = None,
        cluster_dist_neg: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negative: Negative embeddings [batch_size, embed_dim]
            cluster_dist_pos: Not used (for compatibility)
            cluster_dist_neg: Not used (for compatibility)

        Returns:
            Scalar loss
        """
        # Embeddings are already normalized in the model
        # Compute cosine similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        neg_sim = torch.sum(anchor * negative, dim=1) / self.temperature

        # Stack to create logits [batch_size, 2]
        # The goal is to classify positive as index 0
        logits = torch.stack([pos_sim, neg_sim], dim=1)

        # Labels: positive is always at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss for disjoint clustering.

    Uses all examples in the batch as potential positives/negatives.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature for scaling similarities
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        cluster_dist_pos: torch.Tensor = None,
        cluster_dist_neg: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss with in-batch examples.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negative: Negative embeddings [batch_size, embed_dim]
            cluster_dist_pos: Not used (for compatibility)
            cluster_dist_neg: Not used (for compatibility)

        Returns:
            Scalar loss
        """
        batch_size = anchor.shape[0]

        # Concatenate anchor with all positives and negatives in batch
        all_examples = torch.cat([positive, negative], dim=0)  # [2*batch_size, embed_dim]

        # Compute similarities between anchors and all examples
        similarity = torch.matmul(anchor, all_examples.t()) / self.temperature

        # Create mask for positives (first batch_size columns)
        pos_mask = torch.zeros(batch_size, 2 * batch_size).to(anchor.device)
        pos_mask[:, :batch_size] = 1.0

        # Compute log-sum-exp for contrastive loss
        exp_sim = torch.exp(similarity)

        # Denominator: sum over all examples
        log_denominator = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Numerator: sum over positives
        pos_sim = similarity * pos_mask
        log_numerator = torch.logsumexp(pos_sim, dim=1, keepdim=True)

        # Contrastive loss
        loss = -log_numerator + log_denominator
        loss = loss.mean()

        return loss
