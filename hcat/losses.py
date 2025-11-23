"""
Hierarchical triplet loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveMarginTripletLoss(nn.Module):
    """
    Triplet loss with margin adaptive to tree distance.

    The margin scales with the difference in tree distances:
    margin = base_margin + distance_scale * (dist_neg - dist_pos)

    This enforces that:
    - Pairs farther apart in tree should have larger embedding separation
    - The required separation adapts to hierarchical structure
    """

    def __init__(self, base_margin: float = 0.5, distance_scale: float = 0.1):
        """
        Args:
            base_margin: Minimum margin for triplet loss
            distance_scale: How much to scale margin by tree distance difference
        """
        super().__init__()
        self.base_margin = base_margin
        self.distance_scale = distance_scale

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        tree_dist_pos: torch.Tensor,
        tree_dist_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive margin triplet loss.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            positive: Positive embeddings [batch_size, embed_dim]
            negative: Negative embeddings [batch_size, embed_dim]
            tree_dist_pos: Tree distance to positive [batch_size]
            tree_dist_neg: Tree distance to negative [batch_size]

        Returns:
            Scalar loss
        """
        # Compute embedding distances (L2 norm)
        d_pos = torch.norm(anchor - positive, dim=1)
        d_neg = torch.norm(anchor - negative, dim=1)

        # Adaptive margin based on tree distance gap
        distance_gap = tree_dist_neg - tree_dist_pos
        adaptive_margin = self.base_margin + self.distance_scale * distance_gap

        # Triplet loss with adaptive margin
        loss = F.relu(d_pos - d_neg + adaptive_margin)

        return loss.mean()


class SoftAdaptiveTripletLoss(nn.Module):
    """
    Soft version of adaptive triplet loss using log-exp formulation.

    Provides smoother gradients compared to hard margin.
    """

    def __init__(self, base_margin: float = 0.5, distance_scale: float = 0.1,
                 temperature: float = 1.0):
        """
        Args:
            base_margin: Base separation to enforce
            distance_scale: Scaling factor for tree distance
            temperature: Temperature for softmax-like behavior (lower = harder)
        """
        super().__init__()
        self.base_margin = base_margin
        self.distance_scale = distance_scale
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        tree_dist_pos: torch.Tensor,
        tree_dist_neg: torch.Tensor
    ) -> torch.Tensor:
        """Compute soft adaptive triplet loss"""
        # Embedding distances
        d_pos = torch.norm(anchor - positive, dim=1)
        d_neg = torch.norm(anchor - negative, dim=1)

        # Target margin based on tree distances
        distance_gap = tree_dist_neg - tree_dist_pos
        target_margin = self.base_margin + self.distance_scale * distance_gap

        # Soft margin using log-sum-exp
        loss = torch.log(1 + torch.exp(
            (d_pos - d_neg + target_margin) / self.temperature
        ))

        return loss.mean()


class DistanceRegressionLoss(nn.Module):
    """
    Directly regress embedding distance to tree distance.

    Alternative approach: make embedding distances proportional to tree distances.
    """

    def __init__(self, max_tree_distance: float):
        """
        Args:
            max_tree_distance: Maximum tree distance for normalization
        """
        super().__init__()
        self.max_tree_distance = max_tree_distance

    def forward(
        self,
        anchor: torch.Tensor,
        other: torch.Tensor,
        tree_distance: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute MSE between normalized embedding and tree distances.

        Args:
            anchor: Anchor embeddings [batch_size, embed_dim]
            other: Other embeddings [batch_size, embed_dim]
            tree_distance: Tree distances [batch_size]

        Returns:
            MSE loss
        """
        # Normalize tree distance to [0, 1]
        normalized_tree_dist = tree_distance / self.max_tree_distance

        # Compute embedding distance (for normalized embeddings, in [0, 2])
        emb_dist = torch.norm(anchor - other, dim=1)
        normalized_emb_dist = emb_dist / 2.0

        # MSE loss
        loss = F.mse_loss(normalized_emb_dist, normalized_tree_dist)

        return loss


class HybridHierarchicalLoss(nn.Module):
    """
    Combines adaptive triplet loss with distance regression.

    Uses both ranking (triplet) and regression objectives.
    """

    def __init__(
        self,
        base_margin: float = 0.5,
        distance_scale: float = 0.1,
        max_tree_distance: float = 10.0,
        triplet_weight: float = 0.7,
        regression_weight: float = 0.3
    ):
        """
        Args:
            base_margin: Base margin for triplet loss
            distance_scale: Distance scaling for adaptive margin
            max_tree_distance: Max tree distance for normalization
            triplet_weight: Weight for triplet loss component
            regression_weight: Weight for regression loss component
        """
        super().__init__()
        self.triplet_loss = AdaptiveMarginTripletLoss(base_margin, distance_scale)
        self.regression_loss = DistanceRegressionLoss(max_tree_distance)
        self.triplet_weight = triplet_weight
        self.regression_weight = regression_weight

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        tree_dist_pos: torch.Tensor,
        tree_dist_neg: torch.Tensor
    ) -> torch.Tensor:
        """Compute combined loss"""
        # Triplet loss
        triplet = self.triplet_loss(anchor, positive, negative,
                                    tree_dist_pos, tree_dist_neg)

        # Regression losses (for both positive and negative)
        reg_pos = self.regression_loss(anchor, positive, tree_dist_pos)
        reg_neg = self.regression_loss(anchor, negative, tree_dist_neg)
        regression = (reg_pos + reg_neg) / 2

        # Combined loss
        total_loss = (self.triplet_weight * triplet +
                     self.regression_weight * regression)

        return total_loss