"""
DCAT: Disjoint Clustering Augmented Transformers

A module for training transformer models with triplet loss on disjoint clusterings.
Unlike HCAT which uses hierarchical tree structures with adaptive margin losses,
DCAT uses binary cluster membership (same/different cluster) with standard triplet loss.
"""

from .cluster_utils import DisjointClustering
from .dataset import DisjointTripletDataset
from .model import TripletEmbeddingModel
from .losses import TripletLoss, SoftTripletLoss, ContrastiveLoss, SupConLoss
from .trainer import train_epoch, evaluate
from .train import train_model
from .notebook_utils import (
    load_clustering_and_metadata,
    compute_embeddings,
    train_disjoint_model,
    create_test_split,
    plot_training_history
)

__all__ = [
    'DisjointClustering',
    'DisjointTripletDataset',
    'TripletEmbeddingModel',
    'TripletLoss',
    'SoftTripletLoss',
    'ContrastiveLoss',
    'SupConLoss',
    'train_epoch',
    'evaluate',
    'train_model',
    'load_clustering_and_metadata',
    'compute_embeddings',
    'train_disjoint_model',
    'create_test_split',
    'plot_training_history'
]
