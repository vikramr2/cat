"""
DCAT: Disjoint Cluster Training

A package for fine-tuning language models on disjoint cluster structures
using triplet loss with proper train/test splitting.
"""

__version__ = "0.1.0"

from .dataset import DisjointClusterTripletDataset
from .model import TripletEmbeddingModel
from .losses import TripletMarginLoss, SoftTripletLoss, CosineTripletLoss
from .trainer import train_epoch, evaluate
from .split_utils import (
    create_cluster_based_split,
    create_node_based_split,
    get_cluster_statistics,
    print_split_info
)

__all__ = [
    'DisjointClusterTripletDataset',
    'TripletEmbeddingModel',
    'TripletMarginLoss',
    'SoftTripletLoss',
    'CosineTripletLoss',
    'train_epoch',
    'evaluate',
    'create_cluster_based_split',
    'create_node_based_split',
    'get_cluster_statistics',
    'print_split_info'
]
