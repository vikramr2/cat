#!/usr/bin/env python3
"""
Hierarchical Triplet Loss Fine-tuning for Embeddings

This script implements hierarchical triplet loss to fine-tune embeddings based on
a tree topology. Leaf nodes that are closer in the tree hierarchy should have
more similar embeddings.
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import random
from collections import defaultdict
from tqdm import tqdm
import argparse


@dataclass
class TreeNode:
    """Represents a node in the hierarchical tree"""
    id: int
    type: str  # 'leaf' or 'cluster'
    name: Optional[str] = None
    distance: Optional[float] = None
    count: int = 1
    children: List['TreeNode'] = None
    parent: Optional['TreeNode'] = None
    depth: int = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []


class HierarchicalTree:
    """Parse and navigate hierarchical tree structure"""

    def __init__(self, tree_json_path: str):
        """Load tree from JSON file"""
        with open(tree_json_path, 'r') as f:
            data = json.load(f)

        self.root = self._parse_tree(data['hierarchy'])
        self.leaves = {}  # Map leaf name to TreeNode
        self.all_nodes = {}  # Map node id to TreeNode
        self._build_indices(self.root)
        self._compute_depths(self.root, 0)

    def _parse_tree(self, node_dict: Dict, parent=None) -> TreeNode:
        """Recursively parse tree structure"""
        node = TreeNode(
            id=node_dict['id'],
            type=node_dict['type'],
            name=node_dict.get('name'),
            distance=node_dict.get('distance'),
            count=node_dict.get('count', 1),
            parent=parent
        )

        if 'children' in node_dict:
            node.children = [
                self._parse_tree(child, parent=node)
                for child in node_dict['children']
            ]

        return node

    def _build_indices(self, node: TreeNode):
        """Build lookup indices for leaves and all nodes"""
        self.all_nodes[node.id] = node

        if node.type == 'leaf':
            self.leaves[node.name] = node

        for child in node.children:
            self._build_indices(child)

    def _compute_depths(self, node: TreeNode, depth: int):
        """Compute depth for each node"""
        node.depth = depth
        for child in node.children:
            self._compute_depths(child, depth + 1)

    def find_lca(self, node1: TreeNode, node2: TreeNode) -> TreeNode:
        """Find lowest common ancestor of two nodes"""
        # Get ancestors of node1
        ancestors1 = set()
        current = node1
        while current is not None:
            ancestors1.add(current.id)
            current = current.parent

        # Find first common ancestor for node2
        current = node2
        while current is not None:
            if current.id in ancestors1:
                return current
            current = current.parent

        return self.root

    def tree_distance(self, node1: TreeNode, node2: TreeNode) -> float:
        """
        Compute hierarchical distance between two nodes.
        Returns the depth of their lowest common ancestor (LCA).
        Higher values = more distant in tree (lower common ancestor).
        """
        lca = self.find_lca(node1, node2)
        return lca.depth

    def get_leaves_in_subtree(self, node: TreeNode) -> List[TreeNode]:
        """Get all leaf nodes under a given node"""
        if node.type == 'leaf':
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self.get_leaves_in_subtree(child))
        return leaves

    def sample_positive_negative(
        self,
        anchor: TreeNode,
        strategy: str = 'hierarchical'
    ) -> Tuple[TreeNode, TreeNode]:
        """
        Sample positive and negative examples for an anchor.

        Args:
            anchor: Anchor leaf node
            strategy: 'hierarchical' (LCA-based) or 'sibling' (same parent cluster)

        Returns:
            (positive_node, negative_node)
        """
        if strategy == 'sibling':
            # Positive: from same parent cluster
            parent = anchor.parent
            if parent and len(parent.children) > 1:
                siblings = self.get_leaves_in_subtree(parent)
                siblings = [s for s in siblings if s.name != anchor.name]
                if siblings:
                    positive = random.choice(siblings)
                else:
                    # Fallback to any leaf
                    positive = random.choice([l for l in self.leaves.values()
                                            if l.name != anchor.name])
            else:
                positive = random.choice([l for l in self.leaves.values()
                                        if l.name != anchor.name])

            # Negative: from distant cluster
            # Move up the tree and sample from a different branch
            current = anchor.parent
            distant_depth = 0
            while current and current.parent and distant_depth < 3:
                current = current.parent
                distant_depth += 1

            # Get leaves not in anchor's subtree
            all_leaves = list(self.leaves.values())
            anchor_subtree = set(l.name for l in self.get_leaves_in_subtree(current))
            distant_leaves = [l for l in all_leaves if l.name not in anchor_subtree]

            if distant_leaves:
                negative = random.choice(distant_leaves)
            else:
                # Fallback
                negative = random.choice([l for l in all_leaves
                                        if l.name != anchor.name and l.name != positive.name])

        else:  # hierarchical strategy
            # Find all leaves and their distances to anchor
            all_leaves = [l for l in self.leaves.values() if l.name != anchor.name]
            distances = [(l, self.tree_distance(anchor, l)) for l in all_leaves]
            distances.sort(key=lambda x: x[1])

            # Positive: closest leaves (smallest LCA depth)
            close_threshold = distances[0][1] if distances else 0
            positive_candidates = [l for l, d in distances if d <= close_threshold + 1]
            positive = random.choice(positive_candidates) if positive_candidates else distances[0][0]

            # Negative: distant leaves (largest LCA depth)
            far_threshold = distances[-1][1] if distances else 0
            negative_candidates = [l for l, d in distances if d >= far_threshold - 1]
            negative = random.choice(negative_candidates) if negative_candidates else distances[-1][0]

        return positive, negative


class HierarchicalTripletDataset(Dataset):
    """Dataset for hierarchical triplet learning"""

    def __init__(
        self,
        tree: HierarchicalTree,
        metadata_df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        samples_per_leaf: int = 5,
        sampling_strategy: str = 'hierarchical'
    ):
        """
        Args:
            tree: HierarchicalTree instance
            metadata_df: DataFrame with columns [id, title, abstract]
            tokenizer: Hugging Face tokenizer
            max_length: Max token length
            samples_per_leaf: Number of triplets to generate per leaf
            sampling_strategy: 'hierarchical' or 'sibling'
        """
        self.tree = tree
        self.metadata_df = metadata_df.set_index('id')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sampling_strategy = sampling_strategy

        # Generate triplets
        self.triplets = self._generate_triplets(samples_per_leaf)

    def _get_text(self, leaf_name: str) -> str:
        """Get combined title + abstract for a leaf"""
        try:
            row = self.metadata_df.loc[int(leaf_name)]
            title = str(row['title']) if pd.notna(row['title']) else ""
            abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
            return f"{title} {abstract}".strip()
        except (KeyError, ValueError):
            return f"Document {leaf_name}"

    def _generate_triplets(self, samples_per_leaf: int) -> List[Tuple[str, str, str]]:
        """Generate (anchor, positive, negative) triplets"""
        triplets = []
        leaf_names = list(self.tree.leaves.keys())

        for leaf_name in tqdm(leaf_names, desc="Generating triplets"):
            anchor_node = self.tree.leaves[leaf_name]

            for _ in range(samples_per_leaf):
                pos_node, neg_node = self.tree.sample_positive_negative(
                    anchor_node,
                    strategy=self.sampling_strategy
                )

                anchor_text = self._get_text(leaf_name)
                pos_text = self._get_text(pos_node.name)
                neg_text = self._get_text(neg_node.name)

                triplets.append((anchor_text, pos_text, neg_text))

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative = self.triplets[idx]

        # Tokenize
        anchor_encoded = self.tokenizer(
            anchor,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        positive_encoded = self.tokenizer(
            positive,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        negative_encoded = self.tokenizer(
            negative,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'anchor_input_ids': anchor_encoded['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_encoded['attention_mask'].squeeze(0),
            'positive_input_ids': positive_encoded['input_ids'].squeeze(0),
            'positive_attention_mask': positive_encoded['attention_mask'].squeeze(0),
            'negative_input_ids': negative_encoded['input_ids'].squeeze(0),
            'negative_attention_mask': negative_encoded['attention_mask'].squeeze(0),
        }


class TripletLossModel(nn.Module):
    """Model wrapper for triplet loss training"""

    def __init__(self, model_name: str, pooling: str = 'mean'):
        """
        Args:
            model_name: Hugging Face model name
            pooling: 'mean' or 'cls' pooling strategy
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        """Encode text to embedding vector"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == 'mean':
            embeddings = self.mean_pooling(outputs, attention_mask)
        else:  # cls
            embeddings = outputs[0][:, 0, :]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def train_epoch(
    model: TripletLossModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    margin: float = 0.5
):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        # Move to device
        anchor_ids = batch['anchor_input_ids'].to(device)
        anchor_mask = batch['anchor_attention_mask'].to(device)
        pos_ids = batch['positive_input_ids'].to(device)
        pos_mask = batch['positive_attention_mask'].to(device)
        neg_ids = batch['negative_input_ids'].to(device)
        neg_mask = batch['negative_attention_mask'].to(device)

        # Forward pass
        anchor_emb = model(anchor_ids, anchor_mask)
        pos_emb = model(pos_ids, pos_mask)
        neg_emb = model(neg_ids, neg_mask)

        # Compute loss
        loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(dataloader)


def evaluate(
    model: TripletLossModel,
    dataloader: DataLoader,
    device: torch.device,
    margin: float = 0.5
):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            pos_ids = batch['positive_input_ids'].to(device)
            pos_mask = batch['positive_attention_mask'].to(device)
            neg_ids = batch['negative_input_ids'].to(device)
            neg_mask = batch['negative_attention_mask'].to(device)

            anchor_emb = model(anchor_ids, anchor_mask)
            pos_emb = model(pos_ids, pos_mask)
            neg_emb = model(neg_ids, neg_mask)

            loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Hierarchical triplet loss fine-tuning')
    parser.add_argument('--tree_json', type=str, required=True,
                        help='Path to hierarchical tree JSON file')
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to metadata CSV file')
    parser.add_argument('--model_name', type=str, default='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract',
                        help='Hugging Face model name')
    parser.add_argument('--output_dir', type=str, default='./finetuned_model',
                        help='Output directory for fine-tuned model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='Triplet loss margin')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Max token length')
    parser.add_argument('--samples_per_leaf', type=int, default=5,
                        help='Number of triplets per leaf')
    parser.add_argument('--sampling_strategy', type=str, default='hierarchical',
                        choices=['hierarchical', 'sibling'],
                        help='Triplet sampling strategy')
    parser.add_argument('--pooling', type=str, default='mean',
                        choices=['mean', 'cls'],
                        help='Pooling strategy')
    parser.add_argument('--train_split', type=float, default=0.9,
                        help='Train/validation split ratio')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tree and metadata
    print(f"Loading tree from {args.tree_json}...")
    tree = HierarchicalTree(args.tree_json)
    print(f"Tree loaded: {len(tree.leaves)} leaves, {len(tree.all_nodes)} total nodes")

    print(f"Loading metadata from {args.metadata_csv}...")
    metadata_df = pd.read_csv(args.metadata_csv)
    print(f"Metadata loaded: {len(metadata_df)} entries")

    # Load tokenizer and model
    print(f"Loading model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TripletLossModel(args.model_name, pooling=args.pooling).to(device)

    # Create dataset
    print("Creating dataset...")
    dataset = HierarchicalTripletDataset(
        tree=tree,
        metadata_df=metadata_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        samples_per_leaf=args.samples_per_leaf,
        sampling_strategy=args.sampling_strategy
    )
    print(f"Dataset created: {len(dataset)} triplets")

    # Split train/val
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, optimizer, device, margin=args.margin
        )
        print(f"Train loss: {train_loss:.4f}")

        val_loss = evaluate(model, val_loader, device, margin=args.margin)
        print(f"Val loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving model to {args.output_dir}...")
            model.encoder.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to {args.output_dir}")


if __name__ == '__main__':
    main()
