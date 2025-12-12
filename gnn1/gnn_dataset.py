"""
Dataset for GNN triplet training using graph structure
"""

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class GraphTripletDataset(Dataset):
    """
    Dataset for training GNN with triplet loss using graph structure.

    Triplets are generated based on graph connectivity:
    - Anchor: random node
    - Positive: random neighbor of anchor
    - Negative: random non-neighbor node
    """

    def __init__(
        self,
        edge_index: torch.Tensor,
        node_ids: List[str],
        metadata_df: pd.DataFrame,
        tokenizer,
        node_to_idx: Dict[str, int],
        num_samples: int = 10000,
        max_length: int = 512,
        seed: int = 42
    ):
        """
        Args:
            edge_index: Graph edges [2, num_edges]
            node_ids: List of node IDs to sample from
            metadata_df: Metadata with columns [id, title, abstract]
            tokenizer: HuggingFace tokenizer
            node_to_idx: Mapping from node_id (str) -> index (int)
            num_samples: Number of triplets to generate
            max_length: Max sequence length
            seed: Random seed
        """
        self.edge_index = edge_index
        self.node_ids = node_ids
        self.metadata_df = metadata_df.set_index('id')
        self.tokenizer = tokenizer
        self.node_to_idx = node_to_idx
        self.max_length = max_length

        np.random.seed(seed)
        torch.manual_seed(seed)

        # Build adjacency list for fast neighbor lookup
        self.neighbors = self._build_adjacency_list()

        # Filter to nodes that have at least one neighbor
        self.valid_anchors = [
            nid for nid in node_ids
            if len(self.neighbors.get(self.node_to_idx[nid], [])) > 0
        ]

        if len(self.valid_anchors) == 0:
            raise ValueError("No nodes with neighbors found!")

        print(f"  Valid anchor nodes: {len(self.valid_anchors)} / {len(node_ids)}")

        # Generate triplets
        self.triplets = self._generate_triplets(num_samples)

    def _build_adjacency_list(self) -> Dict[int, List[int]]:
        """Build adjacency list from edge_index"""
        adj_list = {}

        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i].item()
            tgt = self.edge_index[1, i].item()

            if src not in adj_list:
                adj_list[src] = []
            adj_list[src].append(tgt)

        return adj_list

    def _generate_triplets(self, num_samples: int) -> List[Tuple[str, str, str]]:
        """Generate triplets: (anchor, positive_neighbor, negative_non_neighbor)"""
        triplets = []

        all_node_indices = set(self.node_to_idx[nid] for nid in self.node_ids)

        for _ in range(num_samples):
            # Sample anchor node (must have neighbors)
            anchor_id = np.random.choice(self.valid_anchors)
            anchor_idx = self.node_to_idx[anchor_id]

            # Get neighbors
            neighbor_indices = self.neighbors[anchor_idx]

            # Sample positive (random neighbor)
            pos_idx = np.random.choice(neighbor_indices)

            # Sample negative (non-neighbor)
            non_neighbors = list(all_node_indices - set(neighbor_indices) - {anchor_idx})
            if len(non_neighbors) == 0:
                # Fallback: use any node except anchor
                non_neighbors = list(all_node_indices - {anchor_idx})

            neg_idx = np.random.choice(non_neighbors)

            # Convert indices back to node IDs
            idx_to_node = {idx: nid for nid, idx in self.node_to_idx.items()}
            pos_id = idx_to_node[pos_idx]
            neg_id = idx_to_node[neg_idx]

            triplets.append((anchor_id, pos_id, neg_id))

        return triplets

    def _get_text(self, node_id: str) -> str:
        """Get combined title + abstract for a node"""
        try:
            row = self.metadata_df.loc[int(node_id)]
            title = str(row['title']) if pd.notna(row['title']) else ""
            abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
            return f"{title} {abstract}".strip()
        except:
            return ""

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Returns dict with:
        - anchor_input_ids, anchor_attention_mask
        - positive_input_ids, positive_attention_mask
        - negative_input_ids, negative_attention_mask
        - edge_index (full graph)
        """
        anchor_id, pos_id, neg_id = self.triplets[idx]

        # Get texts
        anchor_text = self._get_text(anchor_id)
        pos_text = self._get_text(pos_id)
        neg_text = self._get_text(neg_id)

        # Tokenize
        anchor_enc = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        pos_enc = self.tokenizer(
            pos_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        neg_enc = self.tokenizer(
            neg_text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'anchor_input_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_enc['attention_mask'].squeeze(0),
            'positive_input_ids': pos_enc['input_ids'].squeeze(0),
            'positive_attention_mask': pos_enc['attention_mask'].squeeze(0),
            'negative_input_ids': neg_enc['input_ids'].squeeze(0),
            'negative_attention_mask': neg_enc['attention_mask'].squeeze(0),
            'edge_index': self.edge_index
        }
