"""
Dataset for disjoint clustering triplet learning
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from tqdm import tqdm

from cluster_utils import DisjointClustering


class DisjointTripletDataset(Dataset):
    """Dataset for disjoint clustering triplet learning"""

    def __init__(
        self,
        clustering: DisjointClustering,
        metadata_df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        samples_per_node: int = 5,
    ):
        """
        Args:
            clustering: DisjointClustering instance
            metadata_df: DataFrame with columns [id, title, abstract]
            tokenizer: Hugging Face tokenizer
            max_length: Max token length
            samples_per_node: Number of triplets to generate per node
        """
        self.clustering = clustering
        self.metadata_df = metadata_df.set_index('id')
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Generate triplets with distances
        self.triplets = self._generate_triplets(samples_per_node)

    def _get_text(self, node_id: str) -> str:
        """Get combined title + abstract for a node"""
        try:
            row = self.metadata_df.loc[int(node_id)]
            title = str(row['title']) if pd.notna(row['title']) else ""
            abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
            return f"{title} {abstract}".strip()
        except (KeyError, ValueError):
            return f"Document {node_id}"

    def _generate_triplets(self, samples_per_node: int) -> List[Tuple]:
        """Generate (anchor, positive, negative, dist_pos, dist_neg, anchor_id) tuples"""
        triplets = []
        node_ids = self.clustering.all_nodes

        for node_id in tqdm(node_ids, desc="Generating triplets"):
            for _ in range(samples_per_node):
                # Sample with distances
                pos_id, neg_id, dist_pos, dist_neg = \
                    self.clustering.sample_triplet_with_distances(node_id)

                anchor_text = self._get_text(node_id)
                pos_text = self._get_text(pos_id)
                neg_text = self._get_text(neg_id)

                triplets.append((
                    anchor_text, pos_text, neg_text,
                    float(dist_pos), float(dist_neg),
                    node_id  # Store anchor node ID for splitting
                ))

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor, positive, negative, dist_pos, dist_neg, anchor_id = self.triplets[idx]

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
            'cluster_dist_pos': torch.tensor(dist_pos, dtype=torch.float),
            'cluster_dist_neg': torch.tensor(dist_neg, dtype=torch.float),
        }
