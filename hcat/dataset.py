"""
Dataset for hierarchical triplet learning with distance-aware sampling
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from tqdm import tqdm

from tree_utils import HierarchicalTree


class HierarchicalTripletDataset(Dataset):
    """Dataset for hierarchical triplet learning with tree distances"""

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

        # Generate triplets with distances
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

    def _generate_triplets(self, samples_per_leaf: int) -> List[Tuple]:
        """Generate (anchor, positive, negative, dist_pos, dist_neg, anchor_id) tuples"""
        triplets = []
        leaf_names = list(self.tree.leaves.keys())

        for leaf_name in tqdm(leaf_names, desc="Generating triplets"):
            for _ in range(samples_per_leaf):
                # Sample with distances
                pos_name, neg_name, dist_pos, dist_neg = \
                    self.tree.sample_triplet_with_distances(
                        leaf_name,
                        strategy=self.sampling_strategy
                    )

                anchor_text = self._get_text(leaf_name)
                pos_text = self._get_text(pos_name)
                neg_text = self._get_text(neg_name)

                triplets.append((
                    anchor_text, pos_text, neg_text,
                    float(dist_pos), float(dist_neg),
                    leaf_name  # Store anchor node ID for splitting
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
            'tree_dist_pos': torch.tensor(dist_pos, dtype=torch.float),
            'tree_dist_neg': torch.tensor(dist_neg, dtype=torch.float),
        }