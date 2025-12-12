"""
Dataset for disjoint cluster triplet learning
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict
from tqdm import tqdm
import numpy as np


class DisjointClusterTripletDataset(Dataset):
    """Dataset for disjoint cluster triplet learning"""

    def __init__(
        self,
        cluster_df: pd.DataFrame,
        metadata_df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        samples_per_cluster: int = 5,
        train_nodes: List[str] = None,
        train_clusters: List[int] = None,
        seed: int = 42
    ):
        """
        Args:
            cluster_df: DataFrame with columns [node, cluster]
            metadata_df: DataFrame with columns [id, title, abstract]
            tokenizer: Hugging Face tokenizer
            max_length: Max token length
            samples_per_cluster: Number of triplets to generate per cluster
            train_nodes: List of node IDs (as strings) to use for training
            train_clusters: (Deprecated) List of cluster IDs to use
            seed: Random seed for reproducibility
        """
        self.cluster_df = cluster_df
        self.metadata_df = metadata_df.set_index('id')
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seed = seed
        np.random.seed(seed)

        # Filter to train nodes if specified
        if train_nodes is not None:
            train_nodes_int = [int(n) for n in train_nodes]
            self.cluster_df = self.cluster_df[self.cluster_df['node'].isin(train_nodes_int)]
        # Backward compatibility: support train_clusters
        elif train_clusters is not None:
            self.cluster_df = self.cluster_df[self.cluster_df['cluster'].isin(train_clusters)]

        # Build cluster mappings
        self.cluster_to_nodes = self._build_cluster_mapping()
        
        # Filter out clusters with less than 2 nodes in training set
        self.cluster_to_nodes = {
            cid: nodes for cid, nodes in self.cluster_to_nodes.items() 
            if len(nodes) >= 2
        }
        
        self.cluster_ids = list(self.cluster_to_nodes.keys())
        
        # Generate triplets
        self.triplets = self._generate_triplets(samples_per_cluster)

    def _build_cluster_mapping(self) -> Dict[int, List[str]]:
        """Build mapping from cluster ID to list of node IDs"""
        cluster_to_nodes = {}
        for cluster_id in self.cluster_df['cluster'].unique():
            nodes = self.cluster_df[self.cluster_df['cluster'] == cluster_id]['node'].astype(str).tolist()
            cluster_to_nodes[cluster_id] = nodes
        return cluster_to_nodes

    def _get_text(self, node_id: str) -> str:
        """Get combined title + abstract for a node"""
        try:
            row = self.metadata_df.loc[int(node_id)]
            title = str(row['title']) if pd.notna(row['title']) else ""
            abstract = str(row['abstract']) if pd.notna(row['abstract']) else ""
            return f"{title} {abstract}".strip()
        except (KeyError, ValueError):
            return f"Document {node_id}"

    def _generate_triplets(self, samples_per_cluster: int) -> List[Tuple[str, str, str]]:
        """Generate (anchor, positive, negative) triplets"""
        triplets = []
        
        print(f"Generating triplets from {len(self.cluster_ids)} clusters...")
        print(f"Total nodes: {sum(len(nodes) for nodes in self.cluster_to_nodes.values())}")
        
        for cluster_id in tqdm(self.cluster_ids, desc="Generating triplets"):
            cluster_nodes = self.cluster_to_nodes[cluster_id]
            
            for _ in range(samples_per_cluster):
                # Sample anchor and positive from same cluster
                anchor_node, positive_node = np.random.choice(
                    cluster_nodes, size=2, replace=False
                )
                
                # Sample negative from different cluster
                negative_cluster = np.random.choice(
                    [c for c in self.cluster_ids if c != cluster_id]
                )
                negative_node = np.random.choice(
                    self.cluster_to_nodes[negative_cluster]
                )
                
                # Get texts
                anchor_text = self._get_text(anchor_node)
                positive_text = self._get_text(positive_node)
                negative_text = self._get_text(negative_node)
                
                triplets.append((anchor_text, positive_text, negative_text))
        
        print(f"Generated {len(triplets)} triplets")
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
