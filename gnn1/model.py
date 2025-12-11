"""
CAT-GNN Model: Frozen Transformer + Trainable GNN

This model demonstrates that when GNN is trained only on train nodes,
test nodes with no edges degrade to using only transformer embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from torch_geometric.nn import GCNConv, GATConv


class TransformerGNN(nn.Module):
    """
    Combined Transformer + GNN model with frozen transformer.

    Architecture:
    1. Frozen transformer generates initial embeddings
    2. GNN layers propagate information through graph
    3. Only GNN layers are trained

    For nodes without neighbors (test nodes), this degrades to transformer-only.
    """

    def __init__(
        self,
        model_name: str,
        gnn_type: str = 'gcn',
        hidden_dim: int = 768,
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = 'cls',
        freeze_transformer: bool = True
    ):
        """
        Args:
            model_name: Hugging Face transformer model name
            gnn_type: 'gcn' or 'gat'
            hidden_dim: Hidden dimension (should match transformer output)
            num_gnn_layers: Number of GNN layers
            dropout: Dropout rate
            pooling: 'cls' or 'mean' pooling for transformer
            freeze_transformer: Whether to freeze transformer weights
        """
        super().__init__()

        # Transformer encoder (frozen)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.hidden_dim = hidden_dim

        # Freeze transformer if specified
        if freeze_transformer:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"✓ Transformer weights frozen")

        # GNN layers (trainable)
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            if gnn_type == 'gcn':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                # GAT with 4 attention heads
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.num_gnn_layers = num_gnn_layers

        print(f"✓ Created {num_gnn_layers}-layer {gnn_type.upper()} model")
        print(f"  Total GNN parameters: {sum(p.numel() for p in self.gnn_layers.parameters()):,}")

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode_text(self, input_ids, attention_mask):
        """Encode text using frozen transformer"""
        with torch.no_grad():  # Don't compute gradients for transformer
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

            if self.pooling == 'mean':
                embeddings = self.mean_pooling(outputs, attention_mask)
            else:  # cls
                embeddings = outputs[0][:, 0, :]

            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def forward(self, x, edge_index):
        """
        Forward pass through GNN layers.

        Args:
            x: Node features (pre-computed transformer embeddings) [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            Node embeddings after GNN propagation [num_nodes, hidden_dim]
        """
        # Initial embeddings (from frozen transformer)
        h = x

        # GNN layers
        for i, (gnn, bn) in enumerate(zip(self.gnn_layers, self.batch_norms)):
            # Graph convolution
            h_new = gnn(h, edge_index)

            # Batch norm
            h_new = bn(h_new)

            # Activation and dropout (except last layer)
            if i < self.num_gnn_layers - 1:
                h_new = F.relu(h_new)
                h_new = self.dropout(h_new)

            # Residual connection
            h = h + h_new

        # Final normalization
        h = F.normalize(h, p=2, dim=1)

        return h

    def get_embeddings(self, texts, tokenizer, edge_index, device, batch_size=32, max_length=512):
        """
        Complete pipeline: text -> transformer -> GNN -> final embeddings

        Args:
            texts: List of texts (one per node)
            tokenizer: Transformer tokenizer
            edge_index: Graph edges [2, num_edges]
            device: torch device
            batch_size: Batch size for transformer encoding
            max_length: Max sequence length

        Returns:
            Final node embeddings after GNN [num_nodes, hidden_dim]
        """
        self.eval()

        # Step 1: Encode all texts with transformer (in batches)
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(device)

                embeddings = self.encode_text(inputs['input_ids'], inputs['attention_mask'])
                all_embeddings.append(embeddings)

        # Concatenate all transformer embeddings
        x = torch.cat(all_embeddings, dim=0)  # [num_nodes, hidden_dim]

        # Step 2: Apply GNN
        with torch.no_grad():
            final_embeddings = self.forward(x, edge_index)

        return final_embeddings


class TripletTransformerGNN(nn.Module):
    """
    Wrapper for triplet loss training with TransformerGNN.

    This is used during training with triplet loss.
    """

    def __init__(self, base_model: TransformerGNN):
        super().__init__()
        self.model = base_model

    def forward(self, x, edge_index):
        """Forward pass for triplet loss training"""
        return self.model(x, edge_index)

    def encode_text(self, input_ids, attention_mask):
        """Encode text using transformer"""
        return self.model.encode_text(input_ids, attention_mask)
