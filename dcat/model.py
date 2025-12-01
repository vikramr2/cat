"""
Model wrapper for triplet loss training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TripletEmbeddingModel(nn.Module):
    """Model wrapper for triplet loss training"""

    def __init__(self, model_name: str, pooling: str = 'cls'):
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
