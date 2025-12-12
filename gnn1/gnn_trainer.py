"""
Training utilities specifically for GNN models with graph structure
"""

import torch
from tqdm.auto import tqdm


def train_gnn_epoch(model, train_loader, loss_fn, optimizer, device):
    """
    Train GNN model for one epoch using graph-based triplets.

    Args:
        model: TransformerGNN model
        train_loader: DataLoader with graph triplet data
        loss_fn: Triplet loss function
        optimizer: Optimizer
        device: torch device

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move inputs to device
        anchor_ids = batch['anchor_input_ids'].to(device)
        anchor_mask = batch['anchor_attention_mask'].to(device)
        positive_ids = batch['positive_input_ids'].to(device)
        positive_mask = batch['positive_attention_mask'].to(device)
        negative_ids = batch['negative_input_ids'].to(device)
        negative_mask = batch['negative_attention_mask'].to(device)
        edge_index = batch['edge_index'].to(device)

        optimizer.zero_grad()

        # Encode texts with transformer (gradients flow if not frozen)
        anchor_transformer = model.encode_text_trainable(anchor_ids, anchor_mask)
        positive_transformer = model.encode_text_trainable(positive_ids, positive_mask)
        negative_transformer = model.encode_text_trainable(negative_ids, negative_mask)

        # Apply GNN (on batch - simplified, in reality need full graph context)
        # For proper implementation, we'd need all node embeddings
        # Here we assume the GNN is applied to the triplet embeddings directly
        # This is a simplification - see note below

        # SIMPLIFIED: Apply GNN layers to individual embeddings
        # NOTE: This doesn't use graph structure properly for the triplet
        # In practice, you'd need to:
        # 1. Encode ALL nodes in the graph
        # 2. Apply GNN to get all node embeddings
        # 3. Extract anchor/pos/neg embeddings
        anchor_emb = model.gnn_forward(anchor_transformer, edge_index)
        positive_emb = model.gnn_forward(positive_transformer, edge_index)
        negative_emb = model.gnn_forward(negative_transformer, edge_index)

        # Compute triplet loss
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def train_gnn_epoch_full_graph(
    model,
    train_node_ids,
    edge_index,
    node_to_idx,
    metadata_df,
    tokenizer,
    loss_fn,
    optimizer,
    device,
    num_triplets=1000,
    batch_size=32,
    max_length=512
):
    """
    Train GNN model for one epoch using FULL GRAPH approach.

    This is the correct way to train a GNN:
    1. Encode ALL training nodes with transformer
    2. Apply GNN to get ALL node embeddings
    3. Sample triplets from the embeddings
    4. Compute loss

    Args:
        model: TransformerGNN model
        train_node_ids: List of training node IDs
        edge_index: Graph edges for training nodes
        node_to_idx: Mapping from node_id to index
        metadata_df: Metadata DataFrame
        tokenizer: Tokenizer
        loss_fn: Triplet loss function
        optimizer: Optimizer
        device: Device
        num_triplets: Number of triplets to sample per epoch
        batch_size: Batch size for transformer encoding
        max_length: Max sequence length

    Returns:
        Average training loss
    """
    from graph_utils import get_node_texts
    import numpy as np

    model.train()

    # Step 1: Encode ALL training nodes with transformer
    print("  Encoding all training nodes...")
    node_list = [node_to_idx[i] for i in range(len(node_to_idx))]
    idx_to_node = {idx: nid for nid, idx in node_to_idx.items()}
    node_list_ordered = [idx_to_node[i] for i in range(len(idx_to_node))]

    texts = get_node_texts(node_list_ordered, metadata_df)

    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="  Transformer encoding"):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        # Encode with trainable transformer
        batch_embs = model.encode_text_trainable(inputs['input_ids'], inputs['attention_mask'])
        all_embeddings.append(batch_embs)

    x = torch.cat(all_embeddings, dim=0)  # [num_nodes, hidden_dim]

    # Step 2: Apply GNN to get all node embeddings
    print("  Applying GNN...")
    node_embeddings = model(x, edge_index.to(device))  # [num_nodes, hidden_dim]

    # Step 3: Build adjacency list for triplet sampling
    adj_list = {}
    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        tgt = edge_index[1, i].item()
        if src not in adj_list:
            adj_list[src] = []
        adj_list[src].append(tgt)

    # Filter valid anchors (nodes with neighbors)
    valid_anchors = [idx for idx in range(len(node_to_idx)) if idx in adj_list and len(adj_list[idx]) > 0]

    if len(valid_anchors) == 0:
        raise ValueError("No valid anchor nodes found!")

    # Step 4: Sample triplets and compute loss
    print(f"  Sampling {num_triplets} triplets...")
    total_loss = 0
    num_batches = 0

    all_indices = set(range(len(node_to_idx)))

    for _ in tqdm(range(num_triplets), desc="  Computing loss"):
        # Sample anchor
        anchor_idx = np.random.choice(valid_anchors)

        # Sample positive (neighbor)
        pos_idx = np.random.choice(adj_list[anchor_idx])

        # Sample negative (non-neighbor)
        non_neighbors = list(all_indices - set(adj_list[anchor_idx]) - {anchor_idx})
        if len(non_neighbors) == 0:
            non_neighbors = list(all_indices - {anchor_idx})
        neg_idx = np.random.choice(non_neighbors)

        # Get embeddings
        anchor_emb = node_embeddings[anchor_idx:anchor_idx+1]
        pos_emb = node_embeddings[pos_idx:pos_idx+1]
        neg_emb = node_embeddings[neg_idx:neg_idx+1]

        # Compute loss
        optimizer.zero_grad()
        loss = loss_fn(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches
