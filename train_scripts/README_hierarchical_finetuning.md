# Hierarchical Triplet Loss Fine-tuning

This implementation fine-tunes embeddings using hierarchical triplet loss based on a tree topology. Documents (leaves) that are closer in the tree hierarchy will have more similar embeddings after training.

## Overview

The approach works by:
1. **Parsing the tree structure**: Loads the hierarchical Paris clustering JSON
2. **Mining triplets**: For each leaf, samples positive (nearby in tree) and negative (distant in tree) examples
3. **Training with triplet loss**: Minimizes distance between anchor-positive pairs while maximizing distance from anchor-negative pairs

## Key Components

### 1. HierarchicalTree
- Parses the tree JSON structure
- Computes tree distances using Lowest Common Ancestor (LCA)
- Samples positive/negative pairs based on hierarchical proximity

### 2. Sampling Strategies

**Hierarchical (Recommended)**:
- Positives: Leaves with smallest LCA depth (closest common ancestor high in tree)
- Negatives: Leaves with largest LCA depth (common ancestor near root)
- Best for capturing global tree structure

**Sibling**:
- Positives: Leaves from same parent cluster
- Negatives: Leaves from distant branches
- Faster, focuses on local cluster coherence

### 3. Tree Distance Metric
```python
tree_distance(node1, node2) = depth of LCA
```
- Lower depth = closer to root = more distant relationship
- Higher depth = closer to leaves = more similar documents

## Usage

### Basic Command
```bash
python hierarchical_triplet_loss.py \
    --tree_json /path/to/oc_mini_paris.json \
    --metadata_csv /path/to/oc_mini_node_metadata.csv \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output_dir ./finetuned_model \
    --epochs 3
```

### Full Parameters
```bash
python hierarchical_triplet_loss.py \
    --tree_json <path>              # Hierarchical tree JSON
    --metadata_csv <path>            # Metadata with id, title, abstract
    --model_name <hf_model>          # Hugging Face model name
    --output_dir <path>              # Where to save fine-tuned model
    --batch_size 16                  # Batch size
    --epochs 3                       # Training epochs
    --lr 2e-5                        # Learning rate
    --margin 0.5                     # Triplet loss margin
    --max_length 512                 # Max token length
    --samples_per_leaf 5             # Triplets per leaf
    --sampling_strategy hierarchical # or 'sibling'
    --pooling mean                   # or 'cls'
    --train_split 0.9                # Train/val split ratio
```

### Recommended Settings

**For Large Trees (>10K leaves)**:
```bash
--sampling_strategy hierarchical
--samples_per_leaf 3
--batch_size 32
--epochs 2
```

**For Small Trees (<5K leaves)**:
```bash
--sampling_strategy hierarchical
--samples_per_leaf 5
--batch_size 16
--epochs 3
```

**For Fast Training**:
```bash
--sampling_strategy sibling
--samples_per_leaf 2
--batch_size 32
--epochs 2
```

## Understanding the Loss

The triplet loss optimizes:
```
Loss = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```

Where:
- `d(x, y)` is the embedding distance (e.g., Euclidean)
- `margin` controls the minimum separation between positive and negative

The model learns to:
- Pull embeddings of hierarchically-close leaves together
- Push embeddings of hierarchically-distant leaves apart

## After Fine-tuning

Load the fine-tuned model:
```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

model = AutoModel.from_pretrained('./finetuned_model')
tokenizer = AutoTokenizer.from_pretrained('./finetuned_model')

def encode(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

# Use for downstream tasks
embedding = encode("Your document text here")
```

## Data Requirements

**Tree JSON** (from Paris clustering):
```json
{
  "algorithm": "Paris",
  "hierarchy": {
    "id": 28766,
    "type": "cluster",
    "children": [
      {
        "id": 12069,
        "name": "9393161",
        "type": "leaf",
        "count": 1
      },
      ...
    ]
  }
}
```

**Metadata CSV**:
```csv
id,doi,title,abstract
9393161,10.xxx,Title here,Abstract text here...
```

The `name` field in leaf nodes must match the `id` column in the metadata CSV.

## Advanced: Custom Sampling

You can modify the sampling logic in `HierarchicalTree.sample_positive_negative()` to:
- Weight by edge distances instead of LCA depth
- Use k-nearest neighbors in tree
- Sample hard negatives (violating triplets)
- Implement curriculum learning (easy → hard triplets)

## Tips

1. **Monitor validation loss**: Should decrease steadily
2. **Margin tuning**: Smaller margin (0.2-0.3) for fine-grained distinctions, larger (0.7-1.0) for coarse
3. **Learning rate**: Start with 2e-5, reduce if unstable
4. **Batch size**: Larger is better (more diverse triplets per batch)
5. **GPU memory**: Reduce `max_length` or `batch_size` if OOM

## Expected Results

After training, leaves closer in the tree should have:
- Higher cosine similarity in embedding space
- Smaller Euclidean distance
- Form tighter clusters in t-SNE/UMAP visualizations

You can validate this by computing embedding similarities for leaf pairs at different tree distances.
