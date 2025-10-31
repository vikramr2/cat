# CAT

CAT (Community Augmented Transformers) uses an alternative fine tuning method to 'augment' encoder-only transformers with network community structural information, without the use of an exoplicit GNN architecture.

## Approach

CAT has two steps:

1. Standard fine tuning
2. Contrastive community learning

### Standard fine tuning

Traditionally, BERT models use masked-token prediction to learn embeddings. This would generally be the case for this step of the fine tuning process. This step is essentially to start with a base fine tuning method, then in a later step use CAT's actual logic to enhance the embedding.

### Constrastive community learning

In this part, depending on the type of clustering, we use contrastive learning to distinguish pairs that might be part of same of different communities:

- **disjoint/overlapping**: select metadata of two nodes belonging to the same community, then select a node belonging to a different community, then use triplet loss
- **hierarchical**: select metadata from 3 nodes $u,v,w$ such that given their leaf to leaf (l2l) distance, $l2l(u,v)<l2l(u,w)$ and $l2l(u,v)<l2l(v,w)$m then use hierarchical triplet loss.

Selection is done via uniform random sampling given the above constraints. 
