#!/bin/bash
# Example script for running hierarchical triplet loss fine-tuning

# Basic usage
python hierarchical_triplet_loss.py \
    --tree_json /Users/vikram/Documents/School/CS546/oc_mini/clustering/hierarchical/oc_mini_paris.json \
    --metadata_csv /Users/vikram/Documents/School/CS546/oc_mini/metadata/oc_mini_node_metadata.csv \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
    --output_dir ./finetuned_hierarchical_model \
    --batch_size 16 \
    --epochs 3 \
    --lr 2e-5 \
    --margin 0.5 \
    --samples_per_leaf 5 \
    --sampling_strategy hierarchical \
    --pooling mean

# Alternative: Using sibling-based sampling (faster, focuses on local clusters)
# python hierarchical_triplet_loss.py \
#     --tree_json /Users/vikram/Documents/School/CS546/oc_mini/clustering/hierarchical/oc_mini_paris.json \
#     --metadata_csv /Users/vikram/Documents/School/CS546/oc_mini/metadata/oc_mini_node_metadata.csv \
#     --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
#     --output_dir ./finetuned_sibling_model \
#     --batch_size 16 \
#     --epochs 3 \
#     --lr 2e-5 \
#     --margin 0.5 \
#     --samples_per_leaf 3 \
#     --sampling_strategy sibling \
#     --pooling mean
