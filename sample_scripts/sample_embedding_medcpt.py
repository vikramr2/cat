import numpy
import json
import pandas as pd
from tqdm import tqdm
from Bio import Entrez # type: ignore
from transformers import AutoTokenizer, AutoModel
import torch

embedding_dir = "/projects/illinois/eng/shared/shared/CS598GCK-SP25/vikram_ruining_project/data/pubmed_embeddings/"
pmid_table = "/projects/illinois/eng/shared/shared/CS598GCK-SP25/vikram_ruining_project/data/cen/metadata/cen_pmids_filtered.csv"
num_chunks = 38

def load_pmids(embedding_dir, index):
    """
    Load the pmids from the specified directory and file.
    """
    # Load the pmids
    filename = f"pmids_chunk_{index}.json"
    
    with open(embedding_dir + filename, 'r') as f:
        pmids = json.load(f)
    
    # Print the shape of the loaded pmids
    # print(f"Loaded {filename} with shape: {len(pmids)}")
    
    return pmids

def fetch_title_abstract(pmid):
    Entrez.email = 'your_email@example.com'  # Replace with your email address
    handle = Entrez.efetch(db='pubmed', id=pmid, retmode='xml')
    record = Entrez.read(handle)
    title = record['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
    abstract = record['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText']
    return title, abstract

def compute_embeddings(data, model, tokenizer):
    with torch.no_grad():
        encoded = tokenizer(
            data,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=512,
        )
        embeds = model(**encoded).last_hidden_state[:, 0, :]
    return embeds.numpy()

def fetch_embed(pmid):
    """
    Fetch the embedding for the given pmid.
    """
    # Get the index of the pmid in the pmids list
    try:
        title, abstract = fetch_title_abstract(pmid)
    except Exception as e:
        print(f"Error fetching title and abstract for pmid {pmid}: {e}")
        return numpy.zeros((1, 768))  # Return a zero vector if the pmid is not found
    data = f"{title} {abstract}"
    # Compute the embedding
    embed = compute_embeddings(data, model, tokenizer)
    
    return embed

if __name__ == "__main__":
    # Load the tokenizer and model
    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    # Load the pmids
    pmids = []
    for i in tqdm(range(0, num_chunks), desc="Loading pmids"):
        pmids += load_pmids(embedding_dir, i)
    
    # Get pmids from the pmid table
    pmid_table_df = pd.read_csv(pmid_table)
    pmid_table_pmids = pmid_table_df["pmid"].tolist()
    pmids = [int(pmid) for pmid in pmids]
    pmid_table_pmids = [int(pmid) for pmid in pmid_table_pmids]

    # Check if all pmid table pmids are in the pmids list
    pmid_table_pmids_set = set(pmid_table_pmids)
    pmids_set = set(pmids)
    pmid_table_pmids_not_in_pmids = pmid_table_pmids_set - pmids_set
    pmids_not_in_pmid_table = pmids_set - pmid_table_pmids_set
    print(f"PMIDs in pmid table not in pmids list: {len(pmid_table_pmids_not_in_pmids)}")
    if len(pmid_table_pmids_not_in_pmids) > 0:
        print(f"Numbers of PMIDs in pmid table not in pmids list: {len(pmid_table_pmids_not_in_pmids)}")

    # Get the indices of the pmid table pmids in the pmids list
    pmid_to_index = {pmid: idx for idx, pmid in enumerate(pmids)}

    # Convert pmid_table_pmids_not_in_pmids to a list to preserve order
    pmid_table_pmids_not_in_pmids = list(pmid_table_pmids_not_in_pmids)

    not_in_table_embeds = []
    # Get the embeddings for the pmids not in the pmid table
    for pmid in tqdm(pmid_table_pmids_not_in_pmids, desc="Fetching embeddings"):
        embed = fetch_embed(pmid)
        not_in_table_embeds.append(embed)
    not_in_table_embeds = numpy.array(not_in_table_embeds)

    print(f"Fetched {len(not_in_table_embeds)} embeddings with shape: {not_in_table_embeds.shape}")

    # Save the embeddings to a file
    output_file = f"{embedding_dir}/embeds_chunk_{num_chunks}.npy"
    numpy.save(output_file, not_in_table_embeds)

    # Save the pmid list to a json file
    output_json_file = f"{embedding_dir}/pmids_chunk_{num_chunks}.json"
    with open(output_json_file, 'w') as f:
        json.dump(pmid_table_pmids_not_in_pmids, f)
    print(f"Saved pmids to {output_json_file}")
