#%%
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

import time
import os

e5 = SentenceTransformer('intfloat/e5-small-v2')

# %%
raw_dataset = datasets.load_dataset("microsoft/ms_marco",'v1.1', split="test",trust_remote_code=True)
data_truncated = pd.DataFrame(raw_dataset).sample(1000).reset_index(drop=True)
queries = pd.Series(data_truncated['query']).apply(lambda x : 'query: ' + x)
passages = pd.Series(data_truncated['passages']).apply(lambda x: 'passage: ' + x['passage_text'][0])


# %%
class CustomDataset(Dataset):
    def __init__(self,queries,passages,n_samples):
        self.queries = queries
        self.passages = passages
        self.n_samples = n_samples

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, idx):
        negative_samples = self.passages.sample(self.n_samples - 1).to_list()
        return self.queries[idx], [self.passages[idx]] + negative_samples
    
data = CustomDataset(queries,passages,5)

# %%
def pipeline(dataset,e5_model) :
    
    print(f'Number of iterations : {len(dataset)}')
    print('\n\n')
    
    good_preds_e5 = 0
    good_preds_bm25 = 0
    
    start = time.time()
    
    for i in range(len(dataset)) :
        
        query, passages = dataset[i]
        
        emb_query = e5_model.encode(query)
        emb_passages = e5_model.encode(passages)
        scores_e5 = emb_query @ emb_passages.T
        pred_e5 = scores_e5.argmax()
        good_preds_e5 += (pred_e5 == 0)
        
        tokenized_corpus = [doc.split(" ") for doc in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        scores_bm25 = bm25.get_scores(tokenized_query)
        pred_bm25 = scores_bm25.argmax()
        good_preds_bm25 += (pred_bm25 == 0)
        
        if i%50 == 0 :
            end = time.time()
            print(f'Iteration : {i}')
            print(f'Time for this batch : {round(end - start,3)}s')
            print(f'Current e5 accuracy : {round(100*good_preds_e5/(i+1),2)}%')
            print(f'Current bm25 accuracy : {round(100*good_preds_bm25/(i+1),2)}%')
            print('\n\n')
            start = time.time()
    
    print(f'Final e5 accuracy : {round(100*good_preds_e5/(len(dataset)),2)}%')
    print(f'Final bm25 accuracy : {round(100*good_preds_bm25/(len(dataset)),2)}%')

pipeline(data,e5)       
# %%
