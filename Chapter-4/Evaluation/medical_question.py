#%%
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import Tensor

from sentence_transformers import SentenceTransformer

import datasets
import pandas as pd

import time
import os

model = SentenceTransformer('./Finetuned_Models/medical-questions-model')

# %%
raw_dataset = datasets.load_dataset("Malikeh1375/medical-question-answering-datasets", 'all-processed',split="train",trust_remote_code=True)

index = [i for i in range(int(len(raw_dataset)*(80/100)),len(raw_dataset))]
test_dataset = raw_dataset.select(index)

data_truncated = pd.DataFrame(test_dataset).sample(1000).reset_index(drop=True)
queries = pd.Series(data_truncated['input']).apply(lambda x: 'query: '+x)
passages = pd.Series(data_truncated['output']).apply(lambda x: 'passage: '+x)



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
def pipeline(dataset,model) :
    
    print(f'Number of iterations : {len(dataset)}')
    print('\n\n')
    
    good_preds = 0
    
    start = time.time()
    
    for i in range(len(dataset)) :
        
        query, passages = dataset[i]
        
        emb_query = model.encode(query)
        emb_passages = model.encode(passages)
        scores = emb_query @ emb_passages.T
        pred = scores.argmax()
        good_preds += (pred == 0)
        
        if i%50 == 0 :
            end = time.time()
            print(f'Iteration : {i}')
            print(f'Time for this batch : {round(end - start,3)}s')
            print(f'Current accuracy : {round(100*good_preds/(i+1),2)}%')
            print('\n\n')
            start = time.time()
    
    print(f'Final accuracy : {round(100*good_preds/(len(dataset)),2)}%')

pipeline(data,model)       
# %%
