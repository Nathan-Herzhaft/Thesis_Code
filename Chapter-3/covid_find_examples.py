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
# Load dataset and shape it correctly
raw_dataset = datasets.load_dataset("minh21/COVID-QA-question-answering-biencoder-data-75_25", split="train",trust_remote_code=True)
data_truncated = pd.DataFrame(raw_dataset).sample(100).reset_index(drop=True)
queries = pd.Series(data_truncated['question']).apply(lambda x : 'query: ' + x)
passages = pd.Series(data_truncated['context_chunks']).apply(lambda x: 'passage: ' + x[0])



# %%
# Custom class for datasets, that allows to select random negative samples with each inputs
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
# The full experiment pipeline
    
    print(f'Number of iterations : {len(dataset)}')
    print('\n\n')
    
    good_preds_e5 = 0
    good_preds_bm25 = 0
    
    start = time.time()
    
    for i in range(len(dataset)) :
        
        query, passages = dataset[i]
        
        # Predict with e5
        emb_query = e5_model.encode(query)
        emb_passages = e5_model.encode(passages)
        scores_e5 = emb_query @ emb_passages.T
        pred_e5 = scores_e5.argmax()
        good_preds_e5 += (pred_e5 == 0)

        # Predict with bm25
        tokenized_corpus = [doc.split(" ") for doc in passages]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.split(" ")
        scores_bm25 = bm25.get_scores(tokenized_query)
        pred_bm25 = scores_bm25.argmax()
        good_preds_bm25 += (pred_bm25 == 0)

        # Save the examples in sepqrqte txt files
        if pred_e5 != 0 :
            if pred_bm25 != 0 :
                
                with open('./Results/examples-chapter3/both_fail.txt', 'a+') as file:
                    file.write(query + '\n')
                    file.write('true passage : ' + passages[0][9:] + '\n')
                    file.write('e5 retrieved passage : ' + passages[pred_e5][9:] + '\n')
                    file.write('bm25 retrieved passage : ' + passages[pred_bm25][9:] + '\n\n\n')
                    file.close()
            if pred_bm25 == 0 :
                with open('./Results/examples-chapter3/e5_fails.txt', 'a+') as file:
                    file.write(query + '\n')
                    file.write('true passage : ' + passages[0][9:] + '\n')
                    file.write('e5 retrieved passage : ' + passages[pred_e5][9:] + '\n')
                    file.write('bm25 retrieved passage : ' + passages[pred_bm25][9:] + '\n\n\n')
                    file.close()

        if pred_e5 == 0 :
            if pred_bm25 != 0 :
                with open('./Results/examples-chapter3/e5_succeeds.txt', 'a+') as file:
                    file.write(query + '\n')
                    file.write('true passage : ' + passages[0][9:] + '\n')
                    file.write('e5 retrieved passage : ' + passages[pred_e5][9:] + '\n')
                    file.write('bm25 retrieved passage : ' + passages[pred_bm25][9:] + '\n\n\n')
                    file.close()
            if pred_bm25 == 0 :
                with open('./Results/examples-chapter3/both_succeed.txt', 'a+') as file:
                    file.write(query + '\n')
                    file.write('true passage : ' + passages[0][9:] + '\n')
                    file.write('e5 retrieved passage : ' + passages[pred_e5][9:] + '\n')
                    file.write('bm25 retrieved passage : ' + passages[pred_bm25][9:] + '\n\n\n')
                    file.close()

                
        
        if i%10 == 0 :
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
