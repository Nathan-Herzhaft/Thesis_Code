from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from datasets import load_dataset
import datasets
import random
from sentence_transformers.training_args import BatchSamplers
from datasets import concatenate_datasets

synthetic_queries = load_dataset('text', data_files='queries.txt')['train']

raw_passages = datasets.load_dataset("Malikeh1375/medical-question-answering-datasets", 'all-processed',split="train",trust_remote_code=True).select_columns(['output']).select([i for i in range(50000)])

full_data = concatenate_datasets([synthetic_queries,raw_passages],axis=1)

model = SentenceTransformer("intfloat/e5-small-v2")

full_data = full_data.rename_columns({'text':'anchor','output':'positive'})

def add_prefixes(instance) :
    instance['anchor'] = 'query: ' + instance['anchor']
    instance['positive'] = 'passage: ' + instance['positive']
    return instance

train_dataset = full_data.map(add_prefixes)

loss = losses.MultipleNegativesRankingLoss(model)

training_args = SentenceTransformerTrainingArguments(
    output_dir="s3://sagemaker-us-west-2-536930143272/acx-embeddings/",  # output directory for sagemaker to upload to s3
    num_train_epochs=1,  # number of epochs
    per_device_train_batch_size=8,  # training batch size, from the paper
    learning_rate= 3*10**-5,  # learning rate, from the paper
    warmup_steps=400, # warmup steps, from the paper
    lr_scheduler_type="linear", # linear schedule, from the paper 
    optim="adamw_torch_fused",
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    logging_steps=10,  # log every 100 steps
)

trainer = SentenceTransformerTrainer(
    model=model,  # bg-base-en-v1
    args=training_args,  # training arguments
    train_dataset=train_dataset,  # training dataset
    loss=loss,
)


trainer.train()
trainer.save_model(output_dir = '/opt/ml/model')