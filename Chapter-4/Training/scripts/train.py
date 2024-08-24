from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, SentenceTransformerTrainingArguments
from datasets import Dataset
import datasets
import random
from sentence_transformers.training_args import BatchSamplers

# Load model
model = SentenceTransformer("intfloat/e5-small-v2")

# Load dataset and shape the data
raw_dataset = datasets.load_dataset("minh21/COVID-QA-sentence-transformer",split="train")

train_samples = [i for i in range(int(len(raw_dataset)))]
small_dataset = raw_dataset.select(train_samples)

small_dataset = small_dataset.select_columns(['question','positive'])
small_dataset = small_dataset.rename_columns({'question':'anchor','positive':'positive'})

def add_prefixes(instance) :
    instance['anchor'] = 'query: ' + instance['anchor']
    instance['positive'] = 'passage: ' + instance['positive']
    return instance

train_dataset = small_dataset.map(add_prefixes)

loss = losses.MultipleNegativesRankingLoss(model)

# Define training args
training_args = SentenceTransformerTrainingArguments(
    output_dir="s3://sagemaker-us-west-2-536930143272/acx-embeddings/",  # output directory for sagemaker to upload to s3
    num_train_epochs=3,  # number of epochs
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

# Train and save the model
trainer.train()
trainer.save_model(output_dir = '/opt/ml/model')