# Project Title

This is the GitHub repository that contains the code written during my master's thesis at UCL, in partnership with the company Accelex. The subject of the thesis focuses on domain adaptation of embedding models, particularly on the generation of synthetic data used for fine-tuning.

## Table of Contents

1. [Introduction](#introduction)
2. [Experiments](#experiments)
   - [Chapter 3](#chapter-3)
   - [Chapter 4](#chapter-4)
   - [Chapter 5](#chapter-5)
3. [Results](#results)
4. [Dependencies](#dependencies)

## Introduction

The goal of this project is to explore domain adaptation techniques for embedding models, with a focus on generating synthetic data to improve model fine-tuning. This repository contains all the scripts and code necessary to replicate the experiments conducted during the master's thesis.


## Experiments

### Chapter 3

- This chapter provides code for evaluating the performance of embedding models, specifically using the E5 model, on dense retrieval tasks, leveraging custom scripts to measure accuracy and effectiveness compared to traditional methods like BM25.
- Each script completes the same experiment, but each uses a different dataset (according to the script name)
- `covid_find_examples.py`: Does the same experiment but also saves the queries and passages to visualize the results in different cases

### Chapter 4

- This chapter contains the code for domain-specific fine-tuning of the E5 model using specialized datasets, employing AWS SageMaker for training, and demonstrating the model's adaptability to different thematic areas beyond general datasets.
- Contains two folders, `Training/` and `Evaluation/`.
    - `Training/`: uses the script `run_training.py` to run a finetuning job on Sagemaker, using the training scripts and requirements stored in `scripts/`
    - `Evaluation/`: Uses the finetuned model and the evaluation pipeline from the previous chapter to evaluate the performances of the finetuned models. Requires local access to finetuned model, that are not provided with the repo.

### Chapter 5

- This chapter focuses on generating synthetic training data using Large Language Models (LLMs) and fine-tuning the E5 model with this synthetic data to explore its impact on improving performance in domain-specific contexts.
- Includes scripts and directories for data generation, finetuning, and evaluation.
- Key scripts:
  - `Training/`: uses the script `run_training.py` to run a finetuning job on Sagemaker, using the training scripts and requirements stored in `scripts/`. In this Chapter, `scripts/` also contains a copy of the txt files containing the queries. this is required to send them correctly to Sagemaker.
  - `data-generation.py`: Script for generating synthetic data.
  - `evaluation.py`: Script for evaluating the models finetuned with the synthetic data.

## Results

- The `Results` folder contains various outputs from the experiments:
  - `synthetic-queries/`: Stores synthetic query results from Chapter 5
  - `examples-chapter3/`: Contains examples related to Chapter 3 experiments.
  - `Plots/`: Directory for different plots generated from the data.

## Dependencies

All dependencies required to run the experiments are listed in the `requirements.txt` file. To install them, run the following command:

```bash
pip install -r requirements.txt
