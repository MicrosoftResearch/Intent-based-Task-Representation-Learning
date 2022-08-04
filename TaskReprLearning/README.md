# Intent-based Task Representations

This directory contains the code for training an encoder for task representations.

## Overview

### Core scripts

- `src/train.py`: training a Transformer model
- `src/extract_intent_embs.py`: extract intent embedings for single to-do
- `src/extract_intent_embs_pair.py`: extract intent embedings for to-do pairs
- `baseline_encoders/transformer_models.py`: encode single to-do using Transformer models
- `baseline_encoders/transformer_models_pair.py`: encode to-do pairs using Transformer models


### Directories

- `src`: source code for neural nets
- `scripts`: scripts for data conversion, etc.
- `src/models`: defines main neural nets
    - `src/models/simple_enc.py`: defines the main model based on Transformer encoders
- `src/dataset_readers`: defines `torch.Dataset` classes to convert datasets (.json) to a list of dictionaries
- `src/modules`: defines components for neural nets (feed forward networks, loss functions, etc.)
- `config`: training configuration


### Training workflow

1. Hyperparameters are stored in a JSON file (see `config`)
2. Load a pretrained Transformer model using Huggingface's `transformers`
3. Read datasets from the filepaths in the config file.
    - Load a preprocessed cache file (.pkl) if exists
    - Tokenize texts and convert tokens into `torch.Tensor`
    - Save the result to a cache file
4. Set up an optimizer, learning rate scheduler, and data loader
5. Compute loss and update parameters
    - Record training loss every 100 global steps
    - Record validation loss every epoch
    - Save the checkpoint if the validation loss is best


## Setup

``` shell
conda env create -f environment.yml
conda activate todo
```


## Data preparation for the FrameNet task

### 1. Generate FrameNet FE (frame element) embeddings

``` shell
python scripts/extract_framenet_fe_definitions.py data/v1/{trn,vld,tst}.json -o data/v1/fe_definitions.tsv -v
python scripts/embed_framenet_fes_transformer.py data/v1/fe_definitions.tsv -m bert-base-uncased --lowercase -o data/v1/fe_embs.bert-base-uncased.768d.txt -v
python scripts/embed_framenet_fes_transformer.py data/v1/fe_definitions.tsv -m bert-large-uncased --lowercase -o data/v1/fe_embs.bert-large-uncased.1024d.txt -v
python scripts/embed_framenet_fes_transformer.py data/v1/fe_definitions.tsv -m roberta-base --lowercase -o data/v1/fe_embs.roberta-base.768d.txt -v
```

### 2. Calculate FrameNet FE weights

``` shell
python scripts/calculate_fe_weights_for_event_frames.py -o data/framenet/fe_weights.v1.json -v
```


## Training

```shell
python src/train.py config/bert-base-uncased.v1.jsonnet -s save/bert-base-uncased.v1 --cuda 0

## Training on multiple GPUs
CUDA_VISIBLE_DEVICES=<DEVICE NUMBER(s)> python src/train.py  config/bert-base-uncased.v1.jsonnet -s save/bert-base-uncased --parallel
```

This script produces two output directories.

- `save/bert-base-uncased`: trained model file
- `logs/bert-base-uncased...`: tensorboard log file


## Extract intent embeddings

See `cmd/extract_embs.sh` to produce embeddings for given input texts.

```shell
bash cmd/extract_embs.sh <path to model directory> <CUDA device ID (-1 to use CPU)>

# Example
## Pre-trained models are available in save/. Use Git LFS to download tar.gz files and decompress them in save/bert-base-uncased.v1 etc.
## After decompressing the pre-trained BERT-based model (model.best.pth, config.json, and tokenizer/) into save/bert-base-uncased.v1, run:
bash cmd/extract_embs.sh save/bert-base-uncased.v1 0
```

# Baselines

## Transformers

```shell
# Transformers
bash cmd/extract_transformer_embs.sh <model name or path to pretraiend model> <pooling method> <CUDA device ID (-1 to use CPU)>

bash cmd/extract_transformer_embs.sh bert-base-uncased cls 0

bash cmd/extract_transformer_embs.sh bert-base-uncased cls 0
bash cmd/extract_transformer_embs.sh bert-base-uncased mean 0
bash cmd/extract_transformer_embs.sh roberta-base cls 0
bash cmd/extract_transformer_embs.sh roberta-base mean 0
bash cmd/extract_transformer_embs.sh bert-large-uncased cls 0
bash cmd/extract_transformer_embs.sh bert-large-uncased mean 0

## If you have a pretrained Transformer model in save/baselines.pt/bert-base-uncased.v1:
bash cmd/extract_transformer_embs.sh save/baselines.pt/bert-base-uncased.v1 cls

# Sentence-Transformers
bash cmd/extract_sbert_embs.sh bert-base-nli-stsb-mean-tokens 0

# Static embeddings
bash cmd/extract_static_embs.sh word2vec
bash cmd/extract_static_embs.sh fasttext
```
