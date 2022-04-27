# Perform downstream tasks

# Setup

``` shell
conda env create -f environment.yml
conda activate todoeval
```

# 1. Generate random splits

Before running the commands below, follow `../TaskReprLearning/README.md` to download the data to `../TaskReprLearning/data/{CoTL/landes_dieugenio_2018}`

```shell
python scripts/generate_random_splits.py ../TaskReprLearning/data/CoTL/StratTaskRand/Loc{Train,Dev,Test}.txt.tsv -o data/splits.all --task coloc --stratified -v  # 25000 instances (5000 test)
python scripts/generate_random_splits.py ../TaskReprLearning/data/CoTL/StratTaskRand/Tim{Train,Dev,Test}.txt.tsv -o data/splits.all --task cotim --stratified -v  # 25000 instances (5000 test)
python scripts/generate_random_splits.py ../TaskReprLearning/data/landes_dieugenio_2018/todo-dataset.json -o data/splits.all --task ld2018 --test 0.1 --stratified -v  # 253 instances (25 test)
```

The processed data is saved to `data/splits.all`.

Note: we did not use the test split of CoTL when selecting the architecture of a classifier.


# 2. Run experiments

Assume the intent embeddings (see `../TaskReprLearning/README.md`) are saved under `data/out/{model}/{version}/{task}_{train,dev,test}_embs.tsv`.

```shell
# Proposed method (example)
bash cmd/run_random_exp.sh simple_enc bert-base-uncased.v1 data/splits.all 10 --all
# (Main script: scripts/random_exp.py)

# Baselines
bash cmd/run_random_exp.sh baselines bert-base-uncased_cls data/splits.all 5 --all
bash cmd/run_random_exp.sh baselines bert-large-uncased_cls data/splits.all 5 --all
bash cmd/run_random_exp.sh baselines roberta-base_mean data/splits.all 5 --all
bash cmd/run_random_exp.sh baselines word2vec_mean data/splits.all 5 --all
bash cmd/run_random_exp.sh baselines fasttext_mean data/splits.all 5 --all
```

Permutation test

```
bash cmd/run_permtest.sh bert-large-uncased v17.4.1 ld2018; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 uit; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 at; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 coloc; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 cotim
bash cmd/run_permtest.sh bert-large-uncased v17.4.1 ld2018; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 uit; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 at; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 coloc; bash cmd/run_permtest.sh bert-large-uncased v17.4.1 cotim
```

The results are saved to `evaluation/permtest/`.

