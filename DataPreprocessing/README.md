# Process Wunderlist dataset

## Setup

Requirements

- python 3.8
- GNU parallel
- [comet-atomic-2020](https://github.com/allenai/comet-atomic-2020)
- [open-sesame](https://github.com/swabhs/open-sesame)

``` shell
conda env create -f environment.yml
conda activate tododata
```

Starting file: `data/raw/sample.tsv`

## Basic analysis

### POS tagging

```shell
# Extract task descriptions and list names
cat <(cat data/raw/sample.tsv | sed -e '1d' | cut -f 1) <(cat data/raw/sample.tsv | sed -e '1d' | cut -f 2) | sort | uniq > data/raw/texts.txt

# Tokenize the text by spaCy
mkdir data/tagged
python scripts/tokenize_and_tag.py < data/raw/texts.txt > data/tagged/texts.uniq.tsv

# Attach POS tags to the original file
python scripts/attach_tag.py data/raw/sample.tsv --tagged data/tagged/texts.uniq.tsv -o data/tagged/sample.tsv -v
```

### Dependency parsing

```shell
python scripts/run_depparse.py data/tagged/sample.tsv --field task list -o data/tagged/texts.uniq.conllu -v
```

### PAS

```shell
mkdir data/pas
python scripts/extract_predicate_argument_from_conllu.py data/tagged/texts.uniq.conllu -o data/pas/sample.json -v
```

## Specifying task names

Make a specification table (key -> full name)

```shell
mkdir data/taskspec/
python scripts/extract_task_specification.py data/tagged/sample.tsv --pas data/pas/sample.json -o data/taskspec/sample.table.tsv -v

python scripts/create_task_specification_data.py data/tagged/sample.tsv --pas data/pas/sample.json --table data/taskspec/sample.table.tsv -o data/taskspec/sample.spec.tsv --unspec data/taskspec/sample.unspec.tsv -v
# data/taskspec/sample.unspec.tsv contains todo items that do not have longer counterparts. This is for sanity-check.


# Extract specified task descriptions for collecting supervision from COMET and FrameNet (next sections)
python scripts/extract_task-full.py data/taskspec/sample.spec.tsv -o data/taskspec/tasks.uniq.txt -v

# We sampled a subset of the wunderlist dataset by using scripts/sample_dataset_tsv.py in the main experiments
python scripts/sample_dataset_tsv.py <input> -n <n of lines> -o <output> -v
```


## Intent-focused tasks

### COMET

```shell
mkdir data/comet

# Use COMET (https://github.com/allenai/comet-atomic-2020) to obtain pre-action events and goals for data/taskspec/tasks.uniq.txt
# See data/taskspec/tasks.uniq.comet_atomic2020_bart_xneed_xwant_top3.tsv for the format of COMET output.

python scripts/create_dataset.py data/taskspec/sample.spec.tsv --comet data/taskspec/tasks.uniq.comet_atomic2020_bart_xneed_xwant_top3.tsv -o data/comet/sample.json -v
```


## FrameNet

``` shell
mkdir data/framenet

# Run Open-SESAME (https://github.com/swabhs/open-sesame) or any frame-semantic parser on data/taskspec/tasks.uniq.txt
# Obtain a result in the CoNLL-U format (see data/framenet/sample.frameid.connlu)

python scripts/convert_frameid_result.py data/framenet/sample.frameid.connlu -o data/framenet/sample.frameid.tsv -v

python scripts/attach_framenet.py data/comet/sample.json --framenet data/framenet/sample.frameid.tsv -o data/framenet/sample.json -v
```


## DataSplit

```shell
mkdir data/dataset

python scripts/split_dataset_json.py data/framenet/sample.json -o data/dataset/ --num-valid 2 --num-test 2 -v
# Note: As the sample file is very small, the number of validation and test examples is set to 2 in this example.
```

Now `data/dataset/{trn,vld,tst}.json` can be used for training an encoder (Seet `../TaskReprLearning`).
