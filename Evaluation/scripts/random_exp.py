#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict
import argparse
import logging
import os
import pickle
import random
import warnings
warnings.filterwarnings('ignore')  # do not show scikit-learn warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from tqdm import trange
import numpy as np
import pandas as pd

from scripts.utils import init_logger

verbose = False
logger = None

CLFS = {
    'lr': LogisticRegression(max_iter=1000),
    'mlp': MLPClassifier(batch_size=200, max_iter=1000, random_state=42)
}
PARAM_GRIDS = {
    'lr': [
        {
            'C': [2**(-i) for i in range(6)],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        {
            'C': [2**(-i) for i in range(6)],
            'penalty': ['l1'],
            'solver': ['liblinear']
        },
        {
            'penalty': ['none']
        }
    ],
    'mlp': [
        {
            'hidden_layer_sizes': [[dim for _ in range(depth)]
                                   for dim in [128, 256, 512]
                                   for depth in [1, 2]],
            'activation': ['tanh', 'relu']
        }
    ]
}


def main(args):
    global verbose
    verbose = args.verbose

    random.seed(42)
    np.random.seed(42)

    model_name = f'{args.model}.{args.version}'
    dir_in = os.path.join('data/out', args.model, args.version)
    dir_out = os.path.join('evaluation', model_name)
    os.makedirs(dir_out, exist_ok=True)
    if verbose:
        logger.info(f'Model: {model_name}')
        logger.info(f'Input: {dir_in}')
        logger.info(f'Output directory: {dir_out}')

    # Read data
    if args.task == 'ld2018':
        paths_in = [os.path.join(dir_in, 'ld2018_train_embs.txt')]
        with open(os.path.join(dir_in, 'ld2018_train_embs.txt')) as f:
            next(f)
            data = [line.strip().split(' ') for line in f]
        df = pd.DataFrame([row[0].split('@@@') for row in data], columns=['taskname', 'label'])
        X = np.array([[float(v) for v in row[1:]] for row in data])
        indexer = {
            'buy': 0,
            'calendar': 1,
            'call': 2,
            'contact': 3,
            'email': 4,
            'find-service': 5,
            'pay-bill-online': 6,
            'postal': 7,
            'service': 8
        }
        idx2label = [key for key, val in sorted(indexer.items(), key=lambda t: t[1])]
        y = df['label'].apply(lambda val: indexer[val.lower()]).values
    else:
        splits = ['train']
        if args.task in ['coloc', 'cotim']:  # CoLoc and CoTim has a dev split
            splits.append('dev')
        if args.all:
            if verbose:
                logger.info('Use test splits')
            splits.append('test')
        if args.no_list:
            paths_in = [os.path.join(dir_in, f'{args.task}_nolist_{split}_embs.tsv')
                        for split in splits]
        else:
            paths_in = [os.path.join(dir_in, f'{args.task}_{split}_embs.tsv')
                        for split in splits]
        df = pd.concat([pd.read_csv(path_in, delimiter='\t', header=None)
                        for path_in in paths_in])
        if args.task in {'coloc', 'cotim'}:
            X = df.iloc[:, 5:].values
            dim = X.shape[1] // 2
            assert dim * 2 == X.shape[1]
            diff = X[:, :dim] - X[:, dim:]
            mult = X[:, :dim] * X[:, dim:]
            X = np.hstack([X, diff, mult])
            y = df.iloc[:, 0].apply(lambda v: str(v).strip().lower() == 'true').astype(int).values
    num_test = int(args.test_ratio * len(X))
    if verbose:
        logger.info(f'Read {len(df)} instances from {paths_in}')
        logger.info(f'Test data: {num_test}')

    if verbose:
        logger.info(f'Start experiments: # of trials = {args.num_trials}')
        logger.info(f'Classifier: {args.clf}')

    indices = np.arange(X.shape[0])
    param_grid = PARAM_GRIDS[args.clf]
    clf = CLFS[args.clf]
    scoring = {'coloc': 'f1',
               'cotim': 'f1',
               'ld2018': 'accuracy'}[args.task]
    performance = defaultdict(list)
    predictions = defaultdict(list)

    loaded_indices = None
    if args.path_index:
        if verbose:
            logger.info(f'Load {args.path_index}')
        with open(args.path_index, 'rb') as f:
            texts, loaded_indices = pickle.load(f)
    for t in trange(args.num_trials):
        if loaded_indices is None:
            np.random.shuffle(indices)
            indices_train, indices_test = indices[:-num_test], indices[-num_test:]
            num_dev = num_test
        else:
            _indices_train, indices_dev, indices_test = loaded_indices[t]
            num_dev = len(indices_dev)
            num_test = len(indices_test)
            indices_train = np.hstack([_indices_train, indices_dev])
        X_train, y_train = X[indices_train], y[indices_train]
        num_train = X_train.shape[0]
        X_test, y_test = X[indices_test], y[indices_test]

        # Grid search with a fixed split (train, dev)
        # We could do CV here, but it will be time-consuming
        # and probably won't be necessary if num_trials is sufficiently large.
        gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=scoring,
                          cv=[(np.arange(num_train-num_dev),
                               np.arange(num_train-num_dev, num_train))],
                          verbose=0, n_jobs=args.num_jobs)
        gs.fit(X_train, y_train)
        if verbose:
            tqdm.write(f'[{t}] Best score: {gs.best_score_:.4f} / Best params: {gs.best_params_}')
        performance['best_validation_score'].append(gs.best_score_)
        y_pred_proba = gs.best_estimator_.predict_proba(X_test)
        y_pred = gs.best_estimator_.predict(X_test)
        if args.task not in {'ld2018'}:  # binary classification
            # For binary classification
            prec, rec, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred,
                                                               average='binary')
            performance['precision'].append(prec)
            performance['recall'].append(rec)
            performance['f1'].append(f1)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true=y_test, y_pred=y_pred,
                                                           average='macro')
        performance['macro_precision'].append(prec)
        performance['macro_recall'].append(rec)
        performance['macro_f1'].append(f1)
        performance['accuracy'].append(accuracy_score(y_true=y_test, y_pred=y_pred))

        conf = confusion_matrix(y_true=y_test, y_pred=y_pred).flatten()
        for conf_idx, conf_val in enumerate(conf):
            performance[f'confusion_matrix.{conf_idx}'].append(conf_val)

        if not args.no_pred:
            for pred_idx, pred in enumerate(y_pred):
                idx = indices_test[pred_idx]
                predictions['trial'].append(t)
                row = df.iloc[idx]
                if args.task == 'ld2018':
                    predictions['task'].append(row[0])
                    predictions['list'].append('inbox')
                    predictions['gold'].append(row[1].lower())
                    predictions['pred'].append(idx2label[pred])
                elif args.task in {'coloc', 'cotim'}:
                    predictions['task1'].append(row[1])
                    predictions['list1'].append(row[2])
                    predictions['task2'].append(row[3])
                    predictions['list2'].append(row[4])
                    predictions['gold'].append(str(row[0]).lower() == 'true')
                    predictions['pred'].append(bool(pred))
                predictions['score'].append(y_pred_proba[pred_idx][pred])

    # Output
    for key, vals in performance.items():
        mean = np.mean(vals)
        std = np.std(vals)
        performance[key].append(mean)
        performance[key].append(std)
        if verbose:
            logger.info(f'{key}: {mean:.4f} (+/- {std:.4f})')
    out = pd.DataFrame.from_dict(performance)
    out.index = list(range(args.num_trials)) + ['mean', 'std']
    path_out = os.path.join(dir_out, f'{args.task}_{args.clf}_report.csv')
    if verbose:
        logger.info(f'Write the result to {path_out}')
    out.to_csv(path_out)

    if not args.no_pred:
        path_out = os.path.join(dir_out, f'{args.task}_{args.clf}_clf.csv')
        if verbose:
            logger.info(f'Write the predictions to {path_out}')
        pd.DataFrame.from_dict(predictions).to_csv(path_out, index=False)

    return 0


if __name__ == '__main__':
    logger = init_logger('Exp')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='baselines', help='model type')
    parser.add_argument('--clf', choices=list(CLFS.keys()), default='lr', help='classifier')
    parser.add_argument('--version', default='bert-base-uncased_cls', help='model version')
    parser.add_argument('--no-list', action='store_true', help='no list in input')
    parser.add_argument('--task', choices=['coloc', 'cotim', 'ld2018'], default='ld2018')
    parser.add_argument('--test', dest='test_ratio', type=float, default=0.2, help='ratio of test data')
    parser.add_argument('-t', '--trials', dest='num_trials', type=int, default=20, help='number of trials')
    parser.add_argument('--index', dest='path_index', help='data indices')
    parser.add_argument('--no-pred', action='store_true', help='do not write out predictions')
    parser.add_argument('-o', '--output', dest='path_output', help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('--all', action='store_true', help='use all data splits')
    parser.add_argument('-j', '--jobs', dest='num_jobs', type=int, default=10, help='number of jobs')
    args = parser.parse_args()
    main(args)
