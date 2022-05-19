#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import sys

from tqdm import tqdm
import numpy as np
import pandas as pd

from scripts.utils import init_logger
from scripts.utils import open_helper

verbose = False


def read_pas(filepath):
    with open_helper(filepath) as f:
        data = [json.loads(line) for line in f]
    if verbose:
        logger.info(f'Read {len(data)} records from {filepath}')
    buff = {'tok': [], 'predicate': [], 'target': [], 'prep': []}
    for record in data:
        buff['tok'].append(record['text'])
        buff['predicate'].append(record.get('predicate', np.nan))
        args = record.get('arguments', {})
        buff['target'].append('@@@'.join(args.get('dobj', args.get('root', []))))
        buff['prep'].append('@@@'.join(args.get('prep', [])))
    return pd.DataFrame.from_dict(buff)


def read_template(filepath: str):
    df = pd.read_csv(filepath, delimiter='\t')
    indices = df[pd.isnull(df['predicate'])&pd.isnull(df['prefix'])&pd.isnull(df['suffix'])].index
    df.drop(indices, inplace=True)
    if verbose:
        logger.info(f'Read {len(df)} templates from {filepath}')
    predicates, prefixes, suffixes = {}, {}, {}
    for _, row in df.iterrows():
        if isinstance(row['predicate'], str):
            predicates[row['list']] = row['predicate'].split('@@@')
        if isinstance(row['prefix'], str):
            prefixes[row['list']] = row['prefix'].split('@@@')
        if isinstance(row['suffix'], str):
            suffixes[row['list']] = row['suffix'].split('@@@')
    predicates = pd.DataFrame(predicates.items(), columns=['list', 'task.list.predicate'])
    prefixes = pd.DataFrame(prefixes.items(), columns=['list', 'task.list.prefix'])
    suffixes = pd.DataFrame(suffixes.items(), columns=['list', 'task.list.suffix'])
    return predicates, prefixes, suffixes


def main(args):
    global verbose
    verbose = args.verbose

    np.random.seed(42)

    pas = read_pas(args.path_pas).set_index('tok')
    t2pred = pas['predicate'].to_dict()
    t2prep = pas['prep'].to_dict()

    if args.path_input == '-':  # use stdin
        df = pd.read_csv(sys.stdin, delimiter='\t')
    else:
        df = pd.read_csv(args.path_input, delimiter='\t')
    df.drop_duplicates(inplace=True)
    if verbose:
        logger.info(f'Read {len(df)} rows from {args.path_input}')
    # Attach PAS
    for field in ['task', 'list']:
        cols = {'predicate': f'{field}.predicate',
                'target': f'{field}.target',
                'prep': f'{field}.prep'}
        df = pd.merge(df, pas.reset_index(), how='left',
                      left_on=f'{field}.tok', right_on='tok'
        ).drop('tok', axis=1).rename(columns=cols)

    # Attach list-based templates
    if args.path_list_template:
        predicates, prefixes, suffixes = read_template(args.path_list_template)
        df = pd.merge(df, predicates, on='list', how='left')
        df = pd.merge(df, prefixes, on='list', how='left')
        df = pd.merge(df, suffixes, on='list', how='left')

    # Read underspecified -> specified table
    table = pd.read_csv(args.path_table, delimiter='\t').drop_duplicates()
    if verbose:
        logger.info(f'Read {len(table)} rows from {args.path_table}')

    ## Create dicts
    tl2counts = {}
    t2counts = {}
    for _, row in tqdm(table.iterrows(), total=len(table)):
        underspecified_taskname = row['key']
        rel = row['rel']
        taskname, listname = row['task.tok'], row['list.tok']
        key = (underspecified_taskname, listname)
        if key not in tl2counts:
            tl2counts[key] = {}
        c = tl2counts[key].get((taskname, rel), 0)
        tl2counts[key][(taskname, rel)] = c + row['users']
        key = underspecified_taskname
        if key not in tl2counts:
            t2counts[key] = {}
        c = t2counts[key].get((taskname, rel), 0)
        t2counts[key][(taskname, rel)] = c + row['users']

    # Specifying tasks
    df['task.full.tok'] = df['task.tok']
    df['task.full.rel'] = 'original'
    df['task.full.freq'] = 1

    # Use list-based information
    has_predicate = df['task.predicate'].apply(
        lambda txt: isinstance(txt, str) and (len(txt) > 0))
    has_list_suffix = ~pd.isnull(df['task.list.suffix'])
    indices = df[has_predicate&has_list_suffix].index
    df.loc[indices, 'task.full.tok'] = df.loc[
        indices, ['task.full.tok', 'task.list.suffix']].apply(
            lambda t: '@@@'.join(t[0] + ' ' + txt.lower() for txt in t[1]), axis=1)

    # Tasks without action verb
    indices = df[~has_predicate].index
    if verbose:
        logger.info(f'Underspecified tasks: {len(indices)}')

    fullnames, rels, preds, preps, freqs = [], [], [], [], []
    for idx in tqdm(indices):
        taskname, listname = df.loc[idx, 'task.tok'], df.loc[idx, 'list.tok']
        cands = tl2counts.get((taskname, listname),
                              t2counts.get(taskname, {}))
        prefixes_ = df.loc[idx, 'task.list.prefix']
        suffixes_ = df.loc[idx, 'task.list.suffix']
        if len(cands) == 0:  # Not found
            if isinstance(df.loc[idx, 'task.list.predicate'], list) and len(df.loc[idx, 'task.list.predicate']) > 0:
                if not isinstance(prefixes_, list) or len(prefixes_) == 0:
                    prefixes_ = ['']
                if not isinstance(suffixes_, list) or len(suffixes_) == 0:
                    suffixes_ = ['']
                fullname = '@@@'.join(f'{pred} {prefix} {df.loc[idx, "task.tok"]} {suffix}'.strip().replace('  ', ' ')
                                      for pred in df.loc[idx, 'task.list.predicate']
                                      for prefix in prefixes_
                                      for suffix in suffixes_)
                fullnames.append(fullname)
                rels.append('list')
                preds.append('@@@'.join(f'{pred}'.strip().replace('  ', ' ')
                                        for pred in df.loc[idx, 'task.list.predicate']
                                        for prefix in (df.loc[idx, 'task.list.prefix'] if isinstance(df.loc[idx, 'task.list.prefix'], list) else [])
                                        for suffix in (df.loc[idx, 'task.list.suffix'] if isinstance(df.loc[idx, 'task.list.suffix'], list) else [])))
                preps.append('')
                freqs.append('')
            else:
                fullnames.append(np.nan)
                rels.append(np.nan)
                preds.append(np.nan)
                preps.append(np.nan)
                freqs.append(np.nan)
            continue

        if args.how == 'max':
            out, freqs_ = zip(*sorted(cands.items(),
                                     key=lambda t: (-t[1], t[0]))[:args.n_cands])
            taskname, rel = zip(*out)
        else:
            raise NotImplementedError
        preds_ = [t2pred[t] for t in taskname]
        preps_ = [t2prep[t] for t in taskname]

        if isinstance(suffixes_, list) and len(suffixes_) > 0:
            taskname = [t if len(t2prep[t]) > 0 else f'{t} {suffix}'
                        for t in taskname for suffix in suffixes_]

        fullnames.append('@@@'.join(taskname))
        rels.append('@@@'.join(rel))
        preds.append('@@@'.join(preds_))
        preps.append('@@@'.join(preps_))
        freqs.append('@@@'.join(map(str, freqs_)))

    df.loc[indices, 'task.full.tok'] = fullnames
    df.loc[indices, 'task.full.rel'] = rels
    df.loc[indices, 'task.predicate'] = preds
    df.loc[indices, 'task.prep'] = preps
    df.loc[indices, 'task.full.freq'] = freqs

    # Drop task names without fully-specified names
    indices = pd.isnull(df['task.full.tok'])
    if args.path_unspec:
        if verbose:
            logger.info(f'Write {int(indices.sum())} unspecified tasks to {args.path_unspec}')
            df[indices].to_csv(args.path_unspec, sep='\t', index=False)
    df = df[~indices]
    if verbose:
        logger.info(f'{len(df)} rows have specified task names')
        logger.info(f'Write the result to {args.path_output}')
    if args.path_output == '-':
        df.to_csv(sys.stdout, sep='\t', index=False)
    else:
        df.to_csv(args.path_output, sep='\t', index=False)

    return 0


if __name__ == '__main__':
    logger = init_logger('TaskSpec')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to an input file')
    parser.add_argument('--unspec', dest='path_unspec', help='path to save unspecified todo items')
    parser.add_argument('--pas', dest='path_pas', required=True, help='path to pas')
    parser.add_argument('--table', dest='path_table', required=True, help='path to table')
    parser.add_argument('--list-template', dest='path_list_template', default='data/listname_company/sample.tsv', help='path to a list-based template')
    parser.add_argument('--how', choices=['max', 'sample'], default='max')
    parser.add_argument('-n', '--n-cands', type=int, default=5, help='number of specified names')
    parser.add_argument('-o', '--output', dest='path_output', required=True, help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
