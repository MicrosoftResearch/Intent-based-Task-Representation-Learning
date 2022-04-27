#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional
import json

from baseline_encoders.utils import open_helper


def read_CoTL(filepath: str):
    with open_helper(filepath) as f:
        next(f)  # Skip header
        for line in f:
            label, task1, list1, task2, list2 = line.strip('\n').split('\t')
            list1 = list1.replace('default list', 'inbox')
            list2 = list2.replace('default list', 'inbox')
            yield task1, list1, task2, list2, label

def read_LD2018(filepath: str,
                listname: Optional[str] = None):
    with open_helper(filepath) as f:
        for line in f:
            row = json.loads(line)
            label = row['class-label']
            task = row['instance']['panon']['text']
            yield task, listname, label

def read_Tasks(filepath: str):
    with open_helper(filepath) as f:
        for line in f:
            task, listname, label = line.strip('\n').split('\t')
            yield task, listname, label
