#!/usr/bin/env python
# coding: utf-8


import sys
import os
import io
import boto3
import json
import time
import pickle
from functools import reduce
from operator import mul
import numpy as np
import pandas as pd
import subprocess


def call(cmd):
    return subprocess.call(cmd.split())


def to_json(o, bucket, key):
    with io.StringIO() as f:
        json.dump(o, f)
        f.seek(0)
        boto3.resource("s3").Object(bucket, key).put(Body=f.getvalue())


def print_dir(path):
    print(subprocess.call(["ls", "-lR", path]))


def print_dict(title, d):
    print(title)
    for k, v in d.items():
        print(f"{k:<16}:{v}")


def load_pickle(filename):
    with open(filename, "rb") as f:
        print(f"Load {filename}...")
        return pickle.load(f)


def dump_pickle(o, filename):
    with open(filename, "wb") as f:
        print(f"Write {filename}...")
        pickle.dump(o, f)


def dump_results(pids, event_names, labels, scores, filepath):
    """Dump results.

    pids        : list of patient ids
    event_names : list of event names
    labels      : numpy array of binary labels of shape (n_patients, n_classes)
    scores      : numpy array of scores, same shape
    filepath    : path where the csv will be dumped
    """
    event_cols = [f"e_{i}" for i in event_names]
    score_cols = [f"s_{i}" for i in event_names]
    df = pd.concat(
        [
            pd.DataFrame(pids, columns=["patient_id"]),
            pd.DataFrame(labels.astype(int), columns=event_cols),
            pd.DataFrame(scores, columns=score_cols),
        ],
        axis=1,
    )
    df.to_csv(filepath, index=False)


def load_results(filepath):
    df = pd.read_csv(filepath)
    event_names = [c[2:] for c in df.columns if c.startswith("e_")]
    event_cols = [c for c in df.columns if c.startswith("e_")]
    score_cols = [c for c in df.columns if c.startswith("s_")]
    return (
        df["patient_id"].to_list(),
        event_names,
        df[event_cols].values,
        df[score_cols].values,
    )


def cache_path(source_list):
    import hashlib

    s = "+".join([str(s) for s in source_list])
    return "cache_{}.pkl".format(hashlib.md5(s.encode()).hexdigest())


def print_model(model):
    print(model)
    print(
        "Total number of parameters",
        sum(map(lambda p: reduce(mul, p.shape), model.parameters())),
    )


def label_binarize_simple(y, classes):
    yb = np.zeros((len(y), len(classes)), np.int8)
    for i, cl in enumerate(classes):
        yb[:, i] = y == cl
    return yb


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
