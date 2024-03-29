"""Common utility functions."""
import os
import json
import gzip
import spacy
import pickle
import random
import numpy as np


def get_abs_path(save_dir, file_name):
    """Get absolute path of a file."""
    __file__ = os.path.join(save_dir, file_name)
    path = os.path.abspath(__file__)
    return path


def save_pickle(content, path, verbose=True):
    """Save pickle object."""
    pickle.dump(content, open(path, "wb"))
    if verbose:
        print("saved {} pickle..".format(path))


def load_pickle(path, encoding=True):
    """Load pickle object."""
    if encoding == False:
        return pickle.load(open(path, "rb"), encoding="latin1")
    else:
        return pickle.load(open(path, "rb"))


def write_json_list(data, filename):
    "Write data to json"
    with gzip.open(filename, "wt") as fout:
        for d in data:
            fout.write("%s\n" % json.dumps(d))


def load_json_list(filename):
    """Load data from json file"""
    data = []
    with gzip.open(filename, "rt") as fin:
        for line in fin:
            data.append(json.loads(line))
    return data


def create_json_list(data, out_path):
    """Create json file for a list"""
    file = gzip.open(out_path, "w+")
    for i in data:
        i_str = json.dumps(i) + "\n"
        i_bytes = i_str.encode("utf-8")
        file.write(i_bytes)
    file.close()
