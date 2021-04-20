#!/usr/bin/env python
# coding: utf-8

# ## Run SHAP on all test and validation results


import sys
import os
import json
import time
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from urllib.parse import urlparse
import tarfile
import pickle
import shutil
import matplotlib.pyplot as plt
import shap
import numpy as np

import deep_id_pytorch

from lstm_models import *
from lstm_lrp_models import *
from lstm_att_models import *
from lstm_self_att_models import *
from lstm_utils import *
from imp_utils import *


IS_SYNTHETIC = True  # If dataset is synthetic/real
# MODEL_NAME = 'lstm'
MODEL_NAME = "lstm"
USE_SELF_ATTENTION = True

NROWS = 1e9

TRAIN_MODEL = True
SEQ_LEN = 30
DATA_TYPE = "event"  # event/sequence

TRAIN_DATA_PATH = f"../data/synthetic/sample_dataset/{DATA_TYPE}/{SEQ_LEN}/train.csv"
VALID_DATA_PATH = f"../data/synthetic/sample_dataset/{DATA_TYPE}/{SEQ_LEN}/val.csv"
TEST_DATA_PATH = f"../data/synthetic/sample_dataset/{DATA_TYPE}/{SEQ_LEN}/test.csv"
VOCAB_PATH = f"../data/synthetic/sample_dataset/{DATA_TYPE}/{SEQ_LEN}/vocab.pkl"

MODEL_SAVE_PATH_PATTERN = (
    f"./output/synthetic/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/model_weights/model_{'{}'}.pkl"
)
IMP_SAVE_DIR_PATTERN = f"./output/synthetic/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/importances/{'{}'}_imp_{'{}'}.pkl"  # Feature importance values path for a given dataset split

OUTPUT_RESULTS_PATH = (
    f"./output/synthetic/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/train_results/results.csv"
)
PARAMS_PATH = (
    f"./output/synthetic/{DATA_TYPE}/{SEQ_LEN}/{MODEL_NAME}/train_results/model_params.json"
)


BEST_EPOCH = 2

TARGET_COLNAME = "label"
UID_COLNAME = "patient_id"
TARGET_VALUE = "1"

DATA_SPLIT = 'test'
TOTAL_EXAMPLES = 7000

# Results path the val/test data
output_dir = os.path.dirname(IMP_SAVE_DIR_PATTERN)
SPLIT_RESULTS_PATH = os.path.join(output_dir, f"{DATA_SPLIT}_all_shap_{BEST_EPOCH}.pkl")

   
def run_a_batch(start_idx, end_idx, lstm_model, tr_dataloader, patient_ids, labels, idxed_text):
    start = time.time()
    (
        features,
        scores,
        patients,
    ) = get_lstm_features_and_shap_scores_mp(
        lstm_model_best,
        tr_dataloader,
        (patient_ids[start_idx:end_idx], 
         labels[start_idx:end_idx], 
         idxed_text[start_idx:end_idx]),
        SEQ_LEN,
        "",
        save_output=False,
        n_background=MODEL_PARAMS["n_background"],
        background_negative_only=MODEL_PARAMS["background_negative_only"],
        test_positive_only=MODEL_PARAMS["test_positive_only"],
        is_test_random=MODEL_PARAMS["is_test_random"],
        multigpu_lst=["cuda:2", "cuda:3", "cuda:1"],
    )

    start_idx = end_idx
    end = time.time()
    mins, secs = epoch_time(start, end)
    print(f"{end_idx} --> {mins}min: {secs}sec")
    
    
    return (copy.deepcopy(features),
            copy.deepcopy(scores),
            copy.deepcopy(patients))


# ### Load Vocab and Dataset

# Load model params
MODEL_PARAMS = None
with open(PARAMS_PATH, "r") as fp:
    MODEL_PARAMS = json.load(fp)
MODEL_PARAMS


if os.path.exists(VOCAB_PATH):
    with open(VOCAB_PATH, "rb") as fp:
        vocab = pickle.load(fp)
    print(f"vocab len: {len(vocab)}")  # vocab + padding + unknown
else:
    raise ValueError(
        "Vocab path does not exist! Please create vocab from training data and save it first."
    )

train_dataset, _ = build_lstm_dataset(
    TRAIN_DATA_PATH,
    min_freq=MODEL_PARAMS["min_freq"],
    uid_colname=UID_COLNAME,
    target_colname=TARGET_COLNAME,
    max_len=SEQ_LEN,
    target_value=TARGET_VALUE,
    vocab=vocab,
    nrows=NROWS,
    rev=MODEL_PARAMS["rev"],
)

train_dataloader = DataLoader(
    train_dataset, batch_size=MODEL_PARAMS["batch_size"], 
    shuffle=True, num_workers=2
)

num_lim_per_epoch = 200
upper_lim = TOTAL_EXAMPLES
start_idx = 0
batch = list(range(num_lim_per_epoch, upper_lim + 1, num_lim_per_epoch))

if DATA_SPLIT == 'val':
    valid_dataset, _ = build_lstm_dataset(
        VALID_DATA_PATH,
        min_freq=MODEL_PARAMS["min_freq"],
        uid_colname=UID_COLNAME,
        target_colname=TARGET_COLNAME,
        max_len=SEQ_LEN,
        target_value=TARGET_VALUE,
        vocab=vocab,
        nrows=NROWS,
        rev=MODEL_PARAMS["rev"],
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=MODEL_PARAMS["batch_size"], 
        shuffle=False, num_workers=2
    )

    patient_ids, labels, idxed_text = get_eval_data(valid_dataloader, upper_lim)
    
else:
    test_dataset, _ = build_lstm_dataset(
        TEST_DATA_PATH,
        min_freq=MODEL_PARAMS["min_freq"],
        uid_colname=UID_COLNAME,
        target_colname=TARGET_COLNAME,
        max_len=SEQ_LEN,
        target_value=TARGET_VALUE,
        vocab=vocab,
        nrows=NROWS,
        rev=MODEL_PARAMS["rev"],
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=MODEL_PARAMS["batch_size"], 
        shuffle=False, num_workers=2
    )

    patient_ids, labels, idxed_text = get_eval_data(test_dataloader, upper_lim)

print(f'Processing {DATA_SPLIT} data...')

# ### Load Best Model

model_path = MODEL_SAVE_PATH_PATTERN.format(f"{BEST_EPOCH:02}")
model_path

# Check if cuda is available
print(f"Cuda available: {torch.cuda.is_available()}")
model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lstm_model_best = AttNoHtLSTM(
    MODEL_PARAMS["embedding_dim"],
    MODEL_PARAMS["hidden_dim"],
    vocab,
    model_device,
    bidi=MODEL_PARAMS["bidirectional"],
    nlayers=MODEL_PARAMS["nlayers"],
    dropout=MODEL_PARAMS["dropout"],
    init_type=MODEL_PARAMS["init_type"],
    linear_bias=MODEL_PARAMS["linear_bias"],
)

lstm_model_best.load_state_dict(torch.load(model_path))


results_best = {}
results_best[BEST_EPOCH] = {}


all_features = []
all_scores = []
all_patients = []
for end_idx in batch:
    (
        features,
        scores,
        patients,
    ) = run_a_batch(start_idx, end_idx, lstm_model_best, train_dataloader, patient_ids, labels, idxed_text)
    start_idx = end_idx
    
    all_features = all_features + features
    all_scores = all_scores + scores
    all_patients = all_patients + patients
    
    torch.cuda.empty_cache()

#Create dictionary
for idx, pid in enumerate(all_patients):
    if pid not in results_best[BEST_EPOCH].keys():
        results_best[BEST_EPOCH][pid] = {}
        
    if "imp" not in results_best[BEST_EPOCH][pid].keys():
        df = pd.DataFrame()
        df['token'] = all_features[idx]
        df['seq_idx'] = [x for x in range(len(all_features[idx]))]
    else:
        df = results_best[BEST_EPOCH][pid]["imp"]    
    df["shap_scores"] = all_scores[idx]
    
    results_best[BEST_EPOCH][pid]["imp"] = df.copy()
    
    if idx % 500 == 0:
        print(f'{idx} of {TOTAL_EXAMPLES}')

os.makedirs(os.path.dirname(SPLIT_RESULTS_PATH), exist_ok=True)
with open(SPLIT_RESULTS_PATH, 'wb') as fp:
    pickle.dump(results_best, fp)
print("Successfully Completed!")
                            
