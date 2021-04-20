#!/usr/bin/env python
# coding: utf-8

# ## Run LRP on all test and validation results

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
TOTAL_EXAMPLES = 7000 #Total patients in val/test data

TARGET_COLNAME = "label"
UID_COLNAME = "patient_id"
TARGET_VALUE = "1"

# Results path for val & test data
output_dir = os.path.dirname(IMP_SAVE_DIR_PATTERN)
VAL_RESULTS_PATH = os.path.join(output_dir, f"val_all_lrp_{BEST_EPOCH}.pkl")
TEST_RESULTS_PATH = os.path.join(output_dir, f"test_all_lrp_{BEST_EPOCH}.pkl")


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

valid_dataset, vocab = build_lstm_dataset(
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

valid_dataloader = DataLoader(
    valid_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
)

test_dataloader = DataLoader(
    test_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
)


# ### Load Best Model

model_path = MODEL_SAVE_PATH_PATTERN.format(f"{BEST_EPOCH:02}")

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


valid_results_best = {}
valid_results_best[BEST_EPOCH] = {}

test_results_best = {}
test_results_best[BEST_EPOCH] = {}
# calculate relevancy and SHAP
lstm_model_best.eval()
lrp_model = LSTM_LRP_MultiLayer(lstm_model_best.cpu())

# Get test/val data
val_patient_ids, val_labels, val_idxed_text = get_eval_data(valid_dataloader, TOTAL_EXAMPLES)

test_patient_ids, test_labels, test_idxed_text = get_eval_data(test_dataloader, TOTAL_EXAMPLES)


start = time.time()

print('Processing validation data...')
for sel_idx in range(len(val_labels)):
    one_text = [
        int(token.numpy())
        for token in val_idxed_text[sel_idx]
        if int(token.numpy()) != 0
    ]
    lrp_model.set_input(one_text)
    lrp_model.forward_lrp()

    Rx, Rx_rev, _ = lrp_model.lrp(one_text, 0, eps=1e-6, bias_factor=0)
    R_words = np.sum(Rx + Rx_rev, axis=1)

    df = pd.DataFrame()
    df["lrp_scores"] = R_words
    df["idx"] = one_text
    df["seq_idx"] = [x for x in range(len(one_text))]
    df["token"] = [lrp_model.vocab.itos(x) for x in one_text]
    df["att_weights"] = lrp_model.get_attn_values()

    if val_patient_ids[sel_idx] not in valid_results_best[BEST_EPOCH]:
        valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]] = {}
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]] = {}
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]["label"] = val_labels[
        sel_idx
    ]
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]["pred"] = lrp_model.s[0]
    valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]["imp"] = df.copy()
    
    if sel_idx % 500 == 0:
        print(f'{sel_idx} of {TOTAL_EXAMPLES}')
            
end = time.time()
mins, secs = epoch_time(start, end)
print(f"Total Time: {mins}min: {secs}sec")


valid_results_best[BEST_EPOCH][val_patient_ids[sel_idx]]


with open(VAL_RESULTS_PATH, "wb") as fp:
    pickle.dump(valid_results_best, fp)

print('Processing test data...')
start = time.time()
for sel_idx in range(len(test_labels)):
    one_text = [
        int(token.numpy())
        for token in test_idxed_text[sel_idx]
        if int(token.numpy()) != 0
    ]
    lrp_model.set_input(one_text)
    lrp_model.forward_lrp()

    Rx, Rx_rev, _ = lrp_model.lrp(one_text, 0, eps=1e-6, bias_factor=0)
    R_words = np.sum(Rx + Rx_rev, axis=1)

    df = pd.DataFrame()
    df["lrp_scores"] = R_words
    df["idx"] = one_text
    df["seq_idx"] = [x for x in range(len(one_text))]
    df["token"] = [lstm_model_best.vocab.itos(x) for x in one_text]
    df["att_weights"] = lrp_model.get_attn_values()

    if test_patient_ids[sel_idx] not in test_results_best[BEST_EPOCH]:
        test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]] = {}
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]] = {}
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]]["label"] = test_labels[
        sel_idx
    ]
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]]["pred"] = lrp_model.s[0]
    test_results_best[BEST_EPOCH][test_patient_ids[sel_idx]]["imp"] = df.copy()
    
    if sel_idx % 500 == 0:
        print(f'{sel_idx} of {TOTAL_EXAMPLES}')

end = time.time()
mins, secs = epoch_time(start, end)
print(f"Total Time: {mins}min: {secs}sec")


with open(TEST_RESULTS_PATH, "wb") as fp:
    pickle.dump(test_results_best, fp)

print('Success!')



