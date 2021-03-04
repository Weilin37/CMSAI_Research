import argparse

import sys

sys.path.append("../")

import os
import json
import time
import torch
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
import xgboost as xgb
import numpy as np
from numpy import newaxis as na

import deep_id_pytorch

from lstm_models import *
from att_lstm_models import *
from lstm_utils import *
from xgboost_utils import *

# from lrp_att_model import *
import shap_jacc_utils as sj_utils

from cdiff_utils import *

# import rbo

# Construct the argument parser
ap = argparse.ArgumentParser()

# Add the arguments to the parser
ap.add_argument("--model_path", required=True, help="Model Path")
ap.add_argument("--params_path", required=True, help="Model Params Path")
ap.add_argument("--train_path", required=True, help="Train Data Path")
ap.add_argument("--valid_path", required=True, help="Valid Data Path")
ap.add_argument("--test_path", required=True, help="Test Data Path")
ap.add_argument("--vocab_path", required=True, help="Vocab Path")
ap.add_argument("--valid_output_path", required=True, help="Output Path for Valid SHAP")
ap.add_argument("--test_output_path", required=True, help="Output Path for Test SHAP")
ap.add_argument("--seq_len", required=True, type=int, help="Max Sequence Length")
ap.add_argument("--nrows", required=False, default=1e9, type=int, help="Number of data rows to load")
ap.add_argument("--multigpu", required=False, default=False, type=int, help="Whether to use Multi-GPU")

args = vars(ap.parse_args())


def run_shap(model_path, train_path, valid_path, test_path, vocab_path, valid_output_path, test_output_path, params_path, 
             seq_len, nrows, multigpu=False, uid_colname='patient_id', 
             target_colname='d_00845', target_value="1"):
    """Run SHAP for a given model."""
    #Load Parameters
    with open(params_path, "r") as fp:
        MODEL_PARAMS = json.load(fp)

    #Load Vocab
    with open(vocab_path, 'rb') as fp:
        vocab = pickle.load(fp)

    #Load data
    train_dataset, _ = build_lstm_dataset(
        train_path,
        min_freq=MODEL_PARAMS["min_freq"],
        uid_colname=uid_colname,
        target_colname=target_colname,
        max_len=seq_len,
        target_value=target_value,
        vocab=vocab,
        nrows=MODEL_PARAMS['n_background'],
        rev=MODEL_PARAMS["rev"],
        cdiff=True,
    )
    valid_dataset, _ = build_lstm_dataset(
        valid_path,
        min_freq=MODEL_PARAMS["min_freq"],
        uid_colname=uid_colname,
        target_colname=target_colname,
        max_len=seq_len,
        target_value=target_value,
        vocab=vocab,
        nrows=MODEL_PARAMS['batch_size'],
        rev=MODEL_PARAMS["rev"],
    )
    test_dataset, _ = build_lstm_dataset(
        test_path,
        min_freq=MODEL_PARAMS["min_freq"],
        uid_colname=uid_colname,
        target_colname=target_colname,
        max_len=seq_len,
        target_value=target_value,
        vocab=vocab,
        nrows=MODEL_PARAMS['batch_size'],
        rev=MODEL_PARAMS["rev"],
    )
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
    )

    val_patient_ids, val_labels, val_idxed_text = next(iter(valid_dataloader))
    test_patient_ids, test_labels, test_idxed_text = next(iter(test_dataloader))

    #Load Model
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lstm_model = AttNoHtLSTM(
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
    
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model = lstm_model.to(model_device)

    #Delete output directory if exists
    shap_dir = os.path.dirname(valid_output_path)
    if os.path.exists(shap_dir):
        shutil.rmtree(shap_dir)
    if not multigpu:
        (
            val_features,
            val_scores,
            val_patients,
        ) = sj_utils.get_lstm_features_and_shap_scores(
            lstm_model.cuda(),
            train_dataloader,
            valid_dataloader,
            seq_len,
            valid_output_path,
            save_output=True,
            n_background=MODEL_PARAMS["n_background"],
            background_negative_only=MODEL_PARAMS["background_negative_only"],
            n_test=MODEL_PARAMS["n_valid_examples"],
            test_positive_only=MODEL_PARAMS["test_positive_only"],
            is_test_random=MODEL_PARAMS["is_test_random"],
        )

        (
            test_features,
            test_scores,
            test_patients,
        ) = sj_utils.get_lstm_features_and_shap_scores(
            lstm_model.cuda(),
            train_dataloader,
            test_dataloader,
            seq_len,
            test_output_path,
            save_output=True,
            n_background=MODEL_PARAMS["n_background"],
            background_negative_only=MODEL_PARAMS["background_negative_only"],
            n_test=MODEL_PARAMS["n_valid_examples"],
            test_positive_only=MODEL_PARAMS["test_positive_only"],
            is_test_random=MODEL_PARAMS["is_test_random"],
        )
    else:
        MULTIGPU_LST = []
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            for gpu in range(1, n_gpus):
                MULTIGPU_LST.append(f"cuda:{gpu}")
        (
            val_features,
            val_scores,
            val_patients,
        ) = get_lstm_features_and_shap_scores_mp(
            lstm_model.cpu(),
            train_dataloader,
            (val_patient_ids, val_labels, val_idxed_text),
            seq_len,
            valid_output_path,
            save_output=True,
            n_background=MODEL_PARAMS["n_background"],
            background_negative_only=MODEL_PARAMS["background_negative_only"],
            test_positive_only=MODEL_PARAMS["test_positive_only"],
            is_test_random=MODEL_PARAMS["is_test_random"],
            multigpu_lst=MULTIGPU_LST,
        )

        (
            test_features,
            test_scores,
            test_patients,
        ) = get_lstm_features_and_shap_scores_mp(
            lstm_model.cpu(),
            train_dataloader,
            (test_patient_ids, test_labels, test_idxed_text),
            seq_len,
            test_output_path,
            save_output=True,
            n_background=MODEL_PARAMS["n_background"],
            background_negative_only=MODEL_PARAMS["background_negative_only"],
            test_positive_only=MODEL_PARAMS["test_positive_only"],
            is_test_random=MODEL_PARAMS["is_test_random"],
            multigpu_lst=MULTIGPU_LST,  # ["cuda:2", "cuda:3", "cuda:1"],
        )

    print(f'SHAP Successfully Computed and Saved to {os.path.dirname(test_output_path)}!')
    
        
if __name__ == "__main__":
    print(f'Computing SHAP Values for model {args["model_path"]}...')
    run_shap(model_path=args['model_path'], 
             train_path=args['train_path'], 
             valid_path=args['valid_path'], 
             test_path=args['test_path'], 
             vocab_path=args['vocab_path'],
             valid_output_path=args['valid_output_path'], 
             test_output_path=args['test_output_path'], 
             params_path=args['params_path'], 
             seq_len=args['seq_len'],
             nrows=args['nrows'],
             multigpu=args['multigpu'],
            )
    print('Success!')

