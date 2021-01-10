#!/usr/bin/env python
# coding: utf-8

# # LSTM+Attention Model Training and SHAP computation using the Synthetic-events Dataset
# 
# **Author: Tesfagabir Meharizghi<br>Last Updated: 01/07/2021**
# 
# This notebook does the following actions:
# - Model training using the given parameters
# - Model selection using Intersection Similarity Score between ground truth helping features and predicted ones
#     * Early stopping using Intersection similarity score criteria
# - Computes SHAP values and visualizes for a few examples
# - Visualizes the train/val/test probability scores from each trained model
# - Visualizes the Intersection Similarity Scores for val/test splits
# - Finally, after tweaking the parameters, it gets the best model for the given model architecture and dataset
# 
# Outputs:
# - The following artifacts are saved:
#     * Model artifacts
#     * SHAP values and their corresponding scores for the specified number of val/test examples
# 
# Model Architecture Used:
# - LSTM+Attention
# 
# Dataset:
# - Synthetic-events (Toy Dataset)
# 
# Requirements:
# - Make sure that you have already generated the synthetic toy dataset (train/val/test splits) using [Create_toy_dataset.ipynb](../../data/toy_dataset/Create_toy_dataset.ipynb).
# 
# Next Steps:
# - Once you train different models, save the best one you found
# - Do also the same for other models architectures (SimpleLSTM, XGB, etc.) using the separate notebooks
# - Finally, go to [this ipynb]() to compare to compare the models' performances and shap values usig Jaccard Similarity Index


# In[1]:


# pip install nb-black

#! pip install botocore==1.12.201

#! pip install shap
#! pip install xgboost


# In[2]:


#get_ipython().run_line_magic('load_ext', 'lab_black')

# Some issue with reload when using with Jupyter Lab:
# https://stackoverflow.com/questions/43751455/supertype-obj-obj-must-be-an-instance-or-subtype-of-type/52927102#52927102

#%load_ext autoreload
#%autoreload 2


# In[3]:


import os
import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from urllib.parse import urlparse
import tarfile
import pickle
import shutil
from collections import Counter, defaultdict, OrderedDict

import shap
import xgboost as xgb

import sagemaker
import boto3
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.image_uris import retrieve

import deep_id_pytorch

import lstm_utils as l_utils
import att_lstm_models as lstm_att
import shap_jacc_utils as sj_utils


def load_data(train_data_path, 
              valid_data_path, 
              test_data_path,
              seq_len,
              min_freq, 
              batch_size, 
              target_colname,
              uid_colname,
              target_value,
              nrows,
              rev):
    """Load data."""
    train_dataset, vocab = l_utils.build_lstm_dataset(
        train_data_path,
        min_freq=min_freq,
        uid_colname="patient_id",
        target_colname="label",
        max_len=seq_len,
        target_value=target_value,
        vocab=None,
        nrows=nrows,
        rev=rev,
    )
    valid_dataset, _ = l_utils.build_lstm_dataset(
        valid_data_path,
        min_freq=min_freq,
        uid_colname="patient_id",
        target_colname="label",
        max_len=seq_len,
        target_value=target_value,
        vocab=vocab,
        nrows=nrows,
        rev=rev,
    )

    test_dataset, _ = l_utils.build_lstm_dataset(
        test_data_path,
        min_freq=min_freq,
        uid_colname="patient_id",
        target_colname="label",
        max_len=seq_len,
        target_value=target_value,
        vocab=vocab,
        nrows=nrows,
        rev=rev,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return (train_dataloader, valid_dataloader, test_dataloader, vocab)


def run_experiment(train_dataloader, 
                   val_dataloader, 
                   test_dataloader, 
                   vocab,
                   batch_size,
                   seq_len, 
                   model_name, 
                   embedding_dim, 
                   hidden_dim, 
                   nlayers, 
                   dropout, 
                   N_BACKGROUND, 
                   init_type):
    dataset = 'Synthetic-events'

    n_epochs = 10
    stop_num = 2
    bidirectional = True

    # For model early stopping criteria
    EARLY_STOPPING = "intersection_similarity"  # Values are any of these: ['intersection_similarity', 'loss']

    # SHAP related constants
    BACKGROUND_NEGATIVE_ONLY = True  # If negative examples are used as background
    N_VALID_EXAMPLES = 32  # Number of validation examples to be used during model training
    N_TEST_EXAMPLES = 64  # Number of test examples
    TEST_POSITIVE_ONLY = True  # If only positive examples are selected
    IS_TEST_RANDOM = (
        False  # If random test/val examples are selected for shap value computation
    )
    SORT_SHAP_VALUES = False  # Whether to sort per-patient shap values for visualization

    model_save_path = "./output/{}/{}/models/model_{}.pkl".format(seq_len, model_name, "{}")
    shap_save_path = "./output/{}/{}/shap/{}_shap_{}.pkl".format(
        seq_len, model_name, "{}", "{}"
    )  # SHAP values path for a given dataset split (train/val/test) (data format (features, scores, patient_ids))
    
    
    # Model Output Directory
    model_save_dir = os.path.dirname(model_save_path)
    shap_save_dir = os.path.dirname(shap_save_path)
    if os.path.exists(model_save_dir):
        # Remove model save directory if exists
        shutil.rmtree(model_save_dir)
    if os.path.exists(shap_save_dir):
        # Remove model save directory if exists
        shutil.rmtree(shap_save_dir)
    os.makedirs(model_save_dir)
    os.makedirs(shap_save_dir)
    print(f"New directory created: {model_save_dir}")
    print(f"New directory created: {shap_save_dir}")

    print(f"Cuda available: {torch.cuda.is_available()}")
    model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ### Model Training

    # In[8]:

    if model_name == 'lstm-att':
        model = lstm_att.AttLSTM(
            embedding_dim,
            hidden_dim,
            vocab,
            model_device,
            nlayers=nlayers,
            dropout=dropout,
            init_type=init_type,
        )
    else:
        model = lstm.SimpleLSTM(
            embedding_dim,
            hidden_dim,
            vocab,
            model_device,
            nlayers=nlayers,
            dropout=dropout,
            init_type=init_type,
        )        
    model = model.cuda()

    # loss_function = nn.CrossEntropyLoss()
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 4, gamma=0.9)

    if EARLY_STOPPING == "intersection_similarity":
        best_valid = float("-inf")
    else:
        best_valid = float("inf")
    worse_valid = 0  # enable early stopping

    for epoch in range(n_epochs):

        start_time = time.time()

        train_loss, train_auc = l_utils.epoch_train_lstm(
            model, train_dataloader, optimizer, loss_function
        )

        valid_loss, valid_auc = l_utils.epoch_val_lstm(
            model, valid_dataloader, loss_function
        )  # , return_preds=False

        val_shap_path = shap_save_path.format("val", f"{epoch:02}")
        if EARLY_STOPPING == "intersection_similarity":
            print(f"Computing SHAP Intersection Similarity for epoch={epoch}...")
            (features, scores, patients,) = sj_utils.get_lstm_features_and_shap_scores(
                model,
                train_dataloader,
                valid_dataloader,
                seq_len,
                val_shap_path,
                save_output=False,
                n_background=N_BACKGROUND,
                background_negative_only=BACKGROUND_NEGATIVE_ONLY,
                n_test=N_VALID_EXAMPLES,
                test_positive_only=TEST_POSITIVE_ONLY,
                is_test_random=IS_TEST_RANDOM,
            )

            valid_sim, _ = sj_utils.get_model_intersection_similarity((features, scores))
        end_time = time.time()

        epoch_mins, epoch_secs = l_utils.epoch_time(start_time, end_time)

        print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")

        if EARLY_STOPPING == "intersection_similarity":
            if valid_sim > best_valid:
                best_valid = valid_sim
                save_path = model_save_path.format(str(epoch).zfill(2))
                torch.save(model.state_dict(), save_path)
                sj_utils.save_pickle((features, scores, patients), val_shap_path)
                print(f"Saved Model and SHAP values, epoch {epoch:02}")
                worse_valid = 0
            else:
                worse_valid += 1
                if worse_valid == stop_num:
                    print("EARLY STOP ------")
                    break
        else:
            if valid_loss < best_valid:
                best_valid = valid_loss
                save_path = model_save_path.format(str(epoch).zfill(2))
                torch.save(model.state_dict(), save_path)
                print("Saved Model, epoch {}".format(epoch))
                worse_valid = 0
            else:
                worse_valid += 1
                if worse_valid == stop_num:
                    print("EARLY STOP ------")
                    break

        scheduler.step()
        sim_message = ""
        if EARLY_STOPPING == "intersection_similarity":
            sim_message = f"| Val Int. Similarity: {valid_sim:.4f}"
        print(
            f"Train Loss: {train_loss:.3f} | Train AUC: {train_auc:.2f} \t Val. Loss: {valid_loss:.3f} |  Val. AUC: {valid_auc:.4f} {sim_message}"
        )


    # Get paths of each saved model
    models_paths = sj_utils.get_model_paths(model_save_path)

    # ## Model Validation and Visualization

    #Take only the last model (as it is the best)
    models_paths = models_paths[-1:]
    total_models = len(models_paths)
    for i, model_path in enumerate(models_paths):
        print(f"Processing for model {os.path.basename(model_path)} ...")
        # Load trained weights
        print("Loading the trained weights...")
        model.load_state_dict(torch.load(model_path))
        ##Get Train/Val/Test Scores
        print("Computing the models performances for train/val/test splits...")
        train_loss, train_auc, train_labels, train_scores = l_utils.epoch_val_lstm(
            model, train_dataloader, loss_function, return_preds=True
        )
        val_loss, val_auc, val_labels, val_scores = l_utils.epoch_val_lstm(
            model, valid_dataloader, loss_function, return_preds=True
        )
        test_loss, test_auc, test_labels, test_scores = l_utils.epoch_val_lstm(
            model, test_dataloader, loss_function, return_preds=True
        )

        print(f"Computing SHAP for {N_VALID_EXAMPLES} positive val examples...")
        epoch = sj_utils.get_epoch_number_from_path(model_path)
        val_shap_path = shap_save_path.format("val", f"{epoch:02}")
        (
            features,
            scores,
            patients,
        ) = sj_utils.load_pickle(val_shap_path)

        print("Computing Intersection Similarity...")
        val_avg_sim, sim = sj_utils.get_model_intersection_similarity((features, scores))

        # For the best model, get the final performance (test set) (intersection similarity)
        exp_result = ""
        if i == (total_models - 1):
            print(
                f"Computing SHAP for {N_TEST_EXAMPLES} positive TEST examples for the final model..."
            )
            test_shap_path = shap_save_path.format("test", f"{epoch:02}")
            (features, scores, patients,) = sj_utils.get_lstm_features_and_shap_scores(
                model,
                train_dataloader,
                test_dataloader,
                seq_len,
                test_shap_path,
                save_output=True,
                n_background=N_BACKGROUND,
                background_negative_only=BACKGROUND_NEGATIVE_ONLY,
                n_test=N_TEST_EXAMPLES,
                test_positive_only=TEST_POSITIVE_ONLY,
                is_test_random=IS_TEST_RANDOM,
            )
            test_avg_sim, _ = sj_utils.get_model_intersection_similarity((features, scores))
            
            exp_result = f"{model_name},{dataset},{seq_len},{val_avg_sim:.4f},{test_avg_sim:.4f},{val_auc:.4f},{test_auc:.4f},{n_epochs},{stop_num},{batch_size},{embedding_dim},{hidden_dim},{nlayers},{bidirectional},{dropout},{EARLY_STOPPING},{N_BACKGROUND},{BACKGROUND_NEGATIVE_ONLY},{N_VALID_EXAMPLES},{N_TEST_EXAMPLES},{TEST_POSITIVE_ONLY},{IS_TEST_RANDOM}"

        print("All tasks SUCCESSFULLY completed!")
        print("=" * 100)
        return exp_result



def save_experiment(exp_result, output_path, header):
    """Saves the experiment result."""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    data = ""
    if not os.path.exists(output_path):
        data = f"{header}\n{exp_result}"
    else:
        data = f"\n{exp_result}"
        
    with open(output_path, 'a+') as fp:
        fp.write(data)
    

if __name__=="__main__":
    nrows = 1e9
    min_freq = 1
    batch_size = 64  # For model training
    rev = False
    
    target_colname = "label"
    uid_colname = "patient_id"
    target_value = "1"
    
    seq_lens = [30, 300, 900]
    model_names = ['lstm-att', 'lstm']
    embedding_dims = [8]#, 16]
    hidden_dims = [8, 16]
    nlayerss = [1, 2]
    dropouts = [0.2, 0.3]
    init_types = ["zero"] #"learned"]
    N_BACKGROUNDS = [300, 500]

    exp_output_path = './output/experiments/experiments_summary.csv'
    exp_header = "model,dataset,seq_len,val_Intersection_Sim,test_Intersection_Sim,val_AUC,test_AUC,n_epochs,stop_num,batch_size,embedding_dim,hidden_dim,nlayers,bidirectional,dropout,early_stopping_criteria,n_background,background_negative_only,n_valid_examples,n_test_examples,test_positive_only,is_test_random"

    for seq_len in seq_lens:
        #Load data
        train_data_path = "../../data/toy_dataset/data/{}/train.csv".format(seq_len)
        valid_data_path = "../../data/toy_dataset/data/{}/val.csv".format(seq_len)
        test_data_path = "../../data/toy_dataset/data/{}/test.csv".format(seq_len)
        (train_dataloader, valid_dataloader, test_dataloader, vocab) = load_data(train_data_path, 
                                                                          valid_data_path, 
                                                                          test_data_path, 
                                                                          seq_len=seq_len,
                                                                          min_freq=min_freq, 
                                                                          batch_size=batch_size,
                                                                          target_colname=target_colname,
                                                                          uid_colname=uid_colname,
                                                                          target_value=target_value,
                                                                          nrows=nrows, 
                                                                          rev=rev)
        for model_name in model_names:
            for embedding_dim in embedding_dims:
                for hidden_dim in hidden_dims:
                    for nlayers in nlayerss:
                        for dropout in dropouts:
                            for N_BACKGROUND in N_BACKGROUNDS:
                                for init_type in init_types:
                                    start = time.time()
                                    exp_result = run_experiment(
                                                   train_dataloader=train_dataloader,
                                                   val_dataloader=valid_dataloader,                                        
                                                   test_dataloader=test_dataloader,
                                                   vocab=vocab,
                                                   batch_size=batch_size,
                                                   seq_len=seq_len, 
                                                   model_name=model_name,
                                                   embedding_dim=embedding_dim, 
                                                   hidden_dim=hidden_dim, 
                                                   nlayers=nlayers, 
                                                   dropout=dropout, 
                                                   init_type=init_type, 
                                                   N_BACKGROUND=N_BACKGROUND)
                                    end = time.time()
                                    total = (end-start)/60.0
                                    print(f'Total time per experiment: {total:.2f}mins')
                                    save_experiment(exp_result, exp_output_path, exp_header)
        print('Experiment Result Successfully Written for seq_len={seq_len}!')