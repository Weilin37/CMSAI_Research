#!/usr/bin/env python
# coding: utf-8

# ## SHAP, LRP & attention scores with Attention-LSTM
# Author: Lin Lee Cheong <br>
# Modified by: Tesfagabir Meharizghi
# Date created: 1/13/2021 <br>
# Date updated: 2/10/2021 <br>
# 
# **Data:** <br>
# Using the final version of 30 sequence length dataset (sequence-based) generated by Tes<br>
# Train, validation (for model training), test (for performance etc), and example (4 output)
# <br>
# 
# 
# **Steps:** <br>
# 1. Read in datasets [DONE]
# 2. LSTM model training 
#     - TODO: check probab outputs
#     - save epoch train, val, loss, etc [DONE]
#     - calculated SHAP & relevance scores for val and test sets [DONE]
#     - calculate rbo, tau for val and test sets [DONE]
#     - plot rbo, tau
# 3. Extract SHAP, attention and relevance scores for a TEST set
#     - calculate SHAP, relevance scores, performance (AUC, test loss)[DONE]
#     - calculate rbo, tau [DONE]
# 3. Extract SHAP and relevance scores for example set of 4
#     - plot epoch evolution [DONE]
#     - add attention [DONE]
# 4. Save output in dict format[DONE]

# In[1]:


#!pip install rbo


# In[2]:


#!python -m pip install --upgrade pip


# In[3]:


#! pip install shap
#!pip install xgboost
#!pip install -e git+https://github.com/changyaochen/rbo.git@master#egg=rbo

#!pip install nb_black


# In[4]:


# In[5]:

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
from utils import *

from cdiff_utils import *

# import rbo

def run_epoch(lstm_model, 
              scheduler, 
              optimizer, 
              train_auc_lst, 
              train_loss_lst, 
              val_auc_lst, 
              val_loss_lst, 
              test_auc_lst, 
              test_loss_lst,
              val_lrp_sim_lst,
              val_shap_sim_lst,
              val_lrp_shap_rbo_lst,
              val_lrp_shap_tau_lst,
              test_lrp_sim_lst,
              test_shap_sim_lst,
              test_lrp_shap_rbo_lst,
              test_lrp_shap_tau_lst,
              valid_results, 
              test_results, 
              train_dataloader, 
              valid_dataloader, 
              test_dataloader,
              train_path,
              valid_path,
              test_path,
              vocab_path,
              val_batch, 
              test_batch, 
              model_params, 
              params_path,
              epoch, 
              model_save_path_pattern, 
              shap_save_dir_pattern, 
              output_results_path, 
              seq_len,
              nrows,
             ):
    
        start_time = time.time()

        rbo_p = 0.8
        SIMILARITY_FREEDOM = 1  # For Intersection Similarity

        val_patient_ids, val_labels, val_idxed_text = val_batch
        test_patient_ids, test_labels, test_idxed_text = test_batch

        #import pdb; pdb.set_trace()
        lstm_model.train()
        # model training & perf evaluation
        train_loss, train_auc = epoch_train_lstm(
            lstm_model,
            train_dataloader,
            optimizer,
            loss_function,
            clip=model_params["clip"],
            device=model_device,
        )
        train_auc_lst.append(train_auc)
        train_loss_lst.append(train_loss)

        valid_loss, valid_auc = epoch_val_lstm(
            lstm_model, valid_dataloader, loss_function, device=model_device
        )
        val_auc_lst.append(valid_auc)
        val_loss_lst.append(valid_loss)

        test_loss, test_auc = epoch_val_lstm(
            lstm_model, test_dataloader, loss_function, device=model_device
        )
        test_auc_lst.append(test_auc)
        test_loss_lst.append(test_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # save model
        model_save_path = model_save_path_pattern.format(str(epoch).zfill(2))
        torch.save(lstm_model.state_dict(), model_save_path)

        scheduler.step()

        #         print(
        #             f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} "
        #             + f"\t Val. Loss: {valid_loss:.4f} | Val. AUC: {valid_auc:.4f} "
        #             + f"\t Test. Loss: {test_loss:.4f} | Test. AUC: {test_auc:.4f} "
        #         )
        #         continue

    
        # calculate relevancy and SHAP
        lstm_model.eval()
        lrp_model = LSTM_LRP_MultiLayer(lstm_model.cpu())

        # Save valid/test results
        valid_results[epoch] = {}
        test_results[epoch] = {}

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
            df["token"] = [lstm_model.vocab.itos(x) for x in one_text]
            df["att_weights"] = lrp_model.get_attn_values()

            if val_patient_ids[sel_idx] not in valid_results[epoch]:
                valid_results[epoch][val_patient_ids[sel_idx]] = {}
            valid_results[epoch][val_patient_ids[sel_idx]] = {}
            valid_results[epoch][val_patient_ids[sel_idx]]["label"] = val_labels[
                sel_idx
            ]
            valid_results[epoch][val_patient_ids[sel_idx]]["pred"] = lrp_model.s[0]
            valid_results[epoch][val_patient_ids[sel_idx]]["imp"] = df.copy()

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
            df["token"] = [lstm_model.vocab.itos(x) for x in one_text]
            df["att_weights"] = lrp_model.get_attn_values()

            if test_patient_ids[sel_idx] not in test_results[epoch]:
                test_results[epoch][test_patient_ids[sel_idx]] = {}
            test_results[epoch][test_patient_ids[sel_idx]] = {}
            test_results[epoch][test_patient_ids[sel_idx]]["label"] = test_labels[
                sel_idx
            ]
            test_results[epoch][test_patient_ids[sel_idx]]["pred"] = lrp_model.s[0]
            test_results[epoch][test_patient_ids[sel_idx]]["imp"] = df.copy()

        #Run SHAP
        #import pdb; pdb.set_trace()
        shap_start_time = time.time()    
        multigpu = 1
        shap_valid_output_path = shap_save_dir_pattern.format("val", epoch)
        shap_valid_output_path = shap_valid_output_path.replace('shap/', 'shap_intermediate/')
        shap_test_output_path = shap_save_dir_pattern.format("test", epoch)
        shap_test_output_path = shap_test_output_path.replace('shap/', 'shap_intermediate/')        
        command = f'python my_shap.py --model_path={model_save_path} --params_path={params_path} '
        command += f'--train_path={train_path} --valid_path={valid_path} --test_path={test_path} --vocab_path={vocab_path} '
        command += f'--valid_output_path={shap_valid_output_path} --test_output_path={shap_test_output_path} '
        command += f'--seq_len={seq_len} --multigpu={multigpu} --nrows={int(nrows)}'
        os.system(command)
        #Load SHAP Results
        (val_features, val_scores, val_patients) = load_pickle(shap_valid_output_path)
        (test_features, test_scores, test_patients) = load_pickle(shap_test_output_path)

        shap_end_time = time.time()
        shap_mins, shap_secs = epoch_time(shap_start_time, shap_end_time)
        
        for idx, pid in enumerate(val_patients):
            df = valid_results[epoch][pid]["imp"]
            assert len(df) == len(val_scores[idx])
            df["shap_scores"] = val_scores[idx]
            df = df[
                ["idx", "seq_idx", "token", "att_weights", "lrp_scores", "shap_scores"]
            ]
            valid_results[epoch][pid]["imp"] = df.copy()

        for idx, pid in enumerate(test_patients):
            df = test_results[epoch][pid]["imp"]
            assert len(df) == len(test_scores[idx])
            df["shap_scores"] = test_scores[idx]
            df = df[
                ["idx", "seq_idx", "token", "att_weights", "lrp_scores", "shap_scores"]
            ]
            test_results[epoch][pid]["imp"] = df.copy()

        # calculate similarity indexes for val
        epoch_val_lrp_shap_t_corr = []
        epoch_val_lrp_shap_rbo = []
        epoch_val_lrp_sim = []
        epoch_val_shap_sim = []

        for pid in valid_results[epoch].keys():
            imp_df = valid_results[epoch][pid]["imp"]
            imp_df["u_token"] = [
                str(seq) + "_" + str(token)
                for seq, token in zip(imp_df["seq_idx"], imp_df["token"])
            ]
            valid_results[epoch][pid]["lrp_shap_t_corr"] = get_wtau(
                imp_df["lrp_scores"], imp_df["shap_scores"]
            )

            valid_results[epoch][pid]["lrp_shap_rbo"] = get_rbo(
                imp_df["lrp_scores"],
                imp_df["shap_scores"],
                imp_df["u_token"].tolist(),
                p=rbo_p,
            )

            epoch_val_lrp_shap_t_corr.append(
                valid_results[epoch][pid]["lrp_shap_t_corr"]
            )
            epoch_val_lrp_shap_rbo.append(valid_results[epoch][pid]["lrp_shap_rbo"])

            # gt similarity
            lrp_scores = imp_df.lrp_scores.values
            shap_scores = imp_df.shap_scores.values
            tokens = imp_df.u_token
            lrp_sim = get_similarity(lrp_scores, tokens, freedom=1)
            shap_sim = get_similarity(shap_scores, tokens, freedom=1)
            if lrp_sim != -1:
                epoch_val_lrp_sim.append(lrp_sim)
                epoch_val_shap_sim.append(shap_sim)
            valid_results[epoch][pid]["lrp_sim"] = lrp_sim
            valid_results[epoch][pid]["shap_sim"] = shap_sim

        # Save training results to file.
        valid_shap_path = shap_save_dir_pattern.format("val", epoch)
        with open(valid_shap_path, "wb") as fp:
            pickle.dump(valid_results[epoch], fp)

        val_lrp_shap_rbo_lst.append(np.mean(epoch_val_lrp_shap_rbo))
        val_lrp_shap_tau_lst.append(np.mean(epoch_val_lrp_shap_t_corr))
        val_lrp_sim_lst.append(np.mean(epoch_val_lrp_sim))
        val_shap_sim_lst.append(np.mean(epoch_val_shap_sim))

        # calculate similarity indexes for test
        epoch_test_lrp_shap_t_corr = []
        epoch_test_lrp_shap_rbo = []
        epoch_test_lrp_sim = []
        epoch_test_shap_sim = []

        for pid in test_results[epoch].keys():
            imp_df = test_results[epoch][pid]["imp"]
            imp_df["u_token"] = [
                str(seq) + "_" + str(token)
                for seq, token in zip(imp_df["seq_idx"], imp_df["token"])
            ]
            test_results[epoch][pid]["lrp_shap_t_corr"] = get_wtau(
                imp_df["lrp_scores"], imp_df["shap_scores"]
            )

            test_results[epoch][pid]["lrp_shap_rbo"] = get_rbo(
                imp_df["lrp_scores"],
                imp_df["shap_scores"],
                imp_df["u_token"].tolist(),
                p=rbo_p,
            )

            epoch_test_lrp_shap_t_corr.append(
                test_results[epoch][pid]["lrp_shap_t_corr"]
            )
            epoch_test_lrp_shap_rbo.append(test_results[epoch][pid]["lrp_shap_rbo"])

            # gt similarity
            lrp_scores = imp_df.lrp_scores.values
            shap_scores = imp_df.shap_scores.values
            tokens = imp_df.u_token
            lrp_sim = get_similarity(
                lrp_scores, tokens, freedom=SIMILARITY_FREEDOM
            )
            shap_sim = get_similarity(
                shap_scores, tokens, freedom=SIMILARITY_FREEDOM
            )
            if lrp_sim != -1:
                epoch_test_lrp_sim.append(lrp_sim)
                epoch_test_shap_sim.append(shap_sim)
            test_results[epoch][pid]["lrp_sim"] = lrp_sim
            test_results[epoch][pid]["shap_sim"] = shap_sim

        # Save training results to file.
        test_shap_path = shap_save_dir_pattern.format("test", epoch)
        with open(test_shap_path, "wb") as fp:
            pickle.dump(test_results[epoch], fp)

        test_lrp_shap_rbo_lst.append(np.mean(epoch_test_lrp_shap_rbo))
        test_lrp_shap_tau_lst.append(np.mean(epoch_test_lrp_shap_t_corr))
        test_lrp_sim_lst.append(np.mean(epoch_test_lrp_sim))
        test_shap_sim_lst.append(np.mean(epoch_test_shap_sim))

        print(
            f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s | "
            + f"SHAP Time: {shap_mins}m {shap_secs}s"
        )
        print(
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f} "
            + f"\t Val. Loss: {valid_loss:.4f} | Val. AUC: {valid_auc:.4f} "
            + f"| Val LRP Sim: {np.mean(epoch_val_lrp_sim):.4f} | Val SHAP Sim: {np.mean(epoch_val_shap_sim):.4f}"
        )
        
        torch.cuda.empty_cache()

        
        return (
                lstm_model,
                scheduler,
                optimizer,
                train_auc_lst, 
                train_loss_lst, 
                val_auc_lst, 
                val_loss_lst, 
                test_auc_lst, 
                test_loss_lst,
                val_lrp_sim_lst,
                val_shap_sim_lst,
                val_lrp_shap_rbo_lst,
                val_lrp_shap_tau_lst,
                test_lrp_sim_lst,
                test_shap_sim_lst,
                test_lrp_shap_rbo_lst,
                test_lrp_shap_tau_lst,
                valid_results, 
                test_results, 
        )


def run(lstm_model, 
        scheduler, 
        optimizer, 
        train_dataloader, 
        valid_dataloader, 
        test_dataloader,
        train_path,
        valid_path,
        test_path,
        vocab_path,
        model_params, 
        n_epochs, 
        params_path, 
        model_save_path_pattern, 
        shap_save_dir_pattern, 
        output_results_path, 
        seq_len, 
        nrows,
        train_model=True):
    
    valid_results = {}
    test_results = {}
    
    if train_model:
        train_auc_lst = []
        train_loss_lst = []

        val_auc_lst = []
        val_loss_lst = []
        val_lrp_sim_lst = []
        val_shap_sim_lst = []
        val_lrp_shap_rbo_lst = []
        val_lrp_shap_tau_lst = []

        test_auc_lst = []
        test_loss_lst = []
        test_lrp_sim_lst = []
        test_shap_sim_lst = []
        test_lrp_shap_rbo_lst = []
        test_lrp_shap_tau_lst = []

        val_patient_ids, val_labels, val_idxed_text = next(iter(valid_dataloader))
        test_patient_ids, test_labels, test_idxed_text = next(iter(test_dataloader))

        val_batch = (val_patient_ids, val_labels, val_idxed_text)
        test_batch = (test_patient_ids, test_labels, test_idxed_text)

        # Save Model Parameters
        with open(params_path, "w") as fp:
            json.dump(model_params, fp)

        for epoch in range(n_epochs):
            #import pdb; pdb.set_trace()
            (
                lstm_model,
                scheduler,
                optimizer,
                train_auc_lst, 
                train_loss_lst, 
                val_auc_lst, 
                val_loss_lst, 
                test_auc_lst, 
                test_loss_lst,
                val_lrp_sim_lst,
                val_shap_sim_lst,
                val_lrp_shap_rbo_lst,
                val_lrp_shap_tau_lst,
                test_lrp_sim_lst,
                test_shap_sim_lst,
                test_lrp_shap_rbo_lst,
                test_lrp_shap_tau_lst,
                valid_results, 
                test_results, 
            ) = run_epoch(lstm_model, 
                          scheduler=scheduler,
                          optimizer=optimizer,
                          train_auc_lst=train_auc_lst, 
                          train_loss_lst=train_loss_lst, 
                          val_auc_lst=val_auc_lst, 
                          val_loss_lst=val_loss_lst, 
                          test_auc_lst=test_auc_lst, 
                          test_loss_lst=test_loss_lst,
                          val_lrp_sim_lst=val_lrp_sim_lst,
                          val_shap_sim_lst=val_shap_sim_lst,
                          val_lrp_shap_rbo_lst=val_lrp_shap_rbo_lst,
                          val_lrp_shap_tau_lst=val_lrp_shap_tau_lst,
                          test_lrp_sim_lst=test_lrp_sim_lst,
                          test_shap_sim_lst=test_shap_sim_lst,
                          test_lrp_shap_rbo_lst=test_lrp_shap_rbo_lst,
                          test_lrp_shap_tau_lst=test_lrp_shap_tau_lst,                          
                          valid_results=valid_results,
                          test_results=test_results,
                          train_dataloader=train_dataloader, 
                          valid_dataloader=valid_dataloader, 
                          test_dataloader=test_dataloader, 
                          train_path=train_path,
                          valid_path=valid_path,
                          test_path=test_path,
                          vocab_path=vocab_path,
                          val_batch=val_batch,
                          test_batch=test_batch,
                          model_params=model_params,
                          params_path=params_path,
                          epoch=epoch,
                          model_save_path_pattern=model_save_path_pattern, 
                          shap_save_dir_pattern=shap_save_dir_pattern, 
                          output_results_path=output_results_path, 
                          seq_len=seq_len,
                          nrows=nrows,
                         )
    
        df_results = pd.DataFrame()
        df_results["epoch"] = [x for x in range(n_epochs)]
        df_results["train_AUC"] = train_auc_lst
        df_results["train_Loss"] = train_loss_lst
        df_results["val_AUC"] = val_auc_lst
        df_results["val_Loss"] = val_loss_lst
        df_results["test_AUC"] = test_auc_lst
        df_results["test_Loss"] = test_loss_lst
        df_results["val_lrp_shap_rbo"] = val_lrp_shap_rbo_lst
        df_results["val_lrp_shap_tau"] = val_lrp_shap_tau_lst
        df_results["test_lrp_shap_rbo"] = test_lrp_shap_rbo_lst
        df_results["test_lrp_shap_tau"] = test_lrp_shap_tau_lst
        df_results["val_GT_lrp_sim"] = val_lrp_sim_lst
        df_results["val_GT_shap_sim"] = val_shap_sim_lst
        df_results["test_GT_lrp_sim"] = test_lrp_sim_lst
        df_results["test_GT_shap_sim"] = test_shap_sim_lst
        df_results.set_index("epoch", inplace=True)

        # save results summary
        df_results.to_csv(output_results_path)

    else:
        print("Loading Training results....")
        df_results = pd.read_csv(output_results_path)
        df_results.set_index("epoch", inplace=True)

        for epoch in range(n_epochs):
            # Load valid results.
            valid_shap_path = shap_save_dir_pattern.format("val", epoch)
            with open(valid_shap_path, "rb") as fp:
                valid_results[epoch] = pickle.load(fp)
            # Load test results.
            test_shap_path = shap_save_dir_pattern.format("test", epoch)
            with open(test_shap_path, "rb") as fp:
                test_results[epoch] = pickle.load(fp)
        print("SUCCESS!")
        
    return df_results, valid_results, test_results


if __name__ == "__main__":

    MODEL_NAME = "lstm-att-lrp"

    NROWS = 1e9

    TRAIN_MODEL = True
    SEQ_LEN = 100

    N_EPOCHS = 10

    TARGET_COLNAME = "d_00845"
    UID_COLNAME = "patient_id"
    TARGET_VALUE = "1"

    DATA_TYPE = "downsampled"
    FNAME = "all"
    SAVE_DATASET = True

    BATCH_SIZE = 64
    # Model Parameters
    MODEL_PARAMS = {
        # Dataset/vocab related
        "min_freq": 1000,
        "batch_size": BATCH_SIZE,
        # Model related parameters
        "embedding_dim": 10,
        "hidden_dim": 10,
        "nlayers": 1,
        "bidirectional": True,
        "dropout": 0.0,
        "linear_bias": False,
        "init_type": "zero",  # zero/learned
        "learning_rate": 0.01,
        "scheduler_step": 2,
        "clip": True,
        "rev": False,
        # SHAP-related parameters
        "n_background": 300,  # Number of background examples
        "background_negative_only": False,  # If negative examples are used as background
        "background_positive_only": False,
        "test_positive_only": False,
        "is_test_random": False,
        "n_valid_examples": BATCH_SIZE,  # Number of validation examples to be used during shap computation
        "n_test_examples": BATCH_SIZE,  # Number of the final test examples to be used in shap computation #TODO
    }


    # ALL_DOWN_DATA_PATH = (
    #     f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/all.csv"
    # )

    # MONTH_DATA_PATH = (
    #     f"../../../data/AE_CDiff_d00845/output/data/1000/original/preprocessed/20110101.csv"
    # )

    TRAIN_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/train.csv"
    VALID_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/val.csv"
    TEST_DATA_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/test.csv"
    SELECTED_EXAMPLES_PATH = f"../../../data/AE_CDiff_d00845/output/data/1000/{DATA_TYPE}/preprocessed/splits/{FNAME}/visualized_test_patients.txt"

    OUT_TRAIN_DATA_PATH = f"./output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/splits/{FNAME}/{MODEL_PARAMS['min_freq']}/train.csv"
    OUT_VALID_DATA_PATH = f"./output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/splits/{FNAME}/{MODEL_PARAMS['min_freq']}/val.csv"
    OUT_TEST_DATA_PATH = f"./output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/splits/{FNAME}/{MODEL_PARAMS['min_freq']}/test.csv"
    OUT_SELECTED_EXAMPLES_PATH = f"./output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/splits/{FNAME}/{MODEL_PARAMS['min_freq']}/visualized_test_patients.csv"

    VOCAB_PATH = f"./output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/splits/{FNAME}/{MODEL_PARAMS['min_freq']}/vocab.pkl"

    GT_CODES_PATH = "../../../data/AE_CDiff_d00845/cdiff_risk_factors_codes.csv"

    MODEL_SAVE_PATH_PATTERN = f"./output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/{FNAME}/{MODEL_PARAMS['min_freq']}/model_weights/model_{'{}'}.pkl"
    SHAP_SAVE_DIR_PATTERN = f"./output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/{FNAME}/{MODEL_PARAMS['min_freq']}/shap/{'{}'}_shap_{'{}'}.pkl"  # SHAP values path for a given dataset split

    OUTPUT_RESULTS_PATH = f"output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/{FNAME}/{MODEL_PARAMS['min_freq']}/train_results/results.csv"
    PARAMS_PATH = f"output/AE_CDiff/{SEQ_LEN}/{DATA_TYPE}/{MODEL_NAME}/{FNAME}/{MODEL_PARAMS['min_freq']}/train_results/model_params.json"

    # --------------------
    # From the original AE parameters
    # batch_size = 1024
    # N_EPOCHS = 20

    # EMBEDDING_DIM = 30
    # HIDDEN_DIM = 30
    # BIDIRECTIONAL = False
    # DROPOUT = 0.3
    # -------------------


    # Create output directories if needed
    model_dir = os.path.dirname(MODEL_SAVE_PATH_PATTERN)
    shap_dir = os.path.dirname(SHAP_SAVE_DIR_PATTERN)
    output_dir = os.path.dirname(OUTPUT_RESULTS_PATH)

    if TRAIN_MODEL:
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        if os.path.exists(shap_dir):
            shutil.rmtree(shap_dir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(model_dir)
        os.makedirs(shap_dir)
        os.makedirs(output_dir)
        print(f"Directory Created: {model_dir}")
        print(f"Directory Created: {shap_dir}")
        print(f"Directory Created: {output_dir}")

    gt_codes = pd.read_csv(GT_CODES_PATH)
    gt_codes = list(set(gt_codes.Internal_Code))
    print(f'# of Ground Truth Risk Codes: {len(gt_codes)}')


    if TRAIN_MODEL:
        df = pd.read_csv(TRAIN_DATA_PATH)
        df_val = pd.read_csv(VALID_DATA_PATH)
        df_test = pd.read_csv(TEST_DATA_PATH)

        # Take upto seq_len cols
        # cols = df.columns.tolist()
        # exclude = [str(i) for i in range(1000, SEQ_LEN, -1)]
        # cols = [col for col in cols if col not in exclude]
        # df = df[cols]
        # df.head()

        cols = [str(i) for i in range(SEQ_LEN - 1, -1, -1)]
        vocab = Counter(df[cols].values.flatten().tolist())

        # Remove tokens not in gt_codes
        for token in list(vocab):
            if (not token.endswith('_rf')): # or (vocab[token] < MODEL_PARAMS["min_freq"]):
                del vocab[token]
        print(f'Total GT Codes in Vocab: {len(vocab)}')

        df["num_gt_codes"] = df.apply(
            get_gt_code_patient, args=(SEQ_LEN,), axis=1
        )
        df["has_gt_codes"] = (df["num_gt_codes"] > 0).astype(int)
        df = df.sort_values("has_gt_codes", ascending=False)

        df_val["num_gt_codes"] = df_val.apply(
            get_gt_code_patient, args=(SEQ_LEN,), axis=1
        )
        df_val["has_gt_codes"] = (df_val["num_gt_codes"] > 0).astype(int)
        df_val = df_val.sort_values("has_gt_codes", ascending=False)

        df_test["num_gt_codes"] = df_test.apply(
            get_gt_code_patient, args=(SEQ_LEN,), axis=1
        )
        df_test["has_gt_codes"] = (df_test["num_gt_codes"] > 0).astype(int)
        df_test = df_test.sort_values("has_gt_codes", ascending=False)

        out_dir = os.path.dirname(OUT_TRAIN_DATA_PATH)
        os.makedirs(out_dir, exist_ok=True)

        df.to_csv(OUT_TRAIN_DATA_PATH, index=False)
        df_val.to_csv(OUT_VALID_DATA_PATH, index=False)
        df_test.to_csv(OUT_TEST_DATA_PATH, index=False)
    else:
        pass
    #     df_train = pd.read_csv(OUT_TRAIN_DATA_PATH)
    #     df_val = pd.read_csv(OUT_VALID_DATA_PATH)
    #     df_test = pd.read_csv(OUT_TEST_DATA_PATH)

    df_train = pd.read_csv(OUT_TRAIN_DATA_PATH)
    print(df_train.shape)
    #df_train.head()

    print("# of Examples with and without Risk Factors")
    print(df_train.has_gt_codes.value_counts())
    print("-" * 20)
    df_train["num_gt_codes"].plot.hist(bins=10)
    plt.xlabel("Number of Accurrences of Ground Truth Risk Factors/Patient")
    plt.title("Histogram of GT Risk Factors per Patient")
    plt.show()

    del df_train

    df_test = pd.read_csv(OUT_TEST_DATA_PATH)

    SELECTED_EXAMPLES_PATH = OUT_SELECTED_EXAMPLES_PATH.replace(".csv", ".txt")

    if TRAIN_MODEL:
        pos_examples = df_test[df_test["d_00845"] == 1]["patient_id"].iloc[:2].tolist()
        neg_examples = df_test[df_test["d_00845"] == 0]["patient_id"].iloc[:2].tolist()
        test_examples = pos_examples + neg_examples

        with open(SELECTED_EXAMPLES_PATH, "w") as fp:
            fp.write("\n".join(test_examples))
    else:
        with open(SELECTED_EXAMPLES_PATH, "r") as fp:
            test_examples = fp.readlines()
            test_examples = [ex.strip() for ex in test_examples]

    del df_test

    # Check if cuda is available
    print(f"Cuda available: {torch.cuda.is_available()}")
    model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load Selected Patients for later SHAP visualization
    patients = pd.read_csv(SELECTED_EXAMPLES_PATH, sep=" ", header=None)
    patients = patients.values.flatten().tolist()
    print(f"Selected Test Examples: {patients}")
    # create the example set
    selected_patients_path = os.path.join(output_dir, "selected_test_patients.csv")
    test_df = pd.read_csv(OUT_TEST_DATA_PATH)
    test_df[test_df.patient_id.isin(patients)].to_csv(selected_patients_path, index=False)

    del test_df

    train_dataset, vocab = build_lstm_dataset(
        OUT_TRAIN_DATA_PATH,
        min_freq=MODEL_PARAMS["min_freq"],
        uid_colname=UID_COLNAME,
        target_colname=TARGET_COLNAME,
        max_len=SEQ_LEN,
        target_value=TARGET_VALUE,
        vocab=None,
        nrows=NROWS,
        rev=MODEL_PARAMS["rev"],
        cdiff = True,        
    )
    
    #Save Vocab
    with open(VOCAB_PATH, 'wb') as fp:
        pickle.dump(vocab, fp)
        

    valid_dataset, _ = build_lstm_dataset(
        OUT_VALID_DATA_PATH,
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
        OUT_TEST_DATA_PATH,
        min_freq=MODEL_PARAMS["min_freq"],
        uid_colname=UID_COLNAME,
        target_colname=TARGET_COLNAME,
        max_len=SEQ_LEN,
        target_value=TARGET_VALUE,
        vocab=vocab,
        nrows=NROWS,
        rev=MODEL_PARAMS["rev"],
    )

    example_dataset, _ = build_lstm_dataset(
        selected_patients_path,
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
        train_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=True, num_workers=8
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=8
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=8
    )

    example_dataloader = DataLoader(
        example_dataset, batch_size=MODEL_PARAMS["batch_size"], shuffle=False, num_workers=2
    )

    gt_codes, _ = get_ground_truth_codes(GT_CODES_PATH, vocab, SEQ_LEN)

    if not TRAIN_MODEL:
        # LOAD Model Parameters
        with open(PARAMS_PATH, "r") as fp:
            MODEL_PARAMS = json.load(fp)

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


    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        lstm_model.parameters(), lr=MODEL_PARAMS["learning_rate"], weight_decay=0.03
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, MODEL_PARAMS["scheduler_step"], gamma=0.9
    )

    lstm_model = lstm_model.to(model_device)

    _ = run(lstm_model, 
            scheduler=scheduler,
            optimizer=optimizer,
            train_dataloader=train_dataloader, 
            valid_dataloader=valid_dataloader, 
            test_dataloader=test_dataloader,
            train_path=OUT_TRAIN_DATA_PATH,
            valid_path=OUT_VALID_DATA_PATH,
            test_path=OUT_TEST_DATA_PATH,
            vocab_path=VOCAB_PATH,
            model_params=MODEL_PARAMS, 
            n_epochs=N_EPOCHS, 
            params_path=PARAMS_PATH,
            model_save_path_pattern=MODEL_SAVE_PATH_PATTERN, 
            shap_save_dir_pattern=SHAP_SAVE_DIR_PATTERN, 
            output_results_path=OUTPUT_RESULTS_PATH, 
            seq_len=SEQ_LEN, 
            nrows=NROWS,
            train_model=TRAIN_MODEL,
           )