#!/usr/bin/env python
# coding: utf-8

# # XGB Model Training and SHAP computation for AE CDiff Dataset
# 
# **Author: Tesfagabir Meharizghi<br>Last Updated: 02/03/2021**
# 
# In this jupyter notebook, we trained XGB models by manually tuning their parameters to get the best performance and explainability scores from the different datasets.
# 
# Tasks done here are:
# - Training XGB models with the specified parameters and datasets
# - Computing different model performance measures such as AUCs, intersection similarity, RBOs, etc.
# - Visualizing global and local SHAP scores
# - And finally copying the well trained models to S3 if needed
# 
# Outputs:
# - The following artifacts are saved:
#     * Model artifacts
#     * SHAP values and their corresponding scores for the specified number of val/test examples
# 
# Model Architecture Used:
# - XGB
# 
# Dataset:
# - Synthetic Dataset
# 
# Requirements:
# - Make sure that you have already generated sequence-based dataset (train/val/test splits) using [Create_toy_dataset_sequence.ipynb](../../../data/toy_dataset/Create_toy_dataset_sequence.ipynb).
# 
# Discussions/observations:
# - Since the models' parameters are manually specified with the best ones, we could get almost the best performance in terms of its AUCs and explainability scores.
# - In addition, we could also get almost perfect global SHAP importance scores where all the important events are selected.

# Cases and Observations:
# - Increasing the token variety for each category
#       - Performance and importance scores didn't change from that of lower number of tokens
#       - The same also for 30 & 300 seq lengths (didn't change performance)
# - Decreasing the 0.99 prob for the ad_seq to 0.8
#     - The models' performance lowered
#         - 87% --> 83% AUC
#     - Sequence lengths didn't make a difference        

# In[169]:


#! pip install rbo


# In[170]:


# pip install nb-black


# In[171]:


#! pip install botocore==1.12.201

#! pip install shap
#! pip install xgboost


# In[172]:


#!pip install -e git+https://github.com/changyaochen/rbo.git@master#egg=rbo


# In[173]:


# In[174]:


import sys

sys.path.append("../")

import os
import time
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import json

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.inspection import permutation_importance
import re
from scipy import stats

#!pip install -e git+https://github.com/changyaochen/rbo.git@master#egg=rbo
# import rbo

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

import xgboost_utils as xgb_utils
import shap_jacc_utils as sj_utils
import cdiff_utils as cd_utils


# ## XGB Model Training

# ### Constants

# In[175]:


# Feature suffices that help predict the positive or negative class
# HELPING_FEATURES_SUFFICES = ["_A", "_H", "_U"]  # Noises Removed

TRAIN_MODEL = True

# Whether to preprocess data
PREPROCESS_DATA = False

# Whether to save SHAP scores
SAVE_SHAP_OUTPUT = True

# Whether to ouput SHAP explainer expected value
OUTPUT_SHAP_EXPLAINER = True

# Whether to shuffle val & test dataset for shap visualization purposes
SHUFFLE = False

USE_FREQ = True  # Whether to use feature frequencies or one-hot for data preprocessing

nrows = 1e9

target_colname = "d_00845"
uid_colname = "patient_id"
target_value = "1"

rev = False

model_name = "xgb"
dataset = "AE_CDiff"

# For model early stopping criteria
EARLY_STOPPING = "intersection_similarity"  # Values are any of these: ['intersection_similarity', 'loss']

# SHAP related constants
N_BACKGROUND = None  # Number of background examples
BACKGROUND_NEGATIVE_ONLY = False  # If negative examples are used as background
BACKGROUND_POSITIVE_ONLY = False  # If positive examples are used as background
TEST_POSITIVE_ONLY = False  # If only positive examples are selected
IS_TEST_RANDOM = (
    False  # If random test/val examples are selected for shap value computation
)
SORT_SHAP_VALUES = False  # Whether to sort per-patient shap values for visualization

SHAP_SCORE_ABSOLUTE = True  # Whether to consider the absolute value of a shap score #TODO: Check this before running.

# For seq_len=30
seq_len = 100
min_freq = 1000

DATA_TYPE = "original"  # "downsampled"
FNAME = "20110101"  # "all"

train_data_path = f"./output-mix/AE_CDiff/{seq_len}/{DATA_TYPE}/lstm-att-lrp/splits/{FNAME}/{min_freq}/train_balanced.csv"
valid_data_path = f"./output-mix/AE_CDiff/{seq_len}/{DATA_TYPE}/lstm-att-lrp/splits/{FNAME}/{min_freq}/val.csv"
test_data_path = f"./output-mix/AE_CDiff/{seq_len}/{DATA_TYPE}/lstm-att-lrp/splits/{FNAME}/{min_freq}/test.csv"
examples_path = f"./output-mix/AE_CDiff/{seq_len}/{DATA_TYPE}/lstm-att-lrp/splits/{FNAME}/{min_freq}/visualized_test_patients.csv"
vocab_path = f"./output-mix/AE_CDiff/{seq_len}/{DATA_TYPE}/lstm-att-lrp/splits/{FNAME}/{min_freq}/vocab.pkl"

# Dataset preprocessing
x_train_one_hot_path = f"./output-mix/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data/train_one_hot.csv"
x_valid_one_hot_path = f"./output-mix/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data/val_one_hot.csv"
x_test_one_hot_path = f"./output-mix/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data/test_one_hot.csv"

x_train_data_path = f"./output-mix/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data/train.csv"
x_valid_data_path = f"./output-mix/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data/val.csv"
x_test_data_path = f"./output-mix/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data/test.csv"

GT_CODES_PATH = "../../../data/AE_CDiff_d00845/cdiff_risk_factors_codes.csv"

model_save_path = f"./output-mix3/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/models/model_{'{}'}.pkl"
shap_save_path_pattern = f"./output-mix3/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/shap/{'{}'}_shap_{'{}'}.pkl"  # SHAP values path for a given dataset split (train/val/test) (data format (features, scores, patient_ids))

PARAMS_PATH = f"output-mix3/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/train_results/model_params.json"

# Model training
BUCKET = "cmsai-mrk-amzn"
s3_output_data_dir = f"s3://{BUCKET}/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data"

DATA_PREFIX = f"{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/data"
MODEL_PREFIX = f"{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/models"

output_results_path = f"output-mix3/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/train_results/results.csv"

local_model_dir = f"output-mix3/{dataset}/{seq_len}/{DATA_TYPE}/{model_name}/{FNAME}/{min_freq}/models/"
s3_output_path = f"s3://{BUCKET}/{MODEL_PREFIX}/output"

###Algorithm config
ALGORITHM = "xgboost"
REPO_VERSION = "1.2-1"

EARLY_STOPPING_ROUNDS = 2
N_BACKGROUND = 10000  # None if you want to use all (for shap)
N_SHAP_EXAMPLES = 10000  # None if all

MODEL_PARAMS = {
    "n_jobs": 25,
    "random_state": 10,
    "max_depth": 7,
    "n_estimators": 500,
    "eval_metric": "auc",
    "learning_rate": 0.3,
    "reg_alpha": 0.9,
    "reg_lambda": 0.8,
    "colsample_bytree": 0.35,
    "use_label_encoder": False,
}
# EARLY_STOPPING_ROUNDS = 5

# MODEL_PARAMS = {
#     "n_jobs": 25,
#     "random_state": 10,
#     "max_depth": 2,
#     "n_estimators": 500,
#     "eval_metric": "auc",
#     "learning_rate": 0.3,
#     "reg_alpha": 0.2,
#     "reg_lambda": 0.3,
#     "colsample_bytree": 0.35,
#     "use_label_encoder": False,
# }


# In[176]:


if TRAIN_MODEL:
    # Model Output Directory
    model_save_dir = os.path.dirname(model_save_path)
    shap_save_dir = os.path.dirname(shap_save_path_pattern)

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


# ## 2. XGBoost Model Training

# ### Data Preprocessing

# In[177]:


# df = pd.read_csv(train_data_path)

# columns = [str(i) for i in range(seq_len - 1, -1, -1)]
# df = df[columns]
# print(df.shape)
# df.head()


# In[178]:


# tokens = xgb_utils.get_valid_tokens(df, seq_len)
with open(vocab_path, "rb") as fp:
    vocab = pickle.load(fp)


# In[179]:


tokens = list(vocab._vocab.keys())
tokens.remove("<pad>")
tokens.remove("<unk>")
tokens = [token for token in tokens if "day" not in token]
len(tokens)


# In[180]:


if PREPROCESS_DATA:
    xgb_utils.prepare_data(
        train_data_path,
        x_train_one_hot_path,
        x_train_data_path,
        seq_len,
        target_colname,
        tokens,
        s3_output_data_dir,
        use_freq=USE_FREQ,
    )
    xgb_utils.prepare_data(
        valid_data_path,
        x_valid_one_hot_path,
        x_valid_data_path,
        seq_len,
        target_colname,
        tokens,
        s3_output_data_dir,
        use_freq=USE_FREQ,
    )
    xgb_utils.prepare_data(
        test_data_path,
        x_test_one_hot_path,
        x_test_data_path,
        seq_len,
        target_colname,
        tokens,
        s3_output_data_dir,
        use_freq=USE_FREQ,
    )
# else:
#     local_dir = os.path.dirname(x_train_data_path)
#     xgb_utils.copy_data_to_s3(local_dir, s3_output_data_dir)


# ### XGBoost Model Training

# In[181]:


def load_xgb_classifier(model_path, model_params):
    """Load XGBClassifier model from saved Booster model."""
    xgb_model = xgb.XGBClassifier(**model_params)
    # xgb_model.set_params()
    xgb_model._Booster = sj_utils.load_pickle(model_path)
    return xgb_model


def get_model_paths(model_save_dir, sort=True):
    """Get list models paths in sorted order if needed."""
    fnames = os.listdir(model_save_dir)
    if sort:
        fnames.sort(key=lambda f: int(re.sub("\D", "", f)))
    model_paths = [os.path.join(model_save_dir, fname) for fname in fnames]
    return model_paths


def get_wtau(x, y):
    return stats.weightedtau(x, y, rank=None)[0]


# calculate ground truth scores
def is_value(x, cdiff=False):
    outcome = False
    if cdiff:
        if x.endswith("_rf"):
            outcome = True
    else:
        if not x.endswith("_N"):
            outcome = True
    return outcome


def get_model_intersection_similarity_v2(
    features_scores, cdiff, absolute=True, freedom=1
):
    # gt similarity
    all_features, all_scores = features_scores[0], features_scores[1]
    sims = []
    for i, features in enumerate(all_features):
        scores = all_scores[i]
        gt_features = [feature for feature in features if is_value(feature, cdiff)]
        n_gt = len(gt_features)
        if n_gt > 0:
            dict_features_scores = sj_utils.create_dict_features_scores(
                features, scores, absolute
            )
            top_features_scores = sj_utils.top_k(
                dict_features_scores, len(gt_features) + freedom
            )
            top_features = top_features_scores[0]
            pred_features = [
                feature for feature in top_features if is_value(feature, cdiff)
            ]
            sim = len(set(pred_features).intersection(gt_features)) / float(n_gt)
        else:
            sim = -1
        sims.append(sim)
    avg_sim = sum(sims) / len(sims)
    return avg_sim, sims


def compute_shap(xgb_model, df_train0, df_test0, explainer=None):
    # Load the copied model
    # xgb_model = sj_utils.load_pickle(model_path)

    df_train = df_train0.copy()
    df_test = df_test0.copy()

    feature_names = [
        col
        for col in df_train.columns.tolist()
        if col not in ["patient_id", target_colname]
    ]
    X_train = df_train[feature_names]
    X_test = df_test[feature_names]

    if explainer is None:
        explainer = shap.TreeExplainer(xgb_model, X_train)
    shap_scores = explainer.shap_values(X_test).tolist()
    features = [feature_names[:]] * X_test.shape[0]
    patients = df_test.patient_id.tolist()

    return ((features, shap_scores, patients), explainer)


def save_results(patients, features, shap_scores, y_true, y_pred, output_path):
    """Save all model training results to file."""
    results = {}
    for i, patient_id in enumerate(patients):
        results[patient_id] = {}
        results[patient_id]["features_xgb"] = features[i]
        results[patient_id][target_colname] = y_true[i]
        results[patient_id]["xgb_pred"] = y_pred[i]
        results[patient_id]["xgb_shap"] = shap_scores[i]

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    sj_utils.save_pickle(results, output_path, verbose=False)
    return results


def get_features_and_global_shap(results, exp_num=1):
    """Get a dataframe of features and global shap values."""
    features = None
    ##Get features and global shap
    all_shap = []
    for patient_id, result in results.items():
        if features is None:
            features = results[patient_id]["features_xgb"]
        all_shap.append(results[patient_id]["xgb_shap"])
    all_shap = np.absolute(np.array(all_shap)).mean(axis=0)
    df_shap = pd.DataFrame()
    df_shap["features"] = features
    df_shap["scores"] = all_shap
    df_shap["exp_num"] = exp_num
    return df_shap


# In[182]:


df_train = pd.read_csv(x_train_one_hot_path)
df_val = pd.read_csv(x_valid_one_hot_path)
df_test = pd.read_csv(x_test_one_hot_path)


# In[183]:


df_test.head()


# In[184]:


count_0 = df_train[target_colname].value_counts()[0]
count_1 = df_train[target_colname].value_counts()[1]
print(
    f"Train Classes Count: 0-->{count_0}, 1-->{count_1}, Total-->{count_0+count_1}",
)

count_0 = df_val[target_colname].value_counts()[0]
count_1 = df_val[target_colname].value_counts()[1]
print(
    f"Val Classes Count: 0-->{count_0}, 1-->{count_1}, Total-->{count_0+count_1}",
)

count_0 = df_test[target_colname].value_counts()[0]
count_1 = df_test[target_colname].value_counts()[1]
print(
    f"Test Classes Count: 0-->{count_0}, 1-->{count_1}, Total-->{count_0+count_1}",
)

print(df_train.shape)
df_train.head()


# In[185]:


df_train["d_00845"].value_counts(normalize=False)


# In[186]:


# Get feature names
feature_names = [
    col
    for col in df_train.columns.tolist()
    if col not in ["patient_id", target_colname]
]


# In[187]:


df_val["d_00845"].value_counts(normalize=False)


# In[188]:


df_test["d_00845"].value_counts(normalize=False)


# In[189]:


model_save_dir = os.path.dirname(model_save_path)


# In[190]:


if TRAIN_MODEL:
    if os.path.exists(model_save_dir):
        shutil.rmtree(model_save_dir)
    os.makedirs(model_save_dir, exist_ok=True)

    rounds = 1  # Saves model every one-epoch

    check_point = xgb.callback.TrainingCheckPoint(
        directory=model_save_dir,
        iterations=rounds,
        name="model",
        as_pickle=True,
    )

    X_train, y_train = df_train[feature_names], df_train[target_colname]
    X_val, y_val = df_val[feature_names], df_val[target_colname]

    print("Training XGB Classifier...")
    model = XGBClassifier(**MODEL_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        # eval_set=[(X_train, y_train)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=True,
        callbacks=[check_point],
    )

    # Save Model Parameters
    params_dir = os.path.dirname(PARAMS_PATH)
    os.makedirs(params_dir, exist_ok=True)
    with open(PARAMS_PATH, "w") as fp:
        json.dump(MODEL_PARAMS, fp)

    print("Successfully Done!")


# In[191]:


# # Minimize the number of background examples for SHAP
# if N_BACKGROUND is not None:
#     df_train = df_train.iloc[:N_BACKGROUND]

# if N_SHAP_EXAMPLES is not None:
#     df_val = df_val.iloc[:N_SHAP_EXAMPLES]
#     df_test = df_test.iloc[:N_SHAP_EXAMPLES]


# In[198]:


# Get list of models paths
model_paths = get_model_paths(model_save_dir, sort=True)
# model_paths


# In[ ]:


if TRAIN_MODEL:
    p_rbo = 0.9
    SIMILARITY_FREEDOM = 1  # For Intersection Similarity

    train_aucs = []
    val_aucs = []
    test_aucs = []

    train_losses = []
    val_losses = []
    test_losses = []

    gain_permute_rbo_scores = []
    gain_permute_tau_scores = []

    val_shap_sim_lst = []
    test_shap_sim_lst = []

    val_gain_shap_rbo_lst = []
    val_gain_shap_tau_lst = []

    epochs = []
    for model_path in model_paths:
        # Get epoch number from the model path
        model_fname = model_path
        model_fname = os.path.basename(model_path)
        epoch = int(re.sub("\D", "", model_fname))
        epochs.append(epoch)

        xgb_model = load_xgb_classifier(model_path, MODEL_PARAMS)

        # feature_importances = xgb_model.feature_importances_
        feature_names = xgb_model.get_booster().feature_names

        # Compute AUCs
        train_y_true = df_train[target_colname]
        train_y_pred = xgb_model.predict_proba(df_train[feature_names])[:, 1]
        train_auc = roc_auc_score(train_y_true, train_y_pred)
        train_aucs.append(train_auc)

        val_y_true = df_val[target_colname]
        val_y_pred = xgb_model.predict_proba(df_val[feature_names])[:, 1]
        val_auc = roc_auc_score(val_y_true, val_y_pred)
        val_aucs.append(val_auc)

        test_y_true = df_test[target_colname]
        test_y_pred = xgb_model.predict_proba(df_test[feature_names])[:, 1]
        test_auc = roc_auc_score(test_y_true, test_y_pred)
        test_aucs.append(test_auc)

        # Losses
        train_loss = log_loss(train_y_true, train_y_pred)
        train_losses.append(train_loss)
        val_loss = log_loss(val_y_true, val_y_pred)
        val_losses.append(val_loss)
        test_loss = log_loss(test_y_true, test_y_pred)
        test_losses.append(test_loss)
        #import pdb; pdb.set_trace()
#         del xgb_model
#         print(
#             f"Epoch: {epoch:02} | Train Loss={train_loss:.4} | Train AUC={train_auc:.4} | Val Loss={val_loss:.4} | Val AUC={val_auc:.4} | Test Loss={test_loss:.4} | Test AUC={test_auc:.4}"
#         )
#         continue
        xgb_gain = pd.DataFrame(
            (xgb_model.get_booster().get_score(importance_type="gain")).items()
        )
        xgb_gain.columns = ["features", "xgb_gain"]

        feat_imp = pd.DataFrame(
            list(zip(feature_names, xgb_model.feature_importances_))
        )
        feat_imp.columns = ["features", "sk_gain"]

        df_gains = xgb_gain.merge(feat_imp, how="inner", on=["features"])

        df_gains["xgb_rank"] = df_gains["xgb_gain"].rank(ascending=False)
        df_gains["sk_rank"] = df_gains["sk_gain"].rank(ascending=False)
        df_gains.sort_values("xgb_gain", ascending=False)
        # print(df_gains)

        feat_imp = pd.DataFrame(
            list(zip(feature_names, xgb_model.feature_importances_))
        )
        feat_imp.columns = ["features", "scores"]
        feat_imp.set_index("features", inplace=True)
        feat_imp.sort_values("scores", ascending=False, inplace=True)
        # feat_imp.plot.bar(figsize=(10, 5))
        # plt.title("Model Feature Importance Scores")

        # Check permutation
#         perm_imp = permutation_importance(
#             xgb_model, df_val[feature_names], df_val[target_colname]
#         )

        perm_df = pd.DataFrame()
        perm_df["features"] = feature_names
        perm_df["scores"] = [-1.0]*len(feature_names) #perm_imp.importances_mean
        perm_df.sort_values("scores", ascending=False, inplace=True)
        perm_df.set_index("features", inplace=True)
        # perm_df.plot.bar(figsize=(10, 5))
        # plt.title("Permutation Importance Scores")

        vals = feat_imp.merge(perm_df, left_index=True, right_index=True)
        vals["gain_rank"] = vals.scores_x.rank(ascending=False)
        vals["permute_rank"] = vals.scores_y.rank(ascending=False)

        gain_rank = vals.sort_values("gain_rank").index
        permute_rank = vals.sort_values("permute_rank").index

        # Plot for different p values
        #     rbo_scores = []
        #     p_vals = []
        #     p_range = range(1, 10)
        #     for p in p_range:
        #         rbo_score = rbo.RankingSimilarity(gain_rank.values, permute_rank.values).rbo(
        #             p=1.0 / p
        #         )
        #         rbo_scores.append(rbo_score)
        #         p_vals.append(1.0 / p)

        #     # plt.figure(figsize=(10, 5))
        #     plt.plot(p_vals, rbo_scores, marker="x")
        #     plt.title("RBO Rankings for different p values.")
        #     plt.xlabel("p")
        #     plt.ylabel("RBO")

        #         rbo_score = rbo.RankingSimilarity(gain_rank.values, permute_rank.values).rbo(
        #             p=p_rbo
        #         )
        rbo_score = -1.0
        gain_permute_rbo_scores.append(rbo_score)
        tau_score = get_wtau(vals.scores_x, vals.scores_y)
        gain_permute_tau_scores.append(tau_score)

        val_shap_results, explainer = compute_shap(
            xgb_model, df_train.iloc[:10000], df_val.iloc[:10000], explainer=None
        )
        (val_features, val_scores, val_patients) = val_shap_results
        avg_sim, sim0 = get_model_intersection_similarity_v2(
            (val_features, val_scores),
            cdiff=True,
            freedom=SIMILARITY_FREEDOM,
            absolute=True,
        )
        print(f'#unique val sims: {len(set(sim0))}')
        #import pdb; pdb.set_trace()
        val_shap_sim_lst.append(avg_sim)

        # Compute the shap global feature importance
        shap_global = np.absolute(np.array(val_scores)).mean(axis=0)
        df_shap = pd.DataFrame()
        df_shap["features"] = feature_names
        df_shap["scores"] = shap_global.tolist()
        df_shap.set_index("features", inplace=True)
        # df_shap["shap_rank"] = df_shap.scores.rank(ascending=False)
        # df_shap.sort_values("scores", ascending=False, inplace=True)

        vals = feat_imp.merge(df_shap, left_index=True, right_index=True)
        vals["gain_rank"] = vals.scores_x.rank(ascending=False)
        vals["shap_rank"] = vals.scores_y.rank(ascending=False)

        gain_rank = vals.sort_values("gain_rank").index
        shap_rank = vals.sort_values("shap_rank").index

        #         rbo_score_shap = rbo.RankingSimilarity(gain_rank.values, shap_rank.values).rbo(
        #             p=p_rbo
        #         )
        rbo_score_shap = -1.0
        val_gain_shap_rbo_lst.append(rbo_score_shap)

        tau_score_shap = get_wtau(vals.scores_x, vals.scores_y)
        val_gain_shap_tau_lst.append(tau_score_shap)
        test_shap_results, _ = compute_shap(xgb_model, df_train.iloc[:10000], df_test.iloc[:10000], explainer)
        (test_features, test_scores, test_patients) = test_shap_results
        avg_sim, sim0 = get_model_intersection_similarity_v2(
            (test_features, test_scores),
            cdiff=True,
            freedom=SIMILARITY_FREEDOM,
            absolute=True,
        )
        print(f'#unique test sims: {len(set(sim0))}')
        test_shap_sim_lst.append(avg_sim)
        # Save training results to file.
        shap_path = shap_save_path_pattern.format("val", epoch)
        results = save_results(
            val_patients, val_features, val_scores, val_y_true, val_y_pred, shap_path
        )

        shap_path = shap_save_path_pattern.format("test", epoch)
        results = save_results(
            test_patients,
            test_features,
            test_scores,
            test_y_true,
            test_y_pred,
            shap_path,
        )
        del xgb_model
        print(
            f"Epoch: {epoch:02} | Train Loss={train_loss:.4} | Train AUC={train_auc:.4} | Val Loss={val_loss:.4} | Val AUC={val_auc:.4} | RBO(p={p_rbo})={rbo_score:.4} | TAU={tau_score:.4}"
        )


# In[ ]:


if TRAIN_MODEL:
    # Aggregate results
    columns = [
        "train_AUC",
        "val_AUC",
        "test_AUC",
        "train_Loss",
        "val_Loss",
        "test_Loss",
        "gain_permute_rbo",
        "gain_permute_tau",
        "val_GT_shap_sim",
        "test_GT_shap_sim",
        "val_gain_shap_rbo",
        "val_gain_shap_tau",
    ]

    df_results = pd.DataFrame(
        np.array(
            [
                train_aucs,
                val_aucs,
                test_aucs,
                train_losses,
                val_losses,
                test_losses,
                gain_permute_rbo_scores,
                gain_permute_tau_scores,
                val_shap_sim_lst,
                test_shap_sim_lst,
                val_gain_shap_rbo_lst,
                val_gain_shap_tau_lst,
            ]
        ).T,
        columns=columns,
    )
    df_results["epoch"] = epochs
    df_results.set_index("epoch", inplace=True)

    # save results summary
    output_dir = os.path.dirname(output_results_path)
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(output_results_path)
else:
    # save results summary
    df_results = pd.read_csv(output_results_path)
    df_results.set_index("epoch", inplace=True)


# In[ ]:


# df_results.head()


# # In[ ]:


# figsize = (10, 5)
# df_results.reset_index().plot(
#     figsize=figsize,
#     x="epoch",
#     y=["train_AUC", "val_AUC"],
#     kind="line",
#     marker="x",
# )
# plt.title("Training vs Validation AUC for XGB")

# df_results.reset_index().plot(
#     figsize=figsize,
#     x="epoch",
#     y=["train_Loss", "val_Loss"],
#     kind="line",
#     marker="x",
# )
# plt.title("Training vs Validation Loss for XGB")

# df_results.reset_index().plot(
#     figsize=figsize,
#     x="epoch",
#     y=["val_GT_shap_sim"],
#     kind="line",
#     marker="x",
# )
# plt.title("SHAP vs GT Similarity on All Validation Examples")

# df_results.reset_index().plot(
#     figsize=figsize,
#     x="epoch",
#     y=["val_gain_shap_tau"],
#     kind="line",
#     marker="x",
# )
# plt.title("Information Gain vs SHAP with Kendall-T on Validation Examples")

# df_results.reset_index().plot(
#     figsize=figsize,
#     x="epoch",
#     y=["gain_permute_tau"],
#     kind="line",
#     marker="x",
# )
# plt.title("Information Gain vs Permutation Importance with Kendall-T")

# df_results.reset_index().plot(
#     figsize=figsize,
#     x="epoch",
#     y=["gain_permute_tau", "val_AUC"],
#     kind="line",
#     marker="x",
# )
# plt.title("Gain/Permutation withKendall-T vs Validation AUC")

# plt.show()


# # In[ ]:


# # Visualize SHAP


# # In[ ]:


# epochs = range(1, len(model_paths) + 1)
# figsize = (15, 10)
# max_tokens = 30
# for epoch in epochs[:1]:
#     shap_path = shap_save_path_pattern.format("val", epoch)
#     print(shap_path)
#     val_results = sj_utils.load_pickle(shap_path)
#     df_shap = get_features_and_global_shap(val_results)
#     df_shap = df_shap.sort_values("scores", ascending=True)
#     if max_tokens is not None:
#         df_shap = df_shap.iloc[-max_tokens:, :]
#     df_shap.plot(
#         figsize=figsize,
#         x="features",
#         y="scores",
#         kind="barh",
#     )
#     plt.title(f"Global Feature Importance (epoch={epoch})")
#     plt.show()


# # In[ ]:


# data_dir = os.path.dirname(train_data_path)
# patients_path = os.path.join(data_dir, "visualized_test_patients.txt")

# with open(patients_path, "r") as fp:
#     selected_patients = fp.readlines()
#     print(selected_patients)
#     selected_patients = [pat.strip() for pat in selected_patients]

# selected_patients


# # In[ ]:


# selected_patients = [
#     "BQ2TYCPYR_20110501",
#     "APDBFR363_20110201",
#     "FMBUO6UJH_20110101",
#     "VCN68G35I_20110701",
# ]


# # In[ ]:


# max_tokens = 50
# sorted_epoch = 1

# n_jobs = len(model_paths)
# step = int(n_jobs / 2) - 1
# # epochs = range(1, len(model_paths) + 1, 2)
# epochs = range(1, len(model_paths) + 1, step)
# for patient_id in selected_patients:
#     df = pd.DataFrame()
#     label = None
#     pred_prob = None
#     for epoch in epochs:
#         shap_path = shap_save_path_pattern.format("test", epoch)
#         test_results = sj_utils.load_pickle(shap_path)
#         if not len(df):
#             features = test_results[patient_id]["features_xgb"]
#             df["features"] = features
#             df["values"] = df_test[features][
#                 df_test["patient_id"] == patient_id
#             ].values[0]
#             df["features"] = df["values"].astype(str) + "_" + df["features"]
#             label = test_results[patient_id][target_colname]
#         pred_prob = test_results[patient_id]["xgb_pred"]
#         df[f"epoch_{epoch}"] = test_results[patient_id]["xgb_shap"]
#     if max_tokens is not None:
#         df = df.sort_values(f"epoch_{sorted_epoch}", ascending=False)
#         df = df.iloc[:max_tokens, :]
#     figsize = (15, 10)
#     epoch_cols = [f"epoch_{epoch}" for epoch in epochs]
#     df.plot(
#         figsize=figsize,
#         x="features",
#         y=epoch_cols,
#         kind="bar",
#         width=0.8,
#     )
#     plt.title(f"SHAP Scores for {patient_id}: label={label}, pred={pred_prob:.4}")
#     plt.show()


# # In[ ]:


# # n_exps = 10
# # df = pd.DataFrame()
# # for n_exp in range(1, n_exps + 1):
# #     summary_path = f"output/Final_Manually_Tuned/{seq_len}/{n_exp:02}/{model_name}/train_results/results.csv"
# #     df_summary = pd.read_csv(summary_path)
# #     df_summary.set_index("epoch", inplace=True)
# #     best_epoch = df_summary["val_GT_shap_sim"].idxmax()

# #     best_results_path = f"./output/Final_Manually_Tuned/{seq_len}/{n_exp:02}/{model_name}/shap/test_shap_{best_epoch}.pkl"
# #     dict_results = sj_utils.load_pickle(best_results_path)
# #     df_shap = get_features_and_global_shap(dict_results, n_exp)

# #     df = pd.concat([df, df_shap], axis=0)


# # In[ ]:


# # df.head()


# # In[ ]:


# # plt.figure(figsize=(15, 10))
# # df_mean = df.groupby("features")["scores"].mean()
# # df_mean.sort_values(ascending=False, inplace=True)
# # df_mean_cols = df_mean.index
# # ax = sns.boxplot(x="features", y="scores", data=df, order=df_mean_cols)
# # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# # plt.show()


# # In[ ]:


# # Copy best models results to S3
# # n_exps = 10
# # for n_exp in range(1, n_exps + 1):
# #     print(f"Copying best model data to s3 for exp={n_exp}...")
# #     summary_path = f"output/Final_Manually_Tuned/{seq_len}/{n_exp:02}/{model_name}/train_results/results.csv"
# #     df_summary = pd.read_csv(summary_path)
# #     df_summary.set_index("epoch", inplace=True)
# #     best_epoch = df_summary["val_GT_shap_sim"].idxmax()

# #     model_path = f"./output/Final_Manually_Tuned/30/10/xgb/models/model_{best_epoch}.pkl"
# #     val_best_path = f"./output/Final_Manually_Tuned/{seq_len}/{n_exp:02}/{model_name}/shap/val_shap_{best_epoch}.pkl"
# #     test_best_path = f"./output/Final_Manually_Tuned/{seq_len}/{n_exp:02}/{model_name}/shap/test_shap_{best_epoch}.pkl"

# #     s3_dir = f"s3://merck-paper-bucket/Synthetic-events/final_event_30_xgb/{n_exp}/"

# #     # copy files to s3
# #     command_pattern = f"aws s3 cp {'{}'} {s3_dir}"

# #     command = command_pattern.format(summary_path)
# #     os.system(command)

# #     command = command_pattern.format(model_path)
# #     os.system(command)

# #     command = command_pattern.format(val_best_path)
# #     os.system(command)

# #     command = command_pattern.format(test_best_path)
# #     os.system(command)
# # print("Models data successfully copied to S3!")


# # In[ ]:




