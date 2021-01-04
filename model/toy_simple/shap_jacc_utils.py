"""
This module contains utility functions for computing SHAP 
and Jaccard similarities between shap values of different models
"""
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
import os
import torch
import shap

from collections import defaultdict, OrderedDict, Counter

import deep_id_pytorch
from utils import *


def get_xgboost_background(data, n_background=500, negative_only=True, target_col='label'):
    """
    Get background examples for computing xgb shap values.
    If n_background=None, selects all examples as background.
    """
    background = data.copy()
    if negative_only:
        background = background[background[target_col]==0]
        
    if n_background is None:
        return background
    
    background = background.iloc[:n_background, :]
    return background


def get_lstm_background(dataloader, n_background=500, negative_only=True):
    """Get background examples for computing lstm shap values."""
    sel_set = []
    for batch in dataloader:
        if negative_only:
            sel_set.extend(
                [
                    (uid, lab, idxes)
                    for (uid, lab, idxes) in zip(batch[0], batch[1], batch[2])
                    if lab == 0
                ]
            )
        else:
            sel_set.extend(
                [
                    (uid, lab, idxes)
                    for (uid, lab, idxes) in zip(batch[0], batch[1], batch[2])
                ]
            )
            
        if n_background is not None:
            if len(sel_set) > n_background:
                sel_set = sel_set[:n_background]
                break

    background_ids = [x[0] for x in sel_set]
    background_labels = [x[1] for x in sel_set]
    background_idxes = [x[2] for x in sel_set]
    
    background = (background_ids, background_labels, background_idxes)
    return background


def get_lstm_data(dataloader, n_examples):
    """Get data from all batches."""
    sel_set = []
    for batch in dataloader:
        sel_set.extend(
            [
                (uid, lab, idxes)
                for (uid, lab, idxes) in zip(batch[0], batch[1], batch[2])
            ]
        )
        
        if n_examples is not None:
            if len(sel_set) > n_examples:
                sel_set = sel_set[:n_examples]
                break
            
    data_ids = [x[0] for x in sel_set]
    data_labels = [x[1] for x in sel_set]
    data_idxes = [x[2] for x in sel_set]
    
    data = (data_ids, data_labels, data_idxes)
    return data


def get_lstm_features_and_shap_scores(model, 
                                      tr_dataloader, 
                                      te_dataloader, 
                                      seq_len,
                                      features_path, 
                                      scores_path,
                                      patients_path,
                                      n_test=None,
                                      n_background=None,
                                      negative_only=False):
    """Get all features and shape importance scores for each example in te_dataloader."""
    #Get background
    background = get_lstm_background(tr_dataloader, 
                                     n_background=n_background, 
                                     negative_only=negative_only)
    #background = next(iter(tr_dataloader))
    background_ids, background_labels, background_idxes = background
    bg_data, bg_masks = model.get_all_ids_masks(background_idxes, seq_len)
    explainer = deep_id_pytorch.CustomPyTorchDeepIDExplainer(model, bg_data, bg_masks,
                                                             gpu_memory_efficient=True)

    model.train() # in case that shap complains that autograd cannot be called
    lstm_values = []
    features = []
    start = 0

    #test = next(iter(te_dataloader))
    test = get_lstm_data(te_dataloader, n_test)
    test_ids, test_labels, test_idxes = test
    test_data, test_masks = model.get_all_ids_masks(test_idxes, seq_len)
    lstm_shap_values = explainer.shap_values(test_data, test_masks)

    features = []
    scores = []
    patients = []
    total = len(test[0])
    for idx in range(total):
        df_shap, patient_id = get_per_patient_shap(lstm_shap_values, test, model.vocab, idx)
        events = df_shap['events'].values.tolist()
        vals = df_shap['shap_vals'].values.tolist()
        
        pad = '<pad>'
        if pad in events:
            pad_indx = events.index(pad)
            events = events[:pad_indx]
        
        vals = vals[:pad_indx]
        features.append(events)
        scores.append(vals[:])
        patients.append(patient_id)

    if not os.path.isdir(os.path.split(features_path)[0]):
        os.makedirs(os.path.split(features_path)[0])

    if not os.path.isdir(os.path.split(scores_path)[0]):
        os.makedirs(os.path.split(scores_path)[0])

    save_pickle(features, features_path)
    save_pickle(scores, scores_path)
    save_pickle(patients, patients_path)
    
    return features, scores, patients
    

def get_xgboost_features_and_shap_scores(model, 
                                         df_tr, 
                                         df_te, 
                                         features_path, 
                                         scores_path,
                                         patients_path,
                                         n_test=None,
                                         n_background=None,
                                         negative_only=False):
    """Get all features and shape importance scores for each example in te_dataloader."""
    target_col = df_tr.columns.tolist()[-1]
    df_tr = get_xgboost_background(df_tr, 
                                   n_background=n_background, 
                                   negative_only=negative_only, 
                                   target_col=target_col)
    if n_test is not None:
        df_te = df_te.iloc[:n_test, :]
        
    X_train = df_tr.iloc[:, 1:-1]
    X_test = df_te.iloc[:, 1:-1]
    
    patients = df_te.iloc[:, 0]
    
    features = X_train.columns.tolist()
    explainer = shap.TreeExplainer(model, X_train)
    xgb_shap_values = explainer.shap_values(X_test)
    features = [features[:]]*len(X_test)
    scores = xgb_shap_values.tolist()
    
    if not os.path.isdir(os.path.split(features_path)[0]):
        os.makedirs(os.path.split(features_path)[0])

    if not os.path.isdir(os.path.split(scores_path)[0]):
        os.makedirs(os.path.split(scores_path)[0])

    save_pickle(features, features_path)
    save_pickle(scores, scores_path)
    save_pickle(patients, patients_path)

    
    return features, scores, patients


def get_per_patient_shap(shap_values, data, vocab, idx=0):
    """Get shap values for a single patient."""
    pat_shap_values = []
    patient_id, _, token_idxes = data
    patient_id = patient_id[idx]
    token_idxes = token_idxes[idx]
    events = []
    
    for (i, w) in enumerate(token_idxes):
        pat_shap_values.append(shap_values[idx, i, w].item())
        events.append(vocab.itos(w.item()))
    df = pd.DataFrame(np.array([events, pat_shap_values]).T, columns=['events', 'shap_vals'])
    df["shap_vals"] = pd.to_numeric(df["shap_vals"])
    return df, patient_id

def plot_shap_values(df, patient_id, sort=False):
    if sort:
        df = df.reindex(df.shap_vals.abs().sort_values(ascending=False).index).reset_index()
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=df.index, y=df.shap_vals, orient='v')
    z = ax.set_xticklabels(df.events, rotation=90)
    plt.title('Patient ID: {}'.format(patient_id))
    plt.show()


def convert_to_absolute(d):
    for k, v in d.items():
        d[k] = abs(v)
    return d


def jacc_simi(list1, list2):
    list1 = set(list1)
    list2 = set(list2)
    words = list(list1 & list2)
    intersection = len(words)
    union = (len(list1) + len(list2)) - intersection
    return words, float(intersection / union)


def top_k(dict_row_scores, k):
    d = convert_to_absolute(dict_row_scores)
    od = OrderedDict(sorted(d.items(), key=lambda x: x[1]))
    top_k_features = list(od.keys())[-k:]
    top_k_scores = list(od.values())[-k:]
    return top_k_features, top_k_scores


def total_jacc(features_scores_a, features_scores_b, k, overlap=False):
    all_features_a, all_scores_a = features_scores_a[0], features_scores_a[1]
    all_features_b, all_scores_b = features_scores_b[0], features_scores_b[1]

    total_jacc, overlap_tokens = [], []
    for idx, row_features_a in enumerate(all_features_a):
        row_scores_a = all_scores_a[idx]
        
        row_features_b = all_features_b[idx]
        row_scores_b = all_scores_b[idx]

        dict_features_scores_a = dict(zip(row_features_a, row_scores_a))
        dict_features_scores_b = dict(zip(row_features_b, row_scores_b))
        
        features_a, scores_a = top_k(dict_features_scores_a, k)
        features_b, scores_b = top_k(dict_features_scores_b, k)

        overlap_words, jacc_score = jacc_simi(features_a, features_b)
        total_jacc.append(jacc_score)
        overlap_tokens.append(overlap_words)
    if overlap:
        return total_jacc, overlap_tokens
    else:
        return total_jacc


def show_heatmap(models, data, k, fig_size, vmin, vmax):
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=fig_size)
    sns_plot = sns.heatmap(df, annot=True, cmap="PuBu", vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(len(models))+0.5, tuple(models))
    plt.yticks(np.arange(len(models))+0.5, tuple(models), va="center")
    fig = sns_plot.get_figure()
    plt.title('k = {}'.format(k))
    plt.show()
    return fig


def generate_heatmap_data(all_features_scores, k):
    data = []
    num_models = len(all_features_scores)
    for i in range(num_models):
        tmp = []
        features_scores_a = all_features_scores[i]
        for j in range(num_models):
            features_scores_b = all_features_scores[j]
            total = total_jacc(features_scores_a, features_scores_b, k)
            avg_jacc = np.mean(total)
            tmp.append(avg_jacc)
        data.append(tmp)
    return data


def generate_heatmap(all_features_scores, models, k):
    data = generate_heatmap_data(all_features_scores, k)
    fig = show_heatmap(models, data, k, (15, 10), 0, 1.0)

    
def generate_k_heatmaps(all_features_scores, models, k_list):
    for k in k_list:
        generate_heatmap(all_features_scores, models, k)