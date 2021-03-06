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
import random

from collections import defaultdict, OrderedDict, Counter

import deep_id_pytorch
from utils import *

import gc

def get_xgboost_background(
    data, n_background=500, negative_only=True, target_col="label", positive_only=False
):
    """
    Get background examples for computing xgb shap values.
    If n_background=None, selects all examples as background.
    """
    background = data.copy()
    if negative_only:
        background = background[background[target_col] == 0]
    elif positive_only:
        background = background[background[target_col] == 1]

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


def get_lstm_data(dataloader, n_examples, positive_only=False, is_random=False):
    """
    Get n_examples of data from all batches.
    Args:
        dataloader(Object): LSTM Dataset Loader
        n_examples(int): Number of examples to be selected. If None, it selects all
        positive_only(bool): Whether it selects only positive examples
        is_random(bool): If examples randomly selected or only the first n_examples
    """
    sel_set = []
    for batch in dataloader:
        if positive_only:
            sel_set.extend(
                [
                    (uid, lab, idxes)
                    for (uid, lab, idxes) in zip(batch[0], batch[1], batch[2])
                    if lab == 1
                ]
            )
        else:
            sel_set.extend(
                [
                    (uid, lab, idxes)
                    for (uid, lab, idxes) in zip(batch[0], batch[1], batch[2])
                ]
            )

        if n_examples is not None:
            if not is_random:
                if len(sel_set) > n_examples:
                    sel_set = sel_set[:n_examples]
                    break

    if (n_examples is not None) and (is_random):
        sel_set = random.sample(sel_set, n_examples)

    data_ids = [x[0] for x in sel_set]
    data_labels = [x[1] for x in sel_set]
    data_idxes = [x[2] for x in sel_set]

    data = (data_ids, data_labels, data_idxes)
    return data


def get_all_lstm_shap(dataloader, seq_len, model, explainer, positive_only=False, n_test=None):
    """
    Get all shap values.
    Args:
        dataloader(Object): LSTM Dataset Loader
        n_examples(int): Number of examples to be selected. If None, it selects all
        positive_only(bool): Whether it selects only positive examples
        is_random(bool): If examples randomly selected or only the first n_examples
    """
    features = []
    scores = []
    patients = []
    i = 0
    stop = False
    for batch in dataloader:
        for (uid, lab, idxes) in zip(batch[0], batch[1], batch[2]):
            if positive_only and (lab != 1):
                continue
            (test_ids, test_labels, test_idxes) = ([uid], [lab], [idxes])
            test_data, test_masks = model.get_all_ids_masks(test_idxes, seq_len)
            lstm_shap_values = explainer.shap_values(test_data, test_masks)
            
            df_shap, patient_id = get_per_patient_shap(
                lstm_shap_values, (test_ids, test_labels, test_idxes), model.vocab, 0
            )
            events = df_shap["events"].values.tolist()
            vals = df_shap["shap_vals"].values.tolist()
            pad = "<pad>"
            if pad in events:
                pad_indx = events.index(pad)
                events = events[:pad_indx]

                vals = vals[:pad_indx]
            features.append(events)
            scores.append(vals[:])
            patients.append(patient_id)
            
            i += 1
            if (n_test is not None) and (i >= n_test):
                stop = True
                break
        print(events)
        print(vals)
        print(patient_id)
        if stop:
            break
            
    
    return (features, scores, patients)

def get_lstm_features_and_shap_scores(
    model,
    tr_dataloader,
    te_dataloader,
    seq_len,
    shap_path,
    save_output=True,
    n_test=None,
    n_background=None,
    background_negative_only=False,
    test_positive_only=False,
    is_test_random=False,
    output_explainer=False
):
    """Get all features and shape importance scores for each example in te_dataloader."""
    # Get background
    print('Getting background')
    background = get_lstm_background(
        tr_dataloader, n_background=n_background, negative_only=background_negative_only
    )
    # background = next(iter(tr_dataloader))
    background_ids, background_labels, background_idxes = background
    print('Done getting background')
    print(os.system('echo nvidia-smi'))
    print('Getting get_all_ids_masks')
    bg_data, bg_masks = model.get_all_ids_masks(background_idxes, seq_len)
    print('Done get_all_ids_masks')
    print(os.system('echo nvidia-smi'))
    print('Getting CustomPyTorchDeepIDExplainer')
    explainer = deep_id_pytorch.CustomPyTorchDeepIDExplainer(
        model, bg_data, bg_masks, gpu_memory_efficient=True
    )
    print('Done get_all_ids_masks')
    print(os.system('echo nvidia-smi'))

    model.train()  # in case that shap complains that autograd cannot be called

    shap_values = None
    if (n_test is None) or (n_test > 32):
        print('Getting shap values (1)')
        shap_values = get_all_lstm_shap(te_dataloader, seq_len, model, explainer, test_positive_only, n_test)
        print('Done getting shap values (1)')
        print(os.system('echo nvidia-smi'))
    else:
        print('Getting shap values (2)')
        # test = next(iter(te_dataloader))
        test = get_lstm_data(
            te_dataloader,
            n_test,
            positive_only=test_positive_only,
            is_random=is_test_random,
        )
        test_ids, test_labels, test_idxes = test
        test_data, test_masks = model.get_all_ids_masks(test_idxes, seq_len)

        lstm_shap_values = explainer.shap_values(test_data, test_masks)

        features = []
        scores = []
        patients = []
        total = len(test[0])
        for idx in range(total):
            df_shap, patient_id = get_per_patient_shap(
                lstm_shap_values, test, model.vocab, idx
            )
            events = df_shap["events"].values.tolist()
            vals = df_shap["shap_vals"].values.tolist()

            pad = "<pad>"
            if pad in events:
                pad_indx = events.index(pad)
                events = events[:pad_indx]

                vals = vals[:pad_indx]
            features.append(events)
            scores.append(vals[:])
            patients.append(patient_id)
        print('Done getting shap values (2)')


        #For explainer
        shap_values = (features, scores, patients)

#     import gc
#     for obj in gc.get_objects():
#         try:
#             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                 print(type(obj), obj.size())
#         except:
#             pass
#     import pdb; pdb.set_trace()

    if save_output:
        if not os.path.isdir(os.path.split(shap_path)[0]):
            os.makedirs(os.path.split(shap_path)[0])
        save_pickle(shap_values, shap_path)

    if output_explainer:
        return shap_values, explainer.expected_value

    return shap_values


def get_xgboost_features_and_shap_scores(
    model,
    df_tr,
    df_te,
    shap_path,
    save_output=True,
    n_test=None,
    n_background=None,
    background_negative_only=False,
    background_positive_only=False,
    test_positive_only=False,
    is_test_random=False,
    output_explainer=False
):
    """Get all features and shape importance scores for each example in te_dataloader."""
    target_col = df_tr.columns.tolist()[-1]
    df_tr = get_xgboost_background(
        df_tr,
        n_background=n_background,
        negative_only=background_negative_only,
        target_col=target_col,
        positive_only=background_positive_only,
    )
    if n_test is not None:
        df_te = df_te.iloc[:n_test, :]

    X_train = df_tr.iloc[:, 1:-1]
    X_test = df_te.iloc[:, 1:-1]

    patients = df_te.iloc[:, 0]

    features = X_train.columns.tolist()
    explainer = shap.TreeExplainer(model, X_train)
    if test_positive_only:
        positive_rows = df_te.iloc[:, -1]==1#Only positive labels
        X_test = X_test[positive_rows]
        patients = patients[positive_rows].tolist()
    
    xgb_shap_values = explainer.shap_values(X_test)
    features = [features[:]] * len(X_test)
    scores = xgb_shap_values.tolist()

    if not os.path.isdir(os.path.split(shap_path)[0]):
        os.makedirs(os.path.split(shap_path)[0])

    shap_values = (features, scores, patients)
    if save_output:
        save_pickle(shap_values, shap_path)

    if output_explainer:
        return shap_values, explainer.expected_value
    
    return shap_values


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
    df = pd.DataFrame(
        np.array([events, pat_shap_values]).T, columns=["events", "shap_vals"]
    )
    df["shap_vals"] = pd.to_numeric(df["shap_vals"])
    return df, patient_id


def plot_shap_values(df, patient_id, sort=False, figsize=(15, 7), num_features=None):
    if sort:
        df = df.reindex(
            df.shap_vals.abs().sort_values(ascending=False).index
        ).reset_index()
    if num_features is not None:
        df = df.iloc[:num_features]
    plt.figure(figsize=figsize)
    ax = sns.barplot(x=df.index, y=df.shap_vals, orient="v")
    z = ax.set_xticklabels(df.events, rotation=90)
    plt.title("Patient ID: {}".format(patient_id))
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
    od = OrderedDict(sorted(dict_row_scores.items(), key=lambda x: x[1]))
    top_k_features = list(od.keys())[-k:]
    top_k_scores = list(od.values())[-k:]
    return top_k_features, top_k_scores


def total_jacc(features_scores_a, features_scores_b, k, overlap=False, absolute=True):
    all_features_a, all_scores_a = features_scores_a[0], features_scores_a[1]
    all_features_b, all_scores_b = features_scores_b[0], features_scores_b[1]

    tot_jacc, overlap_tokens = [], []
    for idx, row_features_a in enumerate(all_features_a):
        row_scores_a = all_scores_a[idx]

        row_features_b = all_features_b[idx]
        row_scores_b = all_scores_b[idx]

        dict_features_scores_a = create_dict_features_scores(
            row_features_a, row_scores_a, absolute
        )
        dict_features_scores_b = create_dict_features_scores(
            row_features_b, row_scores_b, absolute
        )

        features_a, scores_a = top_k(dict_features_scores_a, k)
        features_b, scores_b = top_k(dict_features_scores_b, k)

        overlap_words, jacc_score = jacc_simi(features_a, features_b)
        tot_jacc.append(jacc_score)
        overlap_tokens.append(overlap_words)
    if overlap:
        return tot_jacc, overlap_tokens
    else:
        return tot_jacc


def show_heatmap(models, data, k, fig_size, vmin, vmax):
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=fig_size)
    sns_plot = sns.heatmap(df, annot=True, cmap="PuBu", vmin=vmin, vmax=vmax, linewidths=.5)
    plt.xticks(np.arange(len(models)) + 0.5, tuple(models))
    plt.yticks(np.arange(len(models)) + 0.5, tuple(models), va="center")
    fig = sns_plot.get_figure()
    plt.title("k = {}".format(k))
    plt.show()
    return fig


def generate_heatmap_data(all_features_scores, k, absolute=True):
    data = []
    num_models = len(all_features_scores)
    for i in range(num_models):
        tmp = []
        features_scores_a = all_features_scores[i]
        for j in range(num_models):
            features_scores_b = all_features_scores[j]
            total = total_jacc(features_scores_a, features_scores_b, k, absolute=absolute)
            avg_jacc = np.mean(total)
            tmp.append(avg_jacc)
        data.append(tmp)
    return data


def generate_heatmap(all_features_scores, models, k, absolute=True):
    data = generate_heatmap_data(all_features_scores, k, absolute=absolute)
    fig = show_heatmap(models, data, k, (7, 5), 0, 1.0)


def generate_k_heatmaps(all_features_scores, models, k_list, absolute=True):
    for k in k_list:
        generate_heatmap(all_features_scores, models, k, absolute=absolute)


def get_model_intersection_similarity(all_features_scores, suffices=["_H", "_A"], absolute=True, df_one_hot=None):
    """Get similarity between the ground truth & model predicted helping events (Adverse and Helper)
    Note: df_one_hot is the one-hot encoding of the input data (used only in xgb).
    """

    def _get_helping_features(features, suffices, one_hot=None):
        """Get only helping features (Ending with _H & _A)"""
        helping_features = []
        for suf in suffices:
            if one_hot is None:
                h = [event for event in features if event.endswith(suf)]
            else:
                h = [event for event in features if event.endswith(suf)]
            h = [event for event in features if event.endswith(suf)]
            helping_features += h
        return helping_features

    all_features, all_scores = all_features_scores[0], all_features_scores[1]
    num_examples = len(all_features)
    similarities = []
    for i in range(num_examples):
        row_features = all_features[i]
        row_scores = all_scores[i]
        row_one_hot = None
        if df_one_hot is not None:
            row_one_hot = df_one_hot.iloc[i]

        gt_helpers = _get_helping_features(row_features, suffices, one_hot= row_one_hot)
        if len(gt_helpers) == 0:
            sim = -1
        else:
            dict_features_scores = create_dict_features_scores(row_features, row_scores, absolute)
            top_features_scores = top_k(dict_features_scores, len(gt_helpers))
            top_features = top_features_scores[0]
            pred_helpers = _get_helping_features(top_features, suffices)
            sim = float(len(set(pred_helpers).intersection(gt_helpers))) / len(gt_helpers)
        similarities.append(sim)
    return sum(similarities) / len(similarities), similarities


def get_model_paths(input_path_pattern):
    """Get list of model paths in sorted form."""
    model_dir = os.path.dirname(input_path_pattern)
    fnames = os.listdir(model_dir)
    fnames = [fname for fname in fnames if fname.endswith(".pkl")]
    fnames = sorted(fnames)
    model_paths = [os.path.join(model_dir, fname) for fname in fnames]
    return model_paths


def plot_histogram(
    data,
    title,
    xlabel,
    ylabel,
    axes,
    axes_idx=0,
    xlim=0.0,
    ylim=1.0,
    bins=50,
    kde=False,
):
    """Plots a histogram based on a given data."""
    if axes is None:
        plt.figure(figsize=(10, 5))
        sns.distplot(data, bins=bins, kde=kde, axlabel=xlabel)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.show()
    else:
        sns.distplot(data, bins=bins, kde=kde, ax=axes[axes_idx], axlabel=xlabel)
        axes[axes_idx].set_title(title)
        axes[axes_idx].set_ylabel(ylabel)
        axes[axes_idx].set_xlim(xlim)
        axes[axes_idx].set_ylim(ylim)
    return axes


def get_global_feature_importance(all_features, all_scores, absolute=True):
    """Get global feature importances from the per-patient features and scores."""
    num_examples = len(all_features)
    my_all_features = defaultdict(list)
    for i in range(num_examples):
        features1 = all_features[i]
        scores1 = all_scores[i]
        feat_sc1 = create_dict_features_scores(features1, scores1, absolute)
        for k, v in feat_sc1.items():
            my_all_features[k].append(v)
    # print(my_all_features)
    for k, v in my_all_features.items():
        my_all_features[k] = float(np.mean(my_all_features[k]))
    my_all_features = OrderedDict(
        sorted(my_all_features.items(), key=lambda x: x[1], reverse=False)
    )
    return my_all_features


def plot_global_feature_importance(feature_importances, max_features=None):
    """Plots the global feature importances in horizontal barplot"""
    df = pd.DataFrame(feature_importances, index=range(1)).T
    if max_features is not None:
        df = df.sort_values(0, ascending=False)
        df = df.iloc[:max_features]
        df = df.sort_values(0, ascending=True)
    df.plot.barh(figsize=(10, 20), legend=False)
    plt.show()
    return df


def get_epoch_number_from_path(model_path):
    """Gets the epoch number from the given model path."""
    epoch = model_path.rsplit(".", 1)[0]
    epoch = epoch.rsplit("_", 1)[-1]
    epoch = int(epoch)
    return epoch


def create_dict_features_scores(features, scores, absolute=True):
    """
    Create a dictionary of features and scores.
    If there are duplicate keys, it will average their their scores.
    It also converts each score to its corresponding absolute value
    If absolute, it takes the absolute value of the shap score.
    """
    features_scores = defaultdict(list)
    for i, feature in enumerate(features):
        if absolute:
            features_scores[feature].append(abs(scores[i]))
        else:
            features_scores[feature].append(scores[i])
    for key, vals in features_scores.items():
        features_scores[key] = float(np.mean(vals))
    return features_scores
