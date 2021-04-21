"""
This module contains utility functions for computing SHAP 
and Jaccard similarities between shap values of different models
"""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import os
import torch
import shap
import random
import copy
from multiprocessing.dummy import Pool as dThreadPool
from collections import defaultdict, OrderedDict, Counter
import time
import re

import deep_id_pytorch
from utils import *


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


def get_all_lstm_shap(
    dataloader, seq_len, model, explainer, positive_only=False, n_test=None
):
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
    output_explainer=False,
):
    """Get all features and shape importance scores for each example in te_dataloader."""
    # Get background
    print("Getting background")
    background = get_lstm_background(
        tr_dataloader, n_background=n_background, negative_only=background_negative_only
    )
    # background = next(iter(tr_dataloader))
    background_ids, background_labels, background_idxes = background
    print("Done getting background")
    print(os.system("echo nvidia-smi"))
    print("Getting get_all_ids_masks")
    bg_data, bg_masks = model.get_all_ids_masks(background_idxes, seq_len)
    print("Done get_all_ids_masks")
    print(os.system("echo nvidia-smi"))
    print("Getting CustomPyTorchDeepIDExplainer")
    explainer = deep_id_pytorch.CustomPyTorchDeepIDExplainer(
        model, bg_data, bg_masks, gpu_memory_efficient=True
    )
    print("Done get_all_ids_masks")
    print(os.system("echo nvidia-smi"))

    model.train()  # in case that shap complains that autograd cannot be called

    shap_values = None
    if (n_test is None) or (n_test > 32):
        print("Getting shap values (1)")
        shap_values = get_all_lstm_shap(
            te_dataloader, seq_len, model, explainer, test_positive_only, n_test
        )
        print("Done getting shap values (1)")
        print(os.system("echo nvidia-smi"))
    else:
        print("Getting shap values (2)")
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
        print("Done getting shap values (2)")

        # For explainer
        shap_values = (features, scores, patients)

    if save_output:
        if not os.path.isdir(os.path.split(shap_path)[0]):
            os.makedirs(os.path.split(shap_path)[0])
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
    """Plot shap values."""
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
    """Convert dictionary values to their absolute."""
    for k, v in d.items():
        d[k] = abs(v)
    return d


def jacc_simi(list1, list2):
    """Compute jaccard similarity from two lists."""
    list1 = set(list1)
    list2 = set(list2)
    words = list(list1 & list2)
    intersection = len(words)
    union = (len(list1) + len(list2)) - intersection
    return words, float(intersection / union)


def top_k(dict_row_scores, k):
    """Get top-k features based on their scores/values."""
    od = OrderedDict(sorted(dict_row_scores.items(), key=lambda x: x[1]))
    top_k_features = list(od.keys())[-k:]
    top_k_scores = list(od.values())[-k:]
    return top_k_features, top_k_scores


def total_jacc(features_scores_a, features_scores_b, k, overlap=False, absolute=True):
    """Get total jaccard similarity."""
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
    """Display heatmap."""
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=fig_size)
    sns_plot = sns.heatmap(
        df, annot=True, cmap="PuBu", vmin=vmin, vmax=vmax, linewidths=0.5
    )
    plt.xticks(np.arange(len(models)) + 0.5, tuple(models))
    plt.yticks(np.arange(len(models)) + 0.5, tuple(models), va="center")
    fig = sns_plot.get_figure()
    plt.title("k = {}".format(k))
    plt.show()
    return fig


def generate_heatmap_data(all_features_scores, k, absolute=True):
    """Generate heatmap data based on the given features and scores."""
    data = []
    num_models = len(all_features_scores)
    for i in range(num_models):
        tmp = []
        features_scores_a = all_features_scores[i]
        for j in range(num_models):
            features_scores_b = all_features_scores[j]
            total = total_jacc(
                features_scores_a, features_scores_b, k, absolute=absolute
            )
            avg_jacc = np.mean(total)
            tmp.append(avg_jacc)
        data.append(tmp)
    return data


def generate_heatmap(all_features_scores, models, k, absolute=True):
    """Generate heatmap."""
    data = generate_heatmap_data(all_features_scores, k, absolute=absolute)
    fig = show_heatmap(models, data, k, (7, 5), 0, 1.0)


def generate_k_heatmaps(all_features_scores, models, k_list, absolute=True):
    """Generate k heatmaps."""
    for k in k_list:
        generate_heatmap(all_features_scores, models, k, absolute=absolute)


def get_model_intersection_similarity(
    all_features_scores, suffices=["_H", "_A"], absolute=True, df_one_hot=None
):
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

        gt_helpers = _get_helping_features(row_features, suffices, one_hot=row_one_hot)
        if len(gt_helpers) == 0:
            sim = -1
        else:
            dict_features_scores = create_dict_features_scores(
                row_features, row_scores, absolute
            )
            top_features_scores = top_k(dict_features_scores, len(gt_helpers))
            top_features = top_features_scores[0]
            pred_helpers = _get_helping_features(top_features, suffices)
            sim = float(len(set(pred_helpers).intersection(gt_helpers))) / len(
                gt_helpers
            )
        similarities.append(sim)
    return sum(similarities) / len(similarities), similarities


def get_model_paths(model_save_dir, model_name="lstm", sort=True):
    """Get list models paths in sorted order if needed."""
    if model_name == "lstm":
        model_save_dir = os.path.dirname(model_save_dir)
    fnames = os.listdir(model_save_dir)
    fnames = [fname for fname in fnames if fname.endswith(".pkl")]
    if sort:
        fnames.sort(key=lambda f: int(re.sub("\D", "", f)))
    model_paths = [os.path.join(model_save_dir, fname) for fname in fnames]
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
    for k, v in my_all_features.items():
        my_all_features[k] = float(np.mean(my_all_features[k]))
    my_all_features = OrderedDict(
        sorted(my_all_features.items(), key=lambda x: x[1], reverse=False)
    )
    return my_all_features


def plot_global_feature_importance(feature_importances, max_features=None, title=None):
    """Plots the global feature importances in horizontal barplot"""
    df = pd.DataFrame(feature_importances, index=range(1)).T
    if max_features is not None:
        df = df.sort_values(0, ascending=False)
        df = df.iloc[:max_features]
        df = df.sort_values(0, ascending=True)
    df.plot(kind="barh", figsize=(10, 15), legend=False, colormap="winter")
    if title is not None:
        plt.title(title)
    plt.show()
    # return df


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


def get_best_epoch(df, by="val_AUC"):
    """Get best epoch based on the given dataframe and column"""
    best_epoch = df[by].idxmax()
    best_epoch = int(best_epoch)
    return best_epoch


def get_wtau(x, y):
    """Get tau scores for two sets."""
    return stats.weightedtau(x, y, rank=None)[0]


def ret_label_df(df, colname, dataset):
    """Get data for plotting."""
    len_epoch = df.shape[0]
    plot_df = df[["epoch", colname]].copy()
    plot_df.columns = ["epoch", "AUC"]
    plot_df["dataset"] = dataset

    return plot_df


def get_eval_data(dataloader, num):
    """Get more than one iteration of data"""
    col_pid, col_lab, col_txt = None, None, None
    col_num = 0
    for pid, lab, txt in dataloader:
        if col_pid is None:
            col_pid, col_lab, col_txt = pid, lab, txt
        else:
            col_pid = tuple(list(col_pid) + list(pid))
            col_lab = torch.cat((col_lab, lab), dim=0)
            col_txt = torch.cat((col_txt, txt), dim=0)
        col_num = len(col_pid)
        if col_num > num:
            break

    return col_pid[:num], col_lab[:num], col_txt[:num]


### Multi-Processing version of SHAP Computation
def glfass_single(cpu_model, background, test, seq_len, device):
    """
    Single-thread function for Get Lstm Features And Shap Scores
    Called by get_lstm_features_and_shap_scores_mp
    """
    # start_time = time.time()

    model = cpu_model.to(device)

    try:

        background_ids, background_labels, background_idxes = background
        bg_data, bg_masks = model.get_all_ids_masks(background_idxes, seq_len)

        explainer = deep_id_pytorch.CustomPyTorchDeepIDExplainer(
            model, bg_data, bg_masks, gpu_memory_efficient=True
        )

        model.train()
        test_ids, test_labels, test_idxes = test
        test_data, test_masks = model.get_all_ids_masks(test_idxes, seq_len)

        lstm_shap_values = explainer.shap_values(
            test_data, test_masks, model_device=device
        )

    except Exception as excpt:
        print(excpt)
        raise Exception

    end_time = time.time()
    return (test_ids, test_labels, test_idxes, lstm_shap_values)


def mycallback(x):
    return x


def myerrorcallback(exception):
    print(exception)
    return exception


def get_lstm_features_and_shap_scores_mp(
    model,
    tr_dataloader,
    test,
    seq_len,
    shap_path,
    save_output=True,
    n_background=None,
    background_negative_only=False,
    test_positive_only=False,
    is_test_random=False,
    output_explainer=False,
    multigpu_lst=None,  # cuda:1, cuda:2 ...
):
    """Get all features and shape importance scores for each example in te_dataloader."""
    # Get background dataset
    background = get_lstm_background(
        tr_dataloader, n_background=n_background, negative_only=background_negative_only
    )
    # split up test datasets

    n_gpu = len(multigpu_lst)
    gpu_model_tuple = []
    for gpu in multigpu_lst:
        model = copy.deepcopy(model)
        model.device = gpu
        model = model.to(gpu)
        gpu_model_tuple.append((gpu, model))

    test_ids, test_labels, test_idxes = test

    test_labels_lst, test_idxes_lst, test_ids_lst = [], [], []
    n_per_gpu = int(np.ceil(len(test_ids) / n_gpu))
    for idx in range(n_gpu):
        if idx == (n_gpu - 1):
            test_ids_lst.append(test_ids[idx * n_per_gpu :])
            test_labels_lst.append(test_labels[idx * n_per_gpu :])
            test_idxes_lst.append(test_idxes[idx * n_per_gpu :])
        else:
            test_ids_lst.append(test_ids[idx * n_per_gpu : (idx + 1) * n_per_gpu])
            test_labels_lst.append(test_labels[idx * n_per_gpu : (idx + 1) * n_per_gpu])
            test_idxes_lst.append(test_idxes[idx * n_per_gpu : (idx + 1) * n_per_gpu])

    # multiprocess one core one gpu
    # print(f'Starting multiprocess for {n_gpu} cores')
    try:
        pool = dThreadPool(n_gpu)
        # pool = torch.multiprocessing.Pool(n_gpu)  # one feeding each gpu
        func_call_lst = []
        for cur_test_id, cur_test_label, cur_test_idxes, (gpu, model) in zip(
            test_ids_lst, test_labels_lst, test_idxes_lst, gpu_model_tuple
        ):
            # print(f"\nlength of tests={len(cur_test_id)}")
            # print(f"gpu: {n_gpu}")
            # print(f"model: {model.device}")

            func_call = pool.apply_async(
                glfass_single,
                (
                    model.cpu(),
                    background,
                    (cur_test_id, cur_test_label, cur_test_idxes),
                    seq_len,
                    gpu,
                ),
                callback=mycallback,
                error_callback=myerrorcallback,
            )
            func_call_lst.append(func_call)

        for func_call in func_call_lst:
            func_call.wait()

        test_ids, test_labels, test_idxes, lstm_shap_values = None, None, None, None
        for func_call in func_call_lst:
            init_results = func_call.get()

            # first one
            if test_ids is None:
                test_ids, test_labels, test_idxes, lstm_shap_values = init_results
                test_ids = list(test_ids)
            else:
                test_ids = test_ids + list(init_results[0])
                test_labels = torch.cat([test_labels, init_results[1]], dim=0)
                test_idxes = torch.cat([test_idxes, init_results[2]], dim=0)
                lstm_shap_values = np.concatenate(
                    [lstm_shap_values, init_results[3]], axis=0
                )

    except Exception as excpt:
        print(excpt)

    finally:
        pool.close()
        pool.join()
        pool.terminate()
    try:
        test = (test_ids, test_labels, test_idxes)
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

        shap_values = (features, scores, patients)
    except Exception as excpt:
        print(excpt)
        raise Exception

    if save_output:
        if not os.path.isdir(os.path.split(shap_path)[0]):
            os.makedirs(os.path.split(shap_path)[0])
        save_pickle(shap_values, shap_path)

    if output_explainer:
        return shap_values, explainer.expected_value

    return shap_values


def get_intersection_similarity(scores, tokens, freedom=0, is_synthetic=True):
    """
    Computes Intersection similarity for a given scores (single patient)
    Args:
        scores(list): List of feature importance scores
        tokens(list): List of tokens
        freedom(int): Degree of freedom for computing GT codes/tokens
        is_synthetic(bool): If dataset is synethetic/real
    Returns:
        The intersection similarity score(float) or -1 if not GT tokens available
    """
    if is_synthetic:
        gt_features = [feature for feature in tokens if not feature.endswith("_N")]
    else:
        gt_features = [feature for feature in tokens if feature.endswith("_rf")]
    n_gt = len(gt_features)
    if n_gt > 0:
        dict_features_scores = create_dict_features_scores(
            tokens, scores, absolute=True
        )
        top_features_scores = top_k(dict_features_scores, len(gt_features) + freedom)
        top_features = top_features_scores[0]
        if is_synthetic:
            pred_features = [
                feature for feature in top_features if not feature.endswith("_N")
            ]
        else:
            pred_features = [
                feature for feature in top_features if feature.endswith("_rf")
            ]
        sim = len(set(pred_features).intersection(gt_features)) / float(n_gt + freedom)
    else:
        sim = -1
    return sim


def get_all_intersection_similarity(
    all_scores, all_tokens, freedom=0, is_synthetic=True
):
    """Gets intersection similarity of all patients based on their shap scores and tokens/features"""
    sims = []
    for i, features in enumerate(all_tokens):
        scores = all_scores[i]
        sim = get_intersection_similarity(
            scores, features, freedom=freedom, is_synthetic=is_synthetic
        )
        sims.append(sim)
    sims2 = [sim for sim in sims if sim != -1]
    avg_sim = sum(sims2) / len(sims2)
    return avg_sim, sims


def get_lstm_scores_tokens(results, dtype="shap_scores"):
    """Gets scores and tokens based on the given type."""
    all_scores = []
    all_tokens = []
    for pid in results.keys():
        imp_df = results[pid]["imp"]
        all_scores.append(imp_df[dtype].tolist())
        tokens = imp_df["token"].tolist()
        all_tokens.append(tokens)
    return all_scores, all_tokens


def save_xgb_results(patients, features, shap_scores, y_true, y_pred, output_path):
    """Save all model training results to file."""
    results = {}
    for i, patient_id in enumerate(patients):
        results[patient_id] = {}
        results[patient_id]["features_xgb"] = features[i]
        results[patient_id]["label"] = y_true[i]
        results[patient_id]["xgb_pred"] = y_pred[i]
        results[patient_id]["xgb_shap"] = shap_scores[i]

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    save_pickle(results, output_path, verbose=False)
    return results


def compute_xgb_shap(
    xgb_model,
    df_train0,
    df_test0,
    uid_colname,
    target_colname,
    explainer=None,
    negative_only=False,
):
    """Compute shap of a dataset for a given xgb model."""
    df_train = df_train0.copy()
    df_test = df_test0.copy()

    feature_names = [
        col
        for col in df_train.columns.tolist()
        if col not in [uid_colname, target_colname]
    ]

    if negative_only:
        X_train = df_train.loc[df_train[target_colname] == 0, feature_names]
    else:
        X_train = df_train[feature_names]
    X_test = df_test[feature_names]

    if explainer is None:
        explainer = shap.TreeExplainer(xgb_model, X_train)
    shap_scores = explainer.shap_values(X_test).tolist()
    features = [feature_names[:]] * X_test.shape[0]
    patients = df_test[uid_colname].tolist()

    return ((features, shap_scores, patients), explainer)


def get_xgb_features_and_global_shap(results):
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
    return df_shap


def get_xgb_shap_values_and_features(all_shap):
    """Get XGB shap values and features separately."""
    all_features = []
    all_scores = []
    for pat_id, shap_info in all_shap.items():
        if not all_features:
            all_features = shap_info["features_xgb"]
        all_scores.append(shap_info["xgb_shap"])
    all_scores = np.array(all_scores)
    return all_features, all_scores
