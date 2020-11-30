#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, classification_report
import warnings


def print_report(labels, preds, title, target_names):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = classification_report(labels, preds, target_names=target_names, digits=4)
    print(f'{title}\n\n{s}')

    
def scores_v1(labels, scores, score_threshold, pct=5):
    th = np.percentile(scores, 100 - pct, axis=0)
    fil = scores > th
    for i in range(labels.shape[1]):
        l, p = labels[:, i][fil[:, i]], scores[:, i][fil[:, i]] > score_threshold
        yield precision_score(l, p), recall_score(l, p)

        
def scores_v2(labels, scores, pct=5):
    th = np.percentile(scores, 100 - pct, axis=0)
    fil = scores > th
    for i in range(labels.shape[1]):
        label_i, pred_i = labels[:, i], scores[:, i] > th[i]
        yield precision_score(label_i, pred_i), recall_score(label_i, pred_i)

        
def compute_single_metric(labels, scores):
    """Compute the single metric we want to optimize: macro averaged auc. Returns -1 if auc cannot be computed."""
    try:
        return roc_auc_score(labels, scores, average='macro')
    except ValueError as e:
        print(e)
        return -1
    

def compute_metrics(labels, scores, target_names=None, risk_list=[5, 2, 1, .5, .25]):
    """Compute metrics according to CMS specs.
    
    labels          : the true binary labels, numpy array of shape (n_samples, n_classes)
    scores          : the scores given by the model for each label, numpy array of same shape
    target_names    : list of target names
    risk_list       : list of floats representing the percentage of high score patients we want to consider positive (eg [0.5, 2.5] or [])
    
    return          : dataframe containing the metrics
    
    """
    if len(labels.shape) == 1:
        labels = labels.reshape(-1, 1)
        scores = scores.reshape(-1, 1)
    n_classes = labels.shape[1]
    if target_names is None:
        target_names = [str(i) for i in range(n_classes)]
    elif type(target_names) is str:
        target_names = [target_names]
    if n_classes == 1:
        # duplicate the class to avoid the one class special case in sklearn
        labels = np.concatenate([labels, labels], axis=1)
        scores = np.concatenate([scores, scores], axis=1)
        target_names = target_names * 2
    df = pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df['auroc'] = roc_auc_score(labels, scores, average=None)
        df['avgpr'] = average_precision_score(labels, scores, average=None)
        for risk in risk_list:
            df[f'precis_{risk}% recall_{risk}%'.split()] = pd.DataFrame(scores_v2(labels, scores, risk))
    error = scores - labels
    df['calib_mean'] = np.mean(error,    axis=0)
    df['calib_mse']  = np.mean(error**2, axis=0)
    df.index = target_names
    return df.iloc[0:n_classes, :]  # remove duplication


def randn(m, n):
    if n == 0:
        return np.random.randn(m)
    return np.random.randn(m, n)
    
    
def test_metrics():
    print()
    for nc in [0, 1, 2]:
        labels = randn(1000, nc) > 0
        scores = randn(1000, nc)
        r = compute_metrics(labels, scores)
        print(r.T)
        assert r.shape == (max(nc, 1), 14)
        
